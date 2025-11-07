"""Removal-based interpretability metrics (batch-level).

This module provides batch-level removal-based faithfulness metrics for
evaluating feature attribution methods: Comprehensiveness and Sufficiency.

Key Design:
-----------
- Works at the batch level (not dataset level)
- Object-oriented API with base and derived classes
- Supports different ablation strategies (zero, mean, noise)
- Returns per-sample scores for detailed analysis

The metrics are based on the paper:
    "Towards A Rigorous Science of Interpretable Machine Learning"
    by Finale Doshi-Velez and Been Kim, 2017
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


class RemovalBasedMetric(ABC):
    """Abstract base class for removal-based interpretability metrics.

    This class provides common functionality for computing faithfulness metrics
    by removing or retaining features based on their importance scores.

    Args:
        model: PyTorch model that accepts **kwargs and returns dict with
            'y_prob' or 'logit'.
        percentages: List of percentages to evaluate at.
            Default: [1, 5, 10, 20, 50].
        ablation_strategy: How to ablate features. Options:
            - 'zero': Set ablated features to 0
            - 'mean': Set ablated features to feature mean across batch
            - 'noise': Add Gaussian noise to ablated features
            Default: 'zero'.
    """

    def __init__(
        self,
        model: nn.Module,
        percentages: List[float] = [1, 5, 10, 20, 50],
        ablation_strategy: str = "zero",
        positive_threshold: float = 0.5,
    ):
        self.model = model
        self.percentages = percentages
        self.ablation_strategy = ablation_strategy
        self._positive_threshold = positive_threshold
        self.model.eval()

        # Detect classifier type from model
        self._detect_classifier_type()

    def _detect_classifier_type(self):
        """Detect classifier type from model's output schema.

        Sets self.classifier_type and self.num_classes based on model.

        Expected model types:
            - PyHealth BaseModel with dataset.output_schema
            - Custom models following same interface

        Classifier types:
            - binary: Binary classification, output [batch, 1]
            - multiclass: Multi-class classification, output [batch, C]
            - multilabel: Multi-label classification, output [batch, L]
        """
        # Check if model is a PyHealth BaseModel with dataset
        if not hasattr(self.model, "dataset") or not hasattr(
            self.model.dataset, "output_schema"
        ):
            self.classifier_type = "unknown"
            self.num_classes = None
            print("[RemovalBasedMetric] WARNING: Cannot detect type")
            print("  - Model missing dataset.output_schema")
            print("  - Expected: PyHealth BaseModel or compatible")
            return

        # Get output schema
        output_schema = self.model.dataset.output_schema
        if len(output_schema) == 0:
            self.classifier_type = "unknown"
            self.num_classes = None
            print("[RemovalBasedMetric] WARNING: Empty output_schema")
            return

        # Use first label key (most common case)
        label_key = list(output_schema.keys())[0]
        schema_entry = output_schema[label_key]

        # Use BaseModel's _resolve_mode if available, else manual check
        if hasattr(self.model, "_resolve_mode"):
            try:
                mode = self.model._resolve_mode(schema_entry)
            except Exception as e:
                self.classifier_type = "unknown"
                self.num_classes = None
                print(f"[RemovalBasedMetric] WARNING: {e}")
                return
        else:
            # Fallback: check string or class name
            if isinstance(schema_entry, str):
                mode = schema_entry.lower()
            elif hasattr(schema_entry, "__name__"):
                mode = schema_entry.__name__.lower()
            else:
                mode = "unknown"

        # Set classifier type based on mode
        if mode == "binary":
            self.classifier_type = "binary"
            self.num_classes = 2
            print("[RemovalBasedMetric] Detected BINARY classifier")
            print("  - Output shape: [batch, 1] with P(class=1)")
            print("  - Only evaluates positive predictions (>=threshold)")
        elif mode == "multiclass":
            self.classifier_type = "multiclass"
            # Get num_classes from processor
            if hasattr(self.model.dataset, "output_processors"):
                processor = self.model.dataset.output_processors.get(label_key)
                self.num_classes = processor.size() if processor else None
            else:
                self.num_classes = None
            print("[RemovalBasedMetric] Detected MULTICLASS classifier")
            print(f"  - Num classes: {self.num_classes}")
            print(f"  - Output shape: [batch, {self.num_classes}]")
        elif mode == "multilabel":
            self.classifier_type = "multilabel"
            # Get num_labels from processor
            if hasattr(self.model.dataset, "output_processors"):
                processor = self.model.dataset.output_processors.get(label_key)
                self.num_classes = processor.size() if processor else None
            else:
                self.num_classes = None
            print("[RemovalBasedMetric] Detected MULTILABEL classifier")
            print(f"  - Num labels: {self.num_classes}")
            print(f"  - Output shape: [batch, {self.num_classes}]")
            print("  - NOTE: Multilabel support not fully tested")
        else:
            self.classifier_type = "unknown"
            self.num_classes = None
            print("[RemovalBasedMetric] WARNING: Unknown classifier")
            print(f"  - Mode detected: {mode}")

    @abstractmethod
    def _create_ablated_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create ablated version of inputs based on masks.

        Args:
            inputs: Original model inputs
            masks: Binary masks (1=keep/remove depending on metric)

        Returns:
            Ablated inputs with same structure as inputs
        """
        pass

    def _get_model_predictions(
        self, inputs: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get model predictions and probabilities.

        Args:
            inputs: Model inputs

        Returns:
            Tuple of (probabilities, predicted_classes)
            - For binary: probabilities shape (batch_size, 1), values are P(class=1)
            - For multiclass: probabilities shape (batch_size, num_classes)
            - predicted_classes: shape (batch_size,)
        """
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get probabilities
            if "y_prob" in outputs:
                y_prob = outputs["y_prob"]
            elif "logit" in outputs:
                logit = outputs["logit"]
                # For binary: logit is [batch, 1], apply sigmoid
                # For multiclass: logit is [batch, num_classes], apply softmax
                if logit.shape[-1] == 1:
                    y_prob = torch.sigmoid(logit)
                else:
                    y_prob = F.softmax(logit, dim=-1)
            else:
                raise ValueError("Model output must contain 'y_prob' or 'logit'")

            # Ensure at least 2D
            if y_prob.dim() == 1:
                y_prob = y_prob.unsqueeze(1)

            # Get predicted classes based on classifier type
            if self.classifier_type == "binary":
                # For binary: y_prob is [batch, 1] with P(class=1)
                # Predicted class is 1 if P(class=1) >= 0.5, else 0
                predicted_classes = (y_prob.squeeze(-1) >= 0.5).long()
            else:
                # For multiclass/multilabel: use argmax
                predicted_classes = torch.argmax(y_prob, dim=1)

            return y_prob, predicted_classes

    def _compute_threshold_and_mask(
        self,
        attributions: Dict[str, torch.Tensor],
        percentage: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute binary masks for top-percentage features using quantiles.

        Args:
            attributions: Attribution scores
            percentage: Percentage of features to select (e.g., 10 for top 10%)

        Returns:
            Dictionary mapping feature_key to binary mask
            (1 for top-percentage)
        """
        masks = {}

        for key, attr in attributions.items():
            # Compute per-sample masks
            batch_size = attr.shape[0]
            mask = torch.zeros_like(attr)

            # Convert percentage to quantile (e.g., top 10% = 90th percentile)
            # percentage = 10 means top 10%, so quantile = 1 - 0.10 = 0.90
            quantile = 1.0 - (percentage / 100.0)

            for i in range(batch_size):
                attr_sample = attr[i].flatten()

                # Handle edge case: if all values are the same
                if attr_sample.min() == attr_sample.max():
                    # Select approximately the right percentage of features
                    num_features = attr_sample.numel()
                    num_to_select = max(1, int(num_features * percentage / 100.0))
                    mask_flat = torch.zeros_like(attr_sample)
                    mask_flat[:num_to_select] = 1.0
                else:
                    # Compute threshold using quantile
                    # Use "higher" interpolation to be conservative: when the
                    # quantile falls between two values, we use the higher
                    # threshold, ensuring we select at least the target %.
                    # This matches the behavior of topk which includes all
                    # values tied at the boundary.
                    threshold = torch.quantile(
                        attr_sample, quantile, interpolation="higher"
                    )
                    # Create mask for values >= threshold
                    mask_flat = (attr_sample >= threshold).float()

                mask[i] = mask_flat.reshape(attr[i].shape)

            masks[key] = mask

        return masks

    def _apply_ablation(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply ablation strategy to create modified inputs.

        Args:
            inputs: Original inputs
            masks: Binary masks indicating which features to ablate (1=ablate)

        Returns:
            Modified inputs with ablation applied
        """
        ablated_inputs = {}

        for key in inputs.keys():
            x = inputs[key]

            # Handle tuple inputs (e.g., StageNet's (time, values) format)
            if isinstance(x, tuple):
                # Extract values from tuple (typically (time, values))
                if len(x) >= 2:
                    time_info = x[0]
                    x_values = x[1]

                    # If no mask for this key, keep unchanged
                    if key not in masks:
                        ablated_inputs[key] = x
                        continue

                    mask = masks[key]

                    # Check if values are integers (discrete features like codes)
                    is_discrete = x_values.dtype in [
                        torch.long,
                        torch.int,
                        torch.int32,
                        torch.int64,
                    ]

                    # Apply ablation to values part
                    if is_discrete:
                        # For discrete features (codes), multiply by (1-mask)
                        # Where mask=1 (ablate): set to 0 (padding index)
                        # Where mask=0 (keep): preserve original value
                        ablated_values = x_values * (1 - mask).long()

                        # Safety: prevent complete ablation of discrete sequences
                        # Complete ablation (all zeros) causes issues in StageNet:
                        # - All embeddings become zero (padding_idx=0)
                        # - Mask becomes all zeros
                        # - get_last_visit() tries to index with -1
                        # Solution: keep at least one non-zero element
                        for b in range(ablated_values.shape[0]):
                            if ablated_values[b].sum() == 0:
                                non_zero_mask = x_values[b] != 0
                                if non_zero_mask.any():
                                    # Keep first non-zero element
                                    first_idx = non_zero_mask.nonzero()[0]
                                    ablated_values[b][tuple(first_idx)] = x_values[b][
                                        tuple(first_idx)
                                    ]
                    else:
                        # For continuous features, apply standard ablation
                        if self.ablation_strategy == "zero":
                            ablated_values = x_values * (1 - mask)
                        elif self.ablation_strategy == "mean":
                            x_mean = x_values.mean(dim=0, keepdim=True)
                            ablated_values = x_values * (1 - mask) + x_mean * mask
                        elif self.ablation_strategy == "noise":
                            noise = torch.randn_like(x_values) * x_values.std()
                            ablated_values = x_values * (1 - mask) + noise * mask
                        else:
                            raise ValueError(
                                f"Unknown ablation strategy: "
                                f"{self.ablation_strategy}"
                            )

                    # Reconstruct tuple with ablated values
                    ablated_inputs[key] = (time_info, ablated_values) + x[2:]
                else:
                    # Tuple with unexpected length, keep unchanged
                    ablated_inputs[key] = x
                continue

            # Skip non-tensor, non-tuple inputs (like lists, strings)
            if not isinstance(x, torch.Tensor):
                ablated_inputs[key] = x
                continue

            # If no mask for this key, keep unchanged
            if key not in masks:
                ablated_inputs[key] = x.clone()
                continue

            mask = masks[key]

            # Apply ablation strategy
            if self.ablation_strategy == "zero":
                # Set ablated features to 0
                ablated_inputs[key] = x * (1 - mask)

            elif self.ablation_strategy == "mean":
                # Set ablated features to mean across batch
                x_mean = x.mean(dim=0, keepdim=True)
                ablated_inputs[key] = x * (1 - mask) + x_mean * mask

            elif self.ablation_strategy == "noise":
                # Replace ablated features with Gaussian noise
                noise = torch.randn_like(x) * x.std()
                ablated_inputs[key] = x * (1 - mask) + noise * mask

            else:
                raise ValueError(f"Unknown ablation strategy: {self.ablation_strategy}")

        return ablated_inputs

    def compute(
        self,
        inputs: Dict[str, torch.Tensor],
        attributions: Dict[str, torch.Tensor],
        predicted_class: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute metric averaged over all percentages.

        Args:
            inputs: Model inputs (batch)
            attributions: Attribution scores matching input shapes (batch)
            predicted_class: Optional pre-computed predicted classes (batch)

        Returns:
            Tuple of (metric_scores, valid_mask):
            - metric_scores: Raw scores for ALL samples, shape (batch_size,)
            - valid_mask: Binary mask indicating which samples are valid
              (1 for positive predictions in binary, 1 for all in multiclass)

        Note:
            For binary classifiers, the valid_mask indicates samples with
            P(class=1) >= threshold (default 0.5). Use this mask to filter
            scores during averaging or analysis.
        """
        # Get original predictions
        original_probs, pred_classes = self._get_model_predictions(inputs)

        if predicted_class is not None:
            pred_classes = predicted_class

        batch_size = original_probs.shape[0]

        # Initialize metric scores (will compute for all, filter later)
        metric_scores = torch.zeros(batch_size, device=original_probs.device)

        # Create validity mask based on classifier type
        if self.classifier_type == "binary":
            # For binary: valid = P(class=1) >= threshold
            valid_mask = original_probs.squeeze(-1) >= self._positive_threshold
            original_class_probs = original_probs.squeeze(-1)
        else:
            # For multiclass/multilabel: all samples are valid
            valid_mask = torch.ones(
                batch_size, dtype=torch.bool, device=original_probs.device
            )
            # Get probabilities for predicted classes
            original_class_probs = original_probs.gather(
                1, pred_classes.unsqueeze(1)
            ).squeeze(1)

        # Compute metrics across all percentages
        for percentage in self.percentages:
            # Compute masks for this percentage
            masks = self._compute_threshold_and_mask(attributions, percentage)

            # Create ablated inputs (subclass-specific)
            ablated_inputs = self._create_ablated_inputs(inputs, masks)

            # Get predictions on ablated inputs
            ablated_probs, _ = self._get_model_predictions(ablated_inputs)

            # Get probabilities for same predicted classes
            if self.classifier_type == "binary":
                ablated_class_probs = ablated_probs.squeeze(-1)
            else:
                ablated_class_probs = ablated_probs.gather(
                    1, pred_classes.unsqueeze(1)
                ).squeeze(1)

            # Compute probability drop for ALL samples
            prob_drop = original_class_probs - ablated_class_probs

            # Accumulate for ALL samples (will filter during averaging)
            metric_scores = metric_scores + prob_drop

        # Average across percentages for ALL samples
        metric_scores = metric_scores / len(self.percentages)

        return metric_scores, valid_mask

    def compute_detailed(
        self,
        inputs: Dict[str, torch.Tensor],
        attributions: Dict[str, torch.Tensor],
        predicted_class: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> Dict[float, torch.Tensor]:
        """Compute metric for each percentage separately.

        Args:
            inputs: Model inputs (batch)
            attributions: Attribution scores matching input shapes (batch)
            predicted_class: Optional pre-computed predicted classes (batch)
            debug: If True, print detailed probability information

        Returns:
            Dictionary mapping percentage -> scores (batch_size,)

        Note:
            For binary classifiers, only positive class (class=1) samples
            are evaluated. Negative class samples return NaN to indicate
            they should be excluded from analysis.
        """
        # Get original predictions
        original_probs, pred_classes = self._get_model_predictions(inputs)

        if predicted_class is not None:
            pred_classes = predicted_class

        batch_size = original_probs.shape[0]

        # For binary classification, only compute for positive class
        if self.classifier_type == "binary":
            # Create mask for positive class samples
            positive_mask = pred_classes == 1
            num_positive = positive_mask.sum().item()
            num_negative = (~positive_mask).sum().item()

            # Original probs for positive class
            original_class_probs = original_probs.squeeze(-1)

        else:
            # For multiclass/multilabel: compute for all samples
            positive_mask = torch.ones(
                batch_size, dtype=torch.bool, device=original_probs.device
            )
            num_positive = batch_size
            num_negative = 0

            # Get probabilities for predicted classes
            original_class_probs = original_probs.gather(
                1, pred_classes.unsqueeze(1)
            ).squeeze(1)

        if debug:
            print(f"\n{'='*80}")
            print(f"[DEBUG] compute_detailed for {self.__class__.__name__}")
            print(f"{'='*80}")
            print(f"Batch size: {batch_size}")
            print(f"Classifier type: {self.classifier_type}")

            if self.classifier_type == "binary":
                print(f"Positive class samples: {num_positive}")
                print(f"Negative class samples: {num_negative}")
                print("NOTE: Only computing metrics for POSITIVE class")
                print("      Negative class samples will return NaN")

            print(f"Original probs shape: {original_probs.shape}")
            print(f"Predicted classes: {pred_classes.tolist()}")

            if self.classifier_type == "binary":
                print("\nOriginal probabilities P(class=1):")
                for i, prob in enumerate(original_probs):
                    status = "EVAL" if positive_mask[i] else "SKIP"
                    print(f"  Sample {i} [{status}]: {prob.item():.6f}")
            else:
                print("\nOriginal probabilities (all classes):")
                for i, probs in enumerate(original_probs):
                    print(f"  Sample {i}: {probs.tolist()}")

            print("\nOriginal probs for predicted class:")
            for i, prob in enumerate(original_class_probs):
                if self.classifier_type == "binary":
                    status = "EVAL" if positive_mask[i] else "SKIP"
                    print(f"  Sample {i} [{status}]: {prob.item():.6f}")
                else:
                    print(f"  Sample {i}: {prob.item():.6f}")

        results = {}

        for percentage in self.percentages:
            # Initialize probability drops to 0
            prob_drop = torch.zeros(batch_size, device=original_probs.device)

            # Compute masks for this percentage
            masks = self._compute_threshold_and_mask(attributions, percentage)

            # Create ablated inputs (subclass-specific)
            ablated_inputs = self._create_ablated_inputs(inputs, masks)

            # Get predictions on ablated inputs
            ablated_probs, _ = self._get_model_predictions(ablated_inputs)

            # Get probabilities for predicted classes
            if self.classifier_type == "binary":
                ablated_class_probs = ablated_probs.squeeze(-1)
            else:
                ablated_class_probs = ablated_probs.gather(
                    1, pred_classes.unsqueeze(1)
                ).squeeze(1)

            # Compute probability drop only for relevant samples
            prob_drop[positive_mask] = (
                original_class_probs[positive_mask] - ablated_class_probs[positive_mask]
            )

            if debug:
                print(f"\n{'-'*80}")
                print(f"Percentage: {percentage}%")
                print(f"{'-'*80}")
                print(f"Ablated probs shape: {ablated_probs.shape}")

                if self.classifier_type == "binary":
                    print("\nAblated probabilities P(class=1):")
                    for i, prob in enumerate(ablated_probs):
                        status = "EVAL" if positive_mask[i] else "SKIP"
                        print(f"  Sample {i} [{status}]: {prob.item():.6f}")
                else:
                    print("\nAblated probabilities (all classes):")
                    for i, probs in enumerate(ablated_probs):
                        print(f"  Sample {i}: {probs.tolist()}")

                print("\nAblated probs for predicted class:")
                for i, prob in enumerate(ablated_class_probs):
                    if self.classifier_type == "binary":
                        status = "EVAL" if positive_mask[i] else "SKIP"
                        print(f"  Sample {i} [{status}]: {prob.item():.6f}")
                    else:
                        print(f"  Sample {i}: {prob.item():.6f}")

                print("\nProbability drops (original - ablated):")
                for i, drop in enumerate(prob_drop):
                    if torch.isnan(drop):
                        print(f"  Sample {i} [SKIP]: NaN (negative class)")
                    else:
                        orig = original_class_probs[i].item()
                        abl = ablated_class_probs[i].item()
                        print(
                            f"  Sample {i} [EVAL]: {drop.item():.6f} "
                            f"({orig:.6f} - {abl:.6f})"
                        )

                # Check for unexpected negative values (only in evaluated)
                evaluated_drops = prob_drop[positive_mask]
                neg_mask = evaluated_drops < 0
                if neg_mask.any():
                    neg_count = neg_mask.sum().item()
                    print(f"\n⚠ WARNING: {neg_count} negative detected!")
                    print("  (among evaluated positive class samples)")
                    print("  Negative values mean ablation INCREASED " "confidence,")
                    print("  which suggests:")
                    if self.__class__.__name__ == "ComprehensivenessMetric":
                        print("    - Removed features were HARMING " "predictions")
                        print("    - Attribution may have opposite signs")
                    else:  # SufficiencyMetric
                        print("    - Kept features performed WORSE than " "full")
                        print("    - Attribution quality may be poor")

            results[percentage] = prob_drop

        return results


class ComprehensivenessMetric(RemovalBasedMetric):
    """Comprehensiveness metric for interpretability evaluation.

    Measures the drop in predicted class probability when important features
    are REMOVED (ablated). Higher scores indicate more faithful
    interpretations.

    The metric is computed as:
        COMP = (1/|B|) × Σ[p_c(x)(x) - p_c(x)(x \\ x:q%)]
                        q∈B

    Where:
        - x is the original input
        - x:q% are the top q% most important features
        - x \\ x:q% is input with top q% features removed (ablated)
        - p_c(x)(·) is predicted probability for original predicted class
        - B is the set of percentages (default: {1, 5, 10, 20, 50})

    Examples:
        >>> import torch
        >>> from pyhealth.models import MLP
        >>> from pyhealth.metrics.removal_based import ComprehensivenessMetric
        >>>
        >>> # Assume we have a trained model
        >>> model = MLP(dataset=dataset)
        >>>
        >>> # Initialize metric
        >>> comp = ComprehensivenessMetric(model)
        >>>
        >>> # Prepare inputs and attributions
        >>> inputs = {'conditions': torch.randn(32, 50)}
        >>> attributions = {'conditions': torch.randn(32, 50)}
        >>>
        >>> # Compute metric
        >>> scores = comp.compute(inputs, attributions)
        >>> print(f"Mean comprehensiveness: {scores.mean():.3f}")
        Mean comprehensiveness: 0.234
        >>>
        >>> # Get detailed scores per percentage
        >>> detailed = comp.compute_detailed(inputs, attributions)
        >>> for pct, scores in detailed.items():
        ...     print(f"  {pct}%: {scores.mean():.3f}")
          1%: 0.045
          5%: 0.123
          10%: 0.234
          20%: 0.345
          50%: 0.456
    """

    def _create_ablated_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create ablated inputs by REMOVING (ablating) important features.

        For comprehensiveness, mask==1 indicates important features to remove.

        Args:
            inputs: Original model inputs
            masks: Binary masks (1=remove, 0=keep)

        Returns:
            Ablated inputs with important features removed
        """
        return self._apply_ablation(inputs, masks)


class SufficiencyMetric(RemovalBasedMetric):
    """Sufficiency metric for interpretability evaluation.

    Measures the drop in predicted class probability when ONLY important
    features are KEPT (all others removed). Lower scores indicate more
    faithful interpretations.

    The metric is computed as:
        SUFF = (1/|B|) × Σ[p_c(x)(x) - p_c(x)(x:q%)]
                        q∈B

    Where:
        - x is the original input
        - x:q% are the top q% most important features (all others removed)
        - p_c(x)(·) is predicted probability for original predicted class
        - B is the set of percentages (default: {1, 5, 10, 20, 50})

    Examples:
        >>> import torch
        >>> from pyhealth.models import MLP
        >>> from pyhealth.metrics.removal_based import SufficiencyMetric
        >>>
        >>> # Assume we have a trained model
        >>> model = MLP(dataset=dataset)
        >>>
        >>> # Initialize metric
        >>> suff = SufficiencyMetric(model)
        >>>
        >>> # Prepare inputs and attributions
        >>> inputs = {'conditions': torch.randn(32, 50)}
        >>> attributions = {'conditions': torch.randn(32, 50)}
        >>>
        >>> # Compute metric
        >>> scores = suff.compute(inputs, attributions)
        >>> print(f"Mean sufficiency: {scores.mean():.3f}")
        Mean sufficiency: 0.089
        >>>
        >>> # Get detailed scores per percentage
        >>> detailed = suff.compute_detailed(inputs, attributions)
        >>> for pct, scores in detailed.items():
        ...     print(f"  {pct}%: {scores.mean():.3f}")
          1%: 0.234
          5%: 0.178
          10%: 0.089
          20%: 0.045
          50%: 0.012
    """

    def _create_ablated_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create ablated inputs by KEEPING only important features.

        For sufficiency, mask==1 indicates important features to keep.
        We invert the mask to ablate everything else.

        Args:
            inputs: Original model inputs
            masks: Binary masks (1=keep, 0=remove)

        Returns:
            Ablated inputs with only important features kept
        """
        # Invert masks: ablate where mask==0 (keep only where mask==1)
        inverted_masks = {key: 1 - mask for key, mask in masks.items()}
        return self._apply_ablation(inputs, inverted_masks)


class Evaluator:
    """High-level interface for evaluating interpretations.

    This class provides a convenient API for computing multiple
    interpretability metrics at once, both on individual batches and
    across entire datasets.

    Args:
        model: PyTorch model to evaluate
        percentages: List of percentages to evaluate at.
            Default: [1, 5, 10, 20, 50].
        ablation_strategy: How to ablate features. Options:
            - 'zero': Set ablated features to 0
            - 'mean': Set ablated features to feature mean across batch
            - 'noise': Add Gaussian noise to ablated features
            Default: 'zero'.
        positive_threshold: Threshold for positive class in binary
            classification. Samples with P(class=1) >= threshold are
            considered valid for evaluation. Default: 0.5.

    Examples:
        >>> from pyhealth.models import StageNet
        >>> from pyhealth.metrics.interpretability.removal_based import Evaluator
        >>>
        >>> # Initialize evaluator
        >>> evaluator = Evaluator(model)
        >>>
        >>> # Evaluate on a single batch
        >>> inputs = {'conditions': torch.randn(32, 50)}
        >>> attributions = {'conditions': torch.randn(32, 50)}
        >>> batch_results = evaluator.evaluate(inputs, attributions)
        >>> for metric, (scores, mask) in batch_results.items():
        >>>     print(f"{metric}: {scores[mask].mean():.4f}")
        >>>
        >>> # Evaluate across entire dataset with an attribution method
        >>> from pyhealth.interpret.methods import IntegratedGradients
        >>> ig = IntegratedGradients(model)
        >>> results = evaluator.evaluate_approach(test_loader, ig)
        >>> print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
        >>> print(f"Sufficiency: {results['sufficiency']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        percentages: List[float] = [1, 5, 10, 20, 50],
        ablation_strategy: str = "zero",
        positive_threshold: float = 0.5,
    ):
        self.model = model
        self.percentages = percentages
        self.ablation_strategy = ablation_strategy
        self.positive_threshold = positive_threshold
        self.metrics = {
            "comprehensiveness": ComprehensivenessMetric(
                model,
                percentages=percentages,
                ablation_strategy=ablation_strategy,
                positive_threshold=positive_threshold,
            ),
            "sufficiency": SufficiencyMetric(
                model,
                percentages=percentages,
                ablation_strategy=ablation_strategy,
                positive_threshold=positive_threshold,
            ),
        }

    def evaluate(
        self,
        inputs: Dict[str, torch.Tensor],
        attributions: Dict[str, torch.Tensor],
        metrics: List[str] = ["comprehensiveness", "sufficiency"],
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate multiple metrics at once.

        Args:
            inputs: Model inputs (batch)
            attributions: Attribution scores (batch)
            metrics: List of metrics to compute. Options:
                ["comprehensiveness", "sufficiency"]

        Returns:
            Dictionary mapping metric name -> (scores, valid_mask)
            - scores: Raw metric scores for all samples
            - valid_mask: Binary mask indicating valid samples to average

        Note:
            For binary classifiers, valid_mask indicates samples with
            P(class=1) >= threshold. Use: scores[valid_mask].mean()
        """
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                scores, valid_mask = self.metrics[metric_name].compute(
                    inputs, attributions
                )
                results[metric_name] = (scores, valid_mask)
        return results

    def evaluate_detailed(
        self,
        inputs: Dict[str, torch.Tensor],
        attributions: Dict[str, torch.Tensor],
        metrics: List[str] = ["comprehensiveness", "sufficiency"],
        debug: bool = False,
    ) -> Dict[str, Dict[float, torch.Tensor]]:
        """Evaluate multiple metrics with per-percentage breakdown.

        Args:
            inputs: Model inputs (batch)
            attributions: Attribution scores (batch)
            metrics: List of metrics to compute
            debug: If True, print detailed debug information

        Returns:
            Nested dictionary:
            {metric_name: {percentage: scores}}

        Example:
            >>> detailed = evaluator.evaluate_detailed(inputs, attributions)
            >>> comp_10_pct = detailed['comprehensiveness'][10]
            >>> suff_20_pct = detailed['sufficiency'][20]
        """
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                results[metric_name] = self.metrics[metric_name].compute_detailed(
                    inputs, attributions, debug=debug
                )
        return results

    def evaluate_approach(
        self,
        dataloader,
        method,
        metrics: List[str] = ["comprehensiveness", "sufficiency"],
    ) -> Dict[str, float]:
        """Evaluate an attribution method across an entire dataset.

        This method computes the average faithfulness metrics for an
        attribution approach across all batches in a dataloader. It
        automatically computes attributions using the provided method
        and evaluates them.

        Args:
            dataloader: PyTorch DataLoader with test/validation data.
                Should yield batches compatible with the model.
            method: Attribution method instance (e.g., IntegratedGradients).
                Must implement the BaseInterpreter interface with an
                attribute(**data) method.
            metrics: List of metrics to compute. Options:
                ["comprehensiveness", "sufficiency"]
                Default: both metrics.

        Returns:
            Dictionary mapping metric names to their average scores
            across the entire dataset. For binary classifiers, only
            positive class (predicted class=1) samples are included
            in the average.

            Example: {'comprehensiveness': 0.345, 'sufficiency': 0.123}

        Note:
            For binary classifiers, negative class (predicted class=0)
            samples are excluded from the average, as ablation metrics
            are not meaningful for the default/null class.

        Examples:
            >>> from pyhealth.interpret.methods import IntegratedGradients
            >>> from pyhealth.metrics.interpretability.removal_based import (
            ...     Evaluator
            ... )
            >>>
            >>> # Initialize evaluator and attribution method
            >>> evaluator = Evaluator(model)
            >>> ig = IntegratedGradients(model, use_embeddings=True)
            >>>
            >>> # Evaluate across test set
            >>> results = evaluator.evaluate_approach(
            ...     test_loader, ig, metrics=["comprehensiveness"]
            ... )
            >>> print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
            >>>
            >>> # Compare multiple methods
            >>> from pyhealth.interpret.methods import CheferRelevance
            >>> chefer = CheferRelevance(model)
            >>> ig_results = evaluator.evaluate_approach(test_loader, ig)
            >>> chefer_results = evaluator.evaluate_approach(
            ...     test_loader, chefer
            ... )
            >>> print("Method Comparison:")
            >>> print(f"  IG Comp: {ig_results['comprehensiveness']:.4f}")
            >>> print(f"  Chefer Comp: "
            ...       f"{chefer_results['comprehensiveness']:.4f}")
        """
        # Get model device
        model_device = next(self.model.parameters()).device

        # Tracking for statistics and debug output
        batch_count = 0
        total_samples = 0
        total_valid = {metric_name: 0 for metric_name in metrics}
        running_sum = {metric_name: 0.0 for metric_name in metrics}

        # Process each batch
        for batch in dataloader:
            batch_count += 1
            # Move batch to model device
            batch_on_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_on_device[key] = value.to(model_device)
                elif isinstance(value, tuple) and len(value) >= 2:
                    # Handle (time, values) tuples
                    time_part = value[0]
                    if time_part is not None and isinstance(time_part, torch.Tensor):
                        time_part = time_part.to(model_device)

                    values_part = value[1]
                    if isinstance(values_part, torch.Tensor):
                        values_part = values_part.to(model_device)

                    batch_on_device[key] = (time_part, values_part) + value[2:]
                else:
                    batch_on_device[key] = value

            # Compute attributions for this batch
            attributions = method.attribute(**batch_on_device)

            # Evaluate metrics on this batch (returns scores and masks)
            batch_results = self.evaluate(
                batch_on_device, attributions, metrics=metrics
            )

            # Accumulate statistics incrementally (no tensor storage)
            for metric_name in metrics:
                scores, valid_mask = batch_results[metric_name]

                # Track statistics efficiently
                batch_size = len(scores)
                num_valid = valid_mask.sum().item()
                total_samples += batch_size
                total_valid[metric_name] += num_valid

                # Update running sum (valid scores only)
                valid_scores_batch = (scores * valid_mask).sum().item()
                running_sum[metric_name] += valid_scores_batch

            # Debug output every 10 batches
            if batch_count % 1 == 0:
                print(f"\n[Batch {batch_count}] Progress update:")
                print(f"  Total samples processed: {total_samples}")

                # Compute running averages from accumulated statistics
                for metric_name in metrics:
                    num_valid_so_far = total_valid[metric_name]
                    if num_valid_so_far > 0:
                        running_avg = running_sum[metric_name] / num_valid_so_far
                        print(
                            f"  {metric_name}: {running_avg:.6f} "
                            f"({num_valid_so_far}/{total_samples} valid)"
                        )
                    else:
                        print(f"  {metric_name}: N/A " f"(no valid samples yet)")

        # Compute final averages from accumulated statistics
        results = {}
        for metric_name in metrics:
            if total_valid[metric_name] > 0:
                # Average = running_sum / total_valid
                results[metric_name] = (
                    running_sum[metric_name] / total_valid[metric_name]
                )
            else:
                # No valid samples
                results[metric_name] = float("nan")

        # Final summary
        print(f"\n{'='*70}")
        print("[FINAL] Dataset evaluation complete:")
        print(f"  Total batches: {batch_count}")
        print(f"  Total samples: {total_samples}")
        for metric_name in metrics:
            num_valid_final = total_valid[metric_name]
            if metric_name in results:
                score = results[metric_name]
                if score == score:  # Not NaN
                    print(
                        f"  {metric_name}: {score:.6f} "
                        f"({num_valid_final}/{total_samples} valid)"
                    )
                else:
                    print(f"  {metric_name}: NaN " f"(no valid samples)")

        # Sanity check warnings
        if "comprehensiveness" in results and "sufficiency" in results:
            comp = results["comprehensiveness"]
            suff = results["sufficiency"]
            if comp == comp and suff == suff:  # Both not NaN
                if comp < 0:
                    print("\n⚠ WARNING: Negative comprehensiveness detected!")
                    print("  - Removing 'important' features INCREASED confidence")
                    print("  - Possible causes:")
                    print("    * Attribution scores may be inverted/wrong")
                    print("    * Features with negative attributions")
                    print("    * Model predictions unstable")

                if suff > comp:
                    print("\n⚠ WARNING: Sufficiency > Comprehensiveness!")
                    print("  - Keeping top features worse than removing them")
                    print("  - This suggests:")
                    print("    * Attribution quality is poor")
                    print("    * Important features not correctly identified")
                    print("    * Consider checking attribution method")

        valid_ratio = sum(total_valid.values()) / (len(metrics) * total_samples)
        if valid_ratio < 0.1:
            print(f"\n⚠ WARNING: Only {valid_ratio*100:.1f}% valid samples")
            print("  - Most predictions are negative class")
            print("  - Consider:")
            print("    * Checking model predictions distribution")
            print("    * Adjusting positive_threshold parameter")
            print("    * Using balanced test set")

        print(f"{'='*70}\n")

        return results


# Functional API (wraps Evaluator for convenience)
def evaluate_approach(
    model: nn.Module,
    dataloader,
    method,
    metrics: List[str] = ["comprehensiveness", "sufficiency"],
    percentages: List[float] = [1, 5, 10, 20, 50],
    ablation_strategy: str = "zero",
    positive_threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate an attribution method across a dataset (functional API).

    This is a convenience function that wraps the Evaluator class for
    simple one-off evaluations. For multiple evaluations with the same
    configuration, consider using the Evaluator class directly for
    better efficiency.

    Args:
        model: PyTorch model to evaluate
        dataloader: PyTorch DataLoader with test/validation data
        method: Attribution method instance (e.g., IntegratedGradients).
            Must implement the BaseInterpreter interface.
        metrics: List of metrics to compute. Options:
            ["comprehensiveness", "sufficiency"]
            Default: both metrics.
        percentages: List of percentages to evaluate at.
            Default: [1, 5, 10, 20, 50].
        ablation_strategy: How to ablate features. Options:
            - 'zero': Set ablated features to 0
            - 'mean': Set ablated features to feature mean across batch
            - 'noise': Add Gaussian noise to ablated features
            Default: 'zero'.
        positive_threshold: Threshold for positive class in binary
            classification. Samples with P(class=1) >= threshold are
            considered valid for evaluation. Default: 0.5.

    Returns:
        Dictionary mapping metric names to their average scores
        across the entire dataset. Averaging uses mask-based filtering
        to include only valid samples (positive predictions for binary).

        Example: {'comprehensiveness': 0.345, 'sufficiency': 0.123}

    Note:
        For binary classifiers, only samples with P(class=1) >= threshold
        are included in the average, as ablation metrics are not
        meaningful for negative predictions.

    Examples:
        >>> from pyhealth.interpret.methods import IntegratedGradients
        >>> from pyhealth.metrics.interpretability import evaluate_approach
        >>>
        >>> # Simple one-off evaluation
        >>> ig = IntegratedGradients(model, use_embeddings=True)
        >>> results = evaluate_approach(
        ...     model, test_loader, ig,
        ...     metrics=["comprehensiveness"],
        ...     percentages=[10, 20, 50]
        ... )
        >>> print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
        >>>
        >>> # Custom threshold for binary classification
        >>> results = evaluate_approach(
        ...     model, test_loader, ig,
        ...     positive_threshold=0.7  # Only evaluate high-confidence
        ... )
        >>>
        >>> # For comparing multiple methods efficiently, use Evaluator:
        >>> from pyhealth.metrics.interpretability import Evaluator
        >>> evaluator = Evaluator(model, percentages=[10, 20, 50])
        >>> ig_results = evaluator.evaluate_approach(test_loader, ig)
        >>> chefer_results = evaluator.evaluate_approach(test_loader, chefer)
    """
    evaluator = Evaluator(
        model,
        percentages=percentages,
        ablation_strategy=ablation_strategy,
        positive_threshold=positive_threshold,
    )
    return evaluator.evaluate_approach(dataloader, method, metrics=metrics)
