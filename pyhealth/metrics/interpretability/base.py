"""Base class for removal-based interpretability metrics.

This module provides the abstract base class for removal-based faithfulness
metrics like Comprehensiveness and Sufficiency.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from pyhealth.models import BaseModel

from .utils import create_validity_mask, get_model_predictions


class RemovalBasedMetric(ABC):
    """Abstract base class for removal-based interpretability metrics.

    This class provides common functionality for computing faithfulness metrics
    by removing or retaining features based on their importance scores.

    Args:
        model: PyHealth BaseModel that accepts **kwargs and returns dict with
            'y_prob' or 'logit'.
        percentages: List of percentages to evaluate at.
            Default: [1, 5, 10, 20, 50].
        ablation_strategy: How to ablate features. Options:
            - 'zero': Set ablated features to 0
            - 'mean': Set ablated features to feature mean across batch
            - 'noise': Add Gaussian noise to ablated features
            Default: 'zero'.
        positive_threshold: Threshold for positive class in binary
            classification. Default: 0.5.
    """

    def __init__(
        self,
        model: BaseModel,
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

                    # Check if values are integers (discrete features)
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

                        # Safety: prevent complete ablation of sequences
                        # Complete ablation (all zeros) causes issues in
                        # StageNet:
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
        return_per_percentage: bool = False,
        debug: bool = False,
    ):
        """Compute metric across percentages.

        Args:
            inputs: Model inputs (batch)
            attributions: Attribution scores matching input shapes (batch)
            predicted_class: Optional pre-computed predicted classes (batch)
            return_per_percentage: If True, return dict mapping
                percentage -> scores. If False (default), return averaged
                score across percentages.
            debug: If True, print detailed probability information (only used
                when return_per_percentage=True)

        Returns:
            If return_per_percentage=False (default):
                Tuple of (metric_scores, valid_mask):
                - metric_scores: Average scores across percentages,
                    shape (batch_size,)
                - valid_mask: Binary mask indicating valid samples

            If return_per_percentage=True:
                Dict[float, torch.Tensor]: Maps percentage -> scores
                (batch_size,). For binary classifiers, negative class
                samples have value 0.

        Note:
            For binary classifiers, the valid_mask indicates samples with
            P(class=1) >= threshold (default 0.5). Use this mask to filter
            scores during averaging or analysis.
        """
        # Get original predictions (returns 3 values)
        original_probs, pred_classes, original_class_probs = get_model_predictions(
            self.model,
            inputs,
            self.classifier_type,
            self._positive_threshold,
        )

        if predicted_class is not None:
            pred_classes = predicted_class

        batch_size = original_probs.shape[0]

        # Create validity mask using helper
        valid_mask = create_validity_mask(
            original_probs,
            self.classifier_type,
            self._positive_threshold,
        )

        # For binary: determine which samples to evaluate
        if self.classifier_type == "binary":
            positive_mask = pred_classes == 1
            num_positive = positive_mask.sum().item()
            num_negative = (~positive_mask).sum().item()
        else:
            positive_mask = torch.ones(
                batch_size, dtype=torch.bool, device=original_probs.device
            )
            num_positive = batch_size
            num_negative = 0

        # Debug output (if requested and returning per percentage)
        if debug and return_per_percentage:
            print(f"\n{'='*80}")
            print(f"[DEBUG] compute for {self.__class__.__name__}")
            print(f"{'='*80}")
            print(f"Batch size: {batch_size}")
            print(f"Classifier type: {self.classifier_type}")

            if self.classifier_type == "binary":
                print(f"Positive class samples: {num_positive}")
                print(f"Negative class samples: {num_negative}")
                print("NOTE: Only computing metrics for POSITIVE class")

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

        # Store results per percentage
        if return_per_percentage:
            results = {}
        else:
            # Accumulator for averaging
            metric_scores = torch.zeros(batch_size, device=original_probs.device)

        # Compute metrics across all percentages
        for percentage in self.percentages:
            # Compute masks for this percentage
            masks = self._compute_threshold_and_mask(attributions, percentage)

            # Create ablated inputs (subclass-specific)
            ablated_inputs = self._create_ablated_inputs(inputs, masks)

            # Get predictions on ablated inputs
            ablated_probs, _, ablated_class_probs = get_model_predictions(
                self.model,
                ablated_inputs,
                self.classifier_type,
                self._positive_threshold,
            )

            # Compute probability drop
            prob_drop = torch.zeros(batch_size, device=original_probs.device)
            prob_drop[positive_mask] = (
                original_class_probs[positive_mask] - ablated_class_probs[positive_mask]
            )

            # Debug output for this percentage
            if debug and return_per_percentage:
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

                print("\nProbability drops (original - ablated):")
                for i, drop in enumerate(prob_drop):
                    if drop == 0 and not positive_mask[i]:
                        print(f"  Sample {i} [SKIP]: " f"0.000000 (negative class)")
                    else:
                        orig = original_class_probs[i].item()
                        abl = ablated_class_probs[i].item()
                        print(
                            f"  Sample {i} [EVAL]: {drop.item():.6f} "
                            f"({orig:.6f} - {abl:.6f})"
                        )

                # Check for unexpected negative values
                evaluated_drops = prob_drop[positive_mask]
                neg_mask = evaluated_drops < 0
                if neg_mask.any():
                    neg_count = neg_mask.sum().item()
                    print(f"\n⚠ WARNING: {neg_count} negative detected!")
                    print("  Negative values mean ablation INCREASED " "confidence,")
                    print("  which suggests:")
                    if self.__class__.__name__ == "ComprehensivenessMetric":
                        print("    - Removed features were HARMING " "predictions")
                        print("    - Attribution may have opposite signs")
                    else:  # SufficiencyMetric
                        print("    - Kept features performed WORSE than full")
                        print("    - Attribution quality may be poor")

            if return_per_percentage:
                results[percentage] = prob_drop
            else:
                # Accumulate for averaging
                metric_scores = metric_scores + prob_drop

        # Return appropriate format
        if return_per_percentage:
            return results
        else:
            # Average across percentages
            metric_scores = metric_scores / len(self.percentages)
            return metric_scores, valid_mask
