"""Base class for removal-based interpretability metrics.

This module provides the abstract base class for removal-based faithfulness
metrics like Comprehensiveness and Sufficiency.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from pyhealth.models import BaseModel

from .utils import (
    SampleClass,
    SampleFilterFn,
    get_model_predictions,
)


class RemovalBasedMetric(ABC):
    """Abstract base class for removal-based interpretability metrics.

    This class provides common functionality for computing faithfulness metrics
    by removing or retaining features based on their importance scores.

    Examples:
        >>> from pyhealth.metrics.interpretability import (
        ...     ComprehensivenessMetric,
        ...     RemovalBasedMetric,
        ... )
        >>> issubclass(ComprehensivenessMetric, RemovalBasedMetric)
        True

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
        sample_filter: A callable that classifies each sample for evaluation.
            Signature: (class_probs, classifier_type) -> sample_classes
            where class_probs has shape (batch_size,) and contains the
            probability for the predicted class (sigmoid/softmax output
            with target class already applied), and sample_classes is a
            tensor of SampleClass values.
            - SampleClass.POSITIVE: evaluate with attributions as-is
            - SampleClass.NEGATIVE: evaluate with negated attributions
            - SampleClass.IGNORE: exclude from evaluation
    """

    # Integer dtypes treated as discrete (categorical code indices), which are
    # always zero-ablated to the padding index rather than mean/noise-ablated.
    _DISCRETE_DTYPES = (torch.long, torch.int, torch.int32, torch.int64)

    def __init__(
        self,
        model: BaseModel,
        percentages: List[float] = [1, 5, 10, 20, 50],
        ablation_strategy: str = "zero",
        *,
        sample_filter: SampleFilterFn,
    ):
        self.model = model
        self.percentages = percentages
        self.ablation_strategy = ablation_strategy
        self._sample_filter = sample_filter
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
            print("  - Evaluates both positive and negative predictions")
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

    def _empty_row_safety_repair(
        self,
        x_values: torch.Tensor,
        ablated_values: torch.Tensor,
    ) -> torch.Tensor:
        """Restore one original element in any fully ablated sequence.

        Complete ablation (an all-zero row) breaks downstream models like
        StageNet and Transformer: every embedding collapses to the padding
        vector (``padding_idx=0``), the padding/attention mask becomes all
        zeros, and ``get_last_visit()`` indexes with ``-1`` on an empty
        sequence. To avoid this, each sequence that was fully ablated has
        its first originally non-zero element restored.

        Args:
            x_values: The original (un-ablated) values tensor.
            ablated_values: The ablated values tensor to repair.

        Returns:
            A copy of ``ablated_values`` with at least one non-zero element
            preserved per sequence (where the original had one).
        """
        repaired = ablated_values.clone()
        for b in range(repaired.shape[0]):
            if repaired[b].sum() == 0:
                non_zero_mask = x_values[b] != 0
                if non_zero_mask.any():
                    # Keep first non-zero element
                    first_idx = non_zero_mask.nonzero()[0]
                    repaired[b][tuple(first_idx)] = x_values[b][
                        tuple(first_idx)
                    ]
        return repaired

    def _apply_ablation_to_values(
        self,
        x_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Ablate a single values tensor according to its dtype and strategy.

        Discrete (integer code) features are always zero-ablated regardless
        of ``self.ablation_strategy``: masked positions are set to the
        padding index 0, since "mean"/"noise" are meaningless for
        categorical code indices. Continuous features honor
        ``self.ablation_strategy`` ("zero", "mean", or "noise").

        Args:
            x_values: The values tensor to ablate.
            mask: Binary mask where 1 marks positions to ablate.

        Returns:
            The ablated values tensor (same shape and dtype as
            ``x_values``).
        """
        if x_values.dtype in self._DISCRETE_DTYPES:
            # Discrete features (codes) are always zero-ablated regardless of
            # self.ablation_strategy: multiply by (1-mask) so that
            # mask=1 (ablate) -> 0 (padding index) and mask=0 (keep) ->
            # original value. "mean"/"noise" don't apply to code indices.
            ablated_values = x_values * (1 - mask).long()

            ablated_values = self._empty_row_safety_repair(
                x_values, ablated_values
            )
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

        return ablated_values

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

                    ablated_values = self._apply_ablation_to_values(x_values, mask)

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

            ablated_inputs[key] = self._apply_ablation_to_values(x, mask)

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
                (batch_size,). For binary classifiers, negative-class samples
                are scored from the class-0 perspective.

        Note:
            For binary classifiers, all samples are evaluated
            (both positive and negative predictions). For class 0
            predictions, attributions are negated internally so that
            feature importance is measured relative to the predicted
            class.
        """
        # Get original predictions (returns 3 values)
        y_probs, target_class_idx, sample_class = get_model_predictions(
            model=self.model,
            inputs=inputs,
            classifier_type=self.classifier_type,
            sample_filter=self._sample_filter,
        )
        
        batch_size = y_probs.shape[0]

        # Validity mask: IGNORE samples excluded
        val_mask = sample_class != SampleClass.IGNORE

        # For NEGATIVE samples, negate attributions so that
        # "top features" become those most important for the predicted
        # class (features with low class-1 attribution support class 0).
        neg_mask = sample_class == SampleClass.NEGATIVE
        if neg_mask.any():
            attributions = {
                key: torch.where(
                    neg_mask.view(-1, *([1] * (attr.dim() - 1))),
                    -attr,
                    attr,
                )
                for key, attr in attributions.items()
            }

        # Debug output (if requested and returning per percentage)
        if debug and return_per_percentage:
            print(f"\n{'='*80}")
            print(f"[DEBUG] compute for {self.__class__.__name__}")
            print(f"{'='*80}")
            print(f"Batch size: {batch_size}")
            print(f"Classifier type: {self.classifier_type}")

            if self.classifier_type == "binary":
                print(f"Positive class samples: {(sample_class == SampleClass.POSITIVE).sum().item()}")
                print(f"Negative class samples: {(sample_class == SampleClass.NEGATIVE).sum().item()}")
                print("NOTE: Evaluating BOTH positive and negative predictions")

            print("\nOriginal probs for predicted class:")
            for i, prob in enumerate(y_probs):
                cls = target_class_idx[i].item()
                print(f"  Sample {i} [class={cls}]: {prob.item():.6f}")

        # Store results per percentage
        if return_per_percentage:
            results = {}
        else:
            # Accumulator for averaging
            metric_scores = torch.zeros(batch_size, device=y_probs.device)

        # Compute metrics across all percentages
        for percentage in self.percentages:
            # Compute masks for this percentage
            masks = self._compute_threshold_and_mask(attributions, percentage)

            # Create ablated inputs (subclass-specific)
            ablated_inputs = self._create_ablated_inputs(inputs, masks)

            # Get predictions on ablated inputs
            ablated_probs, _, _ = get_model_predictions(
                model=self.model,
                inputs=ablated_inputs,
                target_class_idx=target_class_idx, # Use same predicted classes from original to avoid shifts
                sample_class=sample_class, # Use same sample classes to ensure consistency
                classifier_type=self.classifier_type,
            )

            # Compute probability drop
            original_class_probs = y_probs.clone()
            original_class_probs[neg_mask] = -original_class_probs[neg_mask]
            
            ablated_class_probs = ablated_probs.clone()
            ablated_class_probs[neg_mask] = -ablated_class_probs[neg_mask]
            
            prob_drop = torch.zeros(batch_size, device=y_probs.device)
            prob_drop[val_mask] = (
                original_class_probs[val_mask] - ablated_class_probs[val_mask]
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
                        cls = target_class_idx[i].item()
                        print(f"  Sample {i} [class={cls}]: {prob.item():.6f}")
                else:
                    print("\nAblated probabilities (all classes):")
                    for i, probs in enumerate(ablated_probs):
                        print(f"  Sample {i}: {probs.tolist()}")

                print("\nProbability drops (original - ablated):")
                for i, drop in enumerate(prob_drop):
                    orig = original_class_probs[i].item()
                    abl = ablated_class_probs[i].item()
                    cls = target_class_idx[i].item()
                    print(
                        f"  Sample {i} [class={cls}]: {drop.item():.6f} "
                        f"({orig:.6f} - {abl:.6f})"
                    )

                # Check for unexpected negative values
                evaluated_drops = prob_drop[val_mask]
                negative_drop_mask = evaluated_drops < 0
                if negative_drop_mask.any():
                    neg_count = negative_drop_mask.sum().item()
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
                results[percentage] = prob_drop # type: ignore
            else:
                # Accumulate for averaging
                metric_scores = metric_scores + prob_drop # type: ignore

        # Return appropriate format
        if return_per_percentage:
            return results # type: ignore
        else:
            # Average across percentages
            metric_scores = metric_scores / len(self.percentages) # type: ignore
            return metric_scores, val_mask
