"""Utility functions for interpretability metrics.

This module provides helper functions used across different interpretability
metrics to avoid code duplication and improve maintainability.
"""

from enum import IntEnum
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from pyhealth.models import BaseModel


class SampleClass(IntEnum):
    """Classification of how a sample should be treated during evaluation.

    Attributes:
        POSITIVE: Evaluate sample with attributions as-is.
            Used for predicted positive class in binary, or all
            samples in multiclass/multilabel.
        NEGATIVE: Evaluate sample with negated attributions.
            Used for predicted negative class in binary classification,
            where feature importance is measured relative to the
            predicted class (class 0).
        IGNORE: Exclude sample from evaluation entirely.
            Useful for filtering out low-confidence predictions or
            samples that should not contribute to the metric.
    """

    POSITIVE = 1
    NEGATIVE = -1
    IGNORE = 0


# Type alias for sample filter functions.
# Signature: (class_probs, classifier_type) -> sample_classes
# class_probs has shape (batch_size,) and contains the probability for each
# sample's predicted class (sigmoid/softmax output with target_id applied).
SampleFilterFn = Callable[[torch.Tensor, str], torch.Tensor]


def threshold_sample_filter(threshold: float = 0.5) -> SampleFilterFn:
    """Create a filter based on a probability threshold.

    For binary and multilabel classifiers, samples whose predicted-class
    probability is at or above ``threshold`` are marked POSITIVE; all
    others are marked IGNORE.

    For multiclass classifiers, all samples are marked POSITIVE
    (the argmax class always has a well-defined probability).

    Args:
        threshold: Minimum predicted-class probability to include
            the sample. Default: 0.5.

    Returns:
        A sample filter function.

    Examples:
        >>> # Create a filter that ignores uncertain predictions
        >>> my_filter = threshold_sample_filter(0.7)
        >>> evaluator = Evaluator(model, sample_filter=my_filter)
    """

    def filter_fn(
        class_probs: torch.Tensor,
        classifier_type: str,
    ) -> torch.Tensor:
        batch_size = class_probs.shape[0]
        result = torch.full(
            (batch_size,),
            SampleClass.POSITIVE,
            dtype=torch.long,
            device=class_probs.device,
        )
        if classifier_type in ("binary", "multilabel"):
            result[class_probs < threshold] = SampleClass.IGNORE
        return result

    return filter_fn


def get_model_predictions(
    model: BaseModel,
    inputs: Dict[str, torch.Tensor],
    classifier_type: str,
    sample_filter: Optional[SampleFilterFn] = None,
    sample_class: Optional[torch.Tensor] = None,
    target_class_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get model predictions, probabilities, and class-specific probabilities.

    Args:
        model: PyHealth BaseModel that returns dict with 'y_prob' or 'logit'
        inputs: Model inputs dict
        classifier_type: One of 'binary', 'multiclass', 'multilabel', 'unknown'
        target_class_idx: (Optional) Pre-computed target class indices, this would ensure ablated runs
                      are consistent with original predictions. If None, will compute from model outputs.
        sample_filter: A callable that classifies each sample for evaluation.
            Signature: (class_probs, classifier_type) -> sample_classes
            where class_probs has shape (batch_size,) and contains the
            probability for the predicted class (sigmoid/softmax output
            with target class already applied), and sample_classes is a
            tensor of SampleClass values.

    Returns:
        Tuple of (y_prob, target_class_idx, sample_classes):
        - y_prob: All class probabilities
            - Binary: shape (batch_size, 1), values are P(class=1)
            - Multiclass: shape (batch_size, num_classes)
        - target_class_idx: Target class indices, shape (batch_size,)
        - sample_classes: SampleClass values for each sample, shape (batch_size,)
    """
    with torch.no_grad():
        outputs = model(**inputs)

        # Get probabilities
        if "y_prob" in outputs:
            y_prob = outputs["y_prob"]
        elif "logit" in outputs:
            logits = outputs["logit"]
            if classifier_type in ["binary", "multilabel"]:
                y_prob = torch.sigmoid(logits)
            else:
                y_prob = F.softmax(logits, dim=-1)
        else:
            raise ValueError("Model output must contain 'y_prob' or 'logit'")

        # Ensure at least 2D
        if y_prob.dim() == 1:
            y_prob = y_prob.unsqueeze(-1)

        if target_class_idx is None:
            target_class_idx = torch.argmax(y_prob, dim=-1)

        y_prob = y_prob[target_class_idx]

        # Apply sample filter
        if sample_class is None:
            if sample_filter is None:
                raise ValueError("sample_filter must be provided if sample_class is None")
            sample_class = sample_filter(y_prob, classifier_type)
        
        y_prob[sample_class == SampleClass.IGNORE] = 0.0  # Set ignored samples' probs to 0
        target_class_idx[sample_class == SampleClass.IGNORE] = 0  # Mark ignored samples with invalid class index
        
        return y_prob, target_class_idx, sample_class



