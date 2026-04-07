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
    pred_classes: Optional[torch.Tensor] = None,
    positive_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get model predictions, probabilities, and class-specific probabilities.

    Args:
        model: PyHealth BaseModel that returns dict with 'y_prob' or 'logit'
        inputs: Model inputs dict
        classifier_type: One of 'binary', 'multiclass', 'multilabel', 'unknown'
        pred_classes: (Optional) Pre-computed predicted classes, this would ensure ablated runs
                      are consistent with original predictions. If None, will compute from model outputs.
        positive_threshold: Threshold for binary classification (default: 0.5)

    Returns:
        Tuple of (y_prob, pred_classes, class_probs):
        - y_prob: All class probabilities
            - Binary: shape (batch_size, 1), values are P(class=1)
            - Multiclass: shape (batch_size, num_classes)
        - pred_classes: Predicted class indices, shape (batch_size,)
        - class_probs: Probability for each sample's predicted class,
            shape (batch_size,)
    """
    with torch.no_grad():
        outputs = model(**inputs)

        # Get probabilities
        if "y_prob" in outputs:
            y_prob = outputs["y_prob"]
        elif "logit" in outputs:
            logits = outputs["logit"]
            if classifier_type == "binary":
                y_prob = torch.sigmoid(logits)
            else:
                y_prob = F.softmax(logits, dim=-1)
        else:
            raise ValueError("Model output must contain 'y_prob' or 'logit'")

        # Ensure at least 2D
        if y_prob.dim() == 1:
            y_prob = y_prob.unsqueeze(-1)

        # Get predicted classes based on classifier type
        if classifier_type == "binary":
            # For binary: class 1 if P(class=1) >= threshold, else 0
            pred_classes = (y_prob.squeeze(-1) >= positive_threshold).long() if pred_classes is None else pred_classes
            # For binary, class_probs is P(predicted_class)
            # class 1: P(class=1), class 0: 1 - P(class=1)
            p1 = y_prob.squeeze(-1)
            class_probs = torch.where(pred_classes == 1, p1, 1 - p1)
        else:
            # For multiclass/multilabel: argmax
            pred_classes = torch.argmax(y_prob, dim=-1) if pred_classes is None else pred_classes
            # Gather probabilities for predicted classes
            class_probs = y_prob.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        assert pred_classes is not None, "pred_classes should have been set either by input or computation."

        return y_prob, pred_classes, class_probs



