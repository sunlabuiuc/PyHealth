"""Utility functions for interpretability metrics.

This module provides helper functions used across different interpretability
metrics to avoid code duplication and improve maintainability.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from pyhealth.models import BaseModel


def get_model_predictions(
    model: BaseModel,
    inputs: Dict[str, torch.Tensor],
    classifier_type: str,
    positive_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get model predictions, probabilities, and class-specific probabilities.

    Args:
        model: PyHealth BaseModel that returns dict with 'y_prob' or 'logit'
        inputs: Model inputs dict
        classifier_type: One of 'binary', 'multiclass', 'multilabel', 'unknown'
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
            pred_classes = (y_prob.squeeze(-1) >= positive_threshold).long()
            # For binary, class_probs is P(class=1)
            class_probs = y_prob.squeeze(-1)
        else:
            # For multiclass/multilabel: argmax
            pred_classes = torch.argmax(y_prob, dim=-1)
            # Gather probabilities for predicted classes
            class_probs = y_prob.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

        return y_prob, pred_classes, class_probs


def create_validity_mask(
    y_prob: torch.Tensor,
    classifier_type: str,
    positive_threshold: float = 0.5,
) -> torch.Tensor:
    """Create a mask indicating which samples are valid for metric computation.

    For binary classifiers, only positive predictions (P(class=1) >= threshold)
    are considered valid. For multiclass/multilabel, all samples are valid.

    Args:
        y_prob: Model probability outputs
        classifier_type: One of 'binary', 'multiclass', 'multilabel'
        positive_threshold: Threshold for binary classification (default: 0.5)

    Returns:
        Boolean tensor of shape (batch_size,) where True indicates valid samples
    """
    batch_size = y_prob.shape[0]

    if classifier_type == "binary":
        # For binary: valid = P(class=1) >= threshold
        return y_prob.squeeze(-1) >= positive_threshold
    else:
        # For multiclass/multilabel: all samples are valid
        return torch.ones(batch_size, dtype=torch.bool, device=y_prob.device)
