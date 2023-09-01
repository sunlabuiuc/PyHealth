from typing import Dict, List, Optional

import numpy as np

from pyhealth.metrics.fairness_utils import disparate_impact, statistical_parity_difference

def fairness_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attributes: np.ndarray,
    favorable_outcome: int = 1,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes metrics for binary classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - disparate_impact:
        - statistical_parity_difference:

    If no metrics are disparate_impact, and statistical_parity_difference are computed by default.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        sensitive_attributes: Sensitive attributes of shape (n_samples,) where 1 is the protected group and 0 is the unprotected group.
        favorable_outcome: Label value which is considered favorable (i.e. "positive").
        metrics: List of metrics to compute. Default is ["disparate_impact", "statistical_parity_difference"].
        threshold: Threshold for binary classification. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.
    """
    if metrics is None:
        metrics = ["disparate_impact", "statistical_parity_difference"]

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "disparate_impact":
            output[metric] = disparate_impact(sensitive_attributes, y_pred, favorable_outcome)
        elif metric == "statistical_parity_difference":
            output[metric] = statistical_parity_difference(sensitive_attributes, y_pred, favorable_outcome)
        else:
            raise ValueError(f"Unknown metric for fairness: {metric}")
    return output

