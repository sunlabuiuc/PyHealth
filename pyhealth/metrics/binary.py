from typing import Dict, List, Optional

import numpy as np
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
import pyhealth.metrics.prediction_set as pset


def binary_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.5,
    y_predset: np.ndarray | None = None,
) -> Dict[str, float]:
    """Computes metrics for binary classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - pr_auc: area under the precision-recall curve
        - roc_auc: area under the receiver operating characteristic curve
        - accuracy: accuracy score
        - balanced_accuracy: balanced accuracy score (usually used for imbalanced
          datasets)
        - f1: f1 score
        - precision: precision score
        - recall: recall score
        - cohen_kappa: Cohen's kappa score
        - jaccard: Jaccard similarity coefficient score
        - ECE: Expected Calibration Error (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_binary`.
        - ECE_adapt: adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_binary`.

    Conformal-prediction set metrics (require ``y_predset`` of shape
    ``(n_samples, 2)``, one column per class):
        - rejection_rate: Frequency of prediction sets with cardinality != 1. Check :func:`pyhealth.metrics.prediction_set.rejection_rate`.
        - set_size: Average size of the prediction sets. Check :func:`pyhealth.metrics.prediction_set.size`.
        - miscoverage_ps: Prob(k not in prediction set). Check :func:`pyhealth.metrics.prediction_set.miscoverage_ps`.
        - miscoverage_mean_ps: The average (across classes) of miscoverage_ps.
        - miscoverage_overall_ps: Prob(Y not in prediction set). Check :func:`pyhealth.metrics.prediction_set.miscoverage_overall_ps`.
        - error_ps: Same as miscoverage_ps, but restricted to un-rejected samples. Check :func:`pyhealth.metrics.prediction_set.error_ps`.
        - error_mean_ps: The average (across classes) of error_ps.
        - error_overall_ps: Same as miscoverage_overall_ps, but restricted to un-rejected samples. Check :func:`pyhealth.metrics.prediction_set.error_overall_ps`.
    If no metrics are specified, pr_auc, roc_auc and f1 are computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        metrics: List of metrics to compute. Default is ["pr_auc", "roc_auc", "f1"].
        threshold: Threshold for binary classification. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
    """
    if metrics is None:
        metrics = ["pr_auc", "roc_auc", "f1"]

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    if y_predset is not None:
        # Set metrics index class columns; binary model labels can be float
        # tensors with shape (N, 1). Keep scalar metrics' inputs unchanged.
        y_true_set = np.asarray(y_true).reshape(-1).astype(int)

    output = {}
    for metric in metrics:
        if metric == "pr_auc":
            pr_auc = sklearn_metrics.average_precision_score(y_true, y_prob)
            output["pr_auc"] = pr_auc
        elif metric == "roc_auc":
            roc_auc = sklearn_metrics.roc_auc_score(y_true, y_prob)
            output["roc_auc"] = roc_auc
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1":
            f1 = sklearn_metrics.f1_score(y_true, y_pred)
            output["f1"] = f1
        elif metric == "precision":
            precision = sklearn_metrics.precision_score(y_true, y_pred)
            output["precision"] = precision
        elif metric == "recall":
            recall = sklearn_metrics.recall_score(y_true, y_pred)
            output["recall"] = recall
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "jaccard":
            jaccard = sklearn_metrics.jaccard_score(y_true, y_pred)
            output["jaccard"] = jaccard
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = calib.ece_confidence_binary(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
            )
        elif metric in pset.PREDICTION_SET_METRICS:
            if y_predset is None:
                continue
            output[metric] = pset.compute_prediction_set_metric(
                metric, y_predset, y_true_set
            )
        else:
            raise ValueError(f"Unknown metric for binary classification: {metric}")
    return output


if __name__ == "__main__":
    all_metrics = [
        "pr_auc",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "cohen_kappa",
        "jaccard",
    ]
    y_true = np.random.randint(2, size=100000)
    y_prob = np.random.random(size=100000)
    print(binary_metrics_fn(y_true, y_prob, metrics=all_metrics))
