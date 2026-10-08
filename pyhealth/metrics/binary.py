from typing import Dict, List, Optional

import numpy as np
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib


def _logit(y_prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _fit_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 100):
    """Unpenalised logistic regression of y on columns x, plus an offset (Newton)."""
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(x @ beta + offset)))
        grad = x.T @ (y - mu)
        hess = (x * (mu * (1 - mu))[:, None]).T @ x
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return np.full(x.shape[1], np.nan)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def _calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Slope of logistic recalibration: y ~ a + b * logit(p). Ideal: 1."""
    lp = _logit(y_prob)
    x = np.column_stack([np.ones_like(lp), lp])
    return float(_fit_logistic(x, np.asarray(y_true, dtype=float), np.zeros_like(lp))[1])


def _calibration_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calibration-in-the-large: a in y ~ a + offset(logit(p)). Ideal: 0."""
    lp = _logit(y_prob)
    x = np.ones((lp.size, 1))
    return float(_fit_logistic(x, np.asarray(y_true, dtype=float), lp)[0])


def binary_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.5,
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
        - brier: Brier score (mean squared error of the probabilities)
        - oe_ratio: observed / expected events, sum(y_true) / sum(y_prob); ideal 1
        - calibration_slope: slope b of the logistic recalibration
          y ~ a + b * logit(y_prob); ideal 1, below 1 means overconfident
        - calibration_intercept: calibration-in-the-large, a in
          y ~ a + offset(logit(y_prob)); ideal 0, above 0 means risks are
          underestimated
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
        elif metric == "brier":
            output["brier"] = sklearn_metrics.brier_score_loss(y_true, y_prob)
        elif metric == "oe_ratio":
            output["oe_ratio"] = float(np.sum(y_true) / np.sum(y_prob))
        elif metric == "calibration_slope":
            output["calibration_slope"] = _calibration_slope(y_true, y_prob)
        elif metric == "calibration_intercept":
            output["calibration_intercept"] = _calibration_intercept(y_true, y_prob)
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = calib.ece_confidence_binary(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
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
