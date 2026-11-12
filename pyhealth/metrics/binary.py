from collections.abc import Sequence

import numpy as np
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
import pyhealth.metrics.prediction_set as pset


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
    metrics: list[str] | None = None,
    threshold: float = 0.5,
    y_predset: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
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

    Prediction-set metrics are also accepted when y_predset is provided:
        - set_size: Mean number of included labels.
        - rejection_rate: Fraction of sets whose size is not one.
        - miscoverage_ps: Per-class miss rates.
        - miscoverage_mean_ps: Mean of the defined class miss rates.
        - miscoverage_overall_ps: Fraction of sets omitting the true label.
        - error_ps: Per-class miss rates among singleton predictions.
        - error_mean_ps: Mean of the defined per-class singleton error rates.
        - error_overall_ps: Overall miss rate among singleton predictions.

    See :mod:`pyhealth.metrics.prediction_set` for denominators and NaN handling.
    Prediction-set metrics are skipped when y_predset is None.

    If no metrics are specified, pr_auc, roc_auc and f1 are computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: Encoded 0/1 labels of shape (n_samples,) or (n_samples, 1).
        y_prob: Positive-class probabilities of shape (n_samples,) or (n_samples, 1).
        metrics: List of metrics to compute. Default is ["pr_auc", "roc_auc", "f1"].
        threshold: Threshold for binary classification. Default is 0.5.
        y_predset: Optional binary membership mask of shape (n_samples, 2).
            Column 0 represents label 0; column 1 represents label 1.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
        >>> result = binary_metrics_fn(
        ...     np.array([[0.], [1.]]), np.array([[0.2], [0.8]]),
        ...     metrics=["miscoverage_overall_ps"], y_predset=np.array([[1, 0], [1, 0]])
        ... )
        >>> float(result["miscoverage_overall_ps"])
        0.5
    """
    if metrics is None:
        metrics = ["pr_auc", "roc_auc", "f1"]

    prediction_set_metrics = [
        "set_size",
        "rejection_rate",
        "miscoverage_ps",
        "miscoverage_mean_ps",
        "miscoverage_overall_ps",
        "error_ps",
        "error_mean_ps",
        "error_overall_ps",
    ]
    if y_predset is not None and any(m in prediction_set_metrics for m in metrics):
        labels = np.asarray(y_true)
        if not (labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1)):
            raise ValueError("binary targets must have shape (N,) or (N, 1)")

        if not np.isin(labels, [0, 1]).all():
            raise ValueError("binary targets must contain labels 0 or 1")

        y_true_set = labels.reshape(-1).astype(int)
        y_predset = np.asarray(y_predset)
        if y_predset.shape != (len(y_true_set), 2):
            raise ValueError("binary prediction sets must have shape (N, 2)")

        pset._validate_prediction_targets(y_predset, y_true_set)

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
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == "set_size":
                output[metric] = pset.size(y_predset)
            elif metric == "rejection_rate":
                output[metric] = pset.rejection_rate(y_predset)
            elif metric == "miscoverage_ps":
                output[metric] = pset.miscoverage_ps(y_predset, y_true_set)
            elif metric == "miscoverage_mean_ps":
                output[metric] = pset.miscoverage_mean_ps(y_predset, y_true_set)
            elif metric == "miscoverage_overall_ps":
                output[metric] = pset.miscoverage_overall_ps(y_predset, y_true_set)
            elif metric == "error_ps":
                output[metric] = pset.error_ps(y_predset, y_true_set)
            elif metric == "error_mean_ps":
                output[metric] = pset.error_mean_ps(y_predset, y_true_set)
            elif metric == "error_overall_ps":
                output[metric] = pset.error_overall_ps(y_predset, y_true_set)
        else:
            raise ValueError(f"Unknown metric for binary classification: {metric}")
    return output



def timeseries_binary_metrics_fn(
    y_true: Sequence[np.ndarray],
    y_prob: Sequence[np.ndarray],
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Computes metrics for binary labels given per time step.

    Each sample contributes the labels and predicted probabilities of its
    labeled time steps. The steps of all samples are pooled and scored with
    :func:`binary_metrics_fn`, as in benchmarks that evaluate every hour of
    every ICU stay (e.g. YAIB).

    Args:
        y_true: One array of 0/1 labels per sample, over its labeled steps.
        y_prob: One array of predicted probabilities per sample, matching
            ``y_true``.
        metrics: Metrics to compute, see :func:`binary_metrics_fn`.
        threshold: Decision threshold, see :func:`binary_metrics_fn`.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics import timeseries_binary_metrics_fn
        >>> y_true = [np.array([0, 0, 1]), np.array([1, 0])]
        >>> y_prob = [np.array([0.1, 0.4, 0.8]), np.array([0.7, 0.2])]
        >>> timeseries_binary_metrics_fn(y_true, y_prob, metrics=["roc_auc"])
        {'roc_auc': 1.0}
    """
    return binary_metrics_fn(
        np.concatenate([np.ravel(y) for y in y_true]),
        np.concatenate([np.ravel(p) for p in y_prob]),
        metrics=metrics,
        threshold=threshold,
    )

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
