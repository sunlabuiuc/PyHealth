"""Point-wise metrics for reconstruction-based time-series anomaly detection.

Contributors: Thi Van Trinh Tran and Zexi Li.
Reference: MedTsLLM, https://arxiv.org/abs/2408.07773.
Follows the authors' ``tasks/anomaly_detection.py`` evaluation. Two parts of
that protocol make scores optimistic, and both can be turned off: the
threshold's contamination taken from the *test* labels ("auto"), and point
adjustment (Xu et al., 2018), which counts a whole true anomalous segment as
detected if any of its steps is flagged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def reconstruction_anomaly_scores(
    y_pred: Any, y_true: Any, normalize_by_feature: bool = True
) -> np.ndarray:
    """Per-time-step anomaly scores from reconstructions.

    The score of a step is its squared reconstruction error averaged over
    channels. With ``normalize_by_feature``, each channel's errors are first
    divided by that channel's mean error, so channels with larger errors do
    not dominate.

    Args:
        y_pred: Reconstructions of shape ``(n_steps, n_features)`` or
            ``(n_windows, window_len, n_features)``.
        y_true: Targets, same shape as ``y_pred``.
        normalize_by_feature: Divide each channel's errors by its mean.

    Returns:
        np.ndarray: Scores of shape ``(n_steps,)``, windows concatenated.

    Raises:
        ValueError: If the shapes differ or are not 2D or 3D.

    Examples:
        >>> reconstruction_anomaly_scores(
        ...     np.array([[1.0, 0.0], [3.0, 0.0]]), np.zeros((2, 2)),
        ...     normalize_by_feature=False).tolist()
        [0.5, 4.5]
    """
    pred, true = _numpy(y_pred).astype(np.float64), _numpy(y_true).astype(np.float64)
    if pred.shape != true.shape or pred.ndim not in (2, 3):
        raise ValueError(
            "y_pred and y_true must have the same 2D or 3D shape, got "
            f"{pred.shape} and {true.shape}."
        )
    errors = ((pred - true) ** 2).reshape(-1, pred.shape[-1])
    if normalize_by_feature:
        errors = errors / errors.mean(axis=0, keepdims=True)
    return np.nanmean(errors, axis=1)


def point_adjust(y_pred: Any, y_true: Any) -> np.ndarray:
    """Marks a whole true anomalous segment as detected if any step is.

    Reproduces MedTsLLM's ``adjust_anomalies``, including its edge case: if
    a segment starting at step 0 is first detected after step 0, step 0 is
    not filled in.

    Args:
        y_pred: Binary predictions of shape ``(n_steps,)``.
        y_true: Binary labels of shape ``(n_steps,)``.

    Returns:
        np.ndarray: Adjusted ``int`` predictions. Steps outside true
        anomalous segments are unchanged.

    Raises:
        ValueError: If the shapes differ or are not 1D.

    Examples:
        >>> point_adjust([0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0]).tolist()
        [0, 1, 1, 1, 0, 1]
    """
    pred = _numpy(y_pred).astype(int).copy()
    true = _numpy(y_true).astype(int)
    if pred.shape != true.shape or pred.ndim != 1:
        raise ValueError("y_pred and y_true must be 1D arrays of the same shape.")
    edges = np.diff(np.concatenate([[0], true, [0]]))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        hits = np.flatnonzero(pred[start:end])
        if hits.size:
            pred[start:end] = 1
            if start == 0 and hits[0] > 0:
                pred[0] = 0
    return pred


def anomaly_detection_metrics(
    scores: Any,
    y_true: Any,
    contamination: float,
    point_adjustment: bool = True,
) -> dict[str, float]:
    """Thresholds anomaly scores and scores the point-wise predictions.

    Steps whose score exceeds the ``1 - contamination`` quantile of
    ``scores`` are predicted anomalous, optionally followed by
    :func:`point_adjust`.

    Args:
        scores: Anomaly scores of shape ``(n_steps,)``, higher = more
            anomalous, e.g. from :func:`reconstruction_anomaly_scores`.
        y_true: Binary labels of shape ``(n_steps,)``.
        contamination: Expected fraction of anomalous steps, in ``[0, 1]``.
        point_adjustment: Apply :func:`point_adjust`. Defaults to True, as in
            MedTsLLM.

    Returns:
        dict[str, float]: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``jaccard`` and ``roc_auc`` of the (adjusted) binary predictions, as
        reported by MedTsLLM; ``score_roc_auc``, the threshold-free AUROC of
        the raw scores (NaN if only one class is present); and the
        ``threshold`` used.

    Raises:
        ValueError: If the shapes differ or ``contamination`` is not in
            ``[0, 1]``.

    Examples:
        >>> m = anomaly_detection_metrics([0.1, 0.2, 5.0, 0.3], [0, 0, 1, 0],
        ...                               contamination=0.25)
        >>> m["f1"], m["score_roc_auc"]
        (1.0, 1.0)
    """
    scores, labels = _numpy(scores).astype(np.float64), _numpy(y_true).astype(int)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and y_true must be 1D arrays of the same shape.")
    if not 0.0 <= contamination <= 1.0:
        raise ValueError("contamination must be in [0, 1].")
    threshold = float(np.quantile(scores, 1.0 - contamination))
    pred = (scores > threshold).astype(int)
    if point_adjustment:
        pred = point_adjust(pred, labels)
    both = len(np.unique(labels)) == 2
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "jaccard": float(jaccard_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, pred)) if both else float("nan"),
        "score_roc_auc": (
            float(roc_auc_score(labels, scores)) if both else float("nan")
        ),
        "threshold": threshold,
    }
