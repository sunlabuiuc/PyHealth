"""Scoring predictions with :mod:`pyhealth.metrics`, and joining folds' predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyhealth.trainer import get_metrics_fn


LOWER_IS_BETTER = frozenset(
    {"loss", "brier", "mse", "mae", "rmse", "kl_divergence", "ece", "ece_adapt",
     "hamming_loss"}
)
"""Metric names where a smaller value is better."""


def greater_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` is better.

    Examples:
        >>> greater_is_better("pr_auc"), greater_is_better("brier")
        (True, False)
    """
    return metric not in LOWER_IS_BETTER


def score_predictions(
    y_true: Any, y_prob: Any, mode: str, metrics: Sequence[str], **kwargs: Any
) -> dict[str, float]:
    """Computes ``metrics`` from predictions, with the label kind's metrics function.

    This is how every attempt is scored: the label processor of ``mode``
    (e.g. ``"binary"``) picks the :mod:`pyhealth.metrics` function, as
    :meth:`~pyhealth.trainer.Trainer.evaluate` does.

    Examples:
        >>> score_predictions(np.array([0, 1]), np.array([0.2, 0.9]), "binary", ["roc_auc"])
        {'roc_auc': 1.0}
    """
    values = get_metrics_fn(mode)(y_true, y_prob, metrics=list(metrics), **kwargs)
    return {k: float(v) for k, v in values.items()}


def out_of_fold(
    indices: Sequence[np.ndarray], predictions: Sequence[tuple], split: str = "train"
) -> dict[str, Any] | None:
    """Joins per-fold predictions into one record per sample.

    Returns:
        ``{"index", "fold", "y_true", "y_prob", "split"}`` as arrays in sample
        order of ``index``, or ``None`` when the predictions are not one row
        per sample (e.g. time-series labels), which guardrails cannot use.

    Examples:
        >>> oof = out_of_fold([np.array([1]), np.array([0])],
        ...                   [(np.array([1]), np.array([0.9]), []),
        ...                    (np.array([0]), np.array([0.2]), [])])
        >>> oof["index"].tolist(), oof["fold"].tolist(), oof["y_prob"].tolist()
        ([0, 1], [1, 0], [0.20000000298023224, 0.8999999761581543])
    """
    try:
        y_true = [np.asarray(p[0]) for p in predictions]
        y_prob = [np.asarray(p[1], dtype=np.float32) for p in predictions]
        if any(t.dtype == object or len(t) != len(i) or len(q) != len(i)
               for t, q, i in zip(y_true, y_prob, indices)):
            return None
        index = np.concatenate([np.asarray(i, dtype=np.int64) for i in indices])
        fold = np.concatenate([np.full(len(i), k, dtype=np.int16)
                               for k, i in enumerate(indices)])
        order = np.argsort(index, kind="stable")
        return {
            "index": index[order], "fold": fold[order],
            "y_true": np.concatenate(y_true)[order],
            "y_prob": np.concatenate(y_prob)[order], "split": split,
        }
    except (ValueError, TypeError):
        return None
