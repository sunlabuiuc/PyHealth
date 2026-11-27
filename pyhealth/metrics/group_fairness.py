"""Subgroup performance and fairness metrics for binary classifiers.

The evaluation protocol of MEDFAIR (Zong, Yang and Hospedales, "MEDFAIR:
Benchmarking Fairness for Medical Imaging", ICLR 2023,
https://arxiv.org/abs/2210.01725).
"""

import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .calibration import ece_confidence_binary

_PER_GROUP = ("auc", "acc", "ece", "bce", "fpr", "fnr", "tpr_at_tnr80")
_SUMMARY = ("worst_auc", "auc_gap", "eqodd")


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Returns the decision threshold that maximises F1.

    MEDFAIR picks the operating point this way. Tune it on validation data
    and pass it to :func:`group_fairness_metrics_fn` for the test set.

    Args:
        y_true: Binary labels of shape (n_samples,).
        y_prob: Positive-class probabilities of shape (n_samples,).

    Returns:
        The threshold ``t`` maximising F1 of ``y_prob >= t``.

    Examples:
        >>> best_f1_threshold(np.array([0, 1, 1]), np.array([0.2, 0.6, 0.9]))
        0.6
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return float(thresholds[np.argmax(f1[:-1])])


def _bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _rate(hit: np.ndarray, condition: np.ndarray) -> float:
    count = condition.sum()
    return float((hit & condition).sum() / count) if count else float("nan")


def _score(name: str, y_true, y_prob, y_pred) -> float:
    if name in ("auc", "tpr_at_tnr80") and len(np.unique(y_true)) < 2:
        return float("nan")
    if name == "auc":
        return float(roc_auc_score(y_true, y_prob))
    if name == "tpr_at_tnr80":
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return float(np.interp(0.2, fpr, tpr))
    if name == "acc":
        return float(np.mean(y_pred == y_true))
    if name == "ece":
        return float(ece_confidence_binary(y_prob, y_true, bins=10))
    if name == "bce":
        return _bce(y_true, y_prob)
    positive = y_true == 1
    if name == "fpr":
        return _rate(y_pred, ~positive)
    return _rate(~y_pred, positive)


def group_fairness_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: Sequence[Hashable],
    metrics: list[str] | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Computes overall and per-subgroup metrics for a binary classifier.

    Per-subgroup values are stored as ``"{metric}_{group}"``, e.g.
    ``"auc_F"``, next to the overall value under ``"{metric}"``. Accepted
    metric names:

        - ``auc``, ``acc``, ``ece``, ``bce``: ROC AUC, accuracy, expected
          calibration error (10 equal-width bins on [0, 1]) and binary
          cross-entropy.
        - ``fpr``, ``fnr``: false positive / negative rate at ``threshold``.
        - ``tpr_at_tnr80``: sensitivity at 80% specificity, linearly
          interpolated on the ROC curve at FPR 0.2 as in MEDFAIR, so it can
          sit between two achievable operating points.
        - ``worst_auc``, ``auc_gap``: lowest subgroup AUC, and highest minus
          lowest.
        - ``eqodd``: equalized odds, ``1 - (|dFPR| + |dTPR|) / 2`` between
          the two subgroups of a binary attribute.

    A subgroup with a single class gets NaN, with a warning, for AUC and
    ``tpr_at_tnr80``, for FPR when it has no negatives or FNR when it has no
    positives, and so for ``eqodd``. NaN AUCs are left out of ``worst_auc``
    and ``auc_gap``; the selection functions reject NaN scores.

    Unlike :func:`~pyhealth.metrics.subgroup_metrics`, this reports error
    rates and equalized odds at a chosen operating point, with MEDFAIR's
    metric set, so results compare directly with that benchmark.

    Args:
        y_true: Binary labels of shape (n_samples,).
        y_prob: Positive-class probabilities of shape (n_samples,) or
            (n_samples, 1).
        groups: Sensitive-attribute value of each sample, e.g. ``"F"``/``"M"``
            or an age bin. Missing values are rejected.
        metrics: Metric names to compute. Defaults to all of them, with
            ``eqodd`` only when there are exactly two subgroups.
        threshold: Decision threshold for ``acc``, ``fpr``, ``fnr`` and
            ``eqodd``. Defaults to None, which tunes it on the data given
            with :func:`best_f1_threshold`, as the MEDFAIR paper does (its
            code uses 0.5 for the per-group FPR/FNR). For a clean test
            estimate pass the threshold tuned on validation. The value used
            is returned as ``"threshold"``.

    Returns:
        Dictionary from metric name to value.

    Raises:
        ValueError: On non-binary labels, probabilities outside [0, 1] or
            NaN, mismatched lengths, missing group values, unknown metric
            names, or ``eqodd`` with other than two groups.

    Examples:
        >>> y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.9])
        >>> groups = np.array(["M"] * 4 + ["F"] * 4)
        >>> s = group_fairness_metrics_fn(y_true, y_prob, groups, threshold=0.5)
        >>> s["auc_M"], s["auc_F"], s["auc_gap"], s["eqodd"]
        (0.75, 1.0, 0.25, 0.5)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not ((y_prob >= 0) & (y_prob <= 1)).all():
        raise ValueError("y_prob must be probabilities in [0, 1], without NaN.")
    y_true = y_true.astype(int)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if not len(y_true) == len(y_prob) == len(groups):
        raise ValueError(
            f"y_true, y_prob and groups must have the same length, got "
            f"{len(y_true)}, {len(y_prob)} and {len(groups)}."
        )
    if pd.isna(groups).any():
        raise ValueError("groups contain missing values.")
    names = sorted(set(groups.tolist()), key=str)
    if metrics is None:
        metrics = [*_PER_GROUP, "worst_auc", "auc_gap"]
        if len(names) == 2:
            metrics.append("eqodd")
    unknown = set(metrics) - set(_PER_GROUP) - set(_SUMMARY)
    if unknown:
        raise ValueError(f"Unknown fairness metrics: {sorted(unknown)}.")
    if "eqodd" in metrics and len(names) != 2:
        raise ValueError(f"eqodd needs exactly two groups, got {len(names)}.")

    if threshold is None:
        threshold = best_f1_threshold(y_true, y_prob)
    y_pred = y_prob >= threshold

    needed = set(metrics)
    if needed & {"worst_auc", "auc_gap"}:
        needed.add("auc")
    if "eqodd" in needed:
        needed |= {"fpr", "fnr"}

    affected = needed & {"auc", "tpr_at_tnr80", "fpr", "fnr"}
    one_class = [str(g) for g in names if len(np.unique(y_true[groups == g])) < 2]
    if affected and one_class:
        warnings.warn(
            f"Groups {one_class} have a single class: their AUC and TPR at 80% "
            "TNR are NaN, as is FPR without negatives or FNR without positives, "
            "and eqodd with them.",
            stacklevel=2,
        )

    scores: dict[str, float] = {}
    for name in (m for m in _PER_GROUP if m in needed):
        scores[name] = _score(name, y_true, y_prob, y_pred)
        for g in names:
            mask = groups == g
            scores[f"{name}_{g}"] = _score(
                name, y_true[mask], y_prob[mask], y_pred[mask]
            )

    if "auc" in needed:
        aucs = np.array([scores[f"auc_{g}"] for g in names])
        defined = aucs[~np.isnan(aucs)]
        scores["worst_auc"] = float(defined.min()) if len(defined) else float("nan")
        scores["auc_gap"] = (
            float(defined.max() - defined.min()) if len(defined) else float("nan")
        )
    if "eqodd" in needed:
        a, b = names
        d_fpr = abs(scores[f"fpr_{a}"] - scores[f"fpr_{b}"])
        d_tpr = abs((1 - scores[f"fnr_{a}"]) - (1 - scores[f"fnr_{b}"]))
        scores["eqodd"] = 1.0 - 0.5 * (d_fpr + d_tpr)

    out = {"threshold": float(threshold)}
    for name in metrics:
        out[name] = scores[name]
        if name in _PER_GROUP:
            out.update({f"{name}_{g}": scores[f"{name}_{g}"] for g in names})
    return out
