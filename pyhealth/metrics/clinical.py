"""Clinical summaries of a binary risk score: alerts, calibration, net benefit, subgroups.

These answer the questions a clinician asks of a risk model, beyond its
AUROC: if we act on the highest-risk patients, how many cases do we catch and
how many alerts are false? Do the predicted risks match observed rates? Does
acting on the model do more good than acting on everyone or no one? Does it
work as well for every group?

Every count that could describe a small group of patients is suppressed below
``min_count`` (default 11, the HIPAA / CMS small-cell convention). When
``groups`` (patient ids) is given, a cell is counted in distinct patients,
not samples, so a patient with many samples cannot make a cell look large.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .binary import binary_metrics_fn

MIN_CELL = 11
"""Smallest count shown in a summary (the HIPAA / CMS small-cell convention)."""


def alert_rate_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    rates: Sequence[float] = (0.05, 0.10, 0.20),
    min_count: int = 0,
) -> dict[str, dict[str, float]]:
    """What happens when the highest-risk fraction of samples is flagged.

    For each rate, the top ``rate`` share of samples by predicted risk is
    flagged (``ceil(rate * n)`` samples).

    Args:
        y_true: Binary labels.
        y_prob: Predicted risks.
        rates: Shares of samples to flag.
        min_count: A rate that flags fewer samples than this is reported as
            ``{"rate", "suppressed": True}`` only.

    Returns:
        ``{"top 10%": {"rate", "n_flagged", "threshold", "sensitivity", "ppv",
        "nne"}}``: the share of all positives caught (sensitivity), the share
        of flags that are positive (PPV), and the number needed to evaluate,
        1 / PPV, the flags per true case.

    Examples:
        >>> out = alert_rate_metrics([1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ...                          [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0], [0.2])
        >>> out["top 20%"]["sensitivity"], out["top 20%"]["ppv"], out["top 20%"]["nne"]
        (0.5, 0.5, 2.0)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    order = np.argsort(-y_prob, kind="stable")
    positives = y_true.sum()
    out = {}
    for rate in rates:
        k = max(1, math.ceil(rate * len(y_true)))
        if k < min_count:
            out[f"top {rate:.0%}"] = {"rate": float(rate), "suppressed": True}
            continue
        flagged = order[:k]
        caught = y_true[flagged].sum()
        ppv = caught / k
        out[f"top {rate:.0%}"] = {
            "rate": float(rate),
            "n_flagged": int(k),
            "threshold": float(y_prob[flagged].min()),
            "sensitivity": float(caught / positives) if positives else float("nan"),
            "ppv": float(ppv),
            "nne": float(1 / ppv) if ppv > 0 else float("inf"),
        }
    return out


def net_benefit(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    thresholds: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> list[dict[str, float]]:
    """Decision-curve net benefit at risk thresholds (Vickers and Elkin, 2006).

    Acting on everyone whose risk is at least ``pt`` has net benefit
    ``TP/n - FP/n * pt / (1 - pt)``: true positives found, minus false
    positives weighted by the odds at which a clinician would act. A model
    is useful at ``pt`` if it beats both acting on everyone and on no one
    (net benefit 0).

    Returns:
        One dict per threshold: ``threshold``, ``model``, ``treat_all``.

    Examples:
        >>> nb = net_benefit([1, 0, 0, 0], [0.9, 0.6, 0.1, 0.1], [0.5])[0]
        >>> round(nb["model"], 3), round(nb["treat_all"], 3)
        (0.0, -0.5)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n, prevalence = len(y_true), y_true.mean()
    out = []
    for pt in thresholds:
        weight = pt / (1 - pt)
        act = y_prob >= pt
        tp = (act & (y_true == 1)).sum() / n
        fp = (act & (y_true == 0)).sum() / n
        out.append({
            "threshold": float(pt),
            "model": float(tp - fp * weight),
            "treat_all": float(prevalence - (1 - prevalence) * weight),
        })
    return out


def calibration_bins(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    n_bins: int = 10,
    min_count: int = MIN_CELL,
    groups: Sequence[Any] | None = None,
) -> list[dict[str, float]]:
    """Mean predicted risk against the observed rate, in bins of predicted risk.

    Bins hold equal numbers of samples (by rank of predicted risk). A bin
    with fewer than ``min_count`` samples, or distinct ``groups`` when
    given, is merged into its neighbour, so every bin shown is large enough
    to publish.

    Returns:
        One dict per bin, lowest risk first: ``mean_predicted``,
        ``observed`` and ``n``.

    Examples:
        >>> bins = calibration_bins([0] * 20 + [1] * 20, [0.1] * 20 + [0.9] * 20, 2)
        >>> [(round(b["mean_predicted"], 2), b["observed"], b["n"]) for b in bins]
        [(0.1, 0.0, 20), (0.9, 1.0, 20)]
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ids = None if groups is None else np.asarray(groups)
    order = np.argsort(y_prob, kind="stable")
    edges = np.linspace(0, len(order), n_bins + 1).round().astype(int)
    chunks = [order[a:b] for a, b in itertools.pairwise(edges) if b > a]

    def big(idx: np.ndarray) -> bool:
        size = len(idx) if ids is None else len(set(ids[idx].tolist()))
        return size >= min_count

    merged: list[np.ndarray] = []
    for chunk in chunks:
        if merged and not big(merged[-1]):
            merged[-1] = np.concatenate([merged[-1], chunk])
        else:
            merged.append(chunk)
    while len(merged) > 1 and not big(merged[-1]):
        last = merged.pop()
        merged[-1] = np.concatenate([merged[-1], last])
    return [
        {"mean_predicted": float(y_prob[idx].mean()),
         "observed": float(y_true[idx].mean()), "n": len(idx)}
        for idx in merged if big(idx)
    ]


def subgroup_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    subgroups: Mapping[str, Sequence[Any]],
    metrics: Sequence[str] = ("roc_auc", "pr_auc"),
    min_count: int = MIN_CELL,
    groups: Sequence[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Discrimination and calibration within each subgroup, and the largest gap.

    Args:
        y_true: Binary labels.
        y_prob: Predicted risks.
        subgroups: Attribute name to one label per sample, e.g.
            ``{"sex": ["F", "M", ...], "age band": [...]}``.
        metrics: Metrics from :func:`binary_metrics_fn` to compute per group.
        min_count: A group gets metrics only with at least this many
            positives and negatives; its counts are shown only if at least
            this large.
        groups: Patient id per sample, so counts are distinct patients.

    Returns:
        ``{attribute: {"groups": {label: {"n", "positives", metric..., "oe_ratio"}},
        "max_gap": {metric: gap}}}``. Counts below ``min_count`` are
        ``None``; a group without enough positives and negatives has counts
        only.

    Examples:
        >>> y = [0, 1] * 20
        >>> p = [0.2, 0.8] * 20
        >>> out = subgroup_metrics(y, p, {"sex": ["F"] * 24 + ["M"] * 16})
        >>> out["sex"]["groups"]["F"]["roc_auc"], out["sex"]["groups"]["M"]["n"]
        (1.0, 16)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ids = None if groups is None else np.asarray(groups)
    out = {}
    for attribute, labels in subgroups.items():
        labels = np.asarray(labels)
        table: dict[str, dict[str, Any]] = {}
        for value in sorted(set(labels.tolist()), key=str):
            idx = labels == value
            if ids is None:
                n, pos = int(idx.sum()), int(y_true[idx].sum())
                neg = n - pos
            else:
                n = len(set(ids[idx]))
                pos = len(set(ids[idx & (y_true == 1)]))
                neg = len(set(ids[idx & (y_true == 0)]))
            row: dict[str, Any] = {
                "n": n if n >= min_count else None,
                "positives": pos if pos >= min_count else None,
            }
            if pos >= min_count and neg >= min_count:
                values = binary_metrics_fn(
                    y_true[idx], y_prob[idx], metrics=[*metrics, "oe_ratio"]
                )
                row.update({k: float(v) for k, v in values.items()})
            table[str(value)] = row
        gaps = {}
        for metric in metrics:
            scored = [r[metric] for r in table.values() if metric in r]
            if len(scored) > 1:
                gaps[metric] = float(max(scored) - min(scored))
        out[attribute] = {"groups": table, "max_gap": gaps}
    return out


def clinical_summary(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    groups: Sequence[Any] | None = None,
    subgroups: Mapping[str, Sequence[Any]] | None = None,
    rates: Sequence[float] = (0.05, 0.10, 0.20),
    thresholds: Sequence[float] | None = None,
    min_count: int = MIN_CELL,
) -> dict[str, Any]:
    """Everything above for one model on one test set, as plain JSON types.

    Thresholds for the decision curve default to half, one and two times the
    prevalence, the range where acting on a risk score is usually debated.

    Examples:
        >>> s = clinical_summary([0, 1] * 30, [0.3, 0.6] * 30)
        >>> sorted(s)  # doctest: +NORMALIZE_WHITESPACE
        ['alerts', 'calibration', 'calibration_bins', 'n', 'net_benefit',
         'positives', 'prevalence', 'subgroups']
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    prevalence = float(y_true.mean())
    if thresholds is None:
        thresholds = sorted({round(min(0.95, f * prevalence), 3) for f in (0.5, 1, 2)})
    calibration = binary_metrics_fn(
        y_true, y_prob,
        metrics=["brier", "oe_ratio", "calibration_slope", "calibration_intercept"],
    )
    n = len(y_true) if groups is None else len(set(np.asarray(groups).tolist()))
    positives = (int(y_true.sum()) if groups is None else
                 len(set(np.asarray(groups)[y_true == 1].tolist())))
    return {
        "n": n if n >= min_count else None,
        "positives": positives if positives >= min_count else None,
        "prevalence": prevalence,
        "calibration": {k: float(v) for k, v in calibration.items()},
        "calibration_bins": calibration_bins(y_true, y_prob, min_count=min_count,
                                             groups=groups),
        "alerts": alert_rate_metrics(y_true, y_prob, rates, min_count),
        "net_benefit": net_benefit(y_true, y_prob, thresholds),
        "subgroups": subgroup_metrics(y_true, y_prob, subgroups or {},
                                      min_count=min_count, groups=groups),
    }
