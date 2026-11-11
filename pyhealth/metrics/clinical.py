"""Clinical summaries of a binary risk score: alerts, calibration, net benefit, subgroups.

These answer the questions a clinician asks of a risk model, beyond its
AUROC: if we act on the highest-risk patients, how many cases do we catch and
how many alerts are false? Do the predicted risks match observed rates? Does
acting on the model do more good than acting on everyone or no one? Does it
work as well for every group?

Small cells are withheld below ``min_count`` (default 11, the HIPAA / CMS
small-cell convention). When ``groups`` (patient ids) is given, a cell is
counted in distinct patients, not samples, so a patient with many samples
cannot make a cell look large. Exactly what is withheld:

- :func:`clinical_summary`: when positives or negatives are below
  ``min_count``, both ``n`` and ``positives`` (so neither can be recovered
  from the other) and everything computed from the labels besides the
  overall calibration scores (``prevalence``, ``alerts``,
  ``calibration_bins``, ``net_benefit``), which are then ``None``, ``{}``,
  ``[]`` and ``[]``.
- :func:`alert_rate_metrics`: an alert rate whose flagged, caught or false
  alarm count is below ``min_count``.
- :func:`net_benefit`: the model's net benefit at a threshold where the
  true or false positives number fewer than ``min_count``, since with ``n``
  and the prevalence it would narrow down those counts.
- :func:`calibration_bins`: bins are merged until each has ``min_count``
  positives and ``min_count`` negatives.
- :func:`subgroup_metrics`: a group's ``n`` and ``positives`` together, and
  its metrics, when its positives or negatives are below ``min_count``. If
  only one group of an attribute would be withheld, the counts of the next
  smallest group are withheld too, so the withheld counts cannot be
  recovered by subtracting the others from the total.

Every function defaults to ``min_count=MIN_CELL``; pass ``min_count=0`` to
use :func:`alert_rate_metrics` or :func:`net_benefit` as plain metrics.
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
    min_count: int = MIN_CELL,
    groups: Sequence[Any] | None = None,
) -> dict[str, dict[str, float]]:
    """What happens when the highest-risk fraction of samples is flagged.

    For each rate, the top ``rate`` share of samples by predicted risk is
    flagged (``ceil(rate * n)`` samples).

    Args:
        y_true: Binary labels.
        y_prob: Predicted risks.
        rates: Shares of samples to flag.
        min_count: A rate whose flagged samples, caught positives or false
            alarms number fewer than this is reported as
            ``{"rate", "suppressed": True}`` only, since sensitivity and PPV
            would reveal those counts. Defaults to :data:`MIN_CELL`; pass 0
            to report every rate.
        groups: Patient id per sample; the counts checked against
            ``min_count`` are then distinct patients. ``n_flagged`` is always
            the number of flagged samples.

    Returns:
        ``{"top 10%": {"rate", "n_flagged", "threshold", "sensitivity", "ppv",
        "nne"}}``. ``threshold`` is the midpoint between the lowest flagged
        and the highest unflagged risk (0 when every sample is flagged), so
        it is not any one sample's predicted risk unless the two are tied.
        The rest are the share of all positives caught (sensitivity), the share
        of flags that are positive (PPV), and the number needed to evaluate,
        1 / PPV, the flags per true case.

    Examples:
        >>> out = alert_rate_metrics([1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ...                          [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0], [0.2],
        ...                          min_count=0)
        >>> out["top 20%"]["sensitivity"], out["top 20%"]["ppv"], out["top 20%"]["nne"]
        (0.5, 0.5, 2.0)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ids = None if groups is None else np.asarray(groups)
    order = np.argsort(-y_prob, kind="stable")
    positives = y_true.sum()

    def count(idx: np.ndarray) -> int:
        return len(idx) if ids is None else len(set(ids[idx].tolist()))

    out = {}
    for rate in rates:
        k = max(1, math.ceil(rate * len(y_true)))
        flagged = order[:k]
        caught = y_true[flagged].sum()
        hit = flagged[y_true[flagged] == 1]
        miss = flagged[y_true[flagged] != 1]
        if min(count(flagged), count(hit), count(miss)) < min_count:
            out[f"top {rate:.0%}"] = {"rate": float(rate), "suppressed": True}
            continue
        ppv = caught / k
        lowest = y_prob[order[k - 1]]
        threshold = (lowest + y_prob[order[k]]) / 2 if k < len(y_prob) else 0.0
        out[f"top {rate:.0%}"] = {
            "rate": float(rate),
            "n_flagged": int(k),
            "threshold": float(threshold),
            "sensitivity": float(caught / positives) if positives else float("nan"),
            "ppv": float(ppv),
            "nne": float(1 / ppv) if ppv > 0 else float("inf"),
        }
    return out


def net_benefit(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    thresholds: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    min_count: int = MIN_CELL,
    groups: Sequence[Any] | None = None,
) -> list[dict[str, float]]:
    """Decision-curve net benefit at risk thresholds (Vickers and Elkin, 2006).

    Acting on everyone whose risk is at least ``pt`` has net benefit
    ``TP/n - FP/n * pt / (1 - pt)``: true positives found, minus false
    positives weighted by the odds at which a clinician would act. A model
    is useful at ``pt`` if it beats both acting on everyone and on no one
    (net benefit 0).

    Args:
        y_true: Binary labels.
        y_prob: Predicted risks.
        thresholds: Risk thresholds ``pt``.
        min_count: At a threshold where the true or false positives number
            fewer than this, ``model`` is left out and ``suppressed`` is
            True: with ``n`` and the prevalence, net benefit would narrow
            down those counts. ``treat_all`` depends only on the prevalence
            and is always given. Defaults to :data:`MIN_CELL`; pass 0 to
            report every threshold.
        groups: Patient id per sample; the counts above are then distinct
            patients.

    Returns:
        One dict per threshold: ``threshold``, ``model`` (or
        ``suppressed``), ``treat_all``.

    Examples:
        >>> nb = net_benefit([1, 0, 0, 0], [0.9, 0.6, 0.1, 0.1], [0.5],
        ...                  min_count=0)[0]
        >>> round(nb["model"], 3), round(nb["treat_all"], 3)
        (0.0, -0.5)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ids = None if groups is None else np.asarray(groups)
    n, prevalence = len(y_true), y_true.mean()

    def count(mask: np.ndarray) -> int:
        return int(mask.sum()) if ids is None else len(set(ids[mask].tolist()))

    out = []
    for pt in thresholds:
        weight = pt / (1 - pt)
        act = y_prob >= pt
        hit, miss = act & (y_true == 1), act & (y_true == 0)
        row: dict[str, Any] = {"threshold": float(pt)}
        if min(count(hit), count(miss)) < min_count:
            row["suppressed"] = True
        else:
            row["model"] = float(hit.sum() / n - miss.sum() / n * weight)
        row["treat_all"] = float(prevalence - (1 - prevalence) * weight)
        out.append(row)
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
    with fewer than ``min_count`` positives or ``min_count`` negatives
    (samples, or distinct ``groups`` when given) is merged into its
    neighbour, so neither the bin's size nor ``observed * n`` reveals a
    small count.

    Returns:
        One dict per bin, lowest risk first: ``mean_predicted``,
        ``observed`` and ``n`` (samples, or distinct ``groups``).

    Examples:
        >>> y = [0] * 15 + [1] * 5 + [0] * 5 + [1] * 15
        >>> bins = calibration_bins(y, [0.2] * 20 + [0.8] * 20, 2, min_count=5)
        >>> [(round(b["mean_predicted"], 2), b["observed"], b["n"]) for b in bins]
        [(0.2, 0.25, 20), (0.8, 0.75, 20)]
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ids = None if groups is None else np.asarray(groups)
    order = np.argsort(y_prob, kind="stable")
    edges = np.linspace(0, len(order), n_bins + 1).round().astype(int)
    chunks = [order[a:b] for a, b in itertools.pairwise(edges) if b > a]

    def count(idx: np.ndarray) -> int:
        return len(idx) if ids is None else len(set(ids[idx].tolist()))

    def big(idx: np.ndarray) -> bool:
        positive = y_true[idx] == 1
        return min(count(idx[positive]), count(idx[~positive])) >= min_count

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
         "observed": float(y_true[idx].mean()), "n": count(idx)}
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
        min_count: A group gets its counts and metrics only with at least
            this many positives and negatives (see the module docstring for
            the extra group withheld when only one would be).
        groups: Patient id per sample, so counts are distinct patients.

    Returns:
        ``{attribute: {"groups": {label: {"n", "positives", metric..., "oe_ratio"}},
        "max_gap": {metric: gap}}}``. Withheld counts are ``None``; a group
        without enough positives and negatives has no metrics.

    Examples:
        >>> y = [0, 1] * 25
        >>> p = [0.2, 0.8] * 25
        >>> out = subgroup_metrics(y, p, {"sex": ["F"] * 24 + ["M"] * 26})
        >>> out["sex"]["groups"]["F"]["roc_auc"], out["sex"]["groups"]["M"]["n"]
        (1.0, 26)
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
            enough = pos >= min_count and neg >= min_count
            row: dict[str, Any] = {
                "n": n if enough else None,
                "positives": pos if enough else None,
            }
            if enough:
                values = binary_metrics_fn(
                    y_true[idx], y_prob[idx], metrics=[*metrics, "oe_ratio"]
                )
                row.update({k: float(v) for k, v in values.items()})
            table[str(value)] = row
        withheld = [r for r in table.values() if r["n"] is None]
        shown = [r for r in table.values() if r["n"] is not None]
        if len(withheld) == 1 and shown:
            # one withheld group = total minus the others: hide one more
            smallest = min(shown, key=lambda r: r["n"])
            smallest["n"] = smallest["positives"] = None
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
    With fewer than ``min_count`` positives or negatives, only ``n``,
    ``positives``, ``calibration`` and ``subgroups`` are given (see the
    module docstring).

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
    ids = None if groups is None else np.asarray(groups)

    def count(mask: np.ndarray) -> int:
        return int(mask.sum()) if ids is None else len(set(ids[mask].tolist()))

    n = count(np.ones(len(y_true), dtype=bool))
    positives, negatives = count(y_true == 1), count(y_true != 1)
    enough = min(positives, negatives) >= min_count
    return {
        "n": n if enough else None,
        "positives": positives if enough else None,
        "prevalence": prevalence if enough else None,
        "calibration": {k: float(v) for k, v in calibration.items()},
        "calibration_bins": calibration_bins(y_true, y_prob, min_count=min_count,
                                             groups=groups) if enough else [],
        "alerts": alert_rate_metrics(y_true, y_prob, rates, min_count,
                                     groups=groups) if enough else {},
        "net_benefit": net_benefit(y_true, y_prob, thresholds, min_count,
                                   groups=groups) if enough else [],
        "subgroups": subgroup_metrics(y_true, y_prob, subgroups or {},
                                      min_count=min_count, groups=groups),
    }
