"""Fairness-audit utilities implementing the FAMEWS methodology.

Provides ``audit_predictions`` — a reusable, framework-agnostic function that
takes per-sample predictions + cohort attributes and returns a fairness report
following Hoche et al., CHIL 2024 (§3.1): bootstrap + Mann-Whitney U +
Bonferroni correction.

Contribution author: Rahul Joshi (rahulpj2@illinois.edu)
Paper: https://proceedings.mlr.press/v248/hoche24a.html
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

_DEFAULT_GROUPINGS = (
    "sex",
    "age_group",
    "ethnicity_4",
    "ethnicity_W",
    "insurance_type",
    "surgical_status",
    "admission_type",
)

_BINARY_METRICS = ("recall", "precision", "npv", "fpr")
_SCORE_METRICS = ("auroc", "auprc")


def _confusion(y: np.ndarray, pred: np.ndarray):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _binary_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = _confusion(y, pred)
    out: Dict[str, float] = {}
    out["recall"] = tp / (tp + fn) if (tp + fn) else float("nan")
    out["precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
    out["npv"] = tn / (tn + fn) if (tn + fn) else float("nan")
    out["fpr"] = fp / (fp + tn) if (fp + tn) else float("nan")
    return out


def _score_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        out["auroc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, prob))
    except Exception:
        out["auprc"] = float("nan")
    return out


def _bootstrap(
    mask: np.ndarray,
    df: pd.DataFrame,
    n_bootstrap: int,
    threshold: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Return bootstrapped metric arrays for the cohort indexed by ``mask``."""
    rng = np.random.default_rng(seed)
    results: Dict[str, List[float]] = {m: [] for m in _BINARY_METRICS + _SCORE_METRICS}
    idx = df.index.to_numpy()
    cohort_pids = set(df.loc[mask, "_pid"].tolist())

    for _ in range(n_bootstrap):
        sample = df.loc[rng.choice(idx, size=len(idx), replace=True)]
        inside = sample[sample["_pid"].isin(cohort_pids)]
        if len(inside) < 3 or inside["label"].nunique() < 2:
            continue
        y = inside["label"].to_numpy()
        prob = inside["prob"].to_numpy()
        pred = (prob >= threshold).astype(int)
        for k, v in _binary_metrics(y, pred).items():
            if not np.isnan(v):
                results[k].append(v)
        for k, v in _score_metrics(y, prob).items():
            if not np.isnan(v):
                results[k].append(v)
    return {k: np.array(v) for k, v in results.items()}


def audit_predictions(
    samples: Sequence[Dict],
    probs: Sequence[float],
    labels: Optional[Sequence[int]] = None,
    *,
    groupings: Iterable[str] = _DEFAULT_GROUPINGS,
    threshold: float = 0.5,
    n_bootstrap: int = 100,
    significance_level: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a FAMEWS-style bootstrap fairness audit on model predictions.

    Implements the paper's §3.1 methodology: for each (grouping, category), draw
    ``n_bootstrap`` resamples of the test set, compute cohort vs rest metrics on
    each resample, then use a one-sided Mann-Whitney U test to detect cohorts
    that are significantly worse than their complement. Reports ``p`` values
    that are **uncorrected**; the returned DataFrame also carries a
    ``significantly_worse`` flag using Bonferroni correction across all
    (category × metric) tests.

    Args:
        samples: List of sample dicts (as returned by
            :func:`mortality_prediction_mimic3_with_fairness_fn`). Must carry
            ``patient_id`` plus the grouping keys.
        probs: Predicted probabilities for the positive class, aligned to
            ``samples``.
        labels: Ground-truth binary labels aligned to ``samples``. If ``None``,
            pulled from ``samples[i]["label"]``.
        groupings: Which cohort attributes to audit. Defaults to the seven
            FAMEWS demographic groupings.
        threshold: Decision threshold for binary metrics. Defaults to 0.5.
        n_bootstrap: Number of bootstrap resamples. Paper uses 100.
        significance_level: Base alpha before Bonferroni correction. Paper
            uses 0.001.
        seed: RNG seed for reproducibility.

    Returns:
        A DataFrame with one row per (grouping, category, metric) with columns:
        ``grouping``, ``category``, ``n_patients``, ``metric``,
        ``median_cohort``, ``median_rest``, ``delta`` (absolute), ``pvalue``,
        ``significantly_worse``.

    Example:
        >>> from pyhealth.tasks.fairness_utils import audit_predictions
        >>> audit = audit_predictions(test_samples, test_probs)
        >>> audit[audit["significantly_worse"]].head()

    References:
        Hoche et al. (2024). FAMEWS: A Fairness Auditing Tool for Medical
        Early-Warning Systems. CHIL 2024, PMLR 248:297-311.
    """
    if labels is None:
        # Support both the new class-based task (key="mortality") and the
        # legacy function task (key="label"); prefer the explicit label_key.
        labels = [int(s.get(label_key, s.get("label", 0))) for s in samples]
    assert len(samples) == len(probs) == len(labels), "length mismatch"

    records = []
    for s, p, y in zip(samples, probs, labels):
        rec = {
            "_pid": s.get("hadm_id", s.get("visit_id", s.get("patient_id"))),
            "prob": float(p),
            "label": int(y),
        }
        for g in groupings:
            rec[g] = s.get(g)
        records.append(rec)
    df = pd.DataFrame(records)

    # Count total category tests for Bonferroni correction
    n_category_tests = sum(df[g].dropna().nunique() for g in groupings if g in df)
    n_metrics = len(_BINARY_METRICS) + len(_SCORE_METRICS)
    corrected_alpha = significance_level / max(1, n_category_tests * n_metrics)

    out_rows: List[Dict] = []
    for g in groupings:
        if g not in df.columns:
            continue
        for cat in df[g].dropna().unique():
            mask = (df[g] == cat).to_numpy()
            if mask.sum() < 3 or (~mask).sum() < 3:
                continue
            m_sub = _bootstrap(mask, df, n_bootstrap, threshold, seed)
            m_rest = _bootstrap(~mask, df, n_bootstrap, threshold, seed + 1)
            for metric in m_sub:
                sub_vals = m_sub[metric]
                rest_vals = m_rest[metric]
                if len(sub_vals) < 10 or len(rest_vals) < 10:
                    continue
                _, p_value = mannwhitneyu(sub_vals, rest_vals, alternative="less")
                delta = abs(float(np.median(sub_vals)) - float(np.median(rest_vals)))
                out_rows.append(
                    {
                        "grouping": g,
                        "category": str(cat),
                        "n_patients": int(mask.sum()),
                        "metric": metric,
                        "median_cohort": float(np.median(sub_vals)),
                        "median_rest": float(np.median(rest_vals)),
                        "delta": delta,
                        "pvalue": float(p_value),
                        "significantly_worse": bool(p_value < corrected_alpha),
                    }
                )
    return pd.DataFrame(out_rows)
