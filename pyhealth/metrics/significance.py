"""Significance testing for benchmark comparisons across repeated runs.

The comparison protocol follows Section 4.1 (Evaluation metrics) of
Nonaka & Seita (2021), *In-depth Benchmarking of Deep Neural Network
Architectures for ECG Diagnosis*, Proceedings of the 6th Machine Learning
for Healthcare Conference, PMLR 149:414-439
(https://proceedings.mlr.press/v149/nonaka21a.html):

1. every setting is trained once per cross-validation split, so each setting
   gets one score per split (five PTB-XL trial splits);
2. settings report the mean and variance of those scores;
3. each candidate setting is compared with the best baseline setting with a
   Welch's t-test at ``alpha = 0.05``, and the p-values are corrected for
   multiple testing with the Benjamini-Hochberg procedure;
4. settings whose improvement over the baseline is not positive are excluded
   from testing, because the question is only whether a setting is
   *significantly better* than the baseline.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

__all__ = ["benjamini_hochberg", "welch_relative_improvement_test"]

ScoreValue = float | int | np.number


def _as_float_array(values: Sequence[ScoreValue] | None) -> np.ndarray:
    """Return ``values`` as a 1-D float array, empty for missing input."""
    if values is None:
        return np.array([], dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"expected a sequence of scores, got the scalar {array}")
    return array.reshape(-1)


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct a family of p-values with the Benjamini-Hochberg procedure.

    Args:
        p_values: p-values of the family of tests, in any order.
        alpha: target false discovery rate.

    Returns:
        A tuple ``(q_values, rejected)`` of arrays the same length as
        ``p_values``. ``q_values[i]`` is the BH adjusted p-value of
        ``p_values[i]``, and ``rejected[i]`` is whether the null hypothesis of
        test ``i`` is rejected at level ``alpha``.

    The adjustment is the step-up procedure: p-values are ranked ascending,
    scaled by ``m / rank``, and made monotone from the largest downwards, so
    adjusted p-values can be compared directly with ``alpha``.

    Examples:
        >>> from pyhealth.metrics import benjamini_hochberg
        >>> p_values = [0.001, 0.008, 0.039, 0.041, 0.206]
        >>> q_values, rejected = benjamini_hochberg(p_values, alpha=0.05)
        >>> [float(round(q, 3)) for q in q_values]
        [0.005, 0.02, 0.051, 0.051, 0.206]
        >>> rejected.tolist()
        [True, True, False, False, False]
    """
    p = _as_float_array(p_values)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if p.size == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, p.size + 1, dtype=float)
    scaled = p[order] * p.size / ranks
    # enforce monotonicity from the largest adjusted p-value downwards
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    q_values = np.empty(p.size, dtype=float)
    q_values[order] = np.minimum(monotone, 1.0)
    rejected = q_values <= alpha
    return q_values, rejected


def _welch_test(
    scores: np.ndarray,
    baseline_scores: np.ndarray,
) -> tuple[float | None, float | None, float | None, str]:
    """Two-sided Welch's t-test, which does not assume equal variances.

    The statistic, the Welch-Satterthwaite degrees of freedom and the p-value
    are computed directly, so the result does not depend on which fields the
    installed scipy version attaches to ``ttest_ind``.

    Returns:
        ``(t_statistic, degrees_of_freedom, p_value, reason)``. ``reason``
        explains why a test could not run, and is ``""`` when it did.
    """
    if scores.size < 2 or baseline_scores.size < 2:
        return None, None, None, "fewer than 2 split scores"

    variance_a = float(np.var(scores, ddof=1))
    variance_b = float(np.var(baseline_scores, ddof=1))
    standard_error_squared = (
        variance_a / scores.size + variance_b / baseline_scores.size
    )
    if standard_error_squared == 0.0:
        # every split gave the same score in both settings: there is no
        # variance to test the observed difference against
        equal = float(np.mean(scores)) == float(np.mean(baseline_scores))
        return None, None, (1.0 if equal else 0.0), "zero variance in both settings"

    t_statistic = (float(np.mean(scores)) - float(np.mean(baseline_scores))) / np.sqrt(
        standard_error_squared
    )
    degrees_of_freedom = standard_error_squared**2 / (
        (variance_a / scores.size) ** 2 / (scores.size - 1)
        + (variance_b / baseline_scores.size) ** 2 / (baseline_scores.size - 1)
    )
    p_value = 2.0 * float(stats.t.sf(abs(t_statistic), degrees_of_freedom))
    return t_statistic, degrees_of_freedom, p_value, ""


def welch_relative_improvement_test(
    scores: Mapping[Hashable, Mapping[Any, Sequence[ScoreValue]]],
    baseline: Any,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Test each setting against a baseline across repeated runs.

    One entry per ``(setting, arm)`` pair, where ``arm`` is the model setting
    being evaluated (depth, width, ``block_reps``, ...) and ``baseline`` is the
    arm the paper compares against. For every setting the arm is compared with
    ``baseline`` with a Welch's t-test over the per-split scores; p-values are
    corrected with :func:`benjamini_hochberg` over the family of tests that
    actually ran. Arms that do not improve on the baseline are reported but
    excluded from testing, following Section 4.1 of Nonaka & Seita (2021).

    Args:
        scores: ``{setting: {arm: [score per split]}}``. Each arm needs at
            least 2 split scores to be testable; 1 (the paper's single run per
            split) is enough to report a mean.
        baseline: arm name to compare every other arm against.
        alpha: significance level for both the Welch test and the
            Benjamini-Hochberg correction.

    Returns:
        One dict per tested or reported arm, sorted by ``setting`` and then
        arm, with keys ``setting``, ``arm``, ``baseline``, ``n``, ``mean``,
        ``std`` (sample standard deviation), ``baseline_n``, ``baseline_mean``,
        ``baseline_std``, ``improvement``, ``relative_improvement`` (percent of
        the baseline mean), ``t_statistic``, ``degrees_of_freedom``,
        ``p_value``, ``q_value`` (BH adjusted), ``significant``, ``tested`` and
        ``reason`` (why an arm was not tested, else ``""``).

    Examples:
        >>> from pyhealth.metrics import welch_relative_improvement_test
        >>> scores = {
        ...     "all/roc_auc_macro": {
        ...         18: [0.914, 0.912, 0.915],
        ...         50: [0.917, 0.916, 0.918],
        ...     },
        ...     "super/roc_auc_macro": {
        ...         18: [0.931, 0.929, 0.930],
        ...         50: [0.929, 0.930, 0.928],
        ...     },
        ... }
        >>> for row in welch_relative_improvement_test(scores, baseline=18):
        ...     reason = row["reason"] or "-"
        ...     print(row["setting"], row["arm"], row["significant"], reason)
        all/roc_auc_macro 50 True -
        super/roc_auc_macro 50 False decreased relative improvement
    """
    if not scores:
        return []

    rows: list[dict[str, Any]] = []
    for setting in sorted(scores, key=str):
        arms = scores[setting]
        baseline_scores = _as_float_array(arms.get(baseline))
        baseline_mean = (
            float(np.mean(baseline_scores)) if baseline_scores.size else None
        )
        baseline_std = (
            float(np.std(baseline_scores, ddof=1))
            if baseline_scores.size > 1
            else None
        )

        for arm in sorted((key for key in arms if key != baseline), key=str):
            values = _as_float_array(arms[arm])
            mean = float(np.mean(values)) if values.size else None
            std = float(np.std(values, ddof=1)) if values.size > 1 else None

            row: dict[str, Any] = {
                "setting": setting,
                "arm": arm,
                "baseline": baseline,
                "n": int(values.size),
                "mean": mean,
                "std": std,
                "baseline_n": int(baseline_scores.size),
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "improvement": None,
                "relative_improvement": None,
                "t_statistic": None,
                "degrees_of_freedom": None,
                "p_value": None,
                "q_value": None,
                "significant": False,
                "tested": False,
                "reason": "",
            }
            if values.size and baseline_mean is not None:
                improvement = mean - baseline_mean
                row["improvement"] = improvement
                if baseline_mean != 0.0:
                    row["relative_improvement"] = 100.0 * improvement / baseline_mean

            if baseline_scores.size == 0:
                row["reason"] = f"baseline arm {baseline!r} missing for this setting"
            elif row["relative_improvement"] is None:
                row["reason"] = "baseline mean is zero, relative improvement undefined"
            elif row["relative_improvement"] <= 0.0:
                row["reason"] = "decreased relative improvement"
            else:
                t_statistic, df, p_value, reason = _welch_test(values, baseline_scores)
                row["t_statistic"] = t_statistic
                row["degrees_of_freedom"] = df
                row["p_value"] = p_value
                row["reason"] = reason
                row["tested"] = p_value is not None
            rows.append(row)

    tested = [row for row in rows if row["tested"]]
    if tested:
        p_values = [row["p_value"] for row in tested]
        q_values, rejected = benjamini_hochberg(p_values, alpha=alpha)
        for row, q_value, reject in zip(tested, q_values, rejected, strict=True):
            row["q_value"] = float(q_value)
            row["significant"] = bool(reject)
    return rows
