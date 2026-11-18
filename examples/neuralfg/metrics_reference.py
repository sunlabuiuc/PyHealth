"""Metrics aligned with the authors' released NeuralFineGray evaluation code.

Based on Jeanselme et al. (CHIL 2023), metrics/discrimination.py's
truncated_concordance_td and metrics/calibration.py's brier_score.

These functions are intentionally separate from metrics.py. That module's
c_index and brier_paper were valid for its declared formulas, but the authors'
released c-index includes earlier competing events and uses pair-specific IPCW
weights; the released Brier code includes all observed competing events.

Weights here use G(t) (right-continuous KM) and clamp at 1e-4, matching the
released source's censoring-weight convention. For continuously distributed
observation times, using G(t) vs. G(t-) has little impact, but for discrete
DeepHit times this may make a numerical difference.
"""
from __future__ import annotations

import numpy as np

from .metrics import CensoringKM, _validate_observations

_EPSILON = 1e-4


class _WeightedFenwick:
    """Fenwick tree for weighted sums indexed by 1-based sorted-score rank."""

    def __init__(self, n: int):
        self.tree = np.zeros(n + 1, dtype=np.float64)

    def add(self, rank: int, weight: float) -> None:
        while rank < len(self.tree):
            self.tree[rank] += weight
            rank += rank & -rank

    def prefix(self, count: int) -> float:
        value = 0.0
        while count > 0:
            value += self.tree[count]
            count -= count & -count
        return value


def _concordant_weight(tree, levels, value, tied_tol):
    # Same comparison as released source: < case_score; tied earns 1/2.
    below = int(np.searchsorted(levels, value - tied_tol, side="left"))
    through = int(np.searchsorted(levels, value + tied_tol, side="right"))
    wt_below = tree.prefix(below)
    wt_through = tree.prefix(through)
    return wt_below + 0.5 * (wt_through - wt_below)


def reference_cindex(
    time,
    event,
    predicted_cif,
    *,
    risk: int,
    horizon: float,
    censoring_km: CensoringKM,
    tied_tol: float = 1e-8,
) -> float:
    """Efficient O(n log n) equivalent of the authors' truncated_concordance_td.

    A case i is any cause-`risk` failure with T_i <= horizon.
    Controls j are either (1) T_j > T_i, OR (2) T_j <= T_i and j
    experienced another *observed* competing event (event_j != 0,risk).

    Weights: type (1) 1/G(T_i)^2; type (2) 1/[G(T_i)*G(T_j)].
    Tied predictions get half credit. Tied observed event times are
    comparable only in type (2), as in the released implementation.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1)
    score = np.asarray(predicted_cif, dtype=float).reshape(-1)
    _validate_observations(time, event)
    if len(score) != len(time) or not np.all(np.isfinite(score)):
        raise ValueError("predicted_cif must be finite and match time length")
    if risk < 1 or not np.isfinite(horizon) or horizon < 0 or tied_tol < 0:
        raise ValueError("invalid risk, horizon, or tied_tol")

    g = np.maximum(censoring_km.survival(time), _EPSILON)
    levels = np.unique(score)
    ranks = np.searchsorted(levels, score) + 1
    numerator = denominator = 0.0
    n = len(time)

    # First part: cases vs patients observed strictly later.
    later = _WeightedFenwick(len(levels))
    order = np.argsort(-time, kind="stable")
    n_later = 0
    start = 0
    while start < n:
        end = start + 1
        t = time[order[start]]
        while end < n and time[order[end]] == t:
            end += 1
        group = order[start:end]
        if t <= horizon and n_later:
            for i in group[event[group] == risk]:
                wt = 1.0 / (g[i] ** 2)
                numerator += wt * _concordant_weight(later, levels, score[i], tied_tol)
                denominator += wt * n_later
        for i in group:
            later.add(int(ranks[i]), 1.0)
        n_later += len(group)
        start = end

    # Second part: cases vs patients with a competing event <= case time.
    before = _WeightedFenwick(len(levels))
    order = np.argsort(time, kind="stable")
    sum_competing_weights = 0.0
    start = 0
    while start < n:
        end = start + 1
        t = time[order[start]]
        while end < n and time[order[end]] == t:
            end += 1
        group = order[start:end]
        # Include same-time competing events (authors' before uses <=).
        competing = group[(event[group] != 0) & (event[group] != risk)]
        for j in competing:
            wj = 1.0 / g[j]
            before.add(int(ranks[j]), wj)
            sum_competing_weights += wj
        if t <= horizon and sum_competing_weights:
            for i in group[event[group] == risk]:
                wi = 1.0 / g[i]
                numerator += wi * _concordant_weight(before, levels, score[i], tied_tol)
                denominator += wi * sum_competing_weights
        start = end

    return float(numerator / denominator) if denominator else float("nan")


def reference_brier(
    time,
    event,
    predicted_cif,
    *,
    risk: int,
    horizon: float,
    censoring_km: CensoringKM,
) -> float:
    """Match authors' released metrics/calibration.py brier_score.

    Any observed (non-censoring) event by the horizon is weighted as a
    known outcome; the outcome is 1 only if the event was cause `risk`.
    Subjects observed beyond the horizon contribute squared predicted CIF.
    Censored subjects <= horizon contribute zero. All scores / N.

    Unlike the paper's printed Appendix A.2 formula, this includes competing
    events before the horizon. KM uses right-continuous G(t) (not G(t-)).
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1)
    predicted = np.asarray(predicted_cif, dtype=float).reshape(-1)
    _validate_observations(time, event)
    if len(predicted) != len(time) or not np.all(np.isfinite(predicted)):
        raise ValueError("predicted_cif must be finite and match time length")
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("predicted CIF must be in [0,1]")
    if risk < 1 or not np.isfinite(horizon) or horizon < 0:
        raise ValueError("invalid risk or horizon")

    # The authors' code returns NaN if no event of the target cause
    # is observed by the evaluation horizon.
    if not np.any((time <= horizon) & (event == risk)):
        return float("nan")
    observed_event = (time <= horizon) & (event != 0)
    after = time > horizon
    total = 0.0
    if np.any(observed_event):
        known_outcome = (event[observed_event] == risk).astype(float)
        g_event = np.maximum(censoring_km.survival(time[observed_event]), _EPSILON)
        total += np.sum((known_outcome - predicted[observed_event]) ** 2 / g_event)
    if np.any(after):
        g_horizon = max(float(censoring_km.survival(horizon)), _EPSILON)
        total += np.sum(predicted[after] ** 2) / g_horizon
    return float(total / len(time))
