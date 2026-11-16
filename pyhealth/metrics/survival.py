"""Metrics for right-censored survival predictions.

Harrell's concordance index ranks risk scores; the inverse-probability-of-
censoring weighted (IPCW) Brier score (Graf et al., 1999) and the cumulative /
dynamic AUC (Uno et al., 2007) evaluate predictions at a time horizon. The
censoring distribution is estimated with the Kaplan-Meier estimator of the
censoring times on the evaluated data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "concordance_index",
    "cumulative_dynamic_auc",
    "ipcw_brier_score",
    "survival_metrics_fn",
]


def _as_arrays(duration, event) -> tuple[np.ndarray, np.ndarray]:
    duration = np.asarray(duration, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=float).reshape(-1)
    if duration.shape != event.shape:
        raise ValueError("duration and event must have the same length")
    if duration.size == 0:
        raise ValueError("duration and event must not be empty")
    if not np.isin(event, (0.0, 1.0)).all():
        raise ValueError("event must contain only 0/1 indicators")
    return duration, event.astype(bool)


def concordance_index(duration, event, risk, chunk_size: int = 2048) -> float:
    """Harrell's concordance index of risk scores.

    A pair ``(i, j)`` is comparable when ``i`` had an observed event before
    ``j``'s time (``duration[i] < duration[j]`` and ``event[i] == 1``). It is
    concordant when ``risk[i] > risk[j]``; tied risks count one half. Higher
    risk should mean earlier events.

    Args:
        duration: Observed times, shape ``[n]``.
        event: 1 for an observed event, 0 for censoring, shape ``[n]``.
        risk: Risk scores, shape ``[n]`` (e.g. a Cox log hazard ratio).
        chunk_size: Rows compared at once, to bound memory.

    Returns:
        The concordance index, or NaN when no pair is comparable.

    Examples:
        >>> concordance_index([1, 2, 3], [1, 1, 0], [3.0, 2.0, 1.0])
        1.0
        >>> concordance_index([1, 2, 3], [1, 1, 0], [1.0, 2.0, 3.0])
        0.0
    """
    duration, event = _as_arrays(duration, event)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    if risk.shape != duration.shape:
        raise ValueError("risk must have the same length as duration")
    concordant = 0.0
    comparable = 0
    cases = np.flatnonzero(event)
    for start in range(0, len(cases), chunk_size):
        rows = cases[start : start + chunk_size]
        later = duration[None, :] > duration[rows, None]
        diff = risk[rows, None] - risk[None, :]
        comparable += int(later.sum())
        concordant += float(((diff > 0) & later).sum())
        concordant += 0.5 * float(((diff == 0) & later).sum())
    return concordant / comparable if comparable else float("nan")


def _censoring_survival(duration: np.ndarray, event: np.ndarray):
    """Kaplan-Meier estimate of the censoring survival function ``G``.

    Returns a function ``G(t, left)``; with ``left=True`` it gives the left
    limit ``G(t-)``.
    """
    times = np.unique(duration)
    at_risk = np.array([(duration >= t).sum() for t in times], dtype=float)
    censored = np.array(
        [((duration == t) & ~event).sum() for t in times], dtype=float
    )
    surv = np.cumprod(1.0 - censored / at_risk)

    def g(t: np.ndarray, left: bool = False) -> np.ndarray:
        idx = np.searchsorted(times, t, side="left" if left else "right") - 1
        return np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)

    return g


def ipcw_brier_score(duration, event, survival_at_t, time: float) -> float:
    """IPCW Brier score of survival probabilities at one time horizon.

    Records with an event by ``time`` should have survival 0, records still
    at risk after ``time`` survival 1; records censored before ``time`` are
    reweighted away by the Kaplan-Meier censoring distribution (Graf et al.,
    1999). Lower is better.

    Args:
        duration: Observed times, shape ``[n]``.
        event: 1 for an observed event, 0 for censoring, shape ``[n]``.
        survival_at_t: Predicted ``S(time | x)``, shape ``[n]``.
        time: The horizon.

    Returns:
        The Brier score.

    Examples:
        >>> round(ipcw_brier_score([1, 3, 4], [1, 0, 1], [0.0, 1.0, 1.0], 2.0), 3)
        0.0
    """
    duration, event = _as_arrays(duration, event)
    survival_at_t = np.asarray(survival_at_t, dtype=float).reshape(-1)
    g = _censoring_survival(duration, event)
    case = (duration <= time) & event
    control = duration > time
    weight_case = np.where(case, 1.0 / np.maximum(g(duration, left=True), 1e-12), 0.0)
    weight_control = 1.0 / max(float(g(np.array([time]))[0]), 1e-12)
    loss = weight_case * survival_at_t**2
    loss = loss + np.where(control, weight_control * (1.0 - survival_at_t) ** 2, 0.0)
    return float(loss.mean())


def cumulative_dynamic_auc(duration, event, risk, time: float) -> float:
    """Cumulative/dynamic AUC of risk scores at one time horizon.

    Cases have an event by ``time`` and are weighted by the inverse censoring
    probability (Uno et al., 2007); controls are still at risk after
    ``time``. Tied risks count one half.

    Args:
        duration: Observed times, shape ``[n]``.
        event: 1 for an observed event, 0 for censoring, shape ``[n]``.
        risk: Risk scores, shape ``[n]``; higher means earlier events.
        time: The horizon.

    Returns:
        The AUC, or NaN without cases or controls.

    Examples:
        >>> cumulative_dynamic_auc([1, 2, 5, 6], [1, 1, 0, 1], [4, 3, 2, 1], 3.0)
        1.0
    """
    duration, event = _as_arrays(duration, event)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    g = _censoring_survival(duration, event)
    cases = np.flatnonzero((duration <= time) & event)
    controls = np.flatnonzero(duration > time)
    if len(cases) == 0 or len(controls) == 0:
        return float("nan")
    weights = 1.0 / np.maximum(g(duration[cases], left=True), 1e-12)
    diff = risk[cases, None] - risk[None, controls]
    wins = (diff > 0).sum(axis=1) + 0.5 * (diff == 0).sum(axis=1)
    return float((weights * wins).sum() / (weights.sum() * len(controls)))


def _survival_at(survival, time_grid, t: float) -> np.ndarray:
    """Survival curves ``[n, m]`` on ``time_grid``, linearly interpolated at ``t``.

    Before the first grid time the curve runs linearly from ``S(0) = 1``.
    A ``t`` after the last grid time raises, since extrapolating a survival
    curve is not well defined.
    """
    grid = np.asarray(time_grid, dtype=float).reshape(-1)
    survival = np.asarray(survival, dtype=float)
    if survival.ndim != 2 or survival.shape[1] != grid.size:
        raise ValueError("survival must have shape [n, len(time_grid)]")
    if t > grid[-1]:
        raise ValueError(
            f"horizon {t:g} is after the last time_grid point {grid[-1]:g}"
        )
    if grid[0] > 0:
        grid = np.concatenate([[0.0], grid])
        survival = np.concatenate([np.ones((survival.shape[0], 1)), survival], 1)
    right = int(np.clip(np.searchsorted(grid, t, side="left"), 1, grid.size - 1))
    left = right - 1
    span = grid[right] - grid[left]
    weight = float(np.clip((t - grid[left]) / span, 0.0, 1.0)) if span > 0 else 1.0
    return (1.0 - weight) * survival[:, left] + weight * survival[:, right]


def survival_metrics_fn(
    duration,
    event,
    risk,
    survival: np.ndarray | None = None,
    time_grid: Sequence[float] | None = None,
    times: Sequence[float] = (),
) -> dict[str, float]:
    """Concordance plus Brier score and AUC at chosen horizons.

    The Brier score at a horizon ``t`` reads each survival curve linearly
    interpolated at ``t``; before the first grid time the curve starts from
    ``S(0) = 1``. A horizon after the last grid time raises ``ValueError``.
    The AUC uses ``risk`` and needs no grid.

    Args:
        duration: Observed times, shape ``[n]``.
        event: 1 for an observed event, 0 for censoring, shape ``[n]``.
        risk: Risk scores, shape ``[n]``; higher means earlier events.
        survival: Optional predicted survival curves, shape ``[n, m]``, on
            ``time_grid``. Needed for the Brier scores.
        time_grid: The ``m`` increasing times of ``survival``.
        times: Horizons for ``brier@t`` (with ``survival``) and ``auc@t``.

    Returns:
        ``{"c_index": ..., "auc@t": ..., "brier@t": ...}``.

    Examples:
        >>> survival_metrics_fn([1, 2, 3], [1, 1, 0], [3.0, 2.0, 1.0])
        {'c_index': 1.0}
        >>> out = survival_metrics_fn(
        ...     [1, 3], [1, 0], [1.0, 0.0], survival=[[0.0, 0.0], [1.0, 0.5]],
        ...     time_grid=[1.0, 3.0], times=[2.0])
        >>> round(out["brier@2"], 4)  # S(2) = 0.75 for the second record
        0.0312
    """
    out = {"c_index": concordance_index(duration, event, risk)}
    if survival is not None and times and time_grid is None:
        raise ValueError("time_grid is needed with survival")
    for t in times:
        out[f"auc@{t:g}"] = cumulative_dynamic_auc(duration, event, risk, t)
        if survival is not None:
            surv = _survival_at(survival, time_grid, t)
            out[f"brier@{t:g}"] = ipcw_brier_score(duration, event, surv, t)
    return out
