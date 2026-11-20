"""Metrics for right-censored survival predictions.

Harrell's concordance index ranks risk scores; the inverse-probability-of-
censoring weighted (IPCW) Brier score (Graf et al., 1999) and the cumulative /
dynamic AUC (Uno et al., 2007) evaluate predictions at a time horizon. The
censoring distribution is estimated with the Kaplan-Meier estimator of the
censoring times on the evaluated data.

For longitudinal (landmark or counting-process) predictions, the
time-dependent concordance index C(t, delta_t) scores each patient's risk from
their latest visit at or before a prediction time ``t`` against events in
``(t, t + delta_t]`` (Lee et al., 2020; Gao et al., MLHC 2024).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "concordance_index",
    "cumulative_dynamic_auc",
    "ipcw_brier_score",
    "mean_tdci_bootstrap_ci",
    "mean_tdci_paired_bootstrap_diff",
    "outcomes_from_labels",
    "survival_metrics_fn",
    "time_dependent_concordance_index",
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


TDCI_PREDICTION_TIMES = (0.5, 1.0, 3.0, 5.0)
"""Prediction times ``t`` (years) of the MAS paper's TDCI grid."""

TDCI_HORIZONS = (1.0, 3.0, 5.0, 7.0)
"""Horizons ``delta_t`` (years) of the MAS paper's TDCI grid."""


def outcomes_from_labels(y_true, landmark_time, patient_id) -> tuple:
    """Per-sample event time and event indicator from survival labels.

    Args:
        y_true: ``[n, 3]`` counting-process labels ``[start, stop, event]``
            (the patient's event time is the ``stop`` of their last interval)
            or ``[n, 2]`` landmark labels ``[duration, event]`` (event time is
            ``landmark_time + duration``).
        landmark_time: Sample times, shape ``[n]``.
        patient_id: Patient of each sample, shape ``[n]``.

    Returns:
        ``(event_time, event)``, each of shape ``[n]`` and constant within a
        patient.

    Raises:
        ValueError: If ``y_true`` has another width, the shapes disagree, or a
            patient's landmark labels disagree on the event time or event.

    Examples:
        >>> outcomes_from_labels(
        ...     [[0, 1, 0], [1, 2.5, 1]], [0, 1], ["a", "a"]
        ... )
        (array([2.5, 2.5]), array([1., 1.]))
    """
    y_true = np.asarray(y_true, dtype=float)
    landmark_time = np.asarray(landmark_time, dtype=float).reshape(-1)
    patient_id = np.asarray(patient_id).reshape(-1)
    if y_true.ndim != 2 or y_true.shape[1] not in (2, 3):
        raise ValueError("y_true must have shape [n, 3] or [n, 2]")
    if not (len(y_true) == len(landmark_time) == len(patient_id)):
        raise ValueError("y_true, landmark_time and patient_id lengths differ")
    _, inverse = np.unique(patient_id, return_inverse=True)
    n_patients = inverse.max() + 1 if len(inverse) else 0
    if y_true.shape[1] == 3:
        stop, event = y_true[:, 1], y_true[:, 2]
        last = np.full(n_patients, -1)
        for i, p in enumerate(inverse):
            if last[p] < 0 or stop[i] > stop[last[p]]:
                last[p] = i
        return stop[last][inverse], event[last][inverse]
    event_time = landmark_time + y_true[:, 0]
    event = y_true[:, 1]
    for values in (event_time, event):
        low = np.full(n_patients, np.inf)
        high = np.full(n_patients, -np.inf)
        np.minimum.at(low, inverse, values)
        np.maximum.at(high, inverse, values)
        if np.any(high - low > 1e-6):
            raise ValueError("landmark labels disagree within a patient")
    return event_time, event


def _risk_at_prediction_times(
    event_time, event, risk, landmark_time, patient_id, prediction_times
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per patient: outcome and risk at each prediction time.

    The risk at ``t`` is the risk of the patient's latest sample with
    ``landmark_time <= t``. It is NaN when the patient has no sample by ``t``
    or their event time is not after ``t`` (no longer at risk).
    """
    event_time = np.asarray(event_time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=float).reshape(-1)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    landmark_time = np.asarray(landmark_time, dtype=float).reshape(-1)
    patient_id = np.asarray(patient_id).reshape(-1)
    n = len(risk)
    if not (
        len(event_time) == len(event) == len(landmark_time) == len(patient_id) == n
    ):
        raise ValueError("all inputs must have one value per sample")
    if n == 0:
        raise ValueError("inputs must not be empty")
    if not np.isin(event, (0.0, 1.0)).all():
        raise ValueError("event must contain only 0/1 indicators")
    if not (np.isfinite(risk).all() and np.isfinite(landmark_time).all()):
        raise ValueError("risk and landmark_time must be finite")
    _, inverse = np.unique(patient_id, return_inverse=True)
    n_patients = inverse.max() + 1
    patient_time = np.zeros(n_patients)
    patient_event = np.zeros(n_patients)
    patient_time[inverse] = event_time
    patient_event[inverse] = event
    times = np.asarray(prediction_times, dtype=float)
    risk_at = np.full((n_patients, len(times)), np.nan)
    latest = np.full((n_patients, len(times)), -np.inf)
    for i in range(n):
        p = inverse[i]
        for k, t in enumerate(times):
            if landmark_time[i] <= t and landmark_time[i] >= latest[p, k]:
                latest[p, k] = landmark_time[i]
                risk_at[p, k] = risk[i]
    risk_at[patient_time[:, None] <= times[None, :]] = np.nan
    return patient_time, patient_event, risk_at


def _tdci_grid(patient_time, patient_event, risk_at, prediction_times, horizons):
    """C(t, delta_t) for every cell, plus at-risk and event counts."""
    times = np.asarray(prediction_times, dtype=float)
    horizons = np.asarray(horizons, dtype=float)
    tdci = np.full((len(times), len(horizons)), np.nan)
    n_events = np.zeros((len(times), len(horizons)), dtype=int)
    n_at_risk = np.zeros(len(times), dtype=int)
    for k, t in enumerate(times):
        eligible = ~np.isnan(risk_at[:, k])
        n_at_risk[k] = int(eligible.sum())
        if not eligible.any():
            continue
        duration = patient_time[eligible]
        for h, delta in enumerate(horizons):
            # Only events by t + delta_t count as cases; any patient who
            # outlives a case is a comparison, as in the authors' c_index.
            case = (patient_event[eligible] == 1) & (duration <= t + delta)
            n_events[k, h] = int(case.sum())
            if case.any():
                tdci[k, h] = concordance_index(
                    duration, case.astype(float), risk_at[eligible, k]
                )
    return tdci, n_at_risk, n_events


def time_dependent_concordance_index(
    event_time,
    event,
    risk,
    landmark_time,
    patient_id,
    prediction_times: Sequence[float] = TDCI_PREDICTION_TIMES,
    horizons: Sequence[float] = TDCI_HORIZONS,
) -> dict:
    """Time-dependent concordance index C(t, delta_t) over a grid, and its mean.

    For each prediction time ``t``, patients still at risk (event time after
    ``t``) are scored by the risk of their latest sample at or before ``t``,
    so no information after ``t`` is used. A pair ``(i, j)`` is comparable
    when ``i`` had the event by ``t + delta_t`` and ``j``'s event time is
    later than ``i``'s; it is concordant when ``i``'s risk is higher. Tied
    risks count one half. The mean TDCI averages the cells that have at least
    one comparable pair. This follows the MAS paper's evaluation (Gao et al.,
    MLHC 2024), which uses the grid ``t`` in {0.5, 1, 3, 5} and ``delta_t``
    in {1, 3, 5, 7} years.

    Args:
        event_time: Patient's event or censoring time per sample, shape
            ``[n]`` (see :func:`outcomes_from_labels`).
        event: Patient's event indicator per sample, shape ``[n]``.
        risk: Risk per sample, shape ``[n]`` (e.g. a Cox log hazard); higher
            means earlier events.
        landmark_time: Time of each sample's visit, shape ``[n]``.
        patient_id: Patient of each sample, shape ``[n]``.
        prediction_times: Prediction times ``t``.
        horizons: Horizons ``delta_t``.

    Returns:
        ``{"mean_tdci": float, "tdci": [len(t), len(delta_t)] array (NaN where
        no pair is comparable), "n_at_risk": [len(t)] array, "n_events":
        [len(t), len(delta_t)] array}``.

    Raises:
        ValueError: If the inputs differ in length, are empty, or contain
            non-finite risks or times or non-0/1 events.

    Examples:
        >>> out = time_dependent_concordance_index(
        ...     event_time=[2, 2, 4, 4], event=[1, 1, 0, 0], risk=[1, 2, 0, 0],
        ...     landmark_time=[0, 1, 0, 1], patient_id=["a", "a", "b", "b"],
        ...     prediction_times=[0.5, 1.5], horizons=[3])
        >>> out["tdci"].ravel().tolist(), out["mean_tdci"]
        ([1.0, 1.0], 1.0)
    """
    rows = _risk_at_prediction_times(
        event_time, event, risk, landmark_time, patient_id, prediction_times
    )
    tdci, n_at_risk, n_events = _tdci_grid(*rows, prediction_times, horizons)
    mean = float(np.nanmean(tdci)) if np.isfinite(tdci).any() else float("nan")
    return {
        "mean_tdci": mean,
        "tdci": tdci,
        "n_at_risk": n_at_risk,
        "n_events": n_events,
    }


def _mean_tdci_metric(prediction_times, horizons):
    """Mean TDCI as ``metric(y_true, y_prob)`` on patient rows, for bootstraps.

    ``y_true`` is ``[n_patients, 2]`` (event time, event) and ``y_prob`` the
    ``[n_patients, len(prediction_times)]`` risks at each prediction time.
    """

    def metric(y_true, y_prob):
        tdci, _, _ = _tdci_grid(
            y_true[:, 0], y_true[:, 1], y_prob, prediction_times, horizons
        )
        return float(np.nanmean(tdci)) if np.isfinite(tdci).any() else float("nan")

    return metric


def mean_tdci_bootstrap_ci(
    event_time,
    event,
    risk,
    landmark_time,
    patient_id,
    prediction_times: Sequence[float] = TDCI_PREDICTION_TIMES,
    horizons: Sequence[float] = TDCI_HORIZONS,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Patient-level percentile bootstrap interval for the mean TDCI.

    Patients are resampled with replacement once per replicate for the whole
    grid, and every copy of a patient is a separate patient, which keeps the
    "latest visit per patient" step correct.

    Args:
        event_time, event, risk, landmark_time, patient_id, prediction_times,
            horizons: As in :func:`time_dependent_concordance_index`.
        n_boot: Number of bootstrap replicates.
        seed: Seed for the resampling.
        alpha: 1 - confidence level (0.05 gives a 95% interval).

    Returns:
        ``{"estimate", "lower", "upper", "n_boot", "n_skipped"}`` as in
        :func:`pyhealth.metrics.bootstrap_ci`.
    """
    from .bootstrap import bootstrap_ci

    patient_time, patient_event, risk_at = _risk_at_prediction_times(
        event_time, event, risk, landmark_time, patient_id, prediction_times
    )
    y_true = np.column_stack([patient_time, patient_event])
    return bootstrap_ci(
        y_true,
        risk_at,
        _mean_tdci_metric(prediction_times, horizons),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )


def mean_tdci_paired_bootstrap_diff(
    event_time,
    event,
    risk_a,
    risk_b,
    landmark_time,
    patient_id,
    prediction_times: Sequence[float] = TDCI_PREDICTION_TIMES,
    horizons: Sequence[float] = TDCI_HORIZONS,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap interval for mean TDCI(model A) - mean TDCI(model B).

    Both models' risks must be for the same samples. They are scored on
    identical patient resamples, so the interval is for their difference; an
    interval excluding 0 indicates a difference at level ``alpha``.

    Args:
        event_time, event, landmark_time, patient_id, prediction_times,
            horizons: As in :func:`time_dependent_concordance_index`.
        risk_a: Model A's risk per sample, shape ``[n]``.
        risk_b: Model B's risk per sample, shape ``[n]``.
        n_boot: Number of bootstrap replicates.
        seed: Seed for the resampling.
        alpha: 1 - confidence level.

    Returns:
        The output of :func:`pyhealth.metrics.paired_bootstrap_diff`.
    """
    from .bootstrap import paired_bootstrap_diff

    patient_time, patient_event, risk_at_a = _risk_at_prediction_times(
        event_time, event, risk_a, landmark_time, patient_id, prediction_times
    )
    _, _, risk_at_b = _risk_at_prediction_times(
        event_time, event, risk_b, landmark_time, patient_id, prediction_times
    )
    y_true = np.column_stack([patient_time, patient_event])
    return paired_bootstrap_diff(
        y_true,
        risk_at_a,
        risk_at_b,
        _mean_tdci_metric(prediction_times, horizons),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
