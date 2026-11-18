"""Event-level metrics of an alarm policy for early event prediction.

Time-step metrics (ROC-AUC, PR-AUC) score every hourly prediction on its own.
An early warning system is used differently: it raises alarms, and what
matters is how many of its alarms are followed by an event within the horizon
and how many events it warns of in time. These are the alarm metrics of
Hyland et al., "Early prediction of circulatory failure in the intensive care
unit using machine learning", Nature Medicine 2020, as used in Section 4.1 of
Yeche et al., "Dynamic Survival Analysis for Early Event Prediction", CHIL
2024, https://proceedings.mlr.press/v248/yeche24a.html

A stay is given by its risk scores per step (e.g. the risk of the event within
the horizon), the time of each step, and the labels of
:class:`~pyhealth.tasks.DecompensationMIMIC3` in survival mode: per step the
hours to the event (``time_to_event``) and whether it is observed (``event``).
A step is *in the window* of an event if the event is observed and falls
within ``horizon`` hours of it, so an alarm there is a true alarm.

All functions only use NumPy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def alarm_policy(
    scores: Sequence[float] | np.ndarray,
    times: Sequence[float] | np.ndarray,
    threshold: float,
    silence: float = 0.0,
) -> np.ndarray:
    """Alarms of one stay: threshold the scores, then silence.

    A step raises an alarm if its score is at least ``threshold`` and the last
    alarm is at least ``silence`` time units earlier. Silencing applies whether
    or not the earlier alarm was right (Eq. 7 of the paper), as the veracity of
    a prognosis is not known until the event.

    Args:
        scores: Risk score per step.
        times: Time of each step, increasing, in the unit of ``silence``.
        threshold: Working threshold.
        silence: Time after an alarm in which no alarm is raised.

    Returns:
        Boolean ``[steps]``, True where an alarm is raised.

    Examples:
        >>> alarm_policy([0.9, 0.9, 0.9, 0.1, 0.9], [0, 1, 2, 3, 4], 0.5, 2).tolist()
        [True, False, True, False, True]
    """
    scores = np.asarray(scores, dtype=float)
    times = np.asarray(times, dtype=float)
    if scores.shape != times.shape or scores.ndim != 1:
        raise ValueError("scores and times must be 1-D and of the same length.")
    if silence < 0:
        raise ValueError(f"silence must not be negative, got {silence!r}.")
    alarms = np.zeros(len(scores), dtype=bool)
    last = -np.inf
    for i in np.flatnonzero(scores >= threshold):
        if times[i] - last >= silence:
            alarms[i] = True
            last = times[i]
    return alarms


def _check_stays(scores, times, time_to_event, event, horizon):
    if not (len(scores) == len(times) == len(time_to_event) == len(event)):
        raise ValueError("scores, times, time_to_event and event need one entry"
                         " per stay.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}.")
    stays = []
    for i, parts in enumerate(zip(scores, times, time_to_event, event)):
        s, t, tte, ev = (np.asarray(p, dtype=float).reshape(-1) for p in parts)
        if not (len(s) == len(t) == len(tte) == len(ev)):
            raise ValueError(f"stay {i}: arrays of different lengths.")
        has_event = bool(ev.any())
        window = (ev > 0) & (tte <= horizon)
        stays.append((s, t, tte, has_event, window))
    return stays


def alarm_curve(
    scores: Sequence[Sequence[float]],
    times: Sequence[Sequence[float]],
    time_to_event: Sequence[Sequence[float]],
    event: Sequence[Sequence[float]],
    horizon: float,
    silence: float = 1.0,
    thresholds: Sequence[float] | None = None,
    n_thresholds: int = 100,
) -> dict[str, np.ndarray]:
    """Alarm precision, event recall and first-alarm distance per threshold.

    Args:
        scores: Per stay, the risk score of each step.
        times: Per stay, the time of each step in hours.
        time_to_event: Per stay and step, the hours to the event (or the hours
            observed if censored), as in the survival mode of the task.
        event: Per stay and step, 1 if the event is observed.
        horizon: Hours before an event in which an alarm is right.
        silence: Hours of silence after an alarm (the paper uses 1 h on
            MIMIC-III).
        thresholds: Working thresholds. Default: ``n_thresholds`` quantiles of
            the scores, so that the curve covers the range the model uses.
        n_thresholds: Number of default thresholds.

    Returns:
        Dict of arrays, one value per threshold in decreasing order:
        ``threshold``; ``alarm_precision`` (alarms in the window of an event
        over all alarms; 1 where there is no alarm); ``event_recall`` (events
        with an alarm in their window over all events);
        ``first_alarm_distance`` (mean hours between the first alarm in the
        window and the event, over the events with one; NaN if none);
        ``n_alarms``. Plus ``n_events``.
    """
    stays = _check_stays(scores, times, time_to_event, event, horizon)
    n_events = sum(s[3] for s in stays)
    if thresholds is None:
        every = np.concatenate([s[0] for s in stays]) if stays else np.zeros(0)
        every = every[np.isfinite(every)]
        if every.size == 0:
            thresholds = np.array([0.5])
        else:
            thresholds = np.unique(
                np.quantile(every, np.linspace(0.0, 1.0, n_thresholds))
            )
    thresholds = np.sort(np.asarray(thresholds, dtype=float))[::-1]

    out = {k: [] for k in ("alarm_precision", "event_recall",
                           "first_alarm_distance", "n_alarms")}
    for tau in thresholds:
        n_alarms = true_alarms = caught = 0
        distances = []
        for s, t, tte, has_event, window in stays:
            alarms = alarm_policy(s, t, tau, silence)
            n_alarms += int(alarms.sum())
            hit = alarms & window
            true_alarms += int(hit.sum())
            if has_event and hit.any():
                caught += 1
                # the earliest alarm in the window is the one furthest away
                distances.append(float(tte[hit].max()))
        out["alarm_precision"].append(true_alarms / n_alarms if n_alarms else 1.0)
        out["event_recall"].append(caught / n_events if n_events else 0.0)
        out["first_alarm_distance"].append(
            float(np.mean(distances)) if distances else float("nan")
        )
        out["n_alarms"].append(n_alarms)
    result = {k: np.asarray(v) for k, v in out.items()}
    result["threshold"] = thresholds
    result["n_events"] = np.asarray(n_events)
    return result


def alarm_event_auprc(alarm_precision: np.ndarray, event_recall: np.ndarray) -> float:
    """Area under the alarm precision / event recall curve.

    Precision is summed over the recall it adds, the average-precision rule:
    ``sum_i (R_i - R_{i-1}) * P_i`` over thresholds from high to low, counting
    only recall above the best recall so far (silencing does not make recall
    monotone in the threshold).
    """
    p = np.asarray(alarm_precision, dtype=float)
    r = np.asarray(event_recall, dtype=float)
    best = 0.0
    area = 0.0
    for pi, ri in zip(p, r):
        if ri > best:
            area += (ri - best) * pi
            best = ri
    return float(area)


def event_alarm_metrics(
    scores: Sequence[Sequence[float]],
    times: Sequence[Sequence[float]],
    time_to_event: Sequence[Sequence[float]],
    event: Sequence[Sequence[float]],
    horizon: float,
    silence: float = 1.0,
    precisions: Sequence[float] = (0.6, 0.7, 0.8),
    thresholds: Sequence[float] | None = None,
    n_thresholds: int = 100,
) -> dict[str, float]:
    """Event-level metrics of Table 2 of the paper for a set of stays.

    Args:
        scores, times, time_to_event, event, horizon, silence, thresholds,
            n_thresholds: See :func:`alarm_curve`.
        precisions: Alarm precisions at which to report event recall and the
            distance of the first alarm.

    Returns:
        ``alarm_event_auprc``, and for each precision ``p`` in percent
        ``event_recall@p`` (the best event recall over thresholds whose alarm
        precision is at least ``p``, 0 if none) and ``first_alarm_distance@p``
        (hours before the event of the first alarm, at that operating point,
        NaN if none). ``n_events`` is the number of events.

    Examples:
        >>> m = event_alarm_metrics(
        ...     scores=[[0.1, 0.9, 0.9], [0.9, 0.1, 0.1]],
        ...     times=[[0, 1, 2], [0, 1, 2]],
        ...     time_to_event=[[3, 2, 1], [3, 2, 1]],
        ...     event=[[1, 1, 1], [0, 0, 0]],
        ...     horizon=2,
        ...     silence=1,
        ...     thresholds=[0.5],
        ... )
        >>> m["event_recall@60"], m["first_alarm_distance@60"]
        (1.0, 2.0)
    """
    curve = alarm_curve(
        scores, times, time_to_event, event, horizon,
        silence=silence, thresholds=thresholds, n_thresholds=n_thresholds,
    )
    precision = curve["alarm_precision"]
    recall = curve["event_recall"]
    distance = curve["first_alarm_distance"]
    metrics: dict[str, float] = {
        "alarm_event_auprc": alarm_event_auprc(precision, recall),
        "n_events": float(curve["n_events"]),
    }
    for p in precisions:
        key = f"{round(p * 100):d}"
        ok = (precision >= p) & (curve["n_alarms"] > 0)
        if ok.any():
            best = np.max(recall[ok])
            # of the thresholds with the best recall, the highest threshold
            idx = int(np.flatnonzero(ok & (recall == best))[0])
            metrics[f"event_recall@{key}"] = float(recall[idx])
            metrics[f"first_alarm_distance@{key}"] = float(distance[idx])
        else:
            metrics[f"event_recall@{key}"] = 0.0
            metrics[f"first_alarm_distance@{key}"] = float("nan")
    return metrics
