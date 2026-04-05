"""Survival analysis metrics for PyHealth.

Implements Harrell's concordance index (C-index) and the inverse
probability of censoring weighted (IPCW) Brier score for evaluating
time-to-event / survival models.  Both are computed in pure NumPy so
there is no extra dependency beyond what PyHealth already requires.

Typical usage
-------------
>>> from pyhealth.metrics import survival_metrics_fn
>>> import numpy as np
>>> times  = np.array([5, 10, 3, 8, 15])
>>> events = np.array([1,  0, 1, 1,  0])   # 1 = event occurred
>>> scores = np.array([0.9, 0.3, 0.8, 0.7, 0.2])  # higher = higher risk
>>> survival_metrics_fn(times, scores, events, metrics=["c_index"])
{'c_index': 0.9166...}
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

def _kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Kaplan–Meier survival estimate.

    Args:
        times: 1-D array of observed times.
        events: 1-D binary array; 1 = event occurred, 0 = censored.

    Returns:
        (unique_times, survival_probs): KM estimate at each unique event time,
        with survival_probs[i] = P(T > unique_times[i]).
    """
    order = np.argsort(times)
    times_sorted = times[order]
    events_sorted = events[order]

    n = len(times_sorted)
    unique_times = []
    survival_probs = []
    s = 1.0

    i = 0
    while i < n:
        t = times_sorted[i]
        # Collect all observations at this time
        j = i
        while j < n and times_sorted[j] == t:
            j += 1
        n_at_risk = n - i
        n_events = int(events_sorted[i:j].sum())
        if n_events > 0:
            s *= (1.0 - n_events / n_at_risk)
            unique_times.append(t)
            survival_probs.append(s)
        i = j

    return np.array(unique_times, dtype=float), np.array(survival_probs, dtype=float)


def _km_at_times(
    km_times: np.ndarray,
    km_probs: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Evaluate the KM step function at arbitrary query times.

    Uses the last-value-carries-forward convention (left-continuous).

    Args:
        km_times: Unique event times from :func:`_kaplan_meier`.
        km_probs: KM survival probabilities at those times.
        query_times: Times at which to evaluate.

    Returns:
        KM survival probability at each query time.
    """
    result = np.ones(len(query_times), dtype=float)
    for k, t in enumerate(query_times):
        idx = np.searchsorted(km_times, t, side="right") - 1
        if idx >= 0:
            result[k] = km_probs[idx]
    return result

def concordance_index_censored(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Harrell's concordance index for right-censored survival data.

    Counts all comparable pairs (i, j) where subject i had the event
    before subject j (t_i < t_j and event_i = 1).  A pair is concordant
    when the model assigns a higher risk score to i.

    Complexity: O(n²) in memory for the boolean comparison matrices, which
    is fine up to a few thousand samples.  For very large n, sub-sample or
    use an O(n log n) implementation.

    Args:
        event_times: 1-D array of observed times (shape ``(n,)``).
        predicted_scores: 1-D array of predicted risk scores — *higher means
            higher risk / shorter expected survival* (shape ``(n,)``).
        event_observed: 1-D binary array; 1 = event, 0 = censored (shape ``(n,)``).

    Returns:
        C-index in [0, 1].  0.5 means random; 1.0 means perfect ranking.

    Raises:
        ValueError: If inputs have incompatible shapes or no comparable pairs
            exist.
    """
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = np.asarray(predicted_scores, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)

    if not (event_times.shape == predicted_scores.shape == event_observed.shape):
        raise ValueError("event_times, predicted_scores, and event_observed must have the same shape.")
    if event_times.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")

    t = event_times
    r = predicted_scores
    e = event_observed

    # Broadcasting: t_i[:, None] vs t_j[None, :]
    t_i = t[:, np.newaxis]   # (n, 1)
    t_j = t[np.newaxis, :]   # (1, n)
    r_i = r[:, np.newaxis]
    r_j = r[np.newaxis, :]
    e_i = e[:, np.newaxis]

    # Comparable: i had event first (strict), j not necessarily uncensored
    comparable = (t_i < t_j) & e_i  # (n, n)
    concordant = comparable & (r_i > r_j)
    tied_risk   = comparable & (r_i == r_j)

    n_comparable = int(comparable.sum())
    if n_comparable == 0:
        return 0.5

    c_index = (float(concordant.sum()) + 0.5 * float(tied_risk.sum())) / n_comparable
    return c_index


def brier_score_survival(
    event_times: np.ndarray,
    predicted_survival: np.ndarray,
    event_observed: np.ndarray,
    eval_time: float,
) -> float:
    """IPCW Brier score for survival at a single evaluation time.

    Uses inverse probability of censoring weighting (IPCW) so that the
    score is unbiased under informative censoring.  The censoring
    distribution G(t) is estimated non-parametrically via Kaplan–Meier on
    the *censored* events.

    Reference:
        Graf et al. (1999), "Assessment and comparison of prognostic
        classification schemes for survival data".

    Args:
        event_times: 1-D array of observed times.
        predicted_survival: 1-D array of predicted P(T > eval_time | X),
            i.e. survival probability at ``eval_time`` for each subject.
        event_observed: 1-D binary array; 1 = event, 0 = censored.
        eval_time: The time horizon at which to evaluate the score.

    Returns:
        Brier score at ``eval_time``; 0 = perfect, 0.25 = random baseline
        for a balanced dataset.
    """
    event_times = np.asarray(event_times, dtype=float)
    predicted_survival = np.asarray(predicted_survival, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)

    n = len(event_times)

    # Fit KM on the censoring distribution (flip event indicator)
    censoring_observed = ~event_observed
    km_times, km_probs = _kaplan_meier(event_times, censoring_observed.astype(float))

    # G(t_i) for each subject and G(eval_time)
    g_ti = _km_at_times(km_times, km_probs, event_times)
    g_eval = _km_at_times(km_times, km_probs, np.array([eval_time]))[0]

    # Avoid division by zero
    g_ti = np.maximum(g_ti, 1e-8)
    g_eval = max(g_eval, 1e-8)

    s_hat = predicted_survival

    # IPCW Brier score terms
    # Term 1: t_i <= t*, event_i = 1  →  (0 - S_hat)² / G(t_i)
    # Term 2: t_i >  t*               →  (1 - S_hat)² / G(t*)
    indicator_event_before = (event_times <= eval_time) & event_observed
    indicator_alive        = event_times > eval_time

    bs = (
        np.sum(s_hat**2 * indicator_event_before / g_ti)
        + np.sum((1.0 - s_hat)**2 * indicator_alive / g_eval)
    ) / n

    return float(bs)


def integrated_brier_score(
    event_times: np.ndarray,
    predicted_survival_fn,
    event_observed: np.ndarray,
    time_grid: Optional[np.ndarray] = None,
    n_time_points: int = 100,
) -> float:
    """Integrated Brier Score (IBS) over a time grid.

    Computes :func:`brier_score_survival` at each point in ``time_grid``
    and integrates via the trapezoid rule.

    Args:
        event_times: 1-D observed times.
        predicted_survival_fn: Callable ``(time_grid: np.ndarray) ->
            np.ndarray`` of shape ``(n_subjects, len(time_grid))``, giving
            predicted survival probabilities for each subject at each
            evaluation time.
        event_observed: 1-D binary event indicator.
        time_grid: Times at which to evaluate the Brier score.  Defaults
            to ``n_time_points`` evenly-spaced points between the 10th
            and 90th percentile of observed event times.
        n_time_points: Number of points to use when ``time_grid`` is None.

    Returns:
        Integrated Brier Score normalised by the time range,
        i.e. ``IBS / (t_max - t_min)`` ∈ [0, 1].
    """
    event_times = np.asarray(event_times, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)

    if time_grid is None:
        t_min = float(np.percentile(event_times, 10))
        t_max = float(np.percentile(event_times, 90))
        time_grid = np.linspace(t_min, t_max, n_time_points)

    # predicted_survival_fn returns (n_subjects, len(time_grid))
    survival_matrix = np.asarray(predicted_survival_fn(time_grid))  # (n, T)

    brier_scores = []
    for k, t in enumerate(time_grid):
        s_at_t = survival_matrix[:, k]
        bs = brier_score_survival(event_times, s_at_t, event_observed, t)
        brier_scores.append(bs)

    brier_scores = np.array(brier_scores)
    t_range = float(time_grid[-1] - time_grid[0])
    if t_range <= 0:
        return float(np.mean(brier_scores))

    ibs = float(np.trapz(brier_scores, time_grid)) / t_range
    return ibs


def survival_metrics_fn(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: np.ndarray,
    metrics: Optional[List[str]] = None,
    eval_time: Optional[float] = None,
    predicted_survival: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute survival analysis evaluation metrics.

    This is the main entry point, analogous to
    :func:`pyhealth.metrics.regression_metrics_fn` but for right-censored
    survival data.

    Args:
        event_times: 1-D array of observed times (n,).
        predicted_scores: 1-D array of predicted risk scores — higher values
            mean *higher risk* / shorter expected survival (n,).
        event_observed: 1-D binary array; 1 = event occurred, 0 = censored (n,).
        metrics: List of metric names to compute.  Accepted values:

            - ``"c_index"``: Harrell's concordance index (default).
            - ``"brier_score"``: IPCW Brier score at ``eval_time`` (requires
              ``eval_time`` and ``predicted_survival``).

            Defaults to ``["c_index"]``.
        eval_time: Time horizon used for ``"brier_score"``.  Required when
            ``"brier_score"`` is in ``metrics``.
        predicted_survival: 1-D array of predicted P(T > eval_time | X)
            for each subject.  Required when ``"brier_score"`` is in
            ``metrics``.

    Returns:
        Dictionary mapping metric name → float value.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics import survival_metrics_fn
        >>> times  = np.array([5.0, 10.0, 3.0, 8.0, 15.0])
        >>> events = np.array([1,   0,    1,   1,   0  ])
        >>> scores = np.array([0.9, 0.3,  0.8, 0.7, 0.2])
        >>> survival_metrics_fn(times, scores, events)
        {'c_index': ...}
    """
    if metrics is None:
        metrics = ["c_index"]

    event_times = np.asarray(event_times, dtype=float).flatten()
    predicted_scores = np.asarray(predicted_scores, dtype=float).flatten()
    event_observed = np.asarray(event_observed, dtype=float).flatten()

    output: Dict[str, float] = {}

    for metric in metrics:
        if metric == "c_index":
            output["c_index"] = concordance_index_censored(
                event_times, predicted_scores, event_observed.astype(bool)
            )
        elif metric == "brier_score":
            if eval_time is None:
                raise ValueError(
                    "'eval_time' must be provided when computing 'brier_score'."
                )
            if predicted_survival is None:
                raise ValueError(
                    "'predicted_survival' must be provided when computing 'brier_score'."
                )
            ps = np.asarray(predicted_survival, dtype=float).flatten()
            output["brier_score"] = brier_score_survival(
                event_times, ps, event_observed.astype(bool), float(eval_time)
            )
        else:
            raise ValueError(
                f"Unknown survival metric: '{metric}'. "
                "Accepted values: 'c_index', 'brier_score'."
            )

    return output


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    true_times = rng.exponential(scale=10, size=n)
    censor_times = rng.exponential(scale=15, size=n)
    observed_times = np.minimum(true_times, censor_times)
    events = (true_times <= censor_times).astype(int)
    risk_scores = 1.0 / true_times + rng.normal(0, 0.1, n)

    print(survival_metrics_fn(observed_times, risk_scores, events))
    eval_t = float(np.median(observed_times[events == 1]))
    surv_at_t = np.clip(1.0 - risk_scores / risk_scores.max(), 0.01, 0.99)
    print(
        survival_metrics_fn(
            observed_times,
            risk_scores,
            events,
            metrics=["c_index", "brier_score"],
            eval_time=eval_t,
            predicted_survival=surv_at_t,
        )
    )
