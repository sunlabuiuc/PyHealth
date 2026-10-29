# Authors: Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html
"""Time-dependent metrics for predicted survival curves.

These metrics score a full predicted survival curve ``S(t | x)`` per patient
against right-censored outcomes, following the evaluation protocol of
Han, Goldstein and Ranganath, "Survival Mixture Density Networks"
(MLHC 2022):

- :func:`concordance_td`: time-dependent concordance (Antolini et al.,
  2005), inverse-probability-of-censoring weighted.
- :func:`integrated_brier_score_td`: integrated Brier score, IPCW
  (Graf et al., 1999).
- :func:`integrated_binomial_loglik_td`: integrated binomial
  log-likelihood, IPCW.
- :func:`survival_curve_metrics_fn`: all three at several horizons.

The censoring distribution ``G(t) = P(C > t)`` is a Kaplan-Meier estimate
fitted on the evaluation set, linearly interpolated between its knots.
Horizons are set by censoring-survival thresholds: the horizon for
threshold ``eps`` is the largest observed time with ``G(t) > eps``.

These complement risk-score metrics such as Harrell's C-index: they need
the whole curve, not one score per patient. Inputs are NumPy arrays or
CPU tensors. Higher is better for concordance and the log-likelihood;
lower is better for the Brier score.
"""

from collections.abc import Callable, Sequence

import numpy as np

# Grid points for the integrated scores, and the floor below which a
# censoring-survival estimate is treated as zero (as in the paper's code).
NUM_INTEGRATION_POINTS = 1000
_G_FLOOR = 1e-8
_LOG_FLOOR = 1e-10
DEFAULT_CENSORING_THRESHOLDS = (1e-8, 0.2, 0.4)


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _check_labels(times: np.ndarray, events: np.ndarray) -> None:
    if times.ndim != 1 or times.size == 0 or events.shape != times.shape:
        raise ValueError("times and events must be 1-D arrays of equal length")
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("times must be finite and strictly positive")
    if not np.all((events == 0) | (events == 1)):
        raise ValueError("events must be 0 (censored) or 1 (event)")
    if events.sum() == 0:
        raise ValueError("at least one event is needed to compute these metrics")


def _check_curves(surv: np.ndarray, n: int, t_grid: np.ndarray) -> None:
    if (
        t_grid.ndim != 1
        or t_grid.size == 0
        or not np.isfinite(t_grid).all()
        or np.any(t_grid < 0)
        or np.any(np.diff(t_grid) <= 0)
    ):
        raise ValueError(
            "t_grid must be finite, non-negative, 1-D and strictly increasing"
        )
    if surv.shape != (n, t_grid.size):
        raise ValueError(
            f"surv must have shape [N, T] = [{n}, {t_grid.size}], got {surv.shape}"
        )
    if not np.isfinite(surv).all() or np.any((surv < 0) | (surv > 1)):
        raise ValueError("survival probabilities must be finite and in [0, 1]")
    if np.any(np.diff(surv, axis=1) > 1e-6):
        raise ValueError("survival curves must be non-increasing within tolerance 1e-6")


def censoring_survival(times, events, eval_times) -> np.ndarray:
    """Kaplan-Meier estimate of ``G(t) = P(C > t)`` at ``eval_times``.

    Censoring is treated as the event (``1 - events``). The step function
    is linearly interpolated between its knots at 0 and at each distinct
    observed time, and held constant after the last one.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]`` (1 = event, 0 = censored).
        eval_times: times at which to evaluate ``G``.

    Returns:
        Array with the shape of ``eval_times``.

    Examples:
        >>> censoring_survival([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0], [2.0, 4.0])
        array([0.66666667, 0.        ])
    """
    times = _as_numpy(times)
    censored = 1.0 - _as_numpy(events)
    knots = np.unique(times)
    at_risk = (times[None, :] >= knots[:, None]).sum(axis=1)
    n_censored = np.array([censored[times == k].sum() for k in knots])
    g = np.cumprod(1.0 - n_censored / at_risk)
    return np.interp(
        _as_numpy(eval_times),
        np.concatenate([[0.0], knots]),
        np.concatenate([[1.0], g]),
    )


def horizon(times, events, censoring_threshold: float) -> float:
    """Largest observed time whose censoring survival exceeds a threshold.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]``.
        censoring_threshold: ``eps`` in ``max{t_i : G(t_i) > eps}``.

    Returns:
        The horizon ``tau``.

    Examples:
        >>> horizon([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0], 0.5)
        3.0
    """
    times = _as_numpy(times)
    g = censoring_survival(times, events, times)
    keep = g > censoring_threshold
    if not keep.any():
        raise ValueError("no observed time has censoring survival above threshold")
    return float(times[keep].max())


def integration_grid(
    times,
    events,
    censoring_threshold: float = DEFAULT_CENSORING_THRESHOLDS[0],
    num_points: int = NUM_INTEGRATION_POINTS,
) -> np.ndarray:
    """Uniform grid from the earliest observed time to the horizon.

    This is the grid on which :func:`integrated_brier_score_td` and
    :func:`integrated_binomial_loglik_td` average, matching the paper.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]``.
        censoring_threshold: sets the horizon; see :func:`horizon`.
        num_points: number of grid points. Default 1000.

    Returns:
        Array of shape ``[num_points]``.

    Examples:
        >>> integration_grid([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0], 0.5, 3)
        array([1., 2., 3.])
    """
    times = _as_numpy(times)
    tau = horizon(times, events, censoring_threshold)
    return np.linspace(times.min(), tau, num_points)


def concordance_td(
    times,
    events,
    surv,
    t_grid,
    censoring_threshold: float = DEFAULT_CENSORING_THRESHOLDS[0],
) -> float:
    """IPCW time-dependent concordance (Antolini's C^td).

    For each patient ``i`` with an observed event at ``T_i``, every patient
    ``j`` still at risk after ``T_i`` (``T_j > T_i``, or ``T_j == T_i`` and
    ``j`` censored) forms a comparable pair. The pair is concordant when the
    model gives ``i`` the lower survival probability at ``T_i``:
    ``S_i(T_i) < S_j(T_i)``. Tied predictions count as discordant. Each of
    ``i``'s pairs is weighted by ``1 / G(T_i)^2``. Events with
    ``G(T_i) < censoring_threshold`` are skipped, which truncates the
    metric at the corresponding horizon.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]``.
        surv: predicted survival curves, shape ``[N, T]``, rows in the
            same order as ``times``.
        t_grid: the times of the curve's columns, shape ``[T]``. Every
            event time must appear in it exactly; ``np.unique(times)``
            works.
        censoring_threshold: truncation threshold on ``G(T_i)``.
            Default 1e-8 (the widest horizon).

    Returns:
        Concordance in ``[0, 1]``; higher is better, 0.5 is random.

    Examples:
        >>> times = [1.0, 2.0, 3.0]
        >>> events = [1, 1, 0]
        >>> grid = [1.0, 2.0, 3.0]
        >>> surv = [[0.5, 0.2, 0.1], [0.8, 0.6, 0.4], [0.9, 0.8, 0.7]]
        >>> concordance_td(times, events, surv, grid)
        1.0
    """
    times, events = _as_numpy(times), _as_numpy(events)
    surv, t_grid = _as_numpy(surv), _as_numpy(t_grid)
    _check_labels(times, events)
    _check_curves(surv, times.size, t_grid)

    g = censoring_survival(times, events, times)
    numerator = 0.0
    denominator = 0.0
    for i in np.flatnonzero(events == 1):
        if g[i] < censoring_threshold:
            continue
        k = np.searchsorted(t_grid, times[i])
        if k >= t_grid.size or t_grid[k] != times[i]:
            raise ValueError("every event time must appear exactly in t_grid")
        comparable = (times > times[i]) | ((times == times[i]) & (events == 0))
        weight = 1.0 / g[i] ** 2
        numerator += weight * np.sum(surv[comparable, k] > surv[i, k])
        denominator += weight * np.sum(comparable)
    if denominator == 0:
        raise ValueError("no comparable pairs below the censoring threshold")
    return float(numerator / denominator)


def _ipcw_terms(times, events, surv, t_grid):
    """Shared weighting for the integrated scores.

    Returns survival, event-before-t indicators, event flags and the two
    censoring weights, with grid points and patients whose censoring
    survival is effectively zero removed.
    """
    times, events = _as_numpy(times), _as_numpy(events)
    surv, t_grid = _as_numpy(surv), _as_numpy(t_grid)
    _check_labels(times, events)
    _check_curves(surv, times.size, t_grid)
    g_patient = censoring_survival(times, events, times)
    g_grid = censoring_survival(times, events, t_grid)
    rows = g_grid > _G_FLOOR
    cols = g_patient > _G_FLOOR
    s = surv.T[rows][:, cols]  # [T, N]
    before = (times[None, cols] <= t_grid[rows, None]).astype(float)
    return (
        s,
        before,
        events[None, cols],
        g_patient[None, cols],
        g_grid[rows, None],
    )


def integrated_brier_score_td(times, events, surv, t_grid) -> float:
    """IPCW integrated Brier score over a time grid.

    At each grid time ``t``, a patient with an event by ``t`` contributes
    ``S(t)^2 / G(T_i)``, a patient still at risk contributes
    ``(1 - S(t))^2 / G(t)``, and a patient censored before ``t``
    contributes nothing. The result is the mean over patients and grid
    times, which equals the time-normalised integral on a uniform grid.
    Use :func:`integration_grid` for the paper's grid.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]``.
        surv: predicted survival curves on ``t_grid``, shape ``[N, T]``.
        t_grid: strictly increasing grid times, shape ``[T]``.

    Returns:
        The integrated Brier score; lower is better, 0.25 is an
        uninformative prediction of 0.5.

    Examples:
        >>> times, events = [1.0, 2.0, 3.0], [1, 1, 0]
        >>> grid = [1.0, 2.0, 3.0]
        >>> perfect = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        >>> integrated_brier_score_td(times, events, perfect, grid)
        0.0
    """
    s, before, ev, g_patient, g_grid = _ipcw_terms(times, events, surv, t_grid)
    score = s**2 * ev * before / g_patient + (1.0 - s) ** 2 * (1.0 - before) / g_grid
    return float(score.mean())


def integrated_binomial_loglik_td(times, events, surv, t_grid) -> float:
    """IPCW integrated binomial log-likelihood over a time grid.

    The log-likelihood analogue of :func:`integrated_brier_score_td`: an
    event by ``t`` contributes ``log(1 - S(t)) / G(T_i)`` and a patient
    still at risk contributes ``log S(t) / G(t)``, with probabilities
    floored at 1e-10 inside the log.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]``.
        surv: predicted survival curves on ``t_grid``, shape ``[N, T]``.
        t_grid: strictly increasing grid times, shape ``[T]``.

    Returns:
        The integrated binomial log-likelihood (at most 0); higher is
        better.

    Examples:
        >>> times, events = [1.0, 2.0, 3.0], [1, 1, 0]
        >>> grid = [1.0, 2.0, 3.0]
        >>> half = [[0.5] * 3] * 3
        >>> round(integrated_binomial_loglik_td(times, events, half, grid), 4)
        -0.6931
    """
    s, before, ev, g_patient, g_grid = _ipcw_terms(times, events, surv, t_grid)
    score = np.log(1.0 - s + _LOG_FLOOR) * ev * before / g_patient
    score = score + np.log(s + _LOG_FLOOR) * (1.0 - before) / g_grid
    return float(score.mean())


def survival_curve_metrics_fn(
    times,
    events,
    predict_survival: Callable[[np.ndarray], np.ndarray],
    censoring_thresholds: Sequence[float] = DEFAULT_CENSORING_THRESHOLDS,
    num_points: int = NUM_INTEGRATION_POINTS,
) -> dict[str, float]:
    """Computes C^td, IBS and IBLL at several horizons, as in the paper.

    The evaluator chooses the time grids, so every model is scored the
    same way: concordance uses the distinct observed times, and the
    integrated scores use :func:`integration_grid` for each horizon.

    Args:
        times: observed times, shape ``[N]``.
        events: event indicators, shape ``[N]``.
        predict_survival: callable mapping a 1-D array of ``T`` times to
            survival probabilities of shape ``[N]`` x ``[T]``, rows in the
            same order as ``times``.
        censoring_thresholds: one horizon per threshold; see
            :func:`horizon`. Default ``(1e-8, 0.2, 0.4)``, widest first.
        num_points: integration grid size. Default 1000.

    Returns:
        A dictionary with ``c_td``, ``ibs`` and ``ibll`` for the first
        (widest) threshold, and ``c_td@eps``, ``ibs@eps``, ``ibll@eps``
        for every threshold.

    Examples:
        >>> times = np.array([1.0, 2.0, 3.0, 4.0])
        >>> events = np.array([1, 1, 0, 1])
        >>> def predict(grid):
        ...     return np.exp(-np.outer([2.0, 1.0, 0.5, 0.25], grid))
        >>> scores = survival_curve_metrics_fn(times, events, predict, (1e-8,))
        >>> sorted(scores)
        ['c_td', 'c_td@1e-08', 'ibll', 'ibll@1e-08', 'ibs', 'ibs@1e-08']
        >>> scores["c_td"]
        1.0
    """
    times, events = _as_numpy(times), _as_numpy(events)
    _check_labels(times, events)
    c_grid = np.unique(times)
    c_surv = _as_numpy(predict_survival(c_grid))
    results: dict[str, float] = {}
    for eps in censoring_thresholds:
        grid = integration_grid(times, events, eps, num_points)
        surv = _as_numpy(predict_survival(grid))
        results[f"c_td@{eps:g}"] = concordance_td(times, events, c_surv, c_grid, eps)
        results[f"ibs@{eps:g}"] = integrated_brier_score_td(times, events, surv, grid)
        results[f"ibll@{eps:g}"] = integrated_binomial_loglik_td(
            times, events, surv, grid
        )
    widest = f"{censoring_thresholds[0]:g}"
    for name in ("c_td", "ibs", "ibll"):
        results[name] = results[f"{name}@{widest}"]
    return results


def d_calibration(survival_probabilities, events, num_bins: int = 10) -> dict:
    """Censoring-aware D-calibration from Haider et al. (2020), Algorithm 2.

    Supply one predicted survival probability at each patient's own observed
    event or censoring time, in the same order as ``events``. An observed
    event contributes one count to its probability bin. A censored patient
    contributes fractional counts over probabilities below their prediction.
    Every patient contributes total mass one, including all-censored cohorts.

    This follows Algorithm 2's probability floor of 1e-5 and ceiling-based
    bin assignment. Bins are upper-inclusive: (0, 1/B], ..., ((B-1)/B, 1].
    Zero predictions are floored before assignment. Counts are returned in
    ascending probability order. The chi-square p-value is a diagnostic;
    a large p-value does not establish individual or clinical calibration.

    Args:
        survival_probabilities: Finite probabilities in [0, 1], shape [N].
        events: Event indicators, shape [N]; 1 observed, 0 right censored.
        num_bins: Number of equal-width probability bins, at least 2.

    Returns:
        Dictionary with ``statistic``, ``p_value``, ``bin_counts``,
        ``bin_edges``, ``n_patients``, ``n_events``, ``n_censored`` and the
        ``probability_floor`` and ``boundary_convention`` used. Arrays are
        lists so the result can be saved directly as JSON.

    Raises:
        ValueError: If inputs are empty, misaligned or outside their ranges.

    Examples:
        >>> result = d_calibration([0.25], [0], num_bins=4)
        >>> result["bin_counts"]
        [1.0, 0.0, 0.0, 0.0]
        >>> d_calibration([1.0], [0], num_bins=4)["bin_counts"]
        [0.25, 0.25, 0.25, 0.25]

    Reference:
        Haider et al., Effective Ways to Build and Evaluate Individual
        Survival Distributions, JMLR 21(85), Appendix B.5 / Algorithm 2.
        https://jmlr.org/papers/v21/18-772.html
    """
    from scipy.stats import chi2

    p, e = _as_numpy(survival_probabilities), _as_numpy(events)
    if p.ndim != 1 or not p.size or e.shape != p.shape:
        raise ValueError("probabilities and events must be nonempty aligned 1-D arrays")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("survival probabilities must be finite and in [0, 1]")
    if not np.all((e == 0) | (e == 1)):
        raise ValueError("events must be 0 (censored) or 1 (event)")
    if (
        isinstance(num_bins, (bool, np.bool_))
        or not isinstance(num_bins, (int, np.integer))
        or num_bins < 2
    ):
        raise ValueError("num_bins must be an integer of at least 2")
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    probabilities = np.maximum(p, 1e-5)
    counts = np.zeros(num_bins)
    bins = np.clip(
        np.searchsorted(edges, probabilities, side="left") - 1, 0, num_bins - 1
    )
    np.add.at(counts, bins[e == 1], 1.0)
    censored = probabilities[e == 0, None]
    counts += (
        np.clip(np.minimum(edges[None, 1:], censored) - edges[None, :-1], 0.0, None)
        / censored
    ).sum(axis=0)
    expected = len(p) / num_bins
    statistic = float(np.square(counts - expected).sum() / expected)
    return {
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, num_bins - 1)),
        "bin_counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "n_patients": len(p),
        "n_events": int(e.sum()),
        "n_censored": int((e == 0).sum()),
        "probability_floor": 1e-5,
        "boundary_convention": "upper-inclusive probability bins, Algorithm 2",
    }
