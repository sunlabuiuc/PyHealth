"""Training-weighted prediction error and concordance for competing events.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import numpy as np


class MetricSupportError(ValueError):
    """An otherwise valid metric is undefined for the available follow-up/pairs.

    Examples:
        >>> issubclass(MetricSupportError, ValueError)
        True
    """


def _labels(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels shared by censoring estimation and both public scores.

    Args:
        times: Nonempty duration vector with finite nonnegative values.
        events: Matching first-event vector; zero is censoring.

    Returns:
        Floating-point durations and nonnegative integer event labels.

    Raises:
        ValueError: Shapes, times, or event labels are invalid.
    """
    times, events = np.asarray(times, float), np.asarray(events, float)
    if (
        times.ndim != 1
        or not times.size
        or times.shape != events.shape
        or not np.isfinite(times).all()
        or not np.isfinite(events).all()
        or (times < 0).any()
        or (events < 0).any()
        or (events != np.floor(events)).any()
    ):
        raise ValueError("Labels must be matching finite vectors with integer events")
    return times, events.astype(int)


class CensoringDistribution:
    """Kaplan-Meier censoring survival fitted on training labels only.

    All medical events are censored observations of censoring time. Reverse
    Kaplan-Meier removes tied medical events before the censoring jump.
    Event weights query G(t-); later-comparator and event-free horizon weights
    query G(t).

    Args:
        times: Nonempty finite nonnegative training-duration vector.
        events: Matching labels, 0=censoring, positive=medical event.

    Examples:
        >>> censoring = CensoringDistribution(
        ...     np.array([1., 2., 3.]), np.array([1, 0, 2])
        ... )
        >>> float(censoring.probability(2.))
        0.5
    """

    def __init__(self, times: np.ndarray, events: np.ndarray) -> None:
        times, events = _labels(times, events)
        self.timeline, positions, counts = np.unique(
            times,
            return_inverse=True,
            return_counts=True,
        )
        censored = np.bincount(
            positions,
            weights=events == 0,
            minlength=len(counts),
        )
        at_risk = len(times) - np.concatenate(([0], counts.cumsum()[:-1]))
        censoring_risk = at_risk - (counts - censored)
        jumps = np.divide(
            censored,
            censoring_risk,
            out=np.zeros_like(censored),
            where=censoring_risk > 0,
        )
        self.survival = np.cumprod(1 - jumps)

    def probability(
        self,
        times: np.ndarray | float,
        before: bool = False,
    ) -> np.ndarray:
        """Query the censoring survival step function or its left limit.

        Args:
            times: Finite nonnegative scalar or array in training time units.
            before: True selects G(t-) instead of G(t).

        Returns:
            Probabilities in the query shape, using 1 before the first removal.

        Raises:
            ValueError: Query times are negative or nonfinite.
        """
        query = np.asarray(times, float)
        if not np.isfinite(query).all() or (query < 0).any():
            raise ValueError("Censoring query times must be finite and nonnegative")
        positions = (
            np.searchsorted(
                self.timeline,
                query,
                side="left" if before else "right",
            )
            - 1
        )
        return np.where(positions < 0, 1.0, self.survival[np.maximum(positions, 0)])

    def _supported(self, times: np.ndarray | float, before: bool = False) -> np.ndarray:
        """Require usable training censoring weights for a metric calculation.

        Args:
            times: Times at which the metric needs censoring probabilities.
            before: Whether the metric needs the left limit G(t-).

        Returns:
            Positive censoring probabilities in the query shape.

        Raises:
            MetricSupportError: Follow-up ends too early or weights are unstable.
        """
        if np.any(np.asarray(times) > self.timeline[-1]):
            raise MetricSupportError("Required time exceeds training follow-up")
        probability = self.probability(times, before)
        if np.any(probability <= 1e-8):
            raise MetricSupportError("Censoring support exhausted")
        return probability


def _metric_inputs(
    times: np.ndarray,
    events: np.ndarray,
    predictions: np.ndarray,
    horizon: float,
    risk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and validate arrays before either public metric is evaluated.

    Args:
        times: Held-out duration vector.
        events: Matching first-event labels.
        predictions: Matching finite risk scores.
        horizon: Finite nonnegative scoring time.
        risk: Positive integer cause, not a boolean.

    Returns:
        Validated duration, integer event, and floating-point prediction arrays.

    Raises:
        ValueError: An input array, horizon, or cause is malformed.
    """
    times, events = _labels(times, events)
    predictions = np.asarray(predictions, float)
    if (
        predictions.shape != times.shape
        or not np.isfinite(predictions).all()
        or not np.isfinite(horizon)
        or horizon < 0
        or not isinstance(risk, (int, np.integer))
        or isinstance(risk, bool)
        or risk < 1
    ):
        raise ValueError("Invalid predictions, horizon, or target risk")
    return times, events, predictions


def competing_risk_brier(
    times: np.ndarray,
    events: np.ndarray,
    predictions: np.ndarray,
    horizon: float,
    risk: int,
    censoring: CensoringDistribution,
) -> float:
    """IPCW Brier error with prior competing first events as known negatives.

    Args:
        times: Held-out duration vector.
        events: Matching first-event labels, 0=censored, positive=cause.
        predictions: Target-cause CIF probabilities at horizon, in [0, 1].
        horizon: Finite nonnegative horizon in training duration units.
        risk: Target cause numbered from 1.
        censoring: Distribution fitted on training patients.

    Returns:
        Weighted squared-error sum divided by held-out patient count.

    Raises:
        ValueError: Inputs are malformed.
        MetricSupportError: Training censoring support through horizon is unavailable.

    Examples:
        >>> times, events = np.array([1., 2., 3.]), np.array([1, 2, 0])
        >>> competing_risk_brier(times, events, np.array([1., 0., 0.]),
        ...     2., 1, CensoringDistribution(times, events))
        0.0
    """
    times, events, predictions = _metric_inputs(
        times,
        events,
        predictions,
        horizon,
        risk,
    )
    if ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("CIF probabilities must lie in [0, 1]")
    horizon_probability = censoring._supported(horizon)
    observed = (times <= horizon) & (events != 0)
    alive = times > horizon
    weights = np.zeros(len(times))
    if observed.any():
        weights[observed] = 1 / censoring._supported(times[observed], before=True)
    if alive.any():
        weights[alive] = 1 / horizon_probability
    target = ((times <= horizon) & (events == risk)).astype(float)
    return float(np.mean(weights * (target - predictions) ** 2))


def competing_risk_concordance(
    times: np.ndarray,
    events: np.ndarray,
    predictions: np.ndarray,
    horizon: float,
    risk: int,
    censoring: CensoringDistribution,
) -> tuple[float, int]:
    """Horizon IPCW concordance including earlier competing first events.

    A target event is compared to later observations and competing events
    observed at/before its time. The latter pairs use 1/(G(target-)*G(competing-));
    later-observation pairs use 1/(G(target-)*G(target)). Tied predictions count half.
    Simultaneous target events are not comparable. This is not an all-time
    Antolini score or the reference's binary-cause convention.

    Args:
        times: Held-out duration vector.
        events: Matching labels, 0=censored, positive=cause.
        predictions: Finite risk scores at horizon; higher means greater risk.
        horizon: Finite nonnegative horizon in training duration units.
        risk: Target cause numbered from 1.
        censoring: Distribution fitted on training patients.

    Returns:
        Weighted score and unweighted comparable-pair count.

    Raises:
        ValueError: Inputs are malformed.
        MetricSupportError: There are no pairs or support through horizon is unavailable.

    Examples:
        >>> times, events = np.array([1., 2., 3.]), np.array([1, 2, 0])
        >>> competing_risk_concordance(times, events, np.array([1., 0., 0.]),
        ...     2., 1, CensoringDistribution(times, events))
        (1.0, 2)
    """
    times, events, predictions = _metric_inputs(
        times,
        events,
        predictions,
        horizon,
        risk,
    )
    censoring._supported(horizon)
    numerator = denominator = 0.0
    pair_count = 0
    for index in np.flatnonzero((events == risk) & (times <= horizon)):
        early_competing = (events != 0) & (events != risk) & (times <= times[index])
        comparable = (times > times[index]) | early_competing
        pair_count += int(comparable.sum())
        if not comparable.any():
            continue
        first = float(censoring._supported(times[index], before=True))
        second = np.full(comparable.sum(), float(censoring._supported(times[index])))
        second[early_competing[comparable]] = censoring._supported(
            times[comparable][early_competing[comparable]],
            before=True,
        )
        weights = 1 / (first * second)
        other = predictions[comparable]
        agreement = (predictions[index] > other).astype(float)
        agreement += 0.5 * (predictions[index] == other)
        numerator += float(np.sum(weights * agreement))
        denominator += float(weights.sum())
    if not denominator:
        raise MetricSupportError("No comparable pairs")
    return numerator / denominator, pair_count
