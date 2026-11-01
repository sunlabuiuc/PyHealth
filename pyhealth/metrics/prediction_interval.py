"""Metrics for regression prediction intervals.

These helpers back the prediction interval metrics of
:func:`pyhealth.metrics.regression_metrics_fn`, the way
:mod:`pyhealth.metrics.prediction_set` backs the prediction set metrics of
the classification metrics functions.
"""

import numpy as np


def _flatten(**arrays: np.ndarray) -> list:
    """Flattens each array to float and checks they all have the same length."""
    flat = {name: np.asarray(a, dtype=float).ravel() for name, a in arrays.items()}
    if len({a.shape for a in flat.values()}) > 1:
        shapes = ", ".join(f"{name} {a.shape}" for name, a in flat.items())
        raise ValueError(f"Inputs must have the same number of values, got {shapes}.")
    return list(flat.values())


def _covered(
    y_lower: np.ndarray, y_upper: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """Whether each target lies in its interval, bounds included."""
    y_lower, y_upper, y_true = _flatten(y_lower=y_lower, y_upper=y_upper, y_true=y_true)
    return (y_lower <= y_true) & (y_true <= y_upper)


def width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Average width of the prediction intervals.

    Infinite if any interval is unbounded.

    Example:
        >>> width(np.asarray([0.0, 1.0]), np.asarray([2.0, 2.0]))
        1.5
    """
    y_lower, y_upper = _flatten(y_lower=y_lower, y_upper=y_upper)
    return float(np.mean(y_upper - y_lower))


def miscoverage_pi(
    y_lower: np.ndarray, y_upper: np.ndarray, y_true: np.ndarray
) -> float:
    """Miscoverage rate, Prob(Y not in [y_lower, y_upper]).

    Example:
        >>> y_lower = np.asarray([0.0, 2.0, 0.0, 1.0])
        >>> y_upper = np.asarray([2.0, 3.0, 4.0, 3.0])
        >>> y_true = np.asarray([1.0, 2.0, 5.0, 3.0])
        >>> miscoverage_pi(y_lower, y_upper, y_true)
        0.25

    Explanation:
    1 is in [0, 2], and 2 and 3 sit on a bound of their intervals, which
    counts as covered. 5 is above [0, 4]. Thus the miscoverage rate is 1/4.
    """
    return float(1 - np.mean(_covered(y_lower, y_upper, y_true)))


def rolling_miscoverage_pi(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_true: np.ndarray,
    window: int,
) -> np.ndarray:
    """Miscoverage rate over each run of ``window`` consecutive time steps.

    Entry ``k`` covers steps ``k`` to ``k + window - 1``, so the result has
    ``len(y_true) - window + 1`` entries. The samples must be in time order.
    Overall miscoverage averages over the whole stream, so it hides periods of
    undercoverage after a distribution shift that show up here.

    Example:
        >>> y_lower = np.zeros(4)
        >>> y_upper = np.full(4, 2.0)
        >>> y_true = np.asarray([1.0, 9.0, 1.0, 1.0])
        >>> rolling_miscoverage_pi(y_lower, y_upper, y_true, window=2)
        array([0.5, 0.5, 0. ])

    Explanation:
    Entry k covers steps k and k + 1. Only step 1 (9 is above [0, 2]) is
    missed, so the windows starting at steps 0 and 1 have miscoverage 1/2.
    """
    missed = (~_covered(y_lower, y_upper, y_true)).astype(float)
    if not 1 <= window <= len(missed):
        raise ValueError(f"window must be in [1, {len(missed)}], got {window}.")
    return np.convolve(missed, np.ones(window) / window, mode="valid")
