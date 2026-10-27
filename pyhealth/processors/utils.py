"""Utilities shared by processors."""

from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np

Normalization = Optional[Literal["zscore", "minmax"]]
NORMALIZATIONS = (None, "zscore", "minmax")


def fit_normalization(
    arrays: Iterable[np.ndarray], method: Normalization
) -> Tuple[List[float], List[float]]:
    """Learns per-feature ``(center, scale)`` from the observed values.

    Missing values (``NaN``) are ignored. ``"zscore"`` uses the mean and the
    population standard deviation, ``"minmax"`` the minimum and the range. A
    feature that is never observed or constant gets a scale of 1.

    Args:
        arrays: Arrays of shape ``(N, F)``, streamed one at a time.
        method: ``"zscore"`` or ``"minmax"``.

    Returns:
        The center and scale of each feature, as lists of floats.
    """
    count = total_mean = total_m2 = low = high = None
    for array in arrays:
        array = np.asarray(array, dtype=np.float64)
        if count is None:
            n_features = array.shape[1]
            count = np.zeros(n_features)
            total_mean = np.zeros(n_features)
            total_m2 = np.zeros(n_features)
            low = np.full(n_features, np.inf)
            high = np.full(n_features, -np.inf)
        observed = ~np.isnan(array)
        n = observed.sum(axis=0)
        if not n.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, np.nansum(array, axis=0) / n, 0.0)
            m2 = np.nansum((array - mean) ** 2, axis=0)
        # Chan et al. parallel update of the running mean and squared deviations.
        combined = count + n
        delta = mean - total_mean
        with np.errstate(invalid="ignore", divide="ignore"):
            total_mean = np.where(
                combined > 0, total_mean + delta * n / combined, 0.0
            )
            total_m2 = total_m2 + m2 + np.where(
                combined > 0, delta**2 * count * n / combined, 0.0
            )
        count = combined
        low = np.fmin(low, np.nanmin(np.where(observed, array, np.inf), axis=0))
        high = np.fmax(high, np.nanmax(np.where(observed, array, -np.inf), axis=0))

    if count is None:
        return [], []
    if method == "zscore":
        center = total_mean
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.where(count > 0, total_m2 / count, 0.0))
    elif method == "minmax":
        center = np.where(count > 0, low, 0.0)
        scale = np.where(count > 0, high - low, 0.0)
    else:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    scale = np.where(scale > 0, scale, 1.0)
    return center.tolist(), scale.tolist()
