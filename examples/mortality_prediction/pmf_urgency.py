"""PMF-based urgency score for 24-hour early-event prediction.

The score combines total event risk, the conditional probability of an
event within the first six hours, and conditional expected-time imminence.
It is a ranking score, not a calibrated event probability.
"""

import numpy as np


def pmf_urgency_score(
    pmf: np.ndarray, near_horizon: int = 6, alpha: float = 0.5
) -> np.ndarray:
    """Compute the urgency score used in the DSA reproduction extension.

    Args:
        pmf: Array of shape (n_samples, 24), containing unconditional event
            probability masses for hours 1 through 24. Rows may sum to less
            than one because an event may occur beyond the horizon.
        near_horizon: Number of near-term hourly bins.
        alpha: Weight on near-term share; 1 - alpha weights imminence.

    Returns:
        One urgency score per row. Risk at or below 1e-8 yields zero,
        matching the original experiment's zero-risk convention.

    Raises:
        ValueError: If the input shape, probability masses or parameters
            are invalid.

    Example:
        >>> pmf = np.zeros((2, 24))
        >>> pmf[0, 0] = pmf[1, 23] = 0.5
        >>> pmf_urgency_score(pmf).tolist()
        [0.5, 0.0]
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim != 2 or pmf.shape[1] != 24:
        raise ValueError("pmf must have shape (n_samples, 24)")
    if not isinstance(near_horizon, (int, np.integer)) or not 1 <= near_horizon <= 24:
        raise ValueError("near_horizon must be an integer in [1, 24]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not np.isfinite(pmf).all() or (pmf < 0).any():
        raise ValueError("pmf must contain finite, nonnegative masses")

    risk = pmf.sum(axis=1)
    if (risk > 1.0 + 1e-6).any():
        raise ValueError("PMF row sums must not exceed one (within tolerance)")
    hours = np.arange(1, 25, dtype=np.float64)
    valid = risk > 1e-8
    near_share = np.divide(
        pmf[:, :near_horizon].sum(axis=1),
        risk,
        out=np.zeros_like(risk),
        where=valid,
    )
    expected_tte = np.divide(
        (pmf * hours[None, :]).sum(axis=1),
        risk,
        out=np.full_like(risk, 24),
        where=valid,
    )
    imminence = np.clip(1.0 - (expected_tte - 1.0) / 23.0, 0.0, 1.0)
    return risk * (alpha * near_share + (1.0 - alpha) * imminence)


if __name__ == "__main__":
    pmf = np.zeros((3, 24))
    pmf[0, 0] = pmf[1, 11] = pmf[2, 23] = 0.5
    for hour, score in zip((1, 12, 24), pmf_urgency_score(pmf)):
        print(f"Event mass at hour {hour:2d}: urgency={score:.6f}")
