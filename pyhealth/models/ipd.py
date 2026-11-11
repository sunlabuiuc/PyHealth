"""Inter-domain Pairwise Distance (IPD) and source weighting.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Measures how similar each source sensor is to the target sensor
    from synchronized (paired) segments, and turns those distances into
    per-source weights and learning-rate scales.

Note:
    The exact IPD definition and learning-rate rule are our reading of the
    paper: the mean DTW distance over paired segments, a weight of
    ``1 / (1 + distance)`` and a learning-rate scale of ``weight * n_sources``.
    Check them against the paper and the authors' code before final runs.
"""

import numpy as np

WEIGHT_MODES = ("ipd", "equal", "inverse")


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    """Computes the DTW distance between two multivariate series.

    Args:
        a: Array of shape ``(T1, C)``.
        b: Array of shape ``(T2, C)``.
        window: Optional Sakoe-Chiba band half-width, for speed.

    Returns:
        The accumulated DTW cost with a Euclidean local cost.

    Examples:
        >>> import numpy as np
        >>> x = np.arange(6, dtype=float).reshape(3, 2)
        >>> dtw_distance(x, x)
        0.0
    """
    n, m = len(a), len(b)
    band = max(window, abs(n - m)) if window is not None else max(n, m)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        local = np.linalg.norm(a[i - 1] - b[lo - 1 : hi], axis=1)
        for k, j in enumerate(range(lo, hi + 1)):
            best = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = local[k] + best
    return float(cost[n, m])


def paired_ipd(
    source: np.ndarray, target: np.ndarray, window: int | None = 10
) -> float:
    """Computes the IPD between a source and the target sensor.

    Row ``i`` of ``source`` and ``target`` is the same time window recorded
    by the two sensors, so the distance is taken over synchronized pairs.

    Args:
        source: Array of shape ``(N, T, C)``.
        target: Array of shape ``(N, T, C)``, paired with ``source``.
        window: Sakoe-Chiba band half-width.

    Returns:
        The mean paired DTW distance; lower means more similar.

    Raises:
        ValueError: If the two arrays have a different number of pairs, or
            no pairs.

    Examples:
        >>> import numpy as np
        >>> x = np.zeros((2, 5, 3))
        >>> paired_ipd(x, x)
        0.0
    """
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of pairs")
    if source.shape[0] == 0:
        raise ValueError("at least one pair is needed")
    dists = [dtw_distance(s, t, window) for s, t in zip(source, target)]
    return float(np.mean(dists))


def source_weights(
    distances: dict[str, float], mode: str = "ipd"
) -> dict[str, float]:
    """Converts per-source distances into weights that sum to one.

    Args:
        distances: Maps each source name to its IPD to the target.
        mode: ``"ipd"`` gives closer sources larger weights, ``"equal"`` gives
            every source the same weight, and ``"inverse"`` reverses the
            direction (farther sources get larger weights).

    Returns:
        Maps each source name to its weight.

    Raises:
        ValueError: If ``mode`` is unknown or ``distances`` is empty.

    Examples:
        >>> w = source_weights({"a": 1.0, "b": 9.0}, mode="ipd")
        >>> w["a"] > w["b"]
        True
        >>> source_weights({"a": 1.0, "b": 9.0}, mode="equal")
        {'a': 0.5, 'b': 0.5}
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got '{mode}'")
    if not distances:
        raise ValueError("distances must not be empty")
    names = list(distances)
    d = np.array([distances[n] for n in names], dtype=float)
    if mode == "equal":
        raw = np.ones_like(d)
    elif mode == "ipd":
        raw = 1.0 / (1.0 + d)
    else:
        raw = d + 1e-12
    raw = raw / raw.sum()
    return dict(zip(names, raw.tolist()))


def learning_rate_scales(weights: dict[str, float]) -> dict[str, float]:
    """Turns weights into learning-rate multipliers with a mean of one.

    Equal weights therefore leave the base learning rate unchanged, and more
    similar sources train with a larger learning rate than less similar ones.

    Args:
        weights: Maps each source name to its weight, from
            :func:`source_weights`.

    Returns:
        Maps each source name to ``weight * number_of_sources``.

    Examples:
        >>> learning_rate_scales({"a": 0.75, "b": 0.25})
        {'a': 1.5, 'b': 0.5}
    """
    n = len(weights)
    return {name: value * n for name, value in weights.items()}
