"""Inter-domain Pairwise Distance (IPD) and source decay rates.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Measures how similar each source sensor is to the target sensor
    from synchronized (paired) segments, and turns the distances into the
    per-source decay rates that set how much each source teaches the model.

Method (Section 4 of the paper):
    1. For every pair, a DTW distance per channel gives a difference vector
       (:func:`dtw_per_channel`, :func:`paired_differences`).
    2. A smooth bootstrap: a Gaussian kernel density estimate of the difference
       vectors is sampled ``m`` times and IPD is ``norm(samples) / m``
       (:func:`smooth_bootstrap_ipd`).
    3. A source's weight is ``alpha = IPD / sum(IPD)`` and its learning rate is
       multiplied by ``1 - alpha`` after every epoch (:func:`decay_rates`), so
       closer sources keep a larger learning rate for longer.

Differences from the paper, where it leaves details open: the kernel bandwidth
    uses Scott's rule, samples are drawn directly from the Gaussian KDE (no
    MCMC), and the matrix norm is the Frobenius norm.
"""

import numpy as np

WEIGHT_MODES = ("ipd", "equal", "inverse")


def dtw_per_channel(
    a: np.ndarray, b: np.ndarray, window: int | None = None
) -> np.ndarray:
    """Computes an independent DTW distance for each channel.

    Args:
        a: Array of shape ``(T1, K)``.
        b: Array of shape ``(T2, K)``.
        window: Optional Sakoe-Chiba band half-width, for speed.

    Returns:
        An array of shape ``(K,)`` with one DTW distance per channel (absolute
        difference as the local cost).

    Examples:
        >>> import numpy as np
        >>> x = np.arange(6, dtype=float).reshape(3, 2)
        >>> dtw_per_channel(x, x).tolist()
        [0.0, 0.0]
    """
    n, m, k = len(a), len(b), a.shape[1]
    band = max(window, abs(n - m)) if window is not None else max(n, m)
    cost = np.full((n + 1, m + 1, k), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            best = np.minimum(
                np.minimum(cost[i - 1, j], cost[i, j - 1]), cost[i - 1, j - 1]
            )
            cost[i, j] = np.abs(a[i - 1] - b[j - 1]) + best
    return cost[n, m].copy()


def paired_differences(
    source: np.ndarray, target: np.ndarray, window: int | None = 10
) -> np.ndarray:
    """Computes the per-channel DTW difference vector of every pair.

    Row ``i`` of ``source`` and ``target`` is the same time window recorded by
    the two sensors.

    Args:
        source: Array of shape ``(N, T, K)``.
        target: Array of shape ``(N, T, K)``, paired with ``source``.
        window: Sakoe-Chiba band half-width.

    Returns:
        An array of shape ``(N, K)``.

    Raises:
        ValueError: If the arrays have a different number of pairs.

    Examples:
        >>> import numpy as np
        >>> x = np.zeros((3, 5, 2))
        >>> paired_differences(x, x).shape
        (3, 2)
    """
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of pairs")
    rows = [dtw_per_channel(s, t, window) for s, t in zip(source, target)]
    return np.stack(rows)


def smooth_bootstrap_ipd(
    differences: np.ndarray, n_samples: int = 1000, seed: int = 0
) -> float:
    """Estimates the IPD from difference vectors by a smooth bootstrap.

    Fits a Gaussian kernel density estimate (Scott's-rule bandwidth) to the
    difference vectors, draws ``n_samples`` from it, and returns the norm of
    the sample matrix divided by ``n_samples``.

    Args:
        differences: Array of shape ``(N, K)`` from :func:`paired_differences`.
        n_samples: Number of samples drawn from the density estimate.
        seed: Seed of the random generator.

    Returns:
        The estimated IPD; lower means more similar domains.

    Raises:
        ValueError: If there are fewer than two difference vectors.

    Examples:
        >>> import numpy as np
        >>> d = np.abs(np.random.default_rng(0).normal(size=(8, 3)))
        >>> smooth_bootstrap_ipd(d) > 0
        True
    """
    n, k = differences.shape
    if n < 2:
        raise ValueError("at least two pairs are needed")
    covariance = np.atleast_2d(np.cov(differences, rowvar=False))
    bandwidth = covariance * n ** (-2.0 / (k + 4)) + 1e-12 * np.eye(k)
    rng = np.random.default_rng(seed)
    centers = differences[rng.integers(0, n, size=n_samples)]
    noise = rng.multivariate_normal(np.zeros(k), bandwidth, size=n_samples)
    return float(np.linalg.norm(centers + noise) / n_samples)


def paired_ipd(
    source: np.ndarray,
    target: np.ndarray,
    window: int | None = 10,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    """Computes the IPD between a source and the target sensor.

    Args:
        source: Array of shape ``(N, T, K)``.
        target: Array of shape ``(N, T, K)``, paired with ``source``.
        window: Sakoe-Chiba band half-width.
        n_samples: Samples drawn in the smooth bootstrap.
        seed: Seed of the smooth bootstrap.

    Returns:
        The IPD; lower means more similar.

    Examples:
        >>> import numpy as np
        >>> x = np.random.default_rng(0).normal(size=(4, 6, 2))
        >>> paired_ipd(x, x, window=2) < paired_ipd(x + 5, x, window=2)
        True
    """
    differences = paired_differences(source, target, window)
    return smooth_bootstrap_ipd(differences, n_samples, seed)


def decay_rates(
    distances: dict[str, float], mode: str = "ipd"
) -> dict[str, float]:
    """Converts per-source IPDs into decay rates ``alpha`` that sum to one.

    The learning rate of a source is multiplied by ``1 - alpha`` after every
    epoch, so a larger ``alpha`` means the source teaches the model less.

    Args:
        distances: Maps each source name to its IPD to the target.
        mode: ``"ipd"`` (the paper) sets ``alpha`` proportional to the
            distance, so closer sources keep their learning rate longer.
            ``"equal"`` gives every source the same ``alpha``. ``"inverse"``
            reverses the direction (``alpha`` proportional to one over the
            distance), so closer sources decay faster.

    Returns:
        Maps each source name to its ``alpha``.

    Raises:
        ValueError: If ``mode`` is unknown or ``distances`` is empty.

    Examples:
        >>> a = decay_rates({"near": 1.0, "far": 3.0}, mode="ipd")
        >>> a["near"] < a["far"]
        True
        >>> decay_rates({"a": 1.0, "b": 3.0}, mode="equal")
        {'a': 0.5, 'b': 0.5}
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got '{mode}'")
    if not distances:
        raise ValueError("distances must not be empty")
    names = list(distances)
    g = np.maximum(np.array([distances[n] for n in names], dtype=float), 1e-12)
    if mode == "equal":
        raw = np.ones_like(g)
    elif mode == "ipd":
        raw = g
    else:
        raw = 1.0 / g
    return dict(zip(names, (raw / raw.sum()).tolist()))
