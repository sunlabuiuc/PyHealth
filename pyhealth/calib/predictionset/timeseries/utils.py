"""Helpers shared by the time series conformal interval methods.

Plain functions, like ``prepare_numpy_dataset`` for the set predictors. Each
method uses the ones it needs and implements its own online loop.
"""

import math

import numpy as np
import torch

__all__ = ["conformal_quantile"]


def conformal_quantile(scores, alpha: float, presorted: bool = False) -> float:
    """Conformal ``(1 - alpha)`` quantile of a window of scores.

    Returns the ``ceil((1 - alpha) * (n + 1))``-th smallest score, the same
    convention as split conformal prediction. Adaptive methods move ``alpha``
    outside ``[0, 1]``, so the edge cases are part of the contract.

    Args:
        scores: Scores of shape (n,). Higher means less conforming.
        alpha: Miscoverage level. Any real value is accepted.
        presorted: Whether ``scores`` is already sorted in ascending order.
            It is then indexed directly, without copying, which lets online
            methods keep a sorted window and read its quantile in constant
            time. Any sequence works, such as a list. Default False.

    Returns:
        The quantile. ``+inf`` when the rank exceeds ``n``, which covers
        ``alpha <= 0`` and windows too small for ``alpha``. ``-inf`` when the
        rank is below 1, which covers ``alpha >= 1``.

    Examples:
        >>> import numpy as np
        >>> conformal_quantile(np.arange(1.0, 10.0), alpha=0.2)
        8.0
        >>> conformal_quantile(np.arange(1.0, 10.0), alpha=0.0)
        inf
    """
    if not presorted:
        scores = np.asarray(scores, dtype=float).ravel()
    n = len(scores)
    # The small tolerance stops float error from bumping an exact rank up by
    # one, e.g. (1 - 0.1) * 10 evaluates to 9.000000000000002.
    rank = math.ceil((1 - alpha) * (n + 1) - 1e-9)
    if rank > n:
        return math.inf
    if rank < 1:
        return -math.inf
    if presorted:
        return float(scores[rank - 1])
    return float(np.partition(scores, rank - 1)[rank - 1])


def as_numpy(values) -> np.ndarray:
    """Flattens a label tensor or array to a float array of shape (n,)."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def as_column(values: np.ndarray) -> torch.Tensor:
    """Converts an array of shape (n,) to a float32 tensor of shape (n, 1)."""
    return torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1, 1)
