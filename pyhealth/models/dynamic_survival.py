"""NumPy helpers of dynamic survival analysis for early event prediction.

Targets and prior of the discrete-time hazard model of Yeche et al.,
"Dynamic Survival Analysis for Early Event Prediction", CHIL 2024,
https://proceedings.mlr.press/v248/yeche24a.html

At a step, the hazard of horizon ``k`` is the probability that the event falls
in hour ``k`` ahead given that it has not happened before. With the hour of the
event, or the observed hours if censored, ``time_to_event`` of the step, the
binary targets of the hazards are ``y_k = 1[k == time_to_event and event]``,
counted only where ``w_k = 1[k <= time_to_event]``, and truncated to the
prediction horizon (Eq. 3 and 4 of the paper).

These functions only use NumPy, so they are the reference that the PyTorch
model of :mod:`pyhealth.models.dynamic_survival_rnn` is tested against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def survival_targets(
    time_to_event: Sequence[float] | np.ndarray,
    event: Sequence[float] | np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hazard targets and weights of steps, truncated to ``horizon`` hours.

    Args:
        time_to_event: Per step, the hour in which the event falls, or the
            whole hours observed if censored (0 if none).
        event: Per step, 1 if the event is observed, else 0.
        horizon: Number of hours ahead to model.

    Returns:
        ``(target, weight)``, each ``[steps, horizon]``: column ``k - 1`` is
        hour ``k``. An event beyond the horizon gives weight 1 and target 0
        for every hour of it.

    Examples:
        >>> y, w = survival_targets([3, 2, 30], [1, 0, 1], horizon=4)
        >>> y.tolist()
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        >>> w.tolist()
        [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    """
    tte = np.asarray(time_to_event, dtype=float)[:, None]
    observed = np.asarray(event, dtype=float)[:, None] > 0.5
    hours = np.arange(1, int(horizon) + 1, dtype=float)[None, :]
    weight = (hours <= tte).astype(float)
    target = ((hours == tte) & observed).astype(float)
    return target, weight


def hazard_prior(
    time_to_event: Sequence[float] | np.ndarray,
    event: Sequence[float] | np.ndarray,
    horizon: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Average hazard of each horizon, to initialize the output bias.

    This is ``y~(k)`` of Section 3.2 of the paper: the mean hazard target of
    hour ``k`` over the steps that count for it. Compute it on the training
    steps only.

    Args:
        time_to_event: As :func:`survival_targets`, over training steps.
        event: As :func:`survival_targets`.
        horizon: Number of hours ahead to model.
        eps: Bounds of the prior, ``[eps, 1 - eps]``, so its log odds are
            finite. A horizon that no step counts for gets the mean of the
            others.

    Returns:
        The prior, ``[horizon]``.

    Examples:
        >>> prior = hazard_prior([2, 1, 3], [1, 1, 0], horizon=3)
        >>> [round(float(p), 3) for p in prior]  # 1/3, 1/2, no event in hour 3
        [0.333, 0.5, 0.0]
    """
    target, weight = survival_targets(time_to_event, event, horizon)
    counted = weight.sum(axis=0)
    rate = np.full(int(horizon), np.nan)
    seen = counted > 0
    rate[seen] = target.sum(axis=0)[seen] / counted[seen]
    if not seen.any():
        raise ValueError("No step has any observed hour: cannot estimate a prior.")
    rate[~seen] = np.nanmean(rate)
    return np.clip(rate, eps, 1.0 - eps)


def risk_from_hazards(hazards: np.ndarray) -> np.ndarray:
    """Cumulative failure ``1 - prod_k (1 - hazard_k)`` over the last axis.

    Examples:
        >>> float(risk_from_hazards(np.array([0.5, 0.5])))
        0.75
    """
    return 1.0 - np.prod(1.0 - np.asarray(hazards, dtype=float), axis=-1)


def survival_nll(
    logits: np.ndarray,
    time_to_event: Sequence[float] | np.ndarray,
    event: Sequence[float] | np.ndarray,
) -> float:
    """Truncated survival loss: mean binary cross-entropy of the hazards.

    The mean is over the (step, hour) entries that count, ``w_k = 1``.
    Truncation is by the width of ``logits``, the horizon.

    Examples:
        >>> round(survival_nll(np.zeros((1, 2)), [2], [1]), 4)  # log 2
        0.6931
    """
    logits = np.asarray(logits, dtype=float)
    target, weight = survival_targets(time_to_event, event, logits.shape[-1])
    if weight.sum() == 0:
        return 0.0
    bce = (
        np.maximum(logits, 0.0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
    )
    return float((bce * weight).sum() / weight.sum())
