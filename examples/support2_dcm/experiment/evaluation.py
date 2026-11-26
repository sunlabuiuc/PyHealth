# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""IPCW metrics: censoring is estimated exclusively from the training set."""

from typing import Any, Optional

import numpy as np
from sksurv.metrics import brier_score, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator


def choose_horizons(
    train: np.ndarray, evaluate: np.ndarray, proposed: Optional[np.ndarray] = None
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep event-time quartiles that IPCW can score; report omitted ones."""
    proposed = (
        np.unique(np.quantile(train["time"][train["event"]], [0.25, 0.5, 0.75]))
        if proposed is None
        else np.asarray(proposed)
    )
    censor = CensoringDistributionEstimator().fit(train)
    kept, omitted = [], []
    for t in proposed:
        reason = None
        if not (evaluate["time"].min() <= t < evaluate["time"].max()):
            reason = "outside evaluation follow-up interval"
        elif t >= train["time"].max():
            reason = "outside training follow-up support"
        elif censor.predict_proba([t])[0] <= 0.01:
            reason = "training censoring survival <= .01"
        (omitted if reason else kept).append(
            {"time": float(t), "reason": reason} if reason else float(t)
        )
    if not kept:
        raise ValueError("No supported IPCW horizons")
    return np.array(kept), omitted


def metrics(
    train: np.ndarray, evaluate: np.ndarray, survival: np.ndarray, horizons: np.ndarray
) -> dict[str, Any]:
    """Return mean/per-horizon IPCW Brier and per-horizon IPCW concordance."""
    if survival.shape != (len(evaluate), len(horizons)):
        raise ValueError("Prediction shape mismatch")
    if not np.isfinite(survival).all() or np.any((survival < 0) | (survival > 1)):
        raise ValueError("Invalid survival probabilities")
    # Administrative truncation past the LAST training-derived horizon leaves
    # Brier at earlier horizons unchanged and avoids censoring extrapolation for
    # late test deaths outside train support. Never estimate G on held-out data.
    end = float(horizons[-1])
    bounded = evaluate.copy()
    bounded["event"] = evaluate["event"] & (evaluate["time"] <= end)
    bounded["time"] = np.minimum(evaluate["time"], np.nextafter(end, np.inf))
    _, bs = brier_score(train, bounded, survival, horizons)
    concordance = []
    for j, t in enumerate(horizons):
        truncated = evaluate.copy()
        truncated["event"] = evaluate["event"] & (evaluate["time"] < t)
        truncated["time"] = np.minimum(evaluate["time"], float(t))
        try:
            c = concordance_index_ipcw(
                train, truncated, 1 - survival[:, j], tau=float(t)
            )[0]
            concordance.append(float(c))
        except ValueError:
            concordance.append(None)
    return {
        "mean_brier": float(np.mean(bs)),
        "brier": bs.tolist(),
        "ipcw_concordance": concordance,
    }
