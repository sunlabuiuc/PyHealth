"""Shared fixtures for the time series conformal prediction tests.

Holds tiny regression streams and fake regression models with known outputs,
for the EnbPI tests now and the ACI and AgACI tests later. It defines no test
cases, so unittest discovery does not collect it.
"""

from typing import List, Optional

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset


def make_stream(
    y: List[float],
    x: Optional[List[float]] = None,
):
    """Regression dataset with one sample per time step.

    Each sample has a 1-d tensor feature ``x`` and a ``regression`` label
    ``y``, in time order.
    """
    x = list(y) if x is None else x
    samples = [
        {"patient_id": "p0", "x": [float(xi)], "y": float(yi)} for xi, yi in zip(x, y)
    ]
    return create_sample_dataset(samples, {"x": "tensor"}, {"y": "regression"})


def make_ar_stream(n: int, phi: float = 0.8, lags: int = 2, seed: int = 0):
    """AR(1) series with unit Gaussian noise, ``lags`` past values as features."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n + lags)
    for t in range(1, len(y)):
        y[t] = phi * y[t - 1] + rng.normal()
    samples = [
        {"patient_id": "p0", "x": y[t - lags : t].tolist(), "y": float(y[t])}
        for t in range(lags, len(y))
    ]
    return create_sample_dataset(samples, {"x": "tensor"}, {"y": "regression"})


class ConstantModel(torch.nn.Module):
    """Fake regression model that predicts ``value`` for every input."""

    def __init__(self, value: float, mode: str = "regression") -> None:
        super().__init__()
        self.value = value
        self.mode = mode
        self.label_keys = ["y"]

    def forward(self, **batch):
        n = len(batch["y"])
        return {"y_prob": torch.full((n, 1), float(self.value))}
