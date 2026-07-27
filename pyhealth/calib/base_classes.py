from abc import ABC
from typing import Dict

import torch


class PostHocCalibrator(ABC, torch.nn.Module):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model

    def calibrate(self, cal_dataset):
        ...

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        ...


    def to(self, device):
        super().to(device)
        self.device = device
        return self


class SetPredictor(ABC, torch.nn.Module):
    """Base class for conformal-prediction-style prediction-set constructors.

    Interface convention for `calibrate()`: every concrete subclass should
    accept `calibrate(self, cal_dataset, train_dataset=None, test_dataset=None,
    **kwargs)`. `cal_dataset` is required; `train_dataset`/`test_dataset` are
    optional and only meaningful to methods that need extra data beyond the
    calibration set (e.g. embeddings-based methods like ClusterLabel,
    CovariateLabel, NeighborhoodLabel) -- methods that don't need them should
    still accept and ignore the two keyword arguments rather than omitting
    them from the signature.

    This lets a generic caller (e.g. a benchmark harness) calibrate *any*
    SetPredictor -- including ones not yet written -- with the same call:
    `cp_model.calibrate(cal_dataset=cal, train_dataset=train, test_dataset=test)`,
    without needing to special-case which extra data each method requires.
    Methods that need embeddings should extract them internally (e.g. via
    `pyhealth.calib.utils.extract_embeddings`) from whichever of
    train_dataset/test_dataset they're given, rather than requiring the
    caller to pre-compute and pass embeddings under method-specific kwarg
    names -- pre-computed embeddings may still be accepted as additional
    optional kwargs for callers who want to reuse/precompute them.
    """

    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model

    def calibrate(self, cal_dataset, train_dataset=None, test_dataset=None):
        ...

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def to(self, device):
        super().to(device)
        self.device = device
        return self