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


class IntervalPredictor(ABC, torch.nn.Module):
    """Interface for regression methods that output prediction intervals.

    ``forward`` returns the point prediction as ``y_prob`` together with
    ``y_lower`` and ``y_upper``. Online methods also learn from revealed
    labels through ``update``.

    Examples:
        >>> from pyhealth.calib.predictionset import EnbPI
        >>> predictor = EnbPI(model_fn=model_fn, alpha=0.1)
        >>> predictor.calibrate(train_ds)
        >>> out = predictor(**batch)  # y_prob, y_lower and y_upper
        >>> predictor.update(revealed_ds)  # learn from newly revealed labels
    """

    def calibrate(self, cal_dataset):
        ...

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        ...

    def update(self, revealed_dataset):
        ...

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class SetPredictor(ABC, torch.nn.Module):
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