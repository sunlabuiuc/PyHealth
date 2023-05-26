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