from abc import ABC
from typing import Dict, List, Optional, Tuple

import torch


class PostHocCalibrator(ABC, torch.nn.Module):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model

    def fit(self, train_dataset):
        ...

    def calibrate(self, cal_dataset):
        ...

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        ...
