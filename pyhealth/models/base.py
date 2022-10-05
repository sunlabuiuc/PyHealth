from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch.nn as nn

from pyhealth.data import TaskDataset

VALID_MODE = ["binary", "multiclass", "multilabel"]


class BaseModel(ABC, nn.Module):
    """Abstract base model for all tasks.

    Args:
        dataset: TaskDataset object
        input_domains: list of input domains (e.g., ["conditions", "procedures"])
        output_domain: output domain (e.g., "drugs")
        mode: "binary", "multiclass", or "multilabel"
    """

    def __init__(
            self,
            dataset: TaskDataset,
            input_domains: Union[List[str], Tuple[str]],
            output_domain: str,
            mode: str,
    ):
        super(BaseModel, self).__init__()
        assert mode in VALID_MODE, f"mode must be one of {VALID_MODE}"
        self.dataset = dataset
        self.input_domains = input_domains
        self.output_domain = output_domain
        self.mode = mode
        return

    @abstractmethod
    def forward(self, device, training, **kwargs):
        raise NotImplementedError
