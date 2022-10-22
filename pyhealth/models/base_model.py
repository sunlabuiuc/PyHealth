from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from pyhealth.models.utils import get_default_loss_module
from pyhealth.datasets import BaseDataset

VALID_MODE = ["binary", "multiclass", "multilabel"]


class BaseModel(ABC, nn.Module):
    """Abstract base model for all tasks.

    Args:
        dataset: BaseDataset object
        tables: list of input domains (e.g., ["conditions", "procedures"])
        target: output domain (e.g., "drugs")
        mode: "binary", "multiclass", or "multilabel"
    """

    def __init__(
        self,
        dataset: BaseDataset,
        tables: Union[List[str], Tuple[str]],
        target: str,
        mode: str,
    ):
        super(BaseModel, self).__init__()
        assert mode in VALID_MODE, f"mode must be one of {VALID_MODE}"
        self.dataset = dataset
        self.tables = tables
        self.target = target
        self.mode = mode
        return

    @abstractmethod
    def forward(self, device, **kwargs):
        raise NotImplementedError

    def cal_loss_and_output(self, logits, device, **kwargs):
        """calculate the loss and output. We support binary, multiclass, and multilabel classification.
        Args:
            logits: the output of the model. shape: (batch_size, num_classes)
            device: torch.device
            **kwargs: other arguments
        Returns:
            loss: the loss of the model
            y: ground truth
            y_pred: the output of the model
            y_prob: the probability of the output of the model
        """
        if self.mode in ["multilabel"]:
            kwargs[self.target] = self.label_tokenizer.batch_encode_2d(
                kwargs[self.target], padding=False, truncation=False
            )
            y = torch.zeros(
                len(kwargs[self.target]),
                self.label_tokenizer.get_vocabulary_size(),
                device=device,
            )
            for idx, sample in enumerate(kwargs[self.target]):
                y[idx, sample] = 1

            loss = get_default_loss_module(self.mode)(logits, y.to(device))
            y_prod = torch.sigmoid(logits)
            y_pred = (y_prod > 0.5).int()

        elif self.mode in ["binary"]:
            y = self.label_tokenizer.convert_tokens_to_indices(kwargs[self.target])
            y = torch.FloatTensor(y)
            loss = get_default_loss_module(self.mode)(logits[:, 0], y.to(device))
            y_prod = torch.sigmoid(logits)[:, 0]
            y_pred = (y_prod > 0.5).int()

        elif self.mode in ["multiclass"]:
            y = self.label_tokenizer.convert_tokens_to_indices(kwargs[self.target])
            y = torch.LongTensor(y)
            loss = get_default_loss_module(self.mode)(logits, y.to(device))
            y_prod = torch.softmax(logits, dim=-1)
            y_pred = torch.argmax(y_prod, dim=-1)
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        return loss, y, y_prod, y_pred
