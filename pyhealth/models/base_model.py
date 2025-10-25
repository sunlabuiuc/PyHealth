from abc import ABC
from typing import Callable, Any
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset
from ..processors import PROCESSOR_REGISTRY


class BaseModel(ABC, nn.Module):
    """Abstract class for PyTorch models.

    Args:
        dataset (SampleDataset): The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
    """

    def __init__(self, dataset: SampleDataset = None):
        """
        Initializes the BaseModel.

        Args:
            dataset (SampleDataset): The dataset to train the model.
        """
        super(BaseModel, self).__init__()
        self.dataset = dataset
        self.feature_keys = []
        self.label_keys = []
        if dataset:
            self.feature_keys = list(dataset.input_schema.keys())
            self.label_keys = list(dataset.output_schema.keys())
            # if single label, try to resolve mode for legacy trainer usage
            if len(self.label_keys) == 1:
                try:
                    m = self._resolve_mode(dataset.output_schema[self.label_keys[0]])
                    if m in {"binary", "multiclass", "multilabel", "regression"}:
                        self.mode = m
                except Exception:
                    pass
        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.mode = getattr(self, "mode", None)  # legacy API

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_mode(self, schema_entry: Any) -> str:
        """Resolve a mode string from an output_schema entry.

        Supports:
          - direct string ("binary", ...)
          - processor class
          - processor instance
        Returns the registered processor name if found.
        """
        if isinstance(schema_entry, str):
            return schema_entry.lower()

        # Get class reference
        cls = schema_entry if inspect.isclass(schema_entry) else schema_entry.__class__
        for name, registered_cls in PROCESSOR_REGISTRY.items():
            if cls is registered_cls or issubclass(
                cls, registered_cls
            ):  # allow subclassing
                return name.lower()
        raise ValueError(
            f"Cannot resolve mode from output_schema entry {schema_entry}. Use a supported string"
        )

    @property
    def device(self) -> torch.device:
        """
        Gets the device of the model.

        Returns:
            torch.device: The device on which the model is located.
        """
        return self._dummy_param.device

    def get_output_size(self) -> int:
        """
        Gets the default output size using the label tokenizer and `self.mode`.

        If the mode is "binary", the output size is 1. If the mode is "multiclass"
        or "multilabel", the output size is the number of classes or labels.

        Returns:
            int: The output size of the model.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_output_size is called"
        output_size = self.dataset.output_processors[self.label_keys[0]].size()
        return output_size

    def get_loss_function(self) -> Callable:
        """
        Gets the default loss function using `self.mode`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`
            - regression: `F.mse_loss`

        Returns:
            Callable: The default loss function.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_loss_function is called"
        label_key = self.label_keys[0]
        mode = self._resolve_mode(self.dataset.output_schema[label_key])
        if mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif mode == "multiclass":
            return F.cross_entropy
        elif mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        elif mode == "regression":
            return F.mse_loss
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Prepares the predicted probabilities for model evaluation.

        This function converts the predicted logits to predicted probabilities
        depending on the mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1) with values in [0, 1],
                which is obtained with `torch.sigmoid()`
            - multiclass: a tensor of shape (batch_size, num_classes) with
                values in [0, 1] and sum to 1, which is obtained with
                `torch.softmax()`
            - multilabel: a tensor of shape (batch_size, num_labels) with values
                in [0, 1], which is obtained with `torch.sigmoid()`
            - regression: a tensor of shape (batch_size, 1) with raw logits

        Args:
            logits (torch.Tensor): The predicted logit tensor.

        Returns:
            torch.Tensor: The predicted probability tensor.
        """
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if get_loss_function is called"
        label_key = self.label_keys[0]
        mode = self._resolve_mode(self.dataset.output_schema[label_key])
        if mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        elif mode in ["regression"]:
            y_prob = logits
        else:
            raise NotImplementedError
        return y_prob
