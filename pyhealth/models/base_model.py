from abc import ABC
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset


class BaseModel(ABC, nn.Module):
    """Abstract class for PyTorch models."""

    def __init__(
        self, 
        dataset: SampleDataset,
        feature_keys: Optional[List[str]] = None,
        label_key: Optional[str] = None,
        mode: Optional[str] = None,
        pretrained_emb: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the BaseModel.

        Args:
            dataset (SampleDataset): The dataset to train the model.
            feature_keys (Optional[List[str]]): Specific feature keys to use. If None, use all from dataset.
            label_key (Optional[str]): Specific label key to use. If None, use first from dataset.
            mode (Optional[str]): Output mode. If None, inferred from dataset.
            pretrained_emb (Optional[str]): Path to pretrained embeddings. If None, train from scratch.
            **kwargs: Additional keyword arguments.
        """
        super(BaseModel, self).__init__()
        self.dataset = dataset
        
        # Handle feature keys - either use provided ones or all from dataset
        if feature_keys is not None:
            self.feature_keys = feature_keys
        else:
            self.feature_keys = list(dataset.input_schema.keys())
            
        # Handle label key - either use provided one or first from dataset
        if label_key is not None:
            self.label_key = label_key
            self.label_keys = [label_key]  # For backward compatibility
        else:
            self.label_keys = list(dataset.output_schema.keys())
            if self.label_keys:
                self.label_key = self.label_keys[0]
                
        # Handle mode - either use provided one or infer from dataset
        if mode is not None:
            self.mode = mode
        elif hasattr(dataset, 'output_schema') and self.label_keys:
            self.mode = dataset.output_schema[self.label_keys[0]]
        
        # Store pretrained embeddings path if provided
        self.pretrained_emb = pretrained_emb
            
        # Used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))

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
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if get_output_size is called"
        )
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
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if get_loss_function is called"
        )
        label_key = self.label_keys[0]
        mode = self.dataset.output_schema[label_key]
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
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if get_loss_function is called"
        )
        label_key = self.label_keys[0]
        mode = self.dataset.output_schema[label_key]
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
