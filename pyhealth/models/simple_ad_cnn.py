from abc import ABC
from typing import Callable, Any, Optional
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset
from ..processors import PROCESSOR_REGISTRY
from pyhealth.models import BaseModel


class SimpleADCNN(BaseModel):
    """Convolutional Neural Network for image classification. Primarily for Alzheimer's Classification.

    This SimpleADCNN is a basic implementation of part of the architecture seen in "Back to the basics
    with inclusion of clinical domain knowledge - A simple, scalable, and effective model of Alzheimer's
    Disease Classification". The MRI images are process through 3 layers with batch normalization,
    dropout, and relu. At the end there are two dense layers and pooling.

    The model is intended to be used with "clinical domain knowledge"; for this paper, it means to
    focus the network on analyzing relevant information (the hippocampus in the case of Alzheimer's
    and the topography of the brain) in order to get accurate classification with a relatively
    simple model.

    
    The paper gives a number of params for the network, for which the SimpleADCNN allows
    these to be recreated with the conv_channels parameter.

    - **I-3D** (inner brain): 120x144x120, ~270k params, ACC 0.79
      - approximate with ``conv_channels=(32, 64, 128)`` (~295k)
    - **P*-3D** (best patch): 30x36x30, ~72k params, ACC 0.81
      - approximate with ``conv_channels=(16, 32, 64), dense_dim=64``
        (~74k)
    - **HC-3D** (hippocampus): 33x45x48, ~140k params, ACC 0.84
      - approximate with ``conv_channels=(16, 32, 128)`` (~142k)

    Paper: https://proceedings.mlr.press/v149/bruningk21a.html

    Args:
        dataset (SampleDataset): The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        in_channels (int): The number of input channels for the first convolutional layer.
        conv_channels (tuple[int, ]): Channels for each layer. 
        kernel_size (int): Convolutional kernel size
        dropout (float): dropout probability
        dense_dim (int): dimension of the hidden dense layer of simple CNN.
            
    Examples:
         from pyhealth.datasets import create_sample_dataset
         import torch
         samples = [
             {
                 "patient_id": "p0",
                 "visit_id": "v0",
                 "mri": torch.randn(1, 33, 45, 48).tolist(),
                 "label": 1,
             },
             {
                 "patient_id": "p1",
                 "visit_id": "v0",
                 "mri": torch.randn(1, 33, 45, 48).tolist(),
                 "label": 0,
             },
         ]
         dataset = create_sample_dataset(
             samples=samples,
             input_schema={"mri": "tensor"},
             output_schema={"label": "binary"},
             dataset_name="demo",
         )
         model = AD3DCNN(dataset=dataset)
    """

    def __init__(
        self, 
        dataset: SampleDataset,
        in_channels: int = 1,
        conv_channels: tuple[int,  ] = (16, 32, 64),
        kernel_size: int = 3,                 
        dropout:float = 0.4, 
        dense_dim: int = 128           
    ):
        """
        Initializes the SimpleADCNN.

        Args:
            dataset (SampleDataset): The dataset to train the model.
        """
        super(SimpleADCNN, self).__init__(dataset = dataset)

        if len(self.feature_keys) != 1:
            raise ValueError(
                f"AD3DCNN expects exactly one input feature (the 3D MRI "
                f"volume), got {len(self.feature_keys)}: {self.feature_keys}"
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                f"AD3DCNN expects exactly one label key, "
                f"got {len(self.label_keys)}: {self.label_keys}"
            )
        self.label_key = self.label_keys[0]

        if in_channels < 1:
            raise ValueError(f"must have at least one inut channel, instead got {in_channels}")
        if not conv_channels:
            raise ValueError("must have at least one output channel")
        if any(ch < 1 for ch in conv_channels):
            raise ValueError(
                f"conv_channels must be positive, got "
                f"{conv_channels}"
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be valid probability, got {dropout}")
        if dense_dim < 1:
            raise ValueError(f"dense_dim must be greater than 1, got {dense_dim}")

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dense_dim = dense_dim

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend(
                [
                    nn.Conv3d(ch_in, ch_out, kernel_size, padding=padding),
                    nn.BatchNorm3d(ch_out),
                    nn.ReLU(),
                    nn.Dropout3d(dropout),
                ]
            )
            ch_in = ch_out
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())

        # sequential instead of blocks like "cnn.py"
        self.features = nn.Sequential(*layers)

        output_size = self.get_output_size()
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, output_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply He-uniform initialization to all layers.

        Conv3d and Linear layers get ``kaiming_uniform_`` on their
        weights (matching the paper's "He uniform initialisation").
        BatchNorm3d layers get weight=1, bias=0.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, 
            **kwargs: torch.Tensor | tuple[torch.Tensor, ]
        ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            **kwargs: A variable number of keyword arguments representing input features.
                Each keyword argument is a tensor or a tuple of tensors of shape (batch_size, ).
        
        Returns:
            A dictionary with the following keys:
                logit: a tensor of predicted logits.
                y_prob: a tensor of predicted probabilities.
                loss [optional]: a scalar tensor representing the final loss, if self.label_keys in kwargs.
                y_true [optional]: a tensor representing the true labels, if self.label_keys in kwargs.
        """
        feature_key = self.feature_keys[0]
        x = kwargs[feature_key]
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.to(self.device).float()

        # (B, D, H, W) or (B, C, D, H, W).
        if x.dim() not in (4, 5):
            raise ValueError(
                f"Expected MRI tensor with 4 or 5 dimensions, got shape "
                f"{tuple(x.shape)}"
            )
        if x.dim() == 4:
            if self.in_channels != 1:
                raise ValueError(
                    "Input is missing an explicit channel dimension. "
                    f"Automatic unsqueeze is only supported when "
                    f"in_channels=1, got in_channels={self.in_channels}."
                )
            x = x.unsqueeze(1)
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected MRI tensor with {self.in_channels} channels, got "
                f"{x.shape[1]}"
            )

        x = self.features(x)
        logits = self.classifier(x)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_mode(self, schema_entry: Any) -> str:
        """Resolve a mode string from an output_schema entry.

        Supports:
          - direct string ("binary", )
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
