# Author: Paul Nguyen, Shayan Jaffar, William Lee
# Description: 3D CNN for Alzheimer's disease classification
# Reference: Liu et al. 2020, "On the Design of Convolutional Neural Networks
#            for Automatic Detection of Alzheimer's Disease"
#            (http://proceedings.mlr.press/v116/liu20a)

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel

# Architecture constants from reference implementation
# https://github.com/NYUMedML/CNN_design_for_AD/blob/master/models/models.py
_BLOCK_KERNELS = [1, 3, 5, 3]  # Conv kernel sizes for each block
_BLOCK_CHANNELS = [4, 32, 64, 64]  # Base channel counts (scaled by widening_factor)
_BLOCK_DILATIONS = [1, 2, 2, 2]  # Dilation rates for dilated convolutions
_BLOCK_PADDINGS = [0, 0, 2, 1]  # Padding values for each conv layer
_POOL_KERNELS = [3, 3, 3, 5]  # MaxPool kernel sizes for each block


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    """Create a normalization layer.

    Args:
        norm_type: Type of normalization - "instance" or "batch".
        num_channels: Number of input channels.

    Returns:
        A normalization module.

    Raises:
        ValueError: If norm_type is not "instance" or "batch".
    """
    if norm_type == "instance":
        # affine=False matches reference implementation default
        return nn.InstanceNorm3d(num_channels)
    elif norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Expected 'instance' or 'batch'.")


class ConvBlock3D(nn.Module):
    """3D convolutional block with normalization and ReLU activation.

    Architecture: Conv3d -> Norm -> ReLU

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        norm_type: Type of normalization ("instance" or "batch"). Default: "instance".
        stride: Stride of the convolution. Default: 1.
        dilation: Dilation rate for the convolution. Default: 1.
        padding: Zero-padding added to all sides. If None, uses kernel_size // 2.

    Note:
        Input shape: [B, in_ch, D, H, W]
        Output shape: [B, out_ch, D', H', W'] where spatial dims depend on kernel/stride/padding

        Unlike reference implementation, we use bias=False since normalization follows.
        This is an intentional deviation for better training stability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "instance",
        stride: int = 1,
        dilation: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=False,  # Intentional deviation: bias=False since norm follows
            ),
            _make_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block.

        Args:
            x: Input tensor of shape [B, in_channels, D, H, W].

        Returns:
            Output tensor of shape [B, out_channels, D', H', W'].
        """
        return self.block(x)


class CNN3DAD(BaseModel):
    """3D CNN for Alzheimer's Disease classification.

    Implements the architecture from Liu et al. 2020 "On the Design of
    Convolutional Neural Networks for Automatic Detection of Alzheimer's Disease".

    The model consists of:
    - 4 dilated convolutional blocks with instance normalization
    - Adaptive global average pooling (improvement over reference's hardcoded flatten)
    - Optional sinusoidal age encoding with MLP fusion
    - Two-layer classifier head

    Args:
        dataset: PyHealth SampleDataset with fitted processors.
        scan_key: Key for the 3D MRI scan in the input data. Default: "scan".
        age_key: Key for the patient age in the input data. Default: "age".
        label_key: Key for the classification label. Default: "label".
        norm_type: Type of normalization ("instance" or "batch"). Default: "instance".
        widening_factor: Channel multiplier for the backbone. Default: 8.
        num_blocks: Number of convolutional blocks. Default: 4.
        age_encoding_dim: Dimension of age encoding. Set to 0 to disable. Default: 512.
        classifier_hidden: Hidden dimension of classifier. Default: 200.
        dropout: Dropout rate after age fusion. Default: 0.1.

    Reference:
        Liu et al. 2020, "On the Design of Convolutional Neural Networks for
        Automatic Detection of Alzheimer's Disease"
        http://proceedings.mlr.press/v116/liu20a

        Code reference: https://github.com/NYUMedML/CNN_design_for_AD

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "scan": np.random.randn(1,96,96,96),
        ...      "age": np.array([60.0]), "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"scan": "tensor", "age": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="adni",
        ... )
        >>> model = CNN3DAD(dataset=dataset)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        scan_key: str = "scan",
        age_key: str = "age",
        label_key: str = "label",
        norm_type: str = "instance",
        widening_factor: int = 8,
        num_blocks: int = 4,
        age_encoding_dim: int = 512,
        classifier_hidden: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__(dataset)

        # Store configuration
        self.scan_key = scan_key
        self.age_key = age_key
        self.label_key = label_key
        self.norm_type = norm_type
        self.use_age_encoding = age_encoding_dim > 0
        self.age_encoding_dim = age_encoding_dim

        # Build backbone: Conv blocks + MaxPool layers
        blocks = []
        in_ch = 1  # Single channel MRI input

        for i in range(num_blocks):
            # Use last valid spec if num_blocks exceeds defined specs
            spec_idx = min(i, len(_BLOCK_KERNELS) - 1)

            out_ch = _BLOCK_CHANNELS[spec_idx] * widening_factor
            kernel = _BLOCK_KERNELS[spec_idx]
            dilation = _BLOCK_DILATIONS[spec_idx]
            padding = _BLOCK_PADDINGS[spec_idx]
            pool_kernel = _POOL_KERNELS[spec_idx]

            # Add conv block
            blocks.append(
                ConvBlock3D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    norm_type=norm_type,
                    dilation=dilation,
                    padding=padding,
                )
            )

            # Add pooling
            blocks.append(nn.MaxPool3d(kernel_size=pool_kernel, stride=2))

            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)

        # Global pooling (improvement over reference's hardcoded 5x5x5 flatten)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # FC layer to feature dimension
        self.fc1 = nn.Linear(in_ch, 1024)

        # Age encoding (sinusoidal positional encoding)
        if self.use_age_encoding:
            # Build positional encoding table (max_len=240 for ages 0-120 at 0.5 year resolution)
            max_len = 240
            d_model = age_encoding_dim

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer("age_pe", pe)

            # Age MLP: Linear -> LayerNorm -> Linear (NO ReLU per reference)
            self.age_fc = nn.Sequential(
                nn.Linear(age_encoding_dim, 512),
                nn.LayerNorm(512),
                nn.Linear(512, 1024),
            )

            # Dropout applied after residual addition
            self.age_dropout = nn.Dropout(p=dropout)
        else:
            self.age_fc = None
            self.age_dropout = None

        # Two-layer classifier head (matches reference's LinearClassifierAlexNet)
        self.classifier = nn.Sequential(
            nn.Linear(1024, classifier_hidden),
            nn.Linear(classifier_hidden, self.get_output_size()),
        )

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using Xavier normal and bias to 0.1.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for the CNN3DAD model.

        Args:
            **kwargs: Dictionary containing:
                - scan_key: 3D MRI scan tensor [B, D, H, W] or [B, 1, D, H, W]
                - age_key: Patient age tensor [B] or [B, 1]
                - label_key: Classification labels [B]

        Returns:
            Dictionary with keys:
                - loss: Scalar loss value
                - y_prob: Predicted probabilities [B, num_classes]
                - y_true: True labels [B]
                - logit: Raw logits [B, num_classes]
        """
        # Extract inputs
        scan = kwargs[self.scan_key]
        age = kwargs[self.age_key]
        label = kwargs[self.label_key]

        # Cast to float
        scan = scan.float()
        age = age.float()

        # Ensure scan has channel dimension [B, 1, D, H, W]
        if scan.dim() == 4:
            scan = scan.unsqueeze(1)

        # Ensure age has shape [B, 1]
        if age.dim() == 1:
            age = age.unsqueeze(1)

        # Move to device
        scan = scan.to(self.device)
        age = age.to(self.device)
        label = label.to(self.device)

        # Forward through backbone
        feat = self.backbone(scan)
        feat = self.global_pool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc1(feat)

        # Apply age encoding if enabled
        if self.use_age_encoding:
            # Convert age to index: age * 2 to get half-year resolution
            age_idx = (age.squeeze(1) * 2).long().clamp(0, 239)
            age_embed = self.age_pe[age_idx]  # [B, age_encoding_dim]
            age_embed = self.age_fc(age_embed)  # [B, 1024]
            feat = self.age_dropout(feat + age_embed)

        # Classifier
        logits = self.classifier(feat)

        # Compute outputs
        y_true = label
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }


if __name__ == "__main__":
    # Create synthetic ADNI-like dataset for testing
    samples = [
        {
            "patient_id": "p0",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([60.0]),
            "label": 0,  # CN (Cognitively Normal)
        },
        {
            "patient_id": "p1",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([72.0]),
            "label": 1,  # MCI (Mild Cognitive Impairment)
        },
        {
            "patient_id": "p2",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([68.0]),
            "label": 2,  # AD (Alzheimer's Disease)
        },
        {
            "patient_id": "p3",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([65.0]),
            "label": 0,  # CN
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"scan": "tensor", "age": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="adni_test",
    )

    # Instantiate model
    model = CNN3DAD(dataset=dataset)

    # Get a batch
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))

    # Forward pass
    result = model(**batch)

    print(result)
