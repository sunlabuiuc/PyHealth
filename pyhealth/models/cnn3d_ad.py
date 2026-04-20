# Author: Paul Nguyen, Shayan Jaffar, William Lee
# Description: 3D CNN for Alzheimer's disease classification

import math
from typing import Dict

import torch
import torch.nn as nn
import numpy as np

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    """Returns a 3D normalization layer based on norm_type.

    Args:
        norm_type: Type of normalization. Either "instance" or "batch".
        num_channels: Number of channels to normalize.

    Returns:
        A 3D normalization layer, InstanceNorm3d or BatchNorm3d.
    """
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    else:
        raise ValueError(f"norm_type must be 'instance' or 'batch', got '{norm_type}'")


class ConvBlock3D(nn.Module):
    """Single 3D neural network convolutional block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        norm_type: Type of normalization layer. Either "instance" or "batch".
        stride: Convolution stride.
        dilation: Convolution dilation factor.
        padding: Explicit padding size.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type="instance", stride=1, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            _make_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: Input tensor of shape [batch_size, in_channels, D, H, W].

        Returns:
            Output tensor of shape [batch_size, out_channels, D, H, W].
        """
        return self.block(x)


# Architecture specs from Table 2
_BLOCK_KERNELS   = [1,  3,  5,  3]   # conv kernel size
_BLOCK_CHANNELS  = [4,  32, 64, 64]  # output channels
_BLOCK_DILATIONS = [1,  2,  2,  2]   # conv dilation
_BLOCK_PADDINGS  = [0,  0,  2,  1]   # conv padding
_POOL_KERNELS    = [3,  3,  3,  5]   # max-pool kernel size (stride=2 throughout)

class CNN3DAD(BaseModel):
    """3D CNN for Alzheimer's disease classification from structural MRI, based on Liu et al. (2020)
    "On the Design of Convolutional Neural Networks for Automatic Detection of Alzheimer's Disease."
    Classifies scans into cognitively normal (CN), mild cognitive impairment (MCI), and Alzheimer's
    disease (AD).

    Args:
        dataset: Dataset with fitted input and output processors.
        scan_key: Input key for the 3D MRI volume.
        age_key: Input key for patient age.
        label_key: Output label key.
        norm_type: "instance" or "batch".
        widening_factor: Channel multiplier applied to all blocks.
        num_blocks: Number of conv blocks.
        age_encoding_dim: Age encoding dimension. 0 disables it.
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
        age_encoding_dim: int = 32,
    ):
        super().__init__(dataset=dataset)

        self.scan_key = scan_key
        self.age_key = age_key
        self.label_key = label_key
        self.norm_type = norm_type
        self.use_age_encoding = age_encoding_dim > 0
        self.age_encoding_dim = age_encoding_dim

        blocks = []
        in_ch = 1
        for i in range(num_blocks):
            idx = min(i, len(_BLOCK_KERNELS) - 1)
            out_ch = _BLOCK_CHANNELS[idx] * widening_factor
            blocks.append(ConvBlock3D(
                in_ch, out_ch,
                kernel_size=_BLOCK_KERNELS[idx],
                norm_type=norm_type,
                dilation=_BLOCK_DILATIONS[idx],
                padding=_BLOCK_PADDINGS[idx],
            ))
            blocks.append(nn.MaxPool3d(kernel_size=_POOL_KERNELS[idx], stride=2))
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_ch, 1024)

        if self.use_age_encoding:
            max_len = 240
            age_pe = torch.zeros(max_len, age_encoding_dim)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, age_encoding_dim, 2).float() * -(math.log(10000.0) / age_encoding_dim))
            age_pe[:, 0::2] = torch.sin(pos * div)
            age_pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("age_pe", age_pe)
            self.age_fc = nn.Sequential(
                nn.Linear(age_encoding_dim, 512),
                nn.LayerNorm(512),
                nn.Linear(512, 1024),
            )
        else:
            self.age_fc = None

        num_classes = self.get_output_size()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Must contain the scan, age, and label tensors under
                their respective keys. Scan shape: [B, 1, D, H, W] or
                [B, D, H, W]. Age shape: [B, 1] or [B,].

        Returns:
            A dictionary with the following keys:
                loss: Cross-entropy loss scalar.
                y_prob: Predicted probabilities of shape [B, num_classes].
                y_true: Ground truth labels of shape [B].
                logit: Raw logits of shape [B, num_classes].
        """
        scan = kwargs[self.scan_key].to(self.device).float()
        age = kwargs[self.age_key].to(self.device).float()
        y_true = kwargs[self.label_key].to(self.device)

        if scan.dim() == 4:         # (B, D, H, W) -> (B, 1, D, H, W)
            scan = scan.unsqueeze(1)
        if age.dim() == 1:          # (B,) -> (B, 1)
            age = age.unsqueeze(1)

        feat = self.backbone(scan)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        feat = self.fc1(feat)

        if self.use_age_encoding:
            age_idx = (age.squeeze(1) * 2).long().clamp(0, 239)
            age_enc = self.age_pe[age_idx]
            age_enc = self.age_fc(age_enc)
            feat = feat + age_enc

        logits = self.classifier(feat)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}


if __name__ == "__main__":
    samples = [
        {
            "patient_id": "p0",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([60.0], dtype="float32"),
            "label": 0,  # CN
        },
        {
            "patient_id": "p1",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([72.0], dtype="float32"),
            "label": 1,  # MCI
        },
        {
            "patient_id": "p2",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([68.0], dtype="float32"),
            "label": 2,  # AD
        },
        {
            "patient_id": "p3",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([65.0], dtype="float32"),
            "label": 0,
        },
    ]

    input_schema = {"scan": "tensor", "age": "tensor"}
    output_schema = {"label": "multiclass"}
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="adni_test",
    )

    model = CNN3DAD(dataset)
    train_loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(train_loader))

    out = model(**batch)
    print(out)
