"""
Author: Zaid Peracha
NetID: Peracha2
Paper Title: Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging
Paper Link: https://dl.acm.org/doi/pdf/10.1145/3368555.3384468

Description:
This file implements the ResNeXt-29 8x64d architecture used in the above paper,
designed for high-capacity convolutional learning. It is adapted to the PyHealth
framework and inherits from BaseModel. It supports reproducible benchmarking
on structured image datasets like CIFAR-100, enabling schema completion and
hidden stratification evaluation.

ResNeXt models use grouped convolutions with cardinality and bottleneck width
parameters to balance computational efficiency and representational power.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel


class ResNeXtBottleneck(nn.Module):
    """ResNeXt bottleneck block with grouped convolutions."""

    expansion: int = 4

    def __init__(self, in_channels: int, cardinality: int,
                 bottleneck_width: int, stride: int) -> None:
        """Initializes a grouped bottleneck residual block.

        Args:
            in_channels: Number of input feature channels.
            cardinality: Number of groups in grouped convolution.
            bottleneck_width: Width of intermediate feature maps.
            stride: Stride for spatial downsampling.
        """
        super().__init__()
        D = cardinality * bottleneck_width

        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)

        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride,
                                   padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)

        self.conv_expand = nn.Conv2d(D, D * self.expansion, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(D * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != D * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, D * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(D * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck block.

        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Output tensor of shape (B, C_out, H_out, W_out)
        """
        out = F.relu(self.bn_reduce(self.conv_reduce(x)))
        out = F.relu(self.bn(self.conv_conv(out)))
        out = self.bn_expand(self.conv_expand(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNeXt29(BaseModel):
    """ResNeXt-29 8x64d image classifier compatible with PyHealth."""

    def __init__(self, dataset, cardinality: int = 8,
                 depth: int = 29, bottleneck_width: int = 64, **kwargs) -> None:
        """Initializes the ResNeXt-29 model.

        Args:
            dataset: A PyHealth dataset object with output_size set.
            cardinality: Number of convolution groups (default = 8).
            depth: Network depth (default = 29).
            bottleneck_width: Width of bottleneck layer (default = 64).
        """
        super().__init__(dataset, **kwargs)

        assert (depth - 2) % 9 == 0, "Depth must be 9n+2 for ResNeXt-29"
        layers = (depth - 2) // 9

        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.stage1 = self._make_stage(layers, stride=1)
        self.stage2 = self._make_stage(layers, stride=2)
        self.stage3 = self._make_stage(layers, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_channels, self.dataset.output_size)

    def _make_stage(self, num_blocks: int, stride: int) -> nn.Sequential:
        """Creates a stage of ResNeXt bottleneck blocks.

        Args:
            num_blocks: Number of bottleneck units.
            stride: Stride of the first block.

        Returns:
            A sequential container of ResNeXt blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            block = ResNeXtBottleneck(self.in_channels,
                                      self.cardinality,
                                      self.bottleneck_width,
                                      s)
            blocks.append(block)
            self.in_channels = self.cardinality * \
                self.bottleneck_width * ResNeXtBottleneck.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNeXt model.

        Args:
            x: A batch of image tensors (B, 3, 32, 32)

        Returns:
            A tensor of class logits (B, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)

    def compute_loss(self, forward_output: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        """Computes cross-entropy loss.

        Args:
            forward_output: Predicted logits of shape (B, num_classes)
            labels: Ground truth class indices of shape (B,)

        Returns:
            A scalar loss tensor.
        """
        return F.cross_entropy(forward_output, labels)
