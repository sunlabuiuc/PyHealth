"""
EnhancedResNet50 Model with CBAM Attention for PyHealth

Author: Selena Lu
NetID: salu2
Paper Title: Detecting Heart Disease from Multi-View Ultrasound Images
via Supervised Attention Multiple Instance Learning
Paper Link: https://arxiv.org/pdf/2306.00003
Description: This model integrates the CBAM attention mechanism into a ResNet50 backbone
             to enhance feature representation for medical image classification tasks.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models
from pyhealth.models import BaseModel


class ChannelAttention(nn.Module):
    """Channel Attention Module."""

    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initializes the ChannelAttention module.

        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction ratio for the MLP.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel attention.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map after applying channel attention.
        """
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = (avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""

    def __init__(self):
        """
        Initializes the SpatialAttention module.
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for spatial attention.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Output feature map after applying spatial attention.
                - Spatial attention weights.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale, scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, in_channels: int):
        """
        Initializes the CBAM module.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CBAM.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Output feature map after applying CBAM.
                - Spatial attention weights.
        """
        x = self.channel_attention(x)
        x, spatial_weights = self.spatial_attention(x)
        return x, spatial_weights


class EnhancedResNet50(BaseModel):
    """Enhanced ResNet50 model with CBAM attention for PyHealth."""

    def __init__(self, num_classes: int = 3):
        """
        Initializes the EnhancedResNet50 model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(EnhancedResNet50, self).__init__(dataset=None)
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        self.cbam = CBAM(in_channels=2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the EnhancedResNet50 model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Logits of shape (B, num_classes).
                - Feature representations of shape (B, 2048).
                - Spatial attention weights of shape (B, 1, H, W).
        """
        x = self.stem(x)  # Shape: [B, 2048, H, W]
        x, attn_weights = self.cbam(x)  # Shape: [B, 2048, H, W]
        pooled = self.pool(x)  # Shape: [B, 2048, 1, 1]
        features = self.flatten(pooled)  # Shape: [B, 2048]
        logits = self.fc(features)  # Shape: [B, num_classes]
        return logits, features, attn_weights
