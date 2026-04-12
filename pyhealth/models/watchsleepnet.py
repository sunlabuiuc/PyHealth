from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

class ResidualBlock(nn.Module):
    """Residual block with strided convolution for temporal downsampling and channel expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x + residual

class TCNBlock(nn.Module):
    """A single TCN block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        num_layers = 3
        self.layers = []
        current_channels = in_channels

        for _ in range(num_layers):
            conv = nn.Conv1d(
                current_channels,
                out_channels, 
                kernel_size,
                dilation=dilation, 
                padding=(kernel_size - 1) * dilation // 2
            )
            bn = nn.BatchNorm1d(out_channels)
            relu = nn.ReLU(inplace=True)
            dropout = nn.Dropout(0.2)
            self.layers.append(nn.Sequential(conv, bn, relu, dropout))
            current_channels = out_channels
            dilation *= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class FeatureExtractor(nn.Module):
    """Feature extractor using residual blocks for each epoch independently."""

    def __init__(
        self,
        in_channels: int
    ):
        super().__init__()

        blocks = []
        out_channels = 16
        stride = 1
        res_channels = [32, 64, 128, 256]

        blocks.append(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        )

        blocks.append(nn.ReLU(inplace=True))
        current_channels = out_channels

        for out_ch in res_channels:
            blocks.append(ResidualBlock(current_channels, out_ch, stride=4))
            current_channels = out_ch

        self.res_blocks = nn.Sequential(*blocks)
        self.conv = nn.Conv1d(256, 256, kernel_size=3, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        return x

class WatchSleepNet(BaseModel):
    """WatchSleepNet model for sleep stage classification.

    Architecture:
        1. Residual convolutional blocks to extract multi-level spatial features
           while preserving info through skip connections.
        2. TCN to address longer-range dependencies.
        3. Bi-directional LSTM to capture temporal dependencies in both directions.
        4. Multi-head attention mechanism to focus on important time steps and features.
        5. Fully connected layer with softmax for classification.

    Args:
        input_size: features per epoch
        num_classes: number of output classes
        tcn_kernel_size: kernel size for TCN layers
        lstm_hidden_size: hidden size for each LSTM direction
        lstm_num_layers: number of LSTM layers
        num_heads: number of multi-head attention heads
        dropout: dropout probability

    Note:
        Default hyperparameters are based on the original WatchSleepNet paper
    """

    def __init__(
        self,
        dataset=None,
        seq_sample_size: int = 750,
        num_features: int = 1,
        num_classes: int = 3,
        tcn_kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        num_heads: int = 4
    ):
        super(WatchSleepNet, self).__init__(dataset)

        self.feature_extractor = FeatureExtractor(in_channels=num_features)
        feature_out_channels = 256

        self.tcn = TCNBlock(feature_out_channels, seq_sample_size, tcn_kernel_size, dilation=1)
        tcn_out_channels = seq_sample_size

        self.lstm = nn.LSTM(
            input_size=tcn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_size = 2 * lstm_hidden_size  # bidirectional

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        self.classifier = nn.Linear(lstm_out_size, num_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward propagation.

        Args:
            x: input tensor of shape (batch, seq_len, input_size).

        Returns:
            output tensor of shape (batch, seq_len, num_classes).
        """
        batch_size, seq_len, seq_sample_size = x.shape

        x = x.view(batch_size * seq_len, 1, seq_sample_size)
        x = self.feature_extractor(x)

        x = x.view(batch_size, seq_len, -1).permute(0, 2, 1)
        x = self.tcn(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)

        output = self.classifier(x)

        return output
