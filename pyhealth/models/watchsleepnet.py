"""WatchSleepNet: ResNet + TCN + BiLSTM + Attention model for IBI sleep staging.

Reference:
    Wang et al. (2025). WatchSleepNet: A Scalable Deep Learning Model for
    Wearable Sleep Staging. CHIL 2025, PMLR 287:1-20.
    https://proceedings.mlr.press/v287/wang25a.html
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyhealth.models import BaseModel


class ResidualBlock(nn.Module):
    """1D residual block with two Conv(k=5) layers and optional downsampling shortcut.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride. Default ``1``.

    Examples:
        >>> block = ResidualBlock(64, 128, stride=2)
        >>> x = torch.randn(4, 64, 100)
        >>> block(x).shape
        torch.Size([4, 128, 50])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, in_channels, L)``.

        Returns:
            Output tensor of shape ``(B, out_channels, L // stride)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class WatchSleepNet(BaseModel):
    """WatchSleepNet for IBI-based sleep staging.

    Architecture: 4-block ResNet → dilated TCN → BiLSTM →
    Multi-head Attention → global average pooling → Linear classifier.

    Args:
        dataset: Optional ``SampleDataset`` (passed to ``BaseModel``).
        num_classes: Number of output sleep stage classes. Default ``5``.
        hidden_dim: Feature dimension throughout the network. Must equal
            ``2 * lstm_hidden``. Default ``256``.
        lstm_hidden: Hidden size per direction in the BiLSTM.
            Default ``128``.
        attn_heads: Number of attention heads. Default ``8``.
        **kwargs: Forwarded to ``BaseModel``.

    Raises:
        ValueError: If ``2 * lstm_hidden != hidden_dim``.

    Examples:
        >>> model = WatchSleepNet(num_classes=3)
        >>> signal = torch.randn(4, 750)
        >>> out = model(signal=signal)
        >>> out["y_prob"].shape
        torch.Size([4, 3])
    """

    def __init__(
        self,
        dataset=None,
        num_classes: int = 5,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        attn_heads: int = 8,
        **kwargs,
    ) -> None:
        if 2 * lstm_hidden != hidden_dim:
            raise ValueError(
                f"BiLSTM output size constraint violated: "
                f"2 * lstm_hidden ({2 * lstm_hidden}) != hidden_dim ({hidden_dim}). "
                "Set lstm_hidden = hidden_dim // 2."
            )
        super().__init__(dataset)
        self.mode = "multiclass"

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.resnet = nn.Sequential(
            ResidualBlock(1, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, hidden_dim, stride=2),
        )

        # Dilated temporal convolutional layer
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        signal: Tensor,
        label: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            signal: IBI epoch batch of shape ``(B, 750)`` float32.
            label: Ground-truth class indices of shape ``(B,)`` int64.
                Optional. When provided, loss is computed.
            **kwargs: Ignored (allows dict-unpacking from DataLoader batches).

        Returns:
            Dict with keys:

            - ``"loss"``: Scalar CrossEntropyLoss, or ``0.0`` if no label.
            - ``"y_prob"``: Softmax probabilities ``(B, num_classes)``.
            - ``"y_true"``: ``label`` passed through, or ``None``.

        Raises:
            ValueError: If ``signal.shape[-1] != 750``.
        """
        if signal.shape[-1] != 750:
            raise ValueError(
                f"Expected signal length 750, got {signal.shape[-1]}."
            )

        x = signal.unsqueeze(1)          # (B, 1, 750)
        x = self.resnet(x)               # (B, hidden_dim, 47)
        x = self.tcn(x)                  # (B, hidden_dim, 47)
        x = x.transpose(1, 2)           # (B, 47, hidden_dim)
        x, _ = self.lstm(x)             # (B, 47, hidden_dim)
        x, _ = self.attention(x, x, x)  # (B, 47, hidden_dim)
        x = x.mean(dim=1)               # (B, hidden_dim)
        logits = self.fc(x)             # (B, num_classes)

        y_prob = F.softmax(logits, dim=-1)
        if label is not None:
            loss = F.cross_entropy(logits, label.long())
        else:
            loss = torch.tensor(0.0, device=signal.device)

        return {"loss": loss, "y_prob": y_prob, "y_true": label}
