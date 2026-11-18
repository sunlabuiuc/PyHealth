"""WatchSleepNet: IBI-based sequence-to-sequence sleep staging.

Paper:
    Wang et al. "WatchSleepNet: A Novel Model and Pretraining Approach for
    Advancing Sleep Staging with Smartwatches." CHIL 2025 (PMLR 287).

The model maps a night of 30 s inter-beat-interval (IBI) epochs to one sleep
stage per epoch (Wake / NREM / REM by default). Pipeline::

    (B, T, L) -> ResNet (per epoch) -> Linear -> TCN -> BiLSTM
              -> multi-head self-attention -> Linear -> (B, T, num_classes)

Note:
    The paper only specifies the ResNet's input (L=750) and output
    (256 maps x 3 points). The strides and channel widths used here are
    assumptions chosen to satisfy that and are configurable; check them
    against the authors' repository.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class ResidualBlock1d(nn.Module):
    """Two-conv residual block with a strided 1D convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (odd).
        stride: Stride of the first convolution (downsamples time).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block to ``x`` of shape ``(N, C_in, L)``."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class CausalTemporalBlock(nn.Module):
    """Causal dilated residual block used by the TCN.

    Args:
        channels: Number of input and output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        dropout: Dropout probability.
    """

    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block to ``x`` of shape ``(B, C, T)``."""
        out = self.dropout(self.relu(self.conv1(F.pad(x, (self.left_pad, 0)))))
        out = self.dropout(self.relu(self.conv2(F.pad(out, (self.left_pad, 0)))))
        return self.relu(out + x)


class WatchSleepNet(BaseModel):
    """WatchSleepNet sequence-to-sequence sleep stager.

    The dataset's input must hold one IBI tensor per sample, shaped
    ``(T, L)`` (or ``(T, C, L)`` when ``num_input_channels > 1``), where
    ``T`` is the number of 30 s epochs and ``L`` the samples per epoch
    (750 in the paper). The label must be a ``(T,)`` integer tensor with one
    stage per epoch; epochs equal to ``ignore_index`` are not scored.

    Since labels are per epoch rather than per sample, the model computes its
    own loss and the dataset should use ``"tensor"`` for both schemas.

    Args:
        dataset: The :class:`~pyhealth.datasets.SampleDataset`.
        feature_key: Input key. Defaults to the dataset's only input key.
        label_key: Label key. Defaults to the dataset's only output key.
        num_classes: Number of sleep stages. Default 3.
        input_length: Samples per epoch ``L``. Default 750.
        num_input_channels: Channels per epoch (1 for IBI only). Default 1.
        res_channels: Output channels of each residual block. The last entry
            is the feature width fed to the TCN. Default (32, 64, 128, 256).
        res_strides: Stride of each residual block. Default (5, 5, 5, 2),
            which reduces 750 points to 3.
        kernel_size: Kernel size for the ResNet and TCN. Default 5.
        tcn_layers: Number of TCN blocks. Default 3.
        hidden_dim: LSTM hidden size per direction. Default 256.
        lstm_layers: Number of LSTM layers. Default 4.
        num_heads: Attention heads. Must divide the LSTM output width.
            Default 16.
        dropout: Dropout used in the TCN, LSTM and attention. Default 0.2.
        use_tcn: Include the TCN. Default True.
        use_attention: Include multi-head attention. Default True.
        bidirectional: Use a BiLSTM. If False the model is causal: the LSTM
            is unidirectional and attention is causally masked. Default True.
        focal_gamma: Focal-loss gamma; 0 gives plain cross-entropy.
        ignore_index: Label value to ignore in the loss. Default -100.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": "p0", "ibi": [[0.0] * 40] * 4,
        ...             "label": [0, 1, 1, 2]}]
        >>> ds = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ibi": "tensor"},
        ...     output_schema={"label": "tensor"},
        ... )
        >>> model = WatchSleepNet(
        ...     ds, input_length=40, res_channels=(4, 8), res_strides=(2, 2),
        ...     hidden_dim=8, lstm_layers=1, num_heads=2,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str | None = None,
        label_key: str | None = None,
        num_classes: int = 3,
        input_length: int = 750,
        num_input_channels: int = 1,
        res_channels: Sequence[int] = (32, 64, 128, 256),
        res_strides: Sequence[int] = (5, 5, 5, 2),
        kernel_size: int = 5,
        tcn_layers: int = 3,
        hidden_dim: int = 256,
        lstm_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.2,
        use_tcn: bool = True,
        use_attention: bool = True,
        bidirectional: bool = True,
        focal_gamma: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(res_channels) != len(res_strides):
            raise ValueError("res_channels and res_strides must match in length.")
        self.feature_key = feature_key or self.feature_keys[0]
        self.label_key = label_key or self.label_keys[0]
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.use_tcn = use_tcn
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index

        # Per-epoch ResNet feature extractor.
        blocks: list[nn.Module] = []
        in_ch = num_input_channels
        for out_ch, stride in zip(res_channels, res_strides):
            blocks.append(ResidualBlock1d(in_ch, out_ch, kernel_size, stride))
            in_ch = out_ch
        self.resnet = nn.Sequential(*blocks)
        feat_dim = res_channels[-1]
        with torch.no_grad():  # infer the temporal length left after the ResNet
            probe = torch.zeros(1, num_input_channels, input_length)
            final_len = self.resnet.eval()(probe).shape[-1]
        self.resnet.train()
        self.collapse = nn.Linear(final_len, 1)

        self.tcn: nn.Sequential | None = None
        if use_tcn:
            self.tcn = nn.Sequential(
                *[
                    CausalTemporalBlock(feat_dim, kernel_size, 2**i, dropout)
                    for i in range(tcn_layers)
                ]
            )

        self.lstm = nn.LSTM(
            feat_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.attention: nn.MultiheadAttention | None = None
        if use_attention:
            self.attention = nn.MultiheadAttention(
                out_dim, num_heads, dropout=dropout, batch_first=True
            )
        self.fc = nn.Linear(out_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Maps IBI epochs to per-epoch logits.

        Args:
            x: Float tensor ``(B, T, L)`` or ``(B, T, C, L)``.

        Returns:
            Logits of shape ``(B, T, num_classes)``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(2)
        batch, steps, channels, length = x.shape
        if channels != self.num_input_channels:
            raise ValueError(
                f"Expected {self.num_input_channels} channels, got {channels}."
            )
        feats = self.resnet(x.reshape(batch * steps, channels, length))
        feats = self.collapse(feats).squeeze(-1)  # (B*T, F)
        feats = feats.reshape(batch, steps, -1)  # (B, T, F)
        if self.tcn is not None:
            feats = self.tcn(feats.transpose(1, 2)).transpose(1, 2)
        feats, _ = self.lstm(feats)
        if self.attention is not None:
            mask = None
            if not self.bidirectional:  # keep the causal model causal
                mask = torch.triu(
                    torch.ones(steps, steps, dtype=torch.bool, device=x.device), 1
                )
            feats, _ = self.attention(feats, feats, feats, attn_mask=mask)
        return self.fc(feats)

    def compute_loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Cross-entropy (optionally class-weighted and/or focal) over epochs.

        Args:
            logits: ``(B, T, num_classes)``.
            y_true: ``(B, T)`` integer labels; ``ignore_index`` is skipped.

        Returns:
            A scalar loss.
        """
        flat_logits = logits.reshape(-1, self.num_classes)
        flat_true = y_true.reshape(-1)
        ce = F.cross_entropy(
            flat_logits,
            flat_true,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = flat_true != self.ignore_index
        if self.focal_gamma > 0:
            ce = (1 - torch.exp(-ce)) ** self.focal_gamma * ce
        return ce.sum() / valid.sum().clamp(min=1)

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: Must contain the input key; may contain the label key.

        Returns:
            A dict with ``logit`` and ``y_prob`` of shape ``(B, T, C)``, and,
            when labels are given, ``loss`` (scalar) and ``y_true`` ``(B, T)``.
        """
        x = kwargs[self.feature_key].to(self.device).float()
        logits = self.encode(x)
        results = {"logit": logits, "y_prob": self.softmax(logits)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).long()
            results["loss"] = self.compute_loss(logits, y_true)
            results["y_true"] = y_true
        return results
