import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.sample_dataset import SampleDataset
from .base_model import BaseModel


class ResidualConvBlock(nn.Module):
    """Residual 1D convolution block for local feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = x + residual
        x = self.relu2(x)
        return x


class DilatedTCNBlock(nn.Module):
    """Residual dilated temporal block."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = x + residual
        x = self.relu2(x)
        return x


class TemporalAttentionPool(nn.Module):
    """Attention pooling over temporal features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return pooled, weights


class WatchSleepNet(BaseModel):
    """Simplified WatchSleepNet-style classifier for sleep staging.

    The architecture is intentionally compact but preserves the main modeling
    ideas from WatchSleepNet:

    - residual 1D convolution for local morphology
    - optional dilated temporal blocks
    - bidirectional LSTM for sequence modeling
    - optional multi-head self-attention
    - temporal pooling and a classification head

    Args:
        dataset: A task-specific ``SampleDataset`` with one tensor feature and
            one multiclass label.
        feature_key: Feature field to read from the dataset. Defaults to the
            first feature key.
        input_dim: Optional feature dimension. If omitted, inferred from the
            first sample.
        hidden_dim: Hidden size of the bidirectional LSTM.
        conv_channels: Channel width of the residual convolution stack.
        conv_blocks: Number of residual convolution blocks.
        tcn_blocks: Number of dilated TCN blocks.
        lstm_layers: Number of bidirectional LSTM layers.
        num_attention_heads: Number of attention heads when attention is used.
        dropout: Dropout rate applied throughout the network.
        num_classes: Optional override for output classes. Defaults to the task
            label processor size.
        use_tcn: Whether to use the dilated TCN stack.
        use_attention: Whether to use multi-head self-attention and attention
            pooling. When disabled, the model uses masked mean pooling instead.
        sequence_output: Whether to emit logits for each timestep instead of a
            single pooled prediction.
        ignore_index: Ignore label used for padded sequence targets.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: Optional[str] = None,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        conv_channels: int = 64,
        conv_blocks: int = 2,
        tcn_blocks: int = 2,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        use_tcn: bool = True,
        use_attention: bool = True,
        sequence_output: bool = False,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("WatchSleepNet supports a single label field.")

        self.label_key = self.label_keys[0]
        self.feature_key = feature_key or self.feature_keys[0]
        self.hidden_dim = hidden_dim
        self.conv_channels = conv_channels
        self.conv_blocks = conv_blocks
        self.tcn_blocks = tcn_blocks
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.use_tcn = use_tcn
        self.use_attention = use_attention
        self.sequence_output = sequence_output
        self.ignore_index = ignore_index

        self.input_dim = input_dim or self._infer_input_dim(self.feature_key)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.input_projection = nn.Conv1d(self.input_dim, conv_channels, kernel_size=1)
        self.residual_conv = nn.ModuleList(
            [
                ResidualConvBlock(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    dropout=dropout,
                )
                for _ in range(conv_blocks)
            ]
        )

        self.tcn = nn.ModuleList(
            [
                DilatedTCNBlock(
                    conv_channels,
                    dilation=2**block_index,
                    kernel_size=3,
                    dropout=dropout,
                )
                for block_index in range(tcn_blocks)
            ]
        )

        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        context_dim = hidden_dim * 2
        self.attention_projection = None
        if use_attention:
            projected_dim = context_dim
            if projected_dim % num_attention_heads != 0:
                projected_dim = (
                    math.ceil(projected_dim / num_attention_heads)
                    * num_attention_heads
                )
                self.attention_projection = nn.Linear(context_dim, projected_dim)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=projected_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(projected_dim)
            self.attention_pool = TemporalAttentionPool(projected_dim)
            classifier_input_dim = projected_dim
        else:
            self.self_attention = None
            self.attention_norm = None
            self.attention_pool = None
            classifier_input_dim = context_dim

        self.final_dropout = nn.Dropout(dropout)
        if self.sequence_output and num_classes is None:
            raise ValueError(
                "num_classes must be provided when sequence_output=True."
            )
        self.classifier = nn.Linear(
            classifier_input_dim,
            num_classes if num_classes is not None else self.get_output_size(),
        )

    def _infer_input_dim(self, feature_key: str) -> int:
        for sample in self.dataset:
            if feature_key not in sample:
                continue
            value = sample[feature_key]
            tensor = (
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            )
            if tensor.dim() == 1:
                return 1
            return int(tensor.shape[-1])
        raise ValueError(
            f"Unable to infer input_dim for feature '{feature_key}' from the dataset."
        )

    def _coerce_input(self, value: Any) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(self.device, dtype=torch.float32)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.dim() != 3:
            raise ValueError(
                "Expected input tensor with 2 or 3 dims, received shape "
                f"{tuple(tensor.shape)}."
            )

        if tensor.shape[-1] == self.input_dim:
            return tensor
        if tensor.shape[1] == self.input_dim:
            return tensor.transpose(1, 2)
        raise ValueError(
            "Unable to infer whether input is channel-last or channel-first. "
            f"Expected one axis to match input_dim={self.input_dim}, got "
            f"shape {tuple(tensor.shape)}."
        )

    @staticmethod
    def _default_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

    def _build_mask(
        self,
        x: torch.Tensor,
        explicit_mask: Optional[Any] = None,
    ) -> torch.Tensor:
        if explicit_mask is None:
            return self._default_mask(x)
        mask = (
            explicit_mask
            if isinstance(explicit_mask, torch.Tensor)
            else torch.as_tensor(explicit_mask)
        )
        mask = mask.to(self.device)
        if mask.dim() != 2:
            raise ValueError(
                f"Expected mask with shape [batch, seq_len], got {tuple(mask.shape)}."
            )
        return mask > 0

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_sequence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        for block in self.residual_conv:
            x = block(x)
        if self.use_tcn:
            for block in self.tcn:
                x = block(x)

        # x: [batch, conv_channels, seq_len] -> [batch, seq_len, conv_channels]
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)

        if self.use_attention and self.self_attention is not None:
            residual = x
            if self.attention_projection is not None:
                residual = self.attention_projection(residual)
            attn_out, _ = self.self_attention(
                residual,
                residual,
                residual,
                key_padding_mask=~mask,
                need_weights=False,
            )
            x = self.attention_norm(attn_out + residual)
        return x

    def _compute_sequence_loss(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y_true.reshape(-1).long(),
            ignore_index=self.ignore_index,
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        x = self._coerce_input(kwargs[self.feature_key])
        mask = self._build_mask(x, kwargs.get("mask"))
        x = self._encode_sequence(x, mask)

        if self.sequence_output:
            embed = self.final_dropout(x)
            logits = self.classifier(embed)
            results = {
                "logit": logits,
                "y_prob": torch.softmax(logits, dim=-1),
            }
            if self.label_key in kwargs:
                y_true = kwargs[self.label_key].to(self.device).long()
                results["loss"] = self._compute_sequence_loss(logits, y_true)
                results["y_true"] = y_true
            if kwargs.get("embed", False):
                results["embed"] = embed
            return results

        if self.use_attention and self.attention_pool is not None:
            pooled, _ = self.attention_pool(x, mask)
        else:
            pooled = self._masked_mean(x, mask)

        embed = self.final_dropout(pooled)
        logits = self.classifier(embed)
        results = {
            "logit": logits,
            "y_prob": self.prepare_y_prob(logits),
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.get_loss_function()(logits, y_true)
            results["y_true"] = y_true

        if kwargs.get("embed", False):
            results["embed"] = embed

        return results
