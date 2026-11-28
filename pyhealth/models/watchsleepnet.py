"""WatchSleepNet for sequence-level sleep staging from IBI signals."""

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from pyhealth.models.base_model import BaseModel
from pyhealth.processors import TimeseriesLabelProcessor


class ResBlock(nn.Module):
    """One-dimensional residual convolution block from WatchSleepNet.

    Examples:
        >>> import torch
        >>> block = ResBlock(1, 8, stride=1)
        >>> x = torch.randn(2, 1, 32)
        >>> block(x).shape
        torch.Size([2, 8, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
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
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + residual


class DeepFeatureExtractor(nn.Module):
    """Extract a 256-dimensional feature vector from each IBI epoch.

    Examples:
        >>> import torch
        >>> extractor = DeepFeatureExtractor().eval()
        >>> x = torch.randn(2, 1, 750)
        >>> features = extractor(x)
        >>> features.shape[0] == 2
        True
    """

    def __init__(self, initial_channels: int = 1) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv1d(
                initial_channels,
                16,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.ReLU(inplace=True),
        ]

        in_channels = 16
        for i in range(4):
            out_channels = 32 * (2**i)
            layers.append(
                ResBlock(
                    in_channels,
                    out_channels,
                    stride=4,
                )
            )
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(
            256,
            256,
            kernel_size=3,
            stride=4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.final_conv(x)
        return x.flatten(start_dim=1)


class TemporalChannelLayerNorm(nn.Module):
    """Normalize channels independently at each temporal position.

    Examples:
        >>> import torch
        >>> norm = TemporalChannelLayerNorm(8)
        >>> x = torch.randn(2, 8, 10)
        >>> norm(x).shape
        torch.Size([2, 8, 10])
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DeepTCNBlock(nn.Module):
    """Dilated temporal convolution block from WatchSleepNet.

    Examples:
        >>> import torch
        >>> tcn = DeepTCNBlock(8, 8, num_layers=1)
        >>> x = torch.randn(2, 8, 10)
        >>> tcn(x).shape
        torch.Size([2, 8, 10])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")

        layers: list[nn.Module] = []
        dilation = 1
        current_channels = in_channels

        for _ in range(num_layers):
            left_padding = (kernel_size - 1) * dilation
            layers.extend(
                [
                    nn.ConstantPad1d((left_padding, 0), 0.0),
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size,
                        padding=0,
                        dilation=dilation,
                    ),
                    TemporalChannelLayerNorm(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ]
            )
            current_channels = out_channels
            dilation *= 2

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LSTMWithMultiheadAttention(nn.Module):
    """Bidirectional LSTM followed by multi-head self-attention.

    Examples:
        >>> import torch
        >>> layer = LSTMWithMultiheadAttention(8, 8, 2, 1)
        >>> x = torch.randn(2, 4, 8)
        >>> lengths = torch.tensor([4, 2])
        >>> output = layer(x, lengths)
        >>> output.shape[:2] == (2, 4)
        True
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        residual_attention: bool = False,
    ) -> None:
        super().__init__()
        self.residual_attention = residual_attention

        feature_size = hidden_size * 2
        if feature_size % num_heads != 0:
            raise ValueError("2 * hidden_dim must be divisible by num_heads.")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.shape[1],
        )

        positions = torch.arange(
            output.shape[1],
            device=output.device,
        )
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)

        attended, _ = self.attention(
            output,
            output,
            output,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        if self.residual_attention:
            return output + attended
        return attended


class LSTMLayer(nn.Module):
    """Bidirectional LSTM without the attention layer.

    Examples:
        >>> import torch
        >>> layer = LSTMLayer(8, 8, 1)
        >>> x = torch.randn(2, 4, 8)
        >>> lengths = torch.tensor([4, 2])
        >>> output = layer(x, lengths)
        >>> output.shape[:2] == (2, 4)
        True
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.shape[1],
        )
        return output


class WatchSleepNet(BaseModel):
    """PyHealth implementation of WatchSleepNet for sequence-level sleep staging.

    Based on Wang et al., "WatchSleepNet: A Novel Model and Pretraining
    Approach for Advancing Sleep Staging with Smartwatches," Proceedings
    of the Sixth Conference on Health, Inference, and Learning, PMLR 287,
    2025.
    Paper: https://proceedings.mlr.press/v287/wang25a.html
    Original repository: https://github.com/WillKeWang/WatchSleepNet_public/

    This implementation adapts the authors' residual CNN feature extractor,
    temporal convolutional network (TCN), bidirectional LSTM, and multi-head
    attention architecture to the PyHealth BaseModel and time-series label
    processor interfaces.

    It uses a 750-sample IBI segment for each 30-second epoch at 25 Hz and
    supports three classes: Wake, NREM, and REM. Variable-length patient
    sequences are handled using label offsets to exclude padded epochs.
    This DREAMT-focused implementation does not reproduce the authors'
    complete multi-dataset pretraining and transfer-learning pipeline.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.tasks import SleepStagingDREAMTSequence
        >>> samples = [{
        ...     'patient_id': 'synthetic',
        ...     'signal': np.zeros((3, 750), dtype=np.float32),
        ...     'label': {'time': [0, 30, 60], 'values': [0, 1, 2]},
        ... }]
        >>> task = SleepStagingDREAMTSequence()
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema=task.input_schema,
        ...     output_schema=task.output_schema,
        ... )
        >>> model = WatchSleepNet(
        ...     dataset, num_channels=8, hidden_dim=8,
        ...     num_heads=2, num_layers=1, tcn_layers=1,
        ... )
        >>> isinstance(model, WatchSleepNet)
        True
    """

    def __init__(
        self,
        dataset: Any,
        num_channels: int = 256,
        kernel_size: int = 5,
        hidden_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 4,
        tcn_layers: int = 3,
        use_tcn: bool = True,
        use_attention: bool = True,
        class_weights: list[float] | None = None,
        residual_attention: bool = False,
    ) -> None:
        super().__init__(dataset)

        if self.feature_keys != ["signal"]:
            raise ValueError(
                "WatchSleepNet expects exactly one input field named 'signal'."
            )
        if len(self.label_keys) != 1:
            raise ValueError("WatchSleepNet expects exactly one output label.")

        if not isinstance(self.label_processor, TimeseriesLabelProcessor):
            raise TypeError(
                "WatchSleepNet requires a time-series label processor, "
                "such as 'timeseries_multiclass'."
            )

        if self.label_processor.output_size() != 3:
            raise ValueError("WatchSleepNet expects exactly three output classes.")

        if num_channels <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("Model dimensions and layer counts must be positive.")
        if num_heads <= 0 or tcn_layers <= 0:
            raise ValueError("num_heads and tcn_layers must be positive.")

        self.label_key = self.label_keys[0]
        self.use_tcn = use_tcn
        self.use_attention = use_attention
        self.residual_attention = residual_attention
        if residual_attention and not use_attention:
            raise ValueError(
                "residual_attention requires use_attention=True."
            )
        self.segment_length = 750

        if class_weights is not None:
            if len(class_weights) != 3 or any(w <= 0 for w in class_weights):
                raise ValueError("class_weights must contain three positive values.")
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        self.feature_extractor = DeepFeatureExtractor(initial_channels=1)
        feature_dim = 256

        if use_tcn:
            self.tcn = DeepTCNBlock(
                in_channels=feature_dim,
                out_channels=num_channels,
                kernel_size=kernel_size,
                num_layers=tcn_layers,
            )
            lstm_input_dim = num_channels
        else:
            lstm_input_dim = feature_dim

        if use_attention:
            self.lstm_layer = LSTMWithMultiheadAttention(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                residual_attention=residual_attention,
            )
        else:
            self.lstm_layer = LSTMLayer(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
            )

        self.classifier = nn.Linear(hidden_dim * 2, 3)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        signal = kwargs["signal"].to(self.device)

        if signal.ndim != 3:
            raise ValueError(
                "Expected signal shape (batch, epochs, samples), "
                f"got {tuple(signal.shape)}."
            )
        if signal.shape[-1] != self.segment_length:
            raise ValueError(
                f"Expected {self.segment_length} samples per epoch, "
                f"got {signal.shape[-1]}."
            )

        batch_size, max_epochs, _ = signal.shape

        if self.label_key not in kwargs:
            raise ValueError(
                f"Missing '{self.label_key}' labels; time-series labels "
                "are needed to determine valid sequence lengths."
            )

        y_true = kwargs[self.label_key]
        if not isinstance(y_true, (tuple, list)) or len(y_true) != 3:
            raise ValueError(
                "Expected packed time-series labels: (values, time, offsets)."
            )

        values, _, offsets = y_true
        values = values.to(self.device)
        offsets = offsets.to(device=self.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]

        if len(lengths) != batch_size:
            raise ValueError("Signal batch and label batch sizes do not match.")
        if bool((lengths <= 0).any()):
            raise ValueError("Every sequence must contain at least one label.")
        if bool((lengths > max_epochs).any()):
            raise ValueError("A label sequence is longer than its signal sequence.")

        # Extract one feature vector for every epoch in the padded batch.
        x = signal.reshape(-1, 1, self.segment_length)
        x = self.feature_extractor(x)
        x = x.reshape(batch_size, max_epochs, -1)

        # Apply temporal convolution across the epoch dimension.
        if self.use_tcn:
            x = self.tcn(x.transpose(1, 2)).transpose(1, 2)

        # Encode each patient's variable-length sequence.
        x = self.lstm_layer(x, lengths)
        logits_padded = self.classifier(x)

        # Keep only real epochs, in the same order as packed label values.
        valid_mask = torch.arange(max_epochs, device=self.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        logits = logits_padded[valid_mask]

        if logits.shape[0] != values.shape[0]:
            raise ValueError(
                "Model predictions and packed labels have different lengths: "
                f"{logits.shape[0]} predictions vs {values.shape[0]} labels."
            )

        output = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
            "y_true": y_true,
            "loss": (
                torch.nn.functional.cross_entropy(
                    logits, values.long(), weight=self.class_weights
                )
                if self.class_weights is not None
                else torch.nn.functional.cross_entropy(
                    logits, values.long()
                )
            ),
        }
        return output
