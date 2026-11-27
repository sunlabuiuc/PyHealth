"""WatchSleepNet: per-epoch sleep staging from inter-beat-interval sequences.

Ported from WatchSleepNet; keep in sync with the reference implementation:
https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L225
"""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from ..processors import LabelProcessor, MultiClassLabelProcessor
from .base_model import BaseModel

_QUANTIZATION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class _ResBlock(nn.Module):
    """Reference ResBlock (ported verbatim; keep in sync)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L8
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L16
        self.relu1 = nn.ReLU(inplace=True)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L17
        self.dropout1 = nn.Dropout(0.2)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L18
        self.conv2 = nn.Conv1d(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L20
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L28
        self.relu2 = nn.ReLU(inplace=True)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L29
        self.dropout2 = nn.Dropout(0.2)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L30
        self.downsample = nn.Sequential()  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L32
        if stride != 1 or in_channels != out_channels:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L33
            self.downsample = nn.Sequential(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L34
                nn.Conv1d(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L35
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L38
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L42
        x = self.conv1(x)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L43
        # BN -> ReLU -> dropout: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L44
        x = self.dropout1(self.relu1(self.bn1(x)))
        x = self.conv2(x)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L47
        # BN -> ReLU -> dropout: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L48
        x = self.dropout2(self.relu2(self.bn2(x)))
        return x + residual  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L51


class _DeepFeatureExtractor(nn.Module):
    """Reference per-epoch feature extractor (ported; keep in sync)."""

    def __init__(self, initial_channels: int = 1):  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L56
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(initial_channels, 16, kernel_size=7, stride=1, padding=3),  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L64
            nn.ReLU(inplace=True),  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L70
        ]
        current_channels = 16  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L71
        for i in range(4):  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L73
            out_channels = 32 * (2**i)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L74
            # Four stride-4 ResBlocks: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L73
            layers.append(_ResBlock(current_channels, out_channels, stride=4))  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L76
            current_channels = out_channels  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L77
        self.layers = nn.Sequential(*layers)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L79
        self.final_conv = nn.Conv1d(256, 256, kernel_size=3, stride=4)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L80

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L82
        for layer in self.layers:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L83
            x = layer(x)
        # Final convolution: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L86
        # Flatten to 256 features: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L88
        return self.final_conv(x).view(x.size(0), -1)


class _DeepTCNBlock(nn.Module):
    """Reference dilated TCN (ported verbatim; keep in sync)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_layers: int,
    ):  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L93
        super().__init__()
        self.layers = nn.ModuleList()  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L95
        current_channels = in_channels  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L96
        dilation = 1  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L97
        for _ in range(num_layers):  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L99
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L102
                        current_channels,
                        out_channels,
                        kernel_size,
                        padding=(kernel_size - 1) * dilation // 2,  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L106
                        dilation=dilation,  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L107
                    ),
                    nn.BatchNorm1d(out_channels),  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L109
                    nn.ReLU(),  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L110
                    nn.Dropout(0.2),  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L111
                )
            )
            current_channels = out_channels  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L114
            dilation *= 2  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L115

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L117
        for layer in self.layers:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L118
            x = layer(x)
        return x


class _LSTMLayer(nn.Module):
    """Reference BiLSTM layer with optional self-attention (ported)."""

    # Attention variant: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L123
    # No-attention variant: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L185

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        use_attention: bool,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L134
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        feature_size = hidden_size * 2  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L142
        if use_attention:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L252
            self.attention = nn.MultiheadAttention(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L143
                feature_size, num_heads, batch_first=True
            )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L147
        # Packing needs CPU int64 lengths: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L148
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.int64)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L155
        packed = nn.utils.rnn.pack_padded_sequence(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L158
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L163
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L166
        weights = None  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L222
        if self.use_attention:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L179
            # Ignore positions beyond each night's true length: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L173
            max_seq_len = output.size(1)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L172
            key_padding_mask = torch.arange(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L173
                max_seq_len, device=output.device
            ).unsqueeze(0) >= lengths.to(output.device).unsqueeze(1)
            output, weights = self.attention(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L179
                output, output, output, key_padding_mask=key_padding_mask
            )
        return output, weights  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L182


class WatchSleepNet(BaseModel):
    """WatchSleepNet for per-epoch sleep staging of IBI sequences.

    This is a simplified PyHealth port of the paper's model. The convolution,
    TCN, BiLSTM, attention, and classifier math matches the reference, while
    the forward method consumes PyHealth batches and handles padding, unknown
    labels, and low-precision dtypes. The model expects one ``"signal"``
    feature of shape ``(epochs, 750)`` per sample and a matching ``"label"``
    tensor; batches may have shape ``(batch, epochs, 750)`` or
    ``(batch, 750)`` for a single epoch per sample. An optional raw
    ``"lengths"`` field overrides lengths inferred from trailing zero rows.

    Checkpoints are bare state dictionaries, identical in layout to the
    reference checkpoint, so ``Trainer.save_ckpt`` / ``load_ckpt`` work with
    original ``.pt`` files.

    Args:
        dataset: The fitted :class:`~pyhealth.datasets.SampleDataset`.
        n_classes: Number of output classes. Default is 3 (W / N / R).
        num_channels: TCN output width. Default is 256.
        kernel_size: TCN convolution kernel. Default is 5.
        hidden_dim: Hidden size of each BiLSTM direction. Default is 256.
        num_heads: Self-attention heads. Default is 16.
        num_layers: Number of stacked BiLSTM layers. Default is 4.
        tcn_layers: Number of dilated TCN blocks. Default is 3.
        use_tcn: Whether to apply the TCN. Default is True.
        use_attention: Whether to apply self-attention. Default is True.
        quantization: Compute dtype, one of ``"fp32"``, ``"fp16"``, or
            ``"bf16"``. Default is ``"fp32"``. Logits are upcast to float32
            for loss and probabilities.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import WatchSleepNet
        >>> samples = [
        ...     {
        ...         "patient_id": str(i),
        ...         "signal": np.random.randn(3, 750).astype("float32"),
        ...         "label": np.array([0, 1, 2]),
        ...         "lengths": 3,
        ...     }
        ...     for i in range(2)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor", "lengths": "raw"},
        ...     output_schema={"label": ("tensor", {"dtype": torch.long})},
        ... )
        >>> model = WatchSleepNet(dataset, hidden_dim=8, num_heads=2,
        ...                       num_channels=8, num_layers=1, tcn_layers=1)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> out["logit"].shape
        torch.Size([2, 3, 3])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_classes: int = 3,
        num_channels: int = 256,
        kernel_size: int = 5,
        hidden_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 4,
        tcn_layers: int = 3,
        use_tcn: bool = True,
        use_attention: bool = True,
        quantization: Literal["fp32", "fp16", "bf16"] = "fp32",
    ):
        super().__init__(dataset=dataset)
        # Keep the state dict identical to the reference bare checkpoint.
        del self._dummy_param
        if "signal" not in self.feature_keys or any(
            key not in {"signal", "lengths"} for key in self.feature_keys
        ):
            raise ValueError(
                "WatchSleepNet needs a 'signal' feature and an optional raw "
                f"'lengths' field, got inputs {self.feature_keys}."
            )
        if len(self.label_keys) != 1 or self.label_keys[0] != "label":
            raise ValueError(
                "WatchSleepNet needs exactly one 'label' target, got "
                f"{self.label_keys}."
            )
        for name, value in (
            ("n_classes", n_classes),
            ("num_channels", num_channels),
            ("kernel_size", kernel_size),
            ("hidden_dim", hidden_dim),
            ("num_heads", num_heads),
            ("num_layers", num_layers),
            ("tcn_layers", tcn_layers),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        if n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {n_classes}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )
        if quantization not in _QUANTIZATION_DTYPES:
            raise ValueError(
                f"quantization must be one of {sorted(_QUANTIZATION_DTYPES)}, "
                f"got {quantization!r}."
            )

        self.feature_key = "signal"
        self.label_key = "label"
        self.n_classes = n_classes
        self.quantization = quantization
        self._label_processor = MultiClassLabelProcessor(labels=n_classes)
        self.use_tcn = use_tcn  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L240
        self.use_attention = use_attention  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L251
        # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L242
        self.feature_extractor = _DeepFeatureExtractor(initial_channels=1)
        feature_dim = 256  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L243
        if use_tcn:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L245
            # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L246
            self.tcn = _DeepTCNBlock(feature_dim, num_channels, kernel_size, tcn_layers)
            lstm_input_dim = num_channels  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L247
        else:
            lstm_input_dim = feature_dim  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L249
        # Attention variant: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L253
        # Plain-LSTM variant: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L262
        self.lstm_layer = _LSTMLayer(  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L253
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_attention=use_attention,
        )
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L271
        # No capability checks: PyTorch reports unsupported dtype/device combinations.
        self.to(dtype=_QUANTIZATION_DTYPES[quantization])

    @property
    def device(self) -> torch.device:
        """Returns the device of the model parameters."""
        return self.classifier.weight.device

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Exposes the flattened per-epoch multiclass label to the Trainer."""
        return {self.label_key: self._label_processor}

    forward_from_embedding = None

    @staticmethod
    def _resolve_lengths(
        signal: torch.Tensor, value: Any, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        if value is None:
            if seq_len == 1 or signal.ndim == 2:
                return torch.full((batch_size,), 1, dtype=torch.int64)
            nonzero = signal.detach().abs().sum(dim=-1) != 0
            trailing = nonzero.flip(dims=[1]).int().argmax(dim=1)
            lengths = torch.where(
                nonzero.any(dim=1), seq_len - trailing, torch.ones_like(trailing)
            )
        elif isinstance(value, torch.Tensor):
            lengths = value.reshape(-1)
        elif isinstance(value, (list, tuple)):
            values = [
                int(v) if not isinstance(v, torch.Tensor) else int(v.item())
                for v in value
            ]
            lengths = torch.as_tensor(values)
        else:
            lengths = torch.full((batch_size,), int(value))
        return lengths.to(dtype=torch.int64).clamp(min=1, max=seq_len)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L273
        """Runs WatchSleepNet and returns masked, flattened multiclass outputs.

        The returned ``logit`` has shape ``(batch, epochs, n_classes)``;
        ``y_prob`` and ``y_true`` are flattened over valid, non-padded epochs
        only so that PyHealth's Trainer scores exactly those positions.

        Returns:
            A dict with ``loss``, ``logit``, ``y_prob``, and ``y_true``.
        """
        signal = kwargs[self.feature_key]
        if not isinstance(signal, torch.Tensor):
            signal = torch.as_tensor(signal)
        dtype = self.classifier.weight.dtype
        signal = signal.to(device=self.device, dtype=dtype)
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)
        if signal.ndim != 3:
            raise ValueError(
                f"'{self.feature_key}' must have shape (batch, epochs, 750) or "
                f"(batch, 750), got {tuple(signal.shape)}."
            )
        if signal.size(-1) != 750:
            raise ValueError(
                f"WatchSleepNet needs 750 samples per epoch, got {signal.size(-1)}."
            )
        batch_size, seq_len, _ = signal.shape  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L274
        lengths = self._resolve_lengths(
            signal, kwargs.get("lengths"), batch_size, seq_len
        )

        labels = kwargs[self.label_key]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(device=self.device, dtype=torch.long)
        if labels.ndim == 1 and seq_len == 1:
            labels = labels.unsqueeze(1)
        labels = labels[:, :seq_len]

        x = signal.reshape(batch_size * seq_len, 1, -1)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L277
        x = self.feature_extractor(x)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L278
        x = x.view(batch_size, seq_len, -1)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L279
        if self.use_tcn:  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L282
            x = x.permute(0, 2, 1)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L283
            x = self.tcn(x)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L284
            x = x.permute(0, 2, 1).contiguous()  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L285
        x, _ = self.lstm_layer(x, lengths)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L288
        logits = self.classifier(x)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/models/watchsleepnet.py#L291
        if logits.size(1) < seq_len:
            # pad_packed_sequence returns max(lengths); restore the batch grid.
            logits = F.pad(logits, (0, 0, 0, seq_len - logits.size(1)))
        logits = logits[:, :seq_len, :].to(torch.float32)

        epoch_index = torch.arange(seq_len, device=self.device).unsqueeze(0)
        valid = (epoch_index < lengths.to(self.device).unsqueeze(1)) & (labels >= 0)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/engine.py#L266; https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/config.py#L142
        flat_logits = logits.reshape(-1, self.n_classes)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/engine.py#L271
        flat_valid = valid.reshape(-1)
        y_true = labels.reshape(-1)[flat_valid]  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/engine.py#L272
        y_prob = self._label_processor.activation(flat_logits)[flat_valid]  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/engine.py#L276
        if y_true.numel():
            loss = self._label_processor.loss(flat_logits[flat_valid], y_true)  # https://github.com/WillKeWang/WatchSleepNet_public/blob/main/modeling/engine.py#L319
        else:
            loss = flat_logits.sum() * 0
        return {
            "loss": loss,
            "logit": logits,
            "y_prob": y_prob,
            "y_true": y_true,
        }


WatchSleepNet.forward_from_embedding = WatchSleepNet.forward
