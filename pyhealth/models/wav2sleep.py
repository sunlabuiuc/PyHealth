"""Multimodal sleep-stage modeling for PyHealth (Wav2Sleep).

This module provides CNN encoders for per-epoch waveforms, multimodal fusion
(including transformer-based aggregation with missing modalities), dilated and
standard temporal mixers, and the :class:`Wav2Sleep` model for use with
:class:`~pyhealth.datasets.SampleDataset`.

Paper-aligned pieces include transformer fusion and a dilated CNN sequence
mixer; a simplified temporal CNN path is available for comparison.

Examples:
    After fitting a dataset whose ``feature_keys`` include ``"ecg"`` and/or
    ``"ppg"`` and a single label field, construct the model and run one step::

        from pyhealth.models import Wav2Sleep

        model = Wav2Sleep(
            dataset,
            embedding_dim=128,
            hidden_dim=128,
            num_classes=5,
        )
        outputs = model(
            ecg=ecg_tensor,
            sleep_stage=label_tensor,
        )
        loss = outputs["loss"]

    Replace ``sleep_stage`` with the actual label key from ``dataset`` (also
    exposed as ``model.label_key``). Pass only modalities present in the
    dataset; each waveform tensor is shaped ``[batch, time]`` per epoch layout
    expected by :class:`SignalEncoders`.

Note:
    Public helpers such as :func:`get_activation` and :func:`get_norm` mirror
    common PyHealth CNN conventions and are reused inside the encoders.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel

# =============================================================================
# Modality Encoders
# =============================================================================

class ConvBlock(nn.Module):
    """Three 1D convolution blocks with optional residual shortcut.

    Stacks three :class:`ConvLayer` modules (kernel 3; strides 1, 1, 2). When
    ``residual`` is True, adds a strided 1x1 projection of the input so the skip
    path matches spatial resolution and channel width before the final activation.

    Attributes:
        residual: Whether the residual branch is enabled.
        layer1: First conv block (same resolution).
        layer2: Second conv block (same resolution).
        layer3: Third conv block (stride 2).
        activation: Nonlinearity applied to the (possibly residual) output.
        down: 1x1 stride-2 conv for the skip connection when ``residual`` is
            True; otherwise a registered ``None`` parameter placeholder.
    """

    def __init__(
        self,
        input_num_channels: int,
        output_num_channels: int,
        activation: str = 'leaky',
        norm: str = 'batch',
        dropout: float = 0.0,
        causal: bool = False,
        eps: float | None = None,
        residual: bool = True,
    ) -> None:
        """Build a ``ConvBlock``.

        Args:
            input_num_channels: Channel width of the input tensor.
            output_num_channels: Channel width after the block.
            activation: Name passed to :func:`get_activation`.
            norm: Normalization style passed to each :class:`ConvLayer`.
            dropout: Dropout probability inside each conv layer.
            causal: Whether underlying convolutions use causal padding/trim.
            eps: Optional epsilon forwarded to normalization when applicable.
            residual: If True, add the strided 1x1 skip connection.
        """
        super().__init__()
        self.residual = residual

        self.layer1 = ConvLayer(
            input_channels=input_num_channels,
            output_channels=output_num_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=causal,
            eps=eps,
        )

        self.layer2 = ConvLayer(
            input_channels=output_num_channels,
            output_channels=output_num_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=causal,
            eps=eps,
        )

        self.layer3 = ConvLayer(
            input_channels=output_num_channels,
            output_channels=output_num_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            kernel_size=3,
            stride=2,
            padding=1,
            causal=causal,
            eps=eps,
        )

        self.activation = get_activation(activation)

        if self.residual:
            self.down = nn.Conv1d(
                input_num_channels,
                output_num_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            )
        else:
            self.register_parameter('down', None)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the three conv layers and optional residual fusion.

        Args:
            x: Input of shape ``[N, C_in, L]``.

        Returns:
            Output of shape ``[N, C_out, L']`` where ``L'`` reflects stride-2 in
            the third block.
        """
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)

        if self.residual:
            output = output + self.down(x)
        output = self.activation(output)
        return output


class ConvLayer(nn.Module):
    """Single 1D convolution with norm, activation, dropout, and optional weight norm.

    Attributes:
        input_channels: Input channel count.
        output_channels: Output channel count.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Base or causal padding (may be overridden when ``causal``).
        dilation: Convolution dilation.
        causal: If True, uses causal padding and trims the right tail of activations.
        groups: Conv1d groups.
        bias: Whether the conv kernel uses a bias vector.
        conv: The :class:`torch.nn.Conv1d` (or weight-norm wrapped) module.
        activation: Activation module from :func:`get_activation`.
        dropout: :class:`torch.nn.Dropout` applied after the activation.
        norm: Normalization or identity (or weight-norm path uses identity here).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        activation: str = 'relu',
        norm: str | None = 'batch',
        eps: float | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        causal: bool = False,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        """Build a ``ConvLayer``.

        Args:
            input_channels: ``C_in`` for ``Conv1d``.
            output_channels: ``C_out`` for ``Conv1d``.
            activation: Activation name for :func:`get_activation`.
            norm: Normalization name for :func:`get_norm`, or ``'weight'`` for
                weight normalization (norm module becomes identity).
            eps: If set, passed as ``norm_eps`` into :func:`get_norm` where
                supported (e.g. instance norm).
            kernel_size: Spatial kernel size.
            stride: Convolution stride along the time axis.
            padding: Padding when not using causal mode; superseded by causal
                padding when ``causal`` is True.
            dilation: Dilation factor.
            dropout: Dropout probability after activation.
            causal: Enables causal padding and optional right-side trimming on the
                forward pass.
            groups: Conv1d groups.
            bias: Conv bias; coordinated with ``norm`` inside ``Conv1d``.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.causal = causal
        self.groups = groups
        self.bias = bias

        if causal:
            self.padding = (self.kernel_size - 1) * self.dilation

        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=bias or norm is None,
            dilation=dilation,
        )

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)

        if norm == 'weight':
            self.norm = nn.Identity()
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        else:
            norm_info = {}
            if eps is not None:
                norm_info = {'norm_eps': eps}
            self.norm = get_norm(
                norm, num_features=output_channels, causal=causal, **norm_info
            )

    def forward(self, x: Tensor) -> Tensor:
        """Convolve, optionally trim causal tail, then norm / activate / dropout.

        Args:
            x: Input of shape ``[N, C_in, L]``.

        Returns:
            Tensor of shape ``[N, C_out, L']`` after conv, optional trim, norm,
            activation, and dropout.
        """
        output = self.conv(x)

        if self.causal and self.padding > 0:
            if isinstance(self.conv.stride, tuple):
                stride = self.conv.stride[0]
            else:
                stride = self.conv.stride
            trim = max(self.padding - stride + 1, 0)
            if trim > 0:
                output = output[:, :, :-trim]

        output = self.norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output


class SignalEncoder(nn.Module):
    """CNN that maps fixed-length waveform chunks to per-epoch embeddings.

    Repeatedly applies :class:`ConvBlock` with capped channel growth, then a
    linear layer maps flattened CNN features to ``epoch_embedding_dim``.

    Attributes:
        input_num_channels: Input channels (typically 1).
        epoch_embedding_dim: Output feature dimension per epoch.
        samples_per_epoch: Samples per epoch (must be a power of two for the
            depth schedule).
        norm: Normalization type per block, or ``'auto'`` to switch by depth.
        init_feature_channels: Base width for the channel schedule.
        max_channels: Upper cap on channel width in the schedule.
        output_norm: ``LayerNorm`` or ``Identity`` on linear outputs.
        residual: Passed through to each :class:`ConvBlock`.
        causal: Global causal flag for conv blocks.
        chunk_causal: Per-epoch vs whole-sequence processing when ``causal``.
        cnn: Sequential stack of ``ConvBlock`` modules.
        linear: Projects flattened CNN features to ``epoch_embedding_dim``.
        epoch_size: Flattened feature size before ``linear``.
        activation: Nonlinearity applied after the linear projection.
    """

    def __init__(
        self,
        input_num_channels: int = 1,
        epoch_embedding_dim: int = 256,
        activation: str = 'gelu',
        samples_per_epoch: int = 1024,
        norm: str = 'instance',
        init_feature_channels: int = 16,
        max_channels: int = 128,
        output_norm: bool = False,
        residual: bool = True,
        causal: bool = False,
        chunk_causal: bool = True,
    ) -> None:
        """Build the encoder for one signal type.

        Args:
            input_num_channels: Leading channel dimension of waveform input.
            epoch_embedding_dim: Size of each output embedding vector.
            activation: Activation name for blocks and post-linear nonlinearity.
            samples_per_epoch: Length of each epoch; must be a power of two.
            norm: Normalization type, or ``'auto'`` for depth-dependent choice.
            init_feature_channels: Initial channel width in the schedule.
            max_channels: Maximum channel width in the schedule.
            output_norm: Whether to apply layer normalization on outputs.
            residual: Whether conv blocks use residual shortcuts.
            causal: Whether to use causal convolutions.
            chunk_causal: Per-epoch vs whole-sequence causal behavior when
                ``causal`` is True.

        Raises:
            ValueError: If ``samples_per_epoch`` is not a power of two.
        """
        super().__init__()
        self.input_num_channels = input_num_channels
        self.epoch_embedding_dim = epoch_embedding_dim
        self.samples_per_epoch = samples_per_epoch
        self.norm = norm
        self.init_feature_channels = init_feature_channels
        self.max_channels = max_channels
        self.residual = residual
        self.causal = causal
        self.chunk_causal = chunk_causal
        self.activation = get_activation(activation)

        blocks = []

        if samples_per_epoch & (samples_per_epoch - 1) != 0:
            raise ValueError("samples_per_epoch must be a power of two")
        num_conv_blocks = int(math.log2(samples_per_epoch)) - 2
        num_channels_per_block = [
            min(init_feature_channels * 2 ** (i // 2), max_channels)
            for i in range(num_conv_blocks)
        ]
        self.epoch_size = num_channels_per_block[-1] * 4

        in_ch = input_num_channels
        for i, dim in enumerate(num_channels_per_block):
            if norm == "auto":
                norm_type = "instance" if i < 2 else "layer"
            else:
                norm_type = norm
            eps = 1e-2 if norm_type == "instance" else None
            blocks.append(
                ConvBlock(
                    input_num_channels=in_ch,
                    output_num_channels=dim,
                    activation=activation,
                    norm=norm_type,
                    eps=eps,
                    causal=(causal and not chunk_causal),
                    residual=residual,
                )
            )
            in_ch = dim

        self.cnn = nn.Sequential(*blocks)
        self.linear = nn.Linear(self.epoch_size, epoch_embedding_dim)
        self.output_norm = (
            nn.LayerNorm(epoch_embedding_dim) if output_norm else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode waveform epochs into shape ``[B, E, D]``.

        Args:
            x: Tensor of shape ``[B, T]`` where ``T`` is a multiple of
                ``samples_per_epoch``.

        Returns:
            Tensor of shape ``[B, E, epoch_embedding_dim]``.

        Raises:
            ValueError: If the time dimension is not divisible by
                ``samples_per_epoch``.
        """
        if x.size(-1) % self.samples_per_epoch:
            raise ValueError(
                f"Input length must be a multiple of {self.samples_per_epoch}"
            )

        batch = x.size(0)
        epochs = x.size(-1) // self.samples_per_epoch

        if self.causal and self.chunk_causal:
            y = x.view(batch, epochs, self.samples_per_epoch)
            y = y.reshape(batch * epochs, 1, self.samples_per_epoch)
            y = self.cnn(y)
            y = y.transpose(-1, -2).reshape(batch, epochs, self.epoch_size)
        else:
            y = x.unsqueeze(1)
            y = self.cnn(y)
            y = y.transpose(-1, -2).reshape(batch, -1, self.epoch_size)

        y = self.output_norm(self.activation(self.linear(y)))
        return y


#: Canonical signal names to samples per epoch for fixed-length CNN encoders.
SIGNAL_TO_SAMPLES_PER_EPOCH: dict[str, int] = {
    'ABD': 256,
    'THX': 256,
    'ECG': 1024,
    'PPG': 1024,
    'EOG_L': 4096,
    'EOG_R': 4096,
}


class SignalEncoders(nn.Module):
    """Shared :class:`SignalEncoder` instances keyed by encoder id.

    ``signal_encoder_map`` assigns each modality name to an encoder key so
    multiple signals can reuse one CNN. Optional per-signal embeddings can be
    added when ``include_signal`` is True.

    Attributes:
        signal_encoder_map: Signal name to encoder-key mapping.
        feature_dim: Per-epoch embedding size ``D``.
        activation: Activation string for each :class:`SignalEncoder`.
        norm: Normalization style for each encoder.
        include_signal: Whether to add a learned embedding per signal type.
        init_feature_channels: CNN width schedule base.
        max_channels: CNN width cap.
        output_norm: LayerNorm flag forwarded to encoders.
        residual: Residual flag for underlying conv blocks.
        causal: Causal flag for underlying encoders.
        chunk_causal: Chunked causal flag for underlying encoders.
        embedding: Signal-type embedding table or ``None``.
        signal_encoders: ``ModuleDict`` of encoders by encoder key.
        signal_to_idx: Index for each signal name (for embeddings).
    """

    def __init__(
        self,
        signal_encoder_map: dict[str, str],
        feature_dim: int,
        activation: str,
        norm: str = 'instance',
        include_signal: bool = False,
        init_feature_channels: int = 16,
        max_channels: int = 128,
        output_norm: bool = False,
        residual: bool = True,
        causal: bool = False,
        chunk_causal: bool = True,
    ) -> None:
        """Create encoders for each distinct encoder key.

        Args:
            signal_encoder_map: Signal names to encoder keys; each signal must
                exist in :data:`SIGNAL_TO_SAMPLES_PER_EPOCH`.
            feature_dim: Output embedding dimension per epoch.
            activation: Passed to :class:`SignalEncoder`.
            norm: Passed to :class:`SignalEncoder`.
            include_signal: If True, instantiate ``nn.Embedding`` over signals.
            init_feature_channels: Passed to :class:`SignalEncoder`.
            max_channels: Passed to :class:`SignalEncoder`.
            output_norm: Passed to :class:`SignalEncoder`.
            residual: Passed to :class:`SignalEncoder`.
            causal: Passed to :class:`SignalEncoder`.
            chunk_causal: Passed to :class:`SignalEncoder`.

        Raises:
            ValueError: If a signal name is missing from
                :data:`SIGNAL_TO_SAMPLES_PER_EPOCH`.
        """
        super().__init__()
        self.signal_encoder_map = signal_encoder_map
        self.feature_dim = feature_dim
        self.activation = activation
        self.norm = norm
        self.include_signal = include_signal
        self.init_feature_channels = init_feature_channels
        self.max_channels = max_channels
        self.output_norm = output_norm
        self.residual = residual
        self.causal = causal
        self.chunk_causal = chunk_causal

        signal_encoders = {}

        for signal, encoder in signal_encoder_map.items():
            if encoder in signal_encoders:
                continue
            if signal not in SIGNAL_TO_SAMPLES_PER_EPOCH:
                raise ValueError(
                    f"Signal {signal} not found in SIGNAL_TO_SAMPLES_PER_EPOCH"
                )
            samples_per_epoch = SIGNAL_TO_SAMPLES_PER_EPOCH[signal]

            signal_encoders[encoder] = SignalEncoder(
                input_num_channels=1,
                epoch_embedding_dim=feature_dim,
                activation=activation,
                samples_per_epoch=samples_per_epoch,
                norm=norm,
                init_feature_channels=init_feature_channels,
                max_channels=max_channels,
                output_norm=output_norm,
                residual=residual,
                causal=causal,
                chunk_causal=chunk_causal,
            )

        if self.include_signal:
            self.embedding = nn.Embedding(
                num_embeddings=len(signal_encoder_map),
                embedding_dim=self.feature_dim,
            )
        else:
            self.register_parameter('embedding', None)

        self.include_signal = include_signal
        self.signal_encoders = nn.ModuleDict(signal_encoders)
        self.signal_to_idx = {
            signal: i for i, signal in enumerate(sorted(signal_encoder_map.keys()))
        }

    def __len__(self) -> int:
        """Return the number of distinct encoder modules."""
        return len(self.signal_encoders)

    def get_signal_encoder(self, signal: str) -> 'SignalEncoder':
        """Return the :class:`SignalEncoder` for ``signal``.

        Args:
            signal: A key in ``signal_encoder_map``.

        Returns:
            The shared encoder for that signal's encoder key.

        Raises:
            KeyError: If ``signal`` or the resolved encoder key is invalid.
        """
        if self.signal_encoder_map is not None:
            return self.signal_encoders[self.signal_encoder_map[signal]]
        else:
            return self.signal_encoders[signal]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode each signal; restore ``-inf`` for batches with no modality data.

        Args:
            x: Signal name to waveform tensor, typically ``[B, T]``.

        Returns:
            Signal name to tensor of shape ``[B, E, D]`` with ``D`` =
            ``feature_dim``.
        """
        out: dict[str, Tensor] = {}

        for signal, x_signal in x.items():
            inf_batch_mask = torch.isinf(x_signal[:, 0])
            x_signal = torch.where(torch.isinf(x_signal), 0.0, x_signal)
            out_bsf = self.get_signal_encoder(signal)(x_signal)
            out_bsf = torch.where(
                inf_batch_mask[:, None, None], float('-inf'), out_bsf
            )

            if self.include_signal:
                embed = self.embedding(
                    torch.tensor(
                        [self.signal_to_idx[signal]],
                        device=out_bsf.device,
                        dtype=torch.int64,
                    )
                )
                out_bsf = out_bsf + embed.view(1, 1, -1)
            out[signal] = out_bsf
        return out


class ConvLayerNorm(nn.Module):
    """Layer normalization over the channel axis for 1D conv features ``[N, C, L]``.

    Attributes:
        eps: Epsilon in the variance denominator.
        weight: Learnable scale ``[1, C, 1]``.
        bias: Learnable bias ``[1, C, 1]``.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """Create layer norm for ``num_features`` channels.

        Args:
            num_features: Channel count ``C``.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` over channels.

        Args:
            x: Tensor of shape ``[N, C, L]``.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        mean = x.mean(1, keepdim=True)
        sigma = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(sigma + self.eps)
        x = self.bias + self.weight * x
        return x


class ConvRMSNorm(nn.Module):
    """RMS normalization over channel dimension for conv activations ``[N, C, L]``.

    Attributes:
        weight: Learnable scale ``[1, C, 1]``.
        eps: Epsilon inside the RMS denominator.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """Build RMS norm for ``num_features`` channels.

        Args:
            num_features: Number of channels ``C``.
            eps: Epsilon for the RMS denominator.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input of shape ``[N, C, L]``.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        sigma = x.pow(2).mean(1, keepdim=True)
        x = x / torch.sqrt(sigma + self.eps)
        x = self.weight * x
        return x


class ConvGroupNorm(nn.Module):
    """Group normalization for 1D conv features via :class:`torch.nn.GroupNorm`.

    Attributes:
        norm: The underlying :class:`torch.nn.GroupNorm` module.
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int = 8,
        eps: float = 1e-5,
        channels: int | None = None,
    ) -> None:
        """Build group norm for ``num_features`` channels.

        Args:
            num_features: ``num_channels`` for GroupNorm.
            num_groups: Desired groups; may be overridden by ``channels``.
            eps: Epsilon for GroupNorm.
            channels: If set, ``num_groups`` becomes ``num_features // channels``.

        Raises:
            ValueError: If ``num_features`` is not divisible by the resolved
                ``num_groups``.
        """
        super().__init__()

        if channels is not None:
            num_groups = num_features // channels
        if num_features < num_groups:
            num_groups = num_features
        if num_features % num_groups != 0:
            raise ValueError("num_features must be divisible by num_groups")
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_features, eps=eps
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply group normalization.

        Args:
            x: Tensor of shape ``[N, C, L]``.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        return self.norm(x)


def get_activation(name: str, **kwargs: Any) -> nn.Module:
    """Return a PyTorch activation module by name.

    Args:
        name: One of ``'relu'``, ``'leaky'``, ``'gelu'``, ``'silu'`` / ``'swish'``,
            or ``'linear'`` (identity).
        **kwargs: Forwarded to the activation constructor.

    Returns:
        An ``nn.Module`` implementing the requested activation.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    if name == 'relu':
        return nn.ReLU(**kwargs)
    elif name == 'leaky':
        return nn.LeakyReLU(**kwargs)
    elif name == 'gelu':
        return nn.GELU(**kwargs)
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(**kwargs)
    elif name == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f'{name=} is unsupported.')


def get_norm(
    name: str | None = 'batch',
    causal: bool = False,
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    """Construct a normalization layer for 1D conv stacks.

    ``causal`` is reserved for API compatibility with call sites.

    Args:
        name: ``'batch'``, ``'layer'``, ``'rms'``, ``'instance'``, ``'group'``,
            or ``None`` (identity).
        causal: Unused; kept for call-site compatibility.
        *args: Positional args forwarded to the chosen norm.
        **kwargs: Keyword args forwarded to the chosen norm. ``norm_eps`` is
            popped and mapped to ``eps`` for instance norm.

    Returns:
        An ``nn.Module`` normalization layer.

    Raises:
        ValueError: If ``name`` is not recognized.
    """
    norm_eps = kwargs.pop('norm_eps', None)

    if name == 'batch':
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == 'layer':
        return ConvLayerNorm(*args, **kwargs)
    elif name == 'rms':
        return ConvRMSNorm(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    elif name == 'instance':
        if norm_eps is not None:
            kwargs['eps'] = norm_eps
        return nn.InstanceNorm1d(*args, **kwargs)
    elif name == 'group':
        return ConvGroupNorm(*args, **kwargs)
    else:
        raise ValueError(f'Normalisation with {name=} and {causal=} unknown.')

# =============================================================================
# Transformer Fusion Module
# =============================================================================
# This file uses the reusable TransformerFusion fusion layer from
# pyhealth.models.fusion to aggregate modality embeddings.
# The transformer fusion module accepts a dict of temporally aligned
# modality tensors of shape [B, T, D] and returns a fused tensor of shape [B, T, D].


# =============================================================================
# Dilated CNN Sequence Mixer (Paper-Faithful)
# =============================================================================

class DilatedConvBlock(nn.Module):
    """Single dilated 1D convolution block with residual add and GELU.

    Expects activations in channels-first form ``[N, C, L]`` and preserves
    length using symmetric padding derived from ``kernel_size`` and
    ``dilation``.

    Attributes:
        padding: Spatial padding applied on both sides of the time axis.
        conv: Depthwise (same-channel) dilated :class:`~torch.nn.Conv1d`.
        batch_norm: :class:`~torch.nn.BatchNorm1d` on channel dimension.
        activation: GELU nonlinearity.
        dropout: Dropout applied after the nonlinearity.

    Args:
        channels: Channel width ``C`` for input and output.
        kernel_size: Convolution kernel along time.
        dilation: Dilation factor along time.
        dropout: Dropout probability after activation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize conv, norm, activation, and dropout modules.

        Args:
            channels: Channel width for the residual path.
            kernel_size: Temporal kernel size.
            dilation: Dilation applied to the conv kernel.
            dropout: Dropout rate after the nonlinearity.
        """
        super().__init__()
        
        # Padding to preserve sequence length with dilation
        # For causal convolution: padding = (kernel_size - 1) * dilation
        # For same convolution: padding = ((kernel_size - 1) * dilation) // 2
        self.padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv1d(
            channels, 
            channels, 
            kernel_size, 
            padding=self.padding,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated conv, norms, nonlinearity, dropout, and residual add.

        Args:
            x: Tensor of shape ``[N, C, L]`` (batch, channels, time).

        Returns:
            Tensor of shape ``[N, C, L]`` after the residual block.
        """
        residual = x
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual  # Residual connection
        return out


class DilatedCNNSequenceMixer(nn.Module):
    """Stacked dilated 1D CNN for temporal modeling (paper-style).

    Applies exponentially increasing dilations by default so the receptive field
    grows across depth while sequence length stays fixed. Input and output use
    batch-major layout ``[B, T, D]``.

    Attributes:
        input_dim: Input feature size per time step.
        hidden_dim: Working channel width inside dilated blocks.
        dilations: Dilation schedule, length ``num_layers``.
        input_proj: Optional ``1x1`` conv when ``input_dim != hidden_dim``.
        dilated_blocks: List of :class:`DilatedConvBlock` modules.
        layer_norm: Final :class:`~torch.nn.LayerNorm` on ``hidden_dim``.
        _receptive_field: Cached receptive field in time steps.

    Args:
        input_dim: Feature dimension ``D`` of inputs ``[B, T, D]``.
        hidden_dim: Channel width inside the dilated stack.
        kernel_size: Temporal kernel size for each dilated block.
        num_layers: Number of dilated blocks (default dilations: ``2**i``).
        dilations: Optional explicit dilation list; length must match depth.
        dropout: Dropout probability inside each block.

    Note:
        Compared to :class:`TemporalConvBlock`, this path targets long-range
        temporal context with fewer parameters at a fixed depth.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        num_layers: int = 5,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        """Build optional input projection and dilated block stack.

        Args:
            input_dim: Input feature size per time step.
            hidden_dim: Working channel width for dilated convolutions.
            kernel_size: Temporal kernel shared by each dilated block.
            num_layers: Number of dilated blocks (default schedule ``2**i``).
            dilations: Optional explicit dilation list of length ``num_layers``.
            dropout: Dropout probability inside each :class:`DilatedConvBlock`.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Default: exponentially increasing dilations
        if dilations is None:
            dilations = [2**i for i in range(num_layers)]  # [1, 2, 4, 8, 16]
        
        self.dilations = dilations
        
        # Input projection if dimensions don't match
        self.input_proj = None
        if input_dim != hidden_dim:
            self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Stacked dilated convolution blocks
        self.dilated_blocks = nn.ModuleList([
            DilatedConvBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Calculate receptive field
        self._receptive_field = self._compute_receptive_field(kernel_size, dilations)
    
    def _compute_receptive_field(self, kernel_size: int, dilations: List[int]) -> int:
        """Return receptive field size in time steps for the dilated stack.

        Args:
            kernel_size: Shared kernel size of each dilated block.
            dilations: Dilation factor per block.

        Returns:
            Integer receptive field along the time axis.
        """
        rf = 1
        for d in dilations:
            rf += (kernel_size - 1) * d
        return rf
    
    @property
    def receptive_field(self) -> int:
        """Return the receptive field in number of time steps."""
        return self._receptive_field
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, T, D_in]`` embeddings to ``[B, T, hidden_dim]``.

        Args:
            x: Tensor of shape ``[B, T, input_dim]``.

        Returns:
            Tensor of shape ``[B, T, hidden_dim]`` after projection, dilated
            depthwise convs, and layer norm.
        """
        # Convert to channels-first format for Conv1d: [B, D, T]
        x = x.transpose(1, 2)
        
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Apply stacked dilated convolutions
        for block in self.dilated_blocks:
            x = block(x)
        
        # Convert back to [B, T, D] format
        x = x.transpose(1, 2)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x


# =============================================================================
# Standard Temporal Conv Block (Simplified Version - for comparison)
# =============================================================================

class TemporalConvBlock(nn.Module):
    """Two-layer temporal Conv1d block with residual path (simplified).

    This block does not use dilations; prefer :class:`DilatedCNNSequenceMixer`
    when matching the reference paper. Shapes follow ``[B, T, D]``.

    Attributes:
        conv1: First Conv-BN-ReLU-Dropout stack (channels-first internally).
        conv2: Second Conv-BN stack.
        residual_proj: Optional ``1x1`` conv when ``input_dim != hidden_dim``.
        relu: Final ReLU after the residual add.
        dropout: Dropout after ReLU.

    Args:
        input_dim: Input feature dimension per time step.
        hidden_dim: Output feature dimension per time step.
        kernel_size: Temporal kernel with same padding.
        dropout: Dropout rate after activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Create conv stacks and optional residual projection.

        Args:
            input_dim: Input feature size per time step.
            hidden_dim: Output feature size per time step.
            kernel_size: Temporal kernel with same padding.
            dropout: Dropout rate after activations.
        """
        super().__init__()

        padding = kernel_size // 2  # Same padding to preserve sequence length
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Residual connection projection if dimensions don't match
        self.residual_proj = None
        if input_dim != hidden_dim:
            self.residual_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two temporal conv layers with residual connection.

        Args:
            x: Tensor of shape ``[B, T, input_dim]``.

        Returns:
            Tensor of shape ``[B, T, hidden_dim]``.
        """
        # Convert to channels-first format for Conv1d: [B, D, T]
        x_transposed = x.transpose(1, 2)
        
        # Apply convolutions
        out = self.conv1(x_transposed)
        out = self.conv2(out)
        
        # Residual connection
        residual = x_transposed
        if self.residual_proj is not None:
            residual = self.residual_proj(x_transposed)
        
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        
        # Convert back to [B, T, D] format
        return out.transpose(1, 2)


class BaseFusionModule(nn.Module, ABC):
    """Abstract base class for multimodal fusion modules."""

    def __init__(self, embed_dim: int, num_modalities: int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self._config = kwargs

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return self._config

    @abstractmethod
    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse multimodal features.

        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: Optional mask indicating present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        pass

    def get_modality_mask(self, modality_features: Union[List[Optional[Tensor]], Dict[str, Optional[Tensor]]]) -> Tensor:
        """Generate modality mask from a feature list or dict."""
        if isinstance(modality_features, dict):
            modality_features = list(modality_features.values())

        device = next((f.device for f in modality_features if f is not None and hasattr(f, 'device')), None)
        batch_size = next((f.size(0) for f in modality_features if f is not None and hasattr(f, 'size')), 1)
        modality_mask = torch.tensor(
            [feat is not None and hasattr(feat, 'device') for feat in modality_features],
            device=device
        ).unsqueeze(0).repeat(batch_size, 1)
        return modality_mask


class AttentionMechanism(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    @abstractmethod
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pass


class MultiHeadAttention(AttentionMechanism):
    """Standard multi-head attention implementation."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = query.size(0)

        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended)
        return output


class CrossModalAttention(BaseFusionModule):
    """Cross-modal attention mechanism for fusing features from different modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 4,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None for missing modalities
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            attended_output: [batch_size, seq_len, embed_dim]
        """
        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Filter present modalities
        present_features = [f for f in modality_features if f is not None]
        if not present_features:
            raise ValueError("At least one modality must be present")

        # Concatenate present features along sequence dimension for cross-attention
        concatenated = torch.cat(present_features, dim=1)  # [batch, total_seq, embed_dim]

        # Use first present feature as query, concatenated as key/value
        query = present_features[0]

        # Create attention mask for missing modalities
        # This is a simplified version - in practice, you'd want more sophisticated masking
        seq_lengths = [f.size(1) for f in present_features]
        total_seq = sum(seq_lengths)

        # For now, use the attention mechanism directly
        output = self.attention(query, concatenated, concatenated)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        attention_type: str = 'self',
    ):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == 'self':
            self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        elif attention_type == 'cross':
            self.attention = CrossModalAttention(embed_dim, num_heads, dropout, num_modalities=1)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.attention_type == 'cross' and context is not None:
            attn_out = self.attention.attention(x, context, context, mask)
        else:
            attn_out = self.attention(x, x, x, mask)

        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerFusion(BaseFusionModule):
    """Transformer-based fusion module for multimodal features with dynamic missing modality handling."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 1000,
        **kwargs
    ):
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.register_buffer('pos_embedding', torch.randn(1, max_seq_len, embed_dim))

        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, 'self')
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def _project_modalities(self, modality_features: List[Optional[Tensor]]) -> List[Tensor]:
        """Project each modality to common embedding space."""
        projected = []
        for i, feat in enumerate(modality_features):
            if feat is not None:
                projected.append(self.modality_projections[i](feat))
            else:
                # Create zero tensor for missing modalities
                batch_size, seq_len = modality_features[0].size(0), modality_features[0].size(1)
                device = modality_features[0].device
                projected.append(torch.zeros(batch_size, seq_len, self.embed_dim, device=device))
        return projected

    def _fuse_features(
        self,
        projected_features: List[Tensor],
        modality_mask: Tensor
    ) -> Tensor:
        """Fuse projected features using weighted average."""
        # Stack features: [num_modalities, batch_size, seq_len, embed_dim]
        stacked = torch.stack(projected_features, dim=0)

        # Apply modality weights and mask
        # weights shape: [num_mod, 1, 1, 1] to broadcast over [num_mod, batch, seq_len, embed_dim]
        weights = self.modality_weights.view(-1, 1, 1, 1)
        # mask_expanded shape: [num_mod, batch, 1, 1]
        mask_expanded = modality_mask.t().unsqueeze(-1).unsqueeze(-1)

        weighted_features = stacked * weights * mask_expanded.float()

        # Average across modalities, accounting for missing ones
        num_present = modality_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [batch, 1]
        fused = weighted_features.sum(dim=0) / num_present.unsqueeze(-1)  # [batch, seq_len, embed_dim]
        return fused

    def forward(
        self,
        modality_features: Union[List[Optional[Tensor]], Dict[str, Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            modality_features: List of tensors or None, or dict of modality->Tensor
            modality_mask: [batch_size, num_modalities] - True for present modalities

        Returns:
            fused_features: [batch_size, seq_len, embed_dim]
        """
        if isinstance(modality_features, dict):
            modality_features = list(modality_features.values())

        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Project modalities
        projected_features = self._project_modalities(modality_features)

        # Fuse features
        fused = self._fuse_features(projected_features, modality_mask)

        batch_size, seq_len, _ = fused.size()

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            extended_pos = torch.cat([
                self.pos_embedding,
                torch.zeros(1, seq_len - self.max_seq_len, self.embed_dim, device=fused.device)
            ], dim=1)
            fused = fused + extended_pos

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, mask=None)  # Self-attention within fused features

        # Final projection
        output = self.output_projection(fused)

        return output


class BaseFusionModule(nn.Module, ABC):
    """Abstract base class for multimodal fusion modules.

    Subclasses implement :meth:`forward` to combine optional per-modality
    tensors of shape ``[B, T, D]`` into a single fused tensor.

    Attributes:
        embed_dim: Feature dimension ``D`` shared across modalities.
        num_modalities: Expected modality count for masking utilities.
        _config: Keyword configuration captured at construction time.
    """

    def __init__(self, embed_dim: int, num_modalities: int, **kwargs: Any) -> None:
        """Initialize the base fusion module.

        Args:
            embed_dim: Dimension of the embedding space.
            num_modalities: Number of input modalities.
            **kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self._config = kwargs

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration parameters.

        Returns:
            Dictionary containing configuration parameters.
        """
        return self._config

    @abstractmethod
    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse multimodal features.

        Args:
            modality_features: List of tensors or None for missing modalities.
                Each tensor has shape [batch_size, seq_len, embed_dim].
            modality_mask: Optional mask indicating present modalities.
                Shape: [batch_size, num_modalities].

        Returns:
            Fused features tensor of shape [batch_size, seq_len, embed_dim].
        """
        pass

    def get_modality_mask(
        self,
        modality_features: Union[List[Optional[Tensor]], Dict[str, Optional[Tensor]]]
    ) -> Tensor:
        """Generate modality mask from feature list or dict.

        Args:
            modality_features: List or dict of modality features.

        Returns:
            Modality mask tensor of shape [batch_size, num_modalities].
        """
        if isinstance(modality_features, dict):
            modality_features = list(modality_features.values())

        device = next(
            (
                f.device
                for f in modality_features
                if f is not None and hasattr(f, "device")
            ),
            None,
        )
        batch_size = next(
            (
                f.size(0)
                for f in modality_features
                if f is not None and hasattr(f, "size")
            ),
            1,
        )
        modality_mask = torch.tensor(
            [
                feat is not None and hasattr(feat, "device")
                for feat in modality_features
            ],
            device=device,
        ).unsqueeze(0).repeat(batch_size, 1)
        return modality_mask


class AttentionMechanism(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        """Store head geometry and validate divisibility.

        Args:
            embed_dim: Model width ``D`` (must divide ``num_heads``).
            num_heads: Parallel attention heads.
            dropout: Dropout probability on attention weights.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute attention output; implemented by subclasses.

        Args:
            query: Query tensor ``[B, L_q, D]``.
            key: Key tensor ``[B, L_k, D]``.
            value: Value tensor ``[B, L_k, D]``.
            mask: Optional attention mask broadcastable to logits.

        Returns:
            Updated representation ``[B, L_q, D]``.
        """
        pass


class MultiHeadAttention(AttentionMechanism):
    """Scaled dot-product multi-head self- or cross-attention.

    Projects ``query``, ``key``, and ``value`` with learned linear maps, applies
    softmax attention per head, then mixes heads with an output projection.

    Attributes:
        q_proj: Query projection ``Linear(embed_dim, embed_dim)``.
        k_proj: Key projection ``Linear(embed_dim, embed_dim)``.
        v_proj: Value projection ``Linear(embed_dim, embed_dim)``.
        out_proj: Output projection ``Linear(embed_dim, embed_dim)``.
        dropout_layer: Dropout on normalized attention weights.
        scale: Pre-softmax scaling ``1 / sqrt(head_dim)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Build projection layers for multi-head attention.

        Args:
            embed_dim: Hidden size ``D`` (divisible by ``num_heads``).
            num_heads: Number of attention heads ``H``.
            dropout: Dropout on attention probabilities.
        """
        super().__init__(embed_dim, num_heads, dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply multi-head scaled dot-product attention.

        Args:
            query: Tensor ``[B, L_q, D]``.
            key: Tensor ``[B, L_k, D]``.
            value: Tensor ``[B, L_k, D]``.
            mask: Optional boolean or 0/1 mask broadcastable to attention logits;
                positions with ``0`` / ``False`` are masked out before softmax.

        Returns:
            Tensor ``[B, L_q, D]`` after attention and output projection.
        """
        batch_size = query.size(0)

        # Linear projections and reshape
        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        output = self.out_proj(attended)
        return output


class CrossModalAttention(BaseFusionModule):
    """Cross-modal attention for fusing optional per-modality feature lists.

    Uses the first available modality as the query and the concatenation of
    present modalities along the sequence axis as keys and values.

    Attributes:
        attention: :class:`MultiHeadAttention` operating on ``embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 4,
        **kwargs: Any,
    ) -> None:
        """Build cross-modal attention on top of :class:`BaseFusionModule`.

        Args:
            embed_dim: Feature dimension ``D`` for each modality tensor.
            num_heads: Attention heads inside :class:`MultiHeadAttention`.
            dropout: Dropout on attention weights.
            num_modalities: Expected modality slots (for base configuration).
            **kwargs: Extra keyword arguments forwarded to ``BaseFusionModule``.
        """
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        modality_features: List[Optional[Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse modalities with cross-attention over concatenated sequences.

        Args:
            modality_features: Length-``num_modalities`` list of tensors
                ``[B, T, D]`` or ``None`` for missing inputs.
            modality_mask: Optional ``[B, M]`` boolean mask of present slots;
                built automatically when omitted.

        Returns:
            Tensor ``[B, T, D]`` for the query modality after attention.

        Raises:
            ValueError: If every modality entry is ``None``.
        """
        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Filter present modalities
        present_features = [f for f in modality_features if f is not None]
        if not present_features:
            raise ValueError("At least one modality must be present")

        # Concatenate present features along sequence dimension for cross-attention
        concatenated = torch.cat(
            present_features, dim=1
        )  # [batch, total_seq, embed_dim]

        # Use first present feature as query, concatenated as key/value
        query = present_features[0]

        # Simplified masking; production code may need finer per-position masks.
        output = self.attention(query, concatenated, concatenated)

        return output


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention and position-wise feed-forward.

    Attributes:
        attention_type: ``"self"`` uses :class:`MultiHeadAttention`; ``"cross"``
            wraps :class:`CrossModalAttention` for fusion-style wiring.
        attention: Either :class:`MultiHeadAttention` or :class:`CrossModalAttention`.
        norm1: Layer norm after the attention residual.
        norm2: Layer norm after the feed-forward residual.
        feed_forward: Two-layer MLP with ReLU and dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        attention_type: str = 'self',
    ) -> None:
        """Construct attention submodule and feed-forward stack.

        Args:
            embed_dim: Hidden dimension ``D``.
            num_heads: Number of attention heads.
            ff_dim: Hidden size of the feed-forward inner layer.
            dropout: Dropout inside attention and feed-forward paths.
            attention_type: ``"self"`` or ``"cross"`` (see ``attention`` attr).

        Raises:
            ValueError: If ``attention_type`` is not recognized.
        """
        super().__init__()
        self.attention_type = attention_type

        if attention_type == 'self':
            self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        elif attention_type == "cross":
            self.attention = CrossModalAttention(
                embed_dim, num_heads, dropout, num_modalities=1
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply attention, residual LayerNorm, feed-forward, and second norm.

        Args:
            x: Input tensor ``[B, L, D]``.
            context: Optional cross-attention context ``[B, L_c, D]`` when
                ``attention_type == "cross"``.
            mask: Optional attention mask forwarded to the attention module.

        Returns:
            Tensor ``[B, L, D]`` after the block.
        """
        if self.attention_type == "cross" and context is not None:
            attn_out = self.attention.attention(x, context, context, mask)
        else:
            attn_out = self.attention(x, x, x, mask)

        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerFusion(BaseFusionModule):
    """Project, mask-average, and run stacked transformer blocks on fused data.

    Attributes:
        max_seq_len: Upper bound used to size the positional embedding buffer.
        pos_embedding: Registered buffer ``[1, max_seq_len, D]`` added to fused
            sequences when ``T <= max_seq_len`` (otherwise padded).
        modality_projections: Per-modality linear maps before fusion.
        layers: ``ModuleList`` of :class:`TransformerBlock` layers.
        output_projection: Final linear map after transformer depth.
        modality_weights: Learnable nonnegative weights per modality slot.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Create modality projections, transformer stack, and fusion weights.

        Args:
            embed_dim: Per-modality embedding size ``D``.
            num_layers: Number of :class:`TransformerBlock` layers.
            num_heads: Attention heads per block.
            ff_dim: Feed-forward hidden size inside each block.
            dropout: Dropout inside transformer blocks.
            num_modalities: Number of modality slots (list or dict length).
            max_seq_len: Maximum sequence length for positional buffer sizing.
            **kwargs: Forwarded to :class:`BaseFusionModule`.
        """
        super().__init__(embed_dim, num_modalities, **kwargs)
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.register_buffer('pos_embedding', torch.randn(1, max_seq_len, embed_dim))

        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, 'self')
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def _project_modalities(
        self, modality_features: List[Optional[Tensor]]
    ) -> List[Tensor]:
        """Project each modality tensor into ``embed_dim`` with zero fill.

        Args:
            modality_features: List of ``[B, T, D]`` tensors or ``None`` slots.

        Returns:
            List of length ``num_modalities`` containing dense tensors.

        Raises:
            AttributeError: If a ``None`` entry is reached without a reference
                tensor for batch/sequence/device inference (callers should pass
                at least one non-``None`` modality earlier).
        """
        projected = []
        for i, feat in enumerate(modality_features):
            if feat is not None:
                projected.append(self.modality_projections[i](feat))
            else:
                # Create zero tensor for missing modalities
                ref = modality_features[0]
                batch_size, seq_len = ref.size(0), ref.size(1)
                device = ref.device
                projected.append(
                    torch.zeros(
                        batch_size, seq_len, self.embed_dim, device=device
                    )
                )
        return projected

    def _fuse_features(
        self,
        projected_features: List[Tensor],
        modality_mask: Tensor,
    ) -> Tensor:
        """Fuse projected tensors with learnable weights and a presence mask.

        Args:
            projected_features: Dense list from :meth:`_project_modalities`.
            modality_mask: ``[B, M]`` tensor with ``1`` for available slots.

        Returns:
            Fused tensor ``[B, T, D]`` averaged across modalities with masking.
        """
        # Stack features: [num_modalities, batch_size, seq_len, embed_dim]
        stacked = torch.stack(projected_features, dim=0)

        # Apply modality weights and mask
        # weights: [num_mod, 1, 1, 1] broadcasts over stacked modalities
        weights = self.modality_weights.view(-1, 1, 1, 1)
        # mask_expanded shape: [num_mod, batch, 1, 1]
        mask_expanded = modality_mask.t().unsqueeze(-1).unsqueeze(-1)

        weighted_features = stacked * weights * mask_expanded.float()

        # Average across modalities, accounting for missing ones
        num_present = modality_mask.sum(dim=1, keepdim=True).float().clamp(
            min=1
        )  # [batch, 1]
        fused = weighted_features.sum(dim=0) / num_present.unsqueeze(
            -1
        )  # [batch, seq_len, embed_dim]
        return fused

    def forward(
        self,
        modality_features: Union[List[Optional[Tensor]], Dict[str, Tensor]],
        modality_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse modalities, add positional encoding, and run transformer layers.

        Args:
            modality_features: Dict mapping names to ``[B, T, D]`` tensors or a
                parallel list with ``None`` for missing modalities.
            modality_mask: Optional ``[B, M]`` mask; inferred when omitted.

        Returns:
            Fused tensor ``[B, T, D]`` after projection, masking, transformer
            depth, and the output projection.

        Raises:
            ValueError: Propagated from attention if tensors are invalid (e.g.
                all modalities missing upstream).
        """
        if isinstance(modality_features, dict):
            modality_features = list(modality_features.values())

        if modality_mask is None:
            modality_mask = self.get_modality_mask(modality_features)

        # Project modalities
        projected_features = self._project_modalities(modality_features)

        # Fuse features
        fused = self._fuse_features(projected_features, modality_mask)

        batch_size, seq_len, _ = fused.size()

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            pad_len = seq_len - self.max_seq_len
            extended_pos = torch.cat(
                [
                    self.pos_embedding,
                    torch.zeros(
                        1,
                        pad_len,
                        self.embed_dim,
                        device=fused.device,
                    ),
                ],
                dim=1,
            )
            fused = fused + extended_pos

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, mask=None)  # Self-attention within fused features

        # Final projection
        output = self.output_projection(fused)

        return output

# =============================================================================
# Main Wav2Sleep Model
# =============================================================================

class Wav2Sleep(BaseModel):
    """Multimodal sleep-stage model with CNN encoders, fusion, and temporal CNN.

    Pipeline: per-modality waveforms ``[B, T]`` → :class:`SignalEncoders` →
    ``[B, E, D]`` embeddings → :class:`TransformerFusion` (or mean fallback) →
    :class:`DilatedCNNSequenceMixer` / :class:`TemporalConvBlock` → linear
    classifier.

    Attributes:
        embedding_dim: Dimension ``D`` of modality embeddings before fusion.
        hidden_dim: Temporal mixer output width (classifier input width).
        num_classes: Number of sleep stages (e.g. 5).
        dropout: Dropout used in fusion and temporal layers when applicable.
        use_paper_faithful: If True, use transformer fusion and dilated mixer.
        label_key: Single label key from the dataset (asserted at init).
        modality_keys: Expected input keys (currently ``ecg``, ``ppg``).
        modality_to_signal: Maps modality key to :class:`SignalEncoders` signal
            name (e.g. ``ecg`` → ``ECG``).
        available_modalities: Subset of ``modality_keys`` present in the dataset.
        signal_encoders: Shared CNN encoders for waveform segments.
        fusion_module: :class:`TransformerFusion` or ``None`` in simplified mode.
        temporal_layer: :class:`DilatedCNNSequenceMixer` or
            :class:`TemporalConvBlock`.
        classifier: Linear map from ``hidden_dim`` to ``num_classes``.

    Note:
        Inputs must include at least one of ``modality_keys`` plus the batch
        label tensor under ``label_key``. Shapes are validated before fusion.

    Examples:
        Typical construction and forward pass (replace ``sleep_stage`` with
        ``model.label_key`` for your dataset)::

            from pyhealth.models import Wav2Sleep

            model = Wav2Sleep(
                dataset,
                embedding_dim=128,
                hidden_dim=128,
                num_classes=5,
            )
            outputs = model(
                ecg=ecg_tensor,
                sleep_stage=label_tensor,
            )
            loss = outputs["loss"]
            logits = outputs["logit"]

        Set ``use_paper_faithful=False`` to swap in mean fusion and
        :class:`TemporalConvBlock` for ablations.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_fusion_heads: int = 4,
        num_fusion_layers: int = 2,
        num_temporal_layers: int = 5,
        temporal_kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_paper_faithful: bool = True,
    ) -> None:
        """Construct ``Wav2Sleep`` from dataset metadata and hyperparameters.

        Args:
            dataset: Fitted dataset with feature and label keys.
            embedding_dim: Encoder and fusion embedding size ``D``.
            hidden_dim: Output width of the temporal stack (classifier input).
            num_classes: Number of classes (e.g. Wake, N1, N2, N3, REM).
            num_fusion_heads: Attention heads in :class:`TransformerFusion`.
            num_fusion_layers: Transformer layers in fusion.
            num_temporal_layers: Layers in the dilated mixer (paper mode).
            temporal_kernel_size: Conv kernel size along time for temporal stack.
            dilations: Dilation schedule for :class:`DilatedCNNSequenceMixer`;
                defaults inside the mixer when ``None``.
            dropout: Dropout for fusion and temporal modules.
            use_paper_faithful: Use transformer fusion + dilated mixer; else
                mean fusion and :class:`TemporalConvBlock`.

        Raises:
            AssertionError: If the dataset does not expose exactly one label key.
            ValueError: If no expected modality appears in ``dataset`` feature keys.
        """
        super().__init__(dataset=dataset)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_paper_faithful = use_paper_faithful

        assert len(self.label_keys) == 1, "Wav2Sleep only supports single label key"
        self.label_key = self.label_keys[0]

        self.modality_keys = ['ecg', 'ppg']
        self.modality_to_signal = {'ecg': 'ECG', 'ppg': 'PPG'}

        self.available_modalities = [
            key for key in self.modality_keys if key in self.feature_keys
        ]
        if not self.available_modalities:
            raise ValueError(
                "At least one modality (ecg, ppg) must be present in dataset"
            )

        self.signal_encoders = SignalEncoders(
            signal_encoder_map={
                self.modality_to_signal[m]: self.modality_to_signal[m]
                for m in self.available_modalities
            },
            feature_dim=embedding_dim,
            activation='relu',
            norm='instance',
            include_signal=False,
        )

        if use_paper_faithful:
            self.fusion_module = TransformerFusion(
                embed_dim=embedding_dim,
                num_heads=num_fusion_heads,
                num_layers=num_fusion_layers,
                dropout=dropout,
                num_modalities=len(self.available_modalities),
            )
        else:
            self.fusion_module = None

        if use_paper_faithful:
            self.temporal_layer = DilatedCNNSequenceMixer(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                kernel_size=temporal_kernel_size,
                num_layers=num_temporal_layers,
                dilations=dilations,
                dropout=dropout,
            )
        else:
            self.temporal_layer = TemporalConvBlock(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                kernel_size=temporal_kernel_size,
                dropout=dropout,
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _validate_modality_shapes(self, modality_embeddings: Dict[str, Tensor]) -> None:
        """Ensure every modality tensor has the same ``[B, T, D]`` shape.

        Args:
            modality_embeddings: Maps modality names to embeddings.

        Raises:
            ValueError: If the mapping is empty or any tensor shape differs.
        """
        if not modality_embeddings:
            raise ValueError("No modality embeddings provided")

        reference_shape = next(iter(modality_embeddings.values())).shape

        for modality, embedding in modality_embeddings.items():
            if embedding.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch in modality '{modality}': expected "
                    f"{reference_shape}, got {embedding.shape}"
                )

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        """Encode modalities, fuse, model time, and compute loss and logits.

        Args:
            **kwargs: Batch fields. Must include at least one of ``ecg`` and
                ``ppg`` (when present in the dataset), the label tensor under
                ``self.label_key``, and optionally ``embed`` (bool) to return
                temporal features.

        Returns:
            Dictionary with:

                * ``loss``: Scalar loss.
                * ``y_prob``: ``[B*T, num_classes]`` probabilities.
                * ``y_true``: ``[B*T]`` flattened labels.
                * ``logit``: ``[B*T, num_classes]`` logits.
                * ``embed`` (optional): ``[B, T, hidden_dim]`` if
                  ``kwargs['embed']`` is true.

        Raises:
            ValueError: If no modality tensor is provided in ``kwargs``.
        """
        if not any(m in kwargs for m in self.available_modalities):
            raise ValueError("At least one modality must be provided in input")

        wave_inputs = {}
        for modality in self.available_modalities:
            if modality not in kwargs:
                continue
            sig = self.modality_to_signal[modality]
            wave_inputs[sig] = kwargs[modality].to(self.device).float()
        z_signal = self.signal_encoders(wave_inputs)

        modality_embeddings = {
            modality: z_signal[self.modality_to_signal[modality]]
            for modality in self.available_modalities
            if modality in kwargs
        }

        self._validate_modality_shapes(modality_embeddings)
        if self.use_paper_faithful and self.fusion_module is not None:
            fused_features = self.fusion_module(modality_embeddings)
        else:
            if len(modality_embeddings) == 1:
                fused_features = next(iter(modality_embeddings.values()))
            else:
                stacked = torch.stack(list(modality_embeddings.values()), dim=0)
                fused_features = stacked.mean(dim=0)

        temporal_features = self.temporal_layer(fused_features)

        logits = self.classifier(temporal_features)

        logits_flat = logits.reshape(-1, logits.size(-1))

        y_true = kwargs[self.label_key].to(self.device)
        y_true_flat = y_true.view(-1)

        loss = self.get_loss_function()(logits_flat, y_true_flat)
        y_prob = self.prepare_y_prob(logits_flat)

        results: Dict[str, Tensor] = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true_flat,
            "logit": logits_flat,
        }

        if kwargs.get("embed", False):
            results["embed"] = temporal_features

        return results

    def get_reproduction_fidelity_report(self) -> Dict[str, str]:
        """Summarize how closely optional components match the reference paper.

        Returns:
            Mapping from component name to a short human-readable status string.
        """
        report = {
            "overall": (
                "paper_faithful" if self.use_paper_faithful else "simplified"
            ),
            "fusion_module": (
                "TransformerFusion (paper-faithful)"
                if self.use_paper_faithful
                else "Mean pooling (simplified)"
            ),
            "temporal_layer": (
                f"Dilated CNN with receptive field "
                f"{self.temporal_layer.receptive_field} epochs (paper-faithful)"
                if self.use_paper_faithful
                and hasattr(self.temporal_layer, "receptive_field")
                else "Standard temporal CNN (simplified)"
            ),
            "modality_encoders": "Placeholder (needs Dhruv's CNN encoders)",
            "classification_head": "Linear layer (paper-faithful)",
        }
        return report


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    # This section can be used for basic testing
    print("Wav2Sleep model loaded successfully!")
    print("\nPaper-faithful components:")
    print("  - TransformerFusion: Transformer-based multimodal aggregation")
    print("  - DilatedCNNSequenceMixer: Dilated temporal convolutions")
    print("\nSimplified components (for comparison):")
    print("  - TemporalConvBlock: Standard temporal CNN")
