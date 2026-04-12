"""
Temporal Pointwise Convolution (TPC) model implementation for hourly ICU
length-of-stay (LoS) prediction.

This module provides a PyTorch implementation of the Temporal Pointwise
Convolution (TPC) architecture described in:

Rocheteau, E., Liò, P., and Hyland, S. (2021).
"Temporal Pointwise Convolutional Networks for Length of Stay Prediction
in the Intensive Care Unit."

Overview:
    The TPC model is designed for multivariate, irregularly sampled EHR
    time-series. It combines two complementary operations at each layer:

    1. Temporal Convolution (TC):
        Feature-wise or shared causal convolutions over time, enabling
        each clinical variable to learn independent temporal dynamics.

    2. Pointwise Convolution (PC):
        Per-time-step feature mixing (1×1 convolution equivalent) to
        capture cross-feature interactions without temporal leakage.

    These components are combined with optional skip connections and
    domain-specific inputs such as decay indicators and static features.

Key Components:
    - TemporalConvBlock: Feature-wise or shared causal temporal convolution
    - PointwiseConvBlock: Per-time-step feature interaction layer
    - TPCLayer: Combined temporal + pointwise layer with skip connections
    - TPC: Full stacked model for LoS regression

Inputs:
    x_values: Tensor of shape [B, T, F]
        Hourly time-series feature values.

    x_decay: Tensor of shape [B, T, F]
        Decay indicators representing time since last observation.

    static: Optional Tensor of shape [B, S]
        Static patient-level features.

Outputs:
    - Sequence mode: [B, T] predictions (default)
    - Final-step mode: [B] prediction

Implementation Notes:
    - Initial feature channels are constructed as [value, decay].
    - Causal padding ensures no future information leakage.
    - Supports ablations:
        * shared vs feature-wise temporal convolutions
        * temporal-only / pointwise-only configurations
        * skip connections on/off
        * decay inclusion in pointwise branch
    - Positive outputs can be enforced via Softplus.

Example:
    >>> model = TPC(input_dim=F, static_dim=S)
    >>> y_pred = model(x_values, x_decay, static)

This implementation is intended for integration with PyHealth task pipelines
for hourly ICU length-of-stay prediction and supports reproducible ablation
studies using synthetic or real EHR datasets.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Feature-wise or shared causal temporal convolution block.

    This block applies a 1-D causal convolution over time for each feature.
    It supports two modes:

    1. Feature-wise temporal convolution:
       A separate ``nn.Conv1d`` is created for each feature so temporal
       filters are not shared across features.

    2. Shared temporal convolution:
       A single ``nn.Conv1d`` is reused for all features, allowing an
       ablation against the feature-specific version.

    Input:
        x: Tensor of shape ``[B, T, F, C_in]``

    Output:
        y: Tensor of shape ``[B, T, F, C_out]``

    Notes:
        - Left padding preserves temporal causality.
        - Output length matches input sequence length.
        - Each feature is processed independently in time even when
          weights are shared.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        shared_temporal: bool = False,
    ) -> None:
        """Initialize the temporal convolution block.

        Args:
            in_channels: Number of per-feature input channels.
            out_channels: Number of per-feature output channels.
            num_features: Number of time-series features.
            kernel_size: Temporal convolution kernel size.
            dilation: Temporal dilation factor.
            dropout: Dropout probability applied after convolution.
            shared_temporal: Whether to share one temporal convolution
                across all features.

        Raises:
            ValueError: If any required dimensional argument is invalid.
        """
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.shared_temporal = shared_temporal

        if self.shared_temporal:
            self.shared_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=True,
            )
            self.feature_convs = None
        else:
            self.shared_conv = None
            self.feature_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        bias=True,
                    )
                    for _ in range(num_features)
                ]
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal temporal convolution to each feature.

        Args:
            x: Input tensor of shape ``[B, T, F, C_in]``.

        Returns:
            Output tensor of shape ``[B, T, F, C_out]``.

        Raises:
            ValueError: If the input tensor has incompatible shape.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape [B, T, F, C], got {tuple(x.shape)}"
            )

        _, _, num_features, in_channels = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}"
            )
        if in_channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {in_channels}"
            )

        outputs = []
        left_pad = self.dilation * (self.kernel_size - 1)

        for feat_idx in range(self.num_features):
            feat_x = x[:, :, feat_idx, :].transpose(1, 2)  # [B, C_in, T]
            feat_x = F.pad(feat_x, (left_pad, 0))

            if self.shared_temporal:
                feat_y = self.shared_conv(feat_x)  # [B, C_out, T]
            else:
                feat_y = self.feature_convs[feat_idx](feat_x)  # [B, C_out, T]

            feat_y = feat_y.transpose(1, 2)  # [B, T, C_out]
            outputs.append(feat_y.unsqueeze(2))  # [B, T, 1, C_out]

        y = torch.cat(outputs, dim=2)  # [B, T, F, C_out]
        y = self.dropout(y)
        return y


class PointwiseConvBlock(nn.Module):
    """Pointwise transformation applied independently at each time step.

    This block is implemented as a linear layer operating on the flattened
    per-time-step representation. It is equivalent in spirit to a 1x1
    convolution across the feature/channel dimension at each hour.

    Input:
        x: Tensor of shape ``[B, T, D_in]``

    Output:
        y: Tensor of shape ``[B, T, D_out]``
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the pointwise block.

        Args:
            input_dim: Input dimension per time step.
            output_dim: Output dimension per time step.
            dropout: Dropout probability applied after projection.

        Raises:
            ValueError: If either dimension is not positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pointwise transformation.

        Args:
            x: Input tensor of shape ``[B, T, D_in]``.

        Returns:
            Output tensor of shape ``[B, T, D_out]``.

        Raises:
            ValueError: If the input tensor shape is invalid.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape [B, T, D], got {tuple(x.shape)}"
            )
        y = self.linear(x)
        y = self.dropout(y)
        return y


class TPCLayer(nn.Module):
    """One Temporal Pointwise Convolution layer.

    This layer combines:

    - optional temporal convolution branch
    - optional pointwise branch
    - optional concatenative skip connections

    Input:
        x: Tensor of shape ``[B, T, F, C_in]``
        decay: Tensor of shape ``[B, T, F]`` or ``None``
        static: Tensor of shape ``[B, S]`` or ``None``

    Output:
        fused: Tensor of shape ``[B, T, F, C_out]``

    Notes:
        - The temporal branch performs feature-wise or shared causal
          temporal convolution.
        - The pointwise branch performs per-time-step feature mixing.
        - Skip connections are implemented as concatenation of the input
          representation with new branch outputs.
    """

    def __init__(
        self,
        num_features: int,
        in_channels: int,
        temporal_channels: int,
        pointwise_channels: int,
        static_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        use_decay_in_pointwise: bool = True,
        shared_temporal: bool = False,
        use_temporal: bool = True,
        use_pointwise: bool = True,
        use_skip_connections: bool = True,
    ) -> None:
        """Initialize the TPC layer.

        Args:
            num_features: Number of time-series features.
            in_channels: Number of input channels per feature.
            temporal_channels: Number of temporal branch output channels.
            pointwise_channels: Number of pointwise branch output channels.
            static_dim: Static feature dimension.
            kernel_size: Temporal kernel size.
            dilation: Temporal dilation factor.
            dropout: Dropout probability.
            use_decay_in_pointwise: Whether decay indicators are concatenated
                into the pointwise branch input.
            shared_temporal: Whether temporal filters are shared across features.
            use_temporal: Whether to enable the temporal branch.
            use_pointwise: Whether to enable the pointwise branch.
            use_skip_connections: Whether to concatenate the prior input
                representation into the layer output.

        Raises:
            ValueError: If layer dimensions are invalid or both branches are off.
        """
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if temporal_channels <= 0:
            raise ValueError("temporal_channels must be positive")
        if pointwise_channels <= 0:
            raise ValueError("pointwise_channels must be positive")
        if static_dim < 0:
            raise ValueError("static_dim must be non-negative")

        self.num_features = num_features
        self.in_channels = in_channels
        self.temporal_channels = temporal_channels
        self.pointwise_channels = pointwise_channels
        self.static_dim = static_dim
        self.use_decay_in_pointwise = use_decay_in_pointwise
        self.shared_temporal = shared_temporal
        self.use_temporal = use_temporal
        self.use_pointwise = use_pointwise
        self.use_skip_connections = use_skip_connections

        if not self.use_temporal and not self.use_pointwise:
            raise ValueError("At least one of use_temporal or use_pointwise must be True")

        self.output_channels = 0
        if self.use_skip_connections:
            self.output_channels += in_channels
        if self.use_temporal:
            self.output_channels += temporal_channels
        if self.use_pointwise:
            self.output_channels += pointwise_channels

        if self.use_temporal:
            self.temporal = TemporalConvBlock(
                in_channels=in_channels,
                out_channels=temporal_channels,
                num_features=num_features,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                shared_temporal=shared_temporal,
            )
        else:
            self.temporal = None

        if self.use_pointwise:
            point_input_dim = (num_features * in_channels) + static_dim
            if use_decay_in_pointwise:
                point_input_dim += num_features

            self.pointwise = PointwiseConvBlock(
                input_dim=point_input_dim,
                output_dim=pointwise_channels,
                dropout=dropout,
            )
        else:
            self.pointwise = None

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        decay: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the TPC layer.

        Args:
            x: Input tensor of shape ``[B, T, F, C_in]``.
            decay: Optional decay tensor of shape ``[B, T, F]``.
            static: Optional static tensor of shape ``[B, S]``.

        Returns:
            Fused tensor of shape ``[B, T, F, C_out]``.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape [B, T, F, C], got {tuple(x.shape)}"
            )

        bsz, seq_len, num_features, in_channels = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}"
            )
        if in_channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {in_channels}"
            )

        if decay is not None:
            if decay.ndim != 3:
                raise ValueError(
                    f"Expected decay to have shape [B, T, F], got {tuple(decay.shape)}"
                )
            if decay.shape[:3] != (bsz, seq_len, num_features):
                raise ValueError(
                    "Expected decay shape "
                    f"{(bsz, seq_len, num_features)}, got {tuple(decay.shape)}"
                )

        if static is not None:
            if static.ndim != 2:
                raise ValueError(
                    f"Expected static to have shape [B, S], got {tuple(static.shape)}"
                )
            if static.shape[0] != bsz:
                raise ValueError(
                    f"Expected static batch size {bsz}, got {static.shape[0]}"
                )

        parts_to_concat = []

        if self.use_skip_connections:
            parts_to_concat.append(x)

        if self.use_temporal:
            temp_out = self.temporal(x)  # [B, T, F, temporal_channels]
            parts_to_concat.append(temp_out)

        if self.use_pointwise:
            flat_x = x.reshape(bsz, seq_len, num_features * in_channels)
            point_parts = [flat_x]

            if static is not None:
                static_rep = static.unsqueeze(1).expand(-1, seq_len, -1)
                point_parts.append(static_rep)

            if self.use_decay_in_pointwise:
                if decay is None:
                    raise ValueError(
                        "decay must be provided when use_decay_in_pointwise=True"
                    )
                point_parts.append(decay)

            point_in = torch.cat(point_parts, dim=-1)  # [B, T, D_in]
            point_out = self.pointwise(point_in)  # [B, T, pointwise_channels]
            point_broadcast = point_out.unsqueeze(2).expand(
                -1, -1, num_features, -1
            )
            parts_to_concat.append(point_broadcast)

        fused = torch.cat(parts_to_concat, dim=-1)
        fused = self.activation(fused)
        return fused


class TPC(nn.Module):
    """Temporal Pointwise Convolution model for hourly LoS regression.

    This implementation expects three logical inputs:

    - ``x_values``: hourly time-series values
    - ``x_decay``: hourly decay indicators
    - ``static``: optional static features

    Inputs:
        x_values: Tensor of shape ``[B, T, F]``
        x_decay: Tensor of shape ``[B, T, F]``
        static: Tensor of shape ``[B, S]`` or ``None``

    Outputs:
        - If ``return_sequence=True``: tensor of shape ``[B, T]``
        - Else: tensor of shape ``[B]``

    Notes:
        - Initial per-feature channels are stacked as ``[value, decay]``.
        - Each TPC layer can enable or disable temporal and pointwise branches
          for architecture ablations.
        - The default output mode is sequence prediction because the task is
          hourly remaining length-of-stay regression.
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        temporal_channels: int = 8,
        pointwise_channels: int = 8,
        num_layers: int = 3,
        kernel_size: int = 3,
        fc_dim: int = 32,
        dropout: float = 0.1,
        return_sequence: bool = True,
        use_decay_in_pointwise: bool = True,
        positive_output: bool = True,
        shared_temporal: bool = False,
        use_temporal: bool = True,
        use_pointwise: bool = True,
        use_skip_connections: bool = True,
    ) -> None:
        """Initialize the TPC model.

        Args:
            input_dim: Number of time-series features.
            static_dim: Static feature dimension.
            temporal_channels: Temporal branch output channels per layer.
            pointwise_channels: Pointwise branch output channels per layer.
            num_layers: Number of stacked TPC layers.
            kernel_size: Temporal kernel size.
            fc_dim: Hidden dimension in the final prediction head.
            dropout: Dropout probability.
            return_sequence: Whether to predict at every hour or only the last.
            use_decay_in_pointwise: Whether to inject decay into the pointwise
                branch.
            positive_output: Whether to enforce positive outputs using Softplus.
            shared_temporal: Whether temporal weights are shared across features.
            use_temporal: Whether to enable the temporal branch.
            use_pointwise: Whether to enable the pointwise branch.
            use_skip_connections: Whether to use concatenative skip connections.

        Raises:
            ValueError: If any dimensional argument is invalid.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if static_dim < 0:
            raise ValueError("static_dim must be non-negative")
        if temporal_channels <= 0:
            raise ValueError("temporal_channels must be positive")
        if pointwise_channels <= 0:
            raise ValueError("pointwise_channels must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if fc_dim <= 0:
            raise ValueError("fc_dim must be positive")

        self.input_dim = input_dim
        self.static_dim = static_dim
        self.temporal_channels = temporal_channels
        self.pointwise_channels = pointwise_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.return_sequence = return_sequence
        self.use_decay_in_pointwise = use_decay_in_pointwise
        self.positive_output = positive_output
        self.shared_temporal = shared_temporal
        self.use_temporal = use_temporal
        self.use_pointwise = use_pointwise
        self.use_skip_connections = use_skip_connections

        layers = []
        in_channels = 2  # value + decay channels at input

        for i in range(num_layers):
            layer = TPCLayer(
                num_features=input_dim,
                in_channels=in_channels,
                temporal_channels=temporal_channels,
                pointwise_channels=pointwise_channels,
                static_dim=static_dim,
                kernel_size=kernel_size,
                dilation=i + 1,
                dropout=dropout,
                use_decay_in_pointwise=use_decay_in_pointwise,
                shared_temporal=shared_temporal,
                use_temporal=use_temporal,
                use_pointwise=use_pointwise,
                use_skip_connections=use_skip_connections,
            )
            layers.append(layer)
            in_channels = layer.output_channels

        self.layers = nn.ModuleList(layers)

        final_input_dim = (input_dim * in_channels) + static_dim
        self.final_fc1 = nn.Linear(final_input_dim, fc_dim)
        self.final_fc2 = nn.Linear(fc_dim, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self,
        x_values: torch.Tensor,
        x_decay: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the TPC model forward.

        Args:
            x_values: Tensor of shape ``[B, T, F]`` containing feature values.
            x_decay: Tensor of shape ``[B, T, F]`` containing decay indicators.
            static: Optional tensor of shape ``[B, S]``.

        Returns:
            A tensor of shape ``[B, T]`` if ``return_sequence=True`` or
            ``[B]`` otherwise.

        Raises:
            ValueError: If input tensor shapes are invalid.
        """
        if x_values.ndim != 3:
            raise ValueError(
                "Expected x_values to have shape [B, T, F], got "
                f"{tuple(x_values.shape)}"
            )
        if x_decay.ndim != 3:
            raise ValueError(
                "Expected x_decay to have shape [B, T, F], got "
                f"{tuple(x_decay.shape)}"
            )
        if x_values.shape != x_decay.shape:
            raise ValueError(
                "x_values and x_decay must have the same shape, got "
                f"{tuple(x_values.shape)} and {tuple(x_decay.shape)}"
            )

        bsz, seq_len, num_features = x_values.shape
        if num_features != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {num_features}"
            )

        if static is not None:
            if static.ndim != 2:
                raise ValueError(
                    f"Expected static to have shape [B, S], got {tuple(static.shape)}"
                )
            if static.shape[0] != bsz:
                raise ValueError(
                    f"Expected static batch size {bsz}, got {static.shape[0]}"
                )
            if static.shape[1] != self.static_dim:
                raise ValueError(
                    f"Expected static_dim={self.static_dim}, got {static.shape[1]}"
                )
        elif self.static_dim != 0:
            raise ValueError(
                f"Model was initialized with static_dim={self.static_dim}, "
                "but static=None was provided"
            )

        x = torch.stack([x_values, x_decay], dim=-1)  # [B, T, F, 2]

        for layer in self.layers:
            x = layer(x, decay=x_decay, static=static)

        if self.return_sequence:
            all_x = x.reshape(bsz, seq_len, -1)

            if static is not None:
                static_rep = static.unsqueeze(1).expand(-1, seq_len, -1)
                all_x = torch.cat([all_x, static_rep], dim=-1)

            h = self.relu(self.final_fc1(all_x))
            y = self.final_fc2(h).squeeze(-1)  # [B, T]

            if self.positive_output:
                y = self.softplus(y)

            return y

        last_x = x[:, -1, :, :].reshape(bsz, -1)  # [B, F*C]

        if static is not None:
            last_x = torch.cat([last_x, static], dim=-1)

        h = self.relu(self.final_fc1(last_x))
        y = self.final_fc2(h).squeeze(-1)  # [B]

        if self.positive_output:
            y = self.softplus(y)

        return y
