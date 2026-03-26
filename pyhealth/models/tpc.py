from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """
    Feature-wise causal temporal convolution.

    Input:
        x: [B, T, F, C_in]

    Output:
        y: [B, T, F, C_out]

    Notes:
        - One Conv1d per feature (non-shared temporal filters).
        - Left padding preserves causality.
        - Each feature is convolved independently through time.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
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
        """
        Args:
            x: [B, T, F, C_in]

        Returns:
            y: [B, T, F, C_out]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape [B, T, F, C], got {tuple(x.shape)}")

        bsz, seq_len, num_features, in_channels = x.shape
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
            # [B, T, C_in] -> [B, C_in, T]
            feat_x = x[:, :, feat_idx, :].transpose(1, 2)
            feat_x = F.pad(feat_x, (left_pad, 0))
            feat_y = self.feature_convs[feat_idx](feat_x)  # [B, C_out, T]
            feat_y = feat_y.transpose(1, 2)  # [B, T, C_out]
            outputs.append(feat_y.unsqueeze(2))  # [B, T, 1, C_out]

        y = torch.cat(outputs, dim=2)  # [B, T, F, C_out]
        y = self.dropout(y)
        return y


class PointwiseConvBlock(nn.Module):
    """
    Pointwise transformation applied independently at each time step.

    Input:
        x: [B, T, D_in]

    Output:
        y: [B, T, D_out]
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        y = self.linear(x)
        y = self.dropout(y)
        return y


class TPCLayer(nn.Module):
    """
    One TPC layer combining:

    - feature-wise causal temporal convolution
    - pointwise transformation over flattened current representation
    - optional static feature injection
    - optional decay injection
    - concatenative skip-style fusion

    Input:
        x: [B, T, F, C_in]
        decay: [B, T, F] or None
        static: [B, S] or None

    Output:
        fused: [B, T, F, C_in + temporal_channels + pointwise_channels]
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
    ) -> None:
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
        self.output_channels = in_channels + temporal_channels + pointwise_channels

        self.temporal = TemporalConvBlock(
            in_channels=in_channels,
            out_channels=temporal_channels,
            num_features=num_features,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )

        point_input_dim = (num_features * in_channels) + static_dim
        if use_decay_in_pointwise:
            point_input_dim += num_features

        self.pointwise = PointwiseConvBlock(
            input_dim=point_input_dim,
            output_dim=pointwise_channels,
            dropout=dropout,
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        decay: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, F, C_in]
            decay: [B, T, F] or None
            static: [B, S] or None

        Returns:
            fused: [B, T, F, C_in + temporal_channels + pointwise_channels]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape [B, T, F, C], got {tuple(x.shape)}")

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
                    f"Expected decay shape {(bsz, seq_len, num_features)}, got {tuple(decay.shape)}"
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

        temp_out = self.temporal(x)  # [B, T, F, temporal_channels]

        flat_x = x.reshape(bsz, seq_len, num_features * in_channels)  # [B, T, F*C_in]
        parts = [flat_x]

        if static is not None:
            static_rep = static.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, S]
            parts.append(static_rep)

        if self.use_decay_in_pointwise:
            if decay is None:
                raise ValueError("decay must be provided when use_decay_in_pointwise=True")
            parts.append(decay)

        point_in = torch.cat(parts, dim=-1)  # [B, T, D_in]
        point_out = self.pointwise(point_in)  # [B, T, pointwise_channels]
        point_broadcast = point_out.unsqueeze(2).expand(-1, -1, num_features, -1)

        fused = torch.cat([x, temp_out, point_broadcast], dim=-1)
        fused = self.activation(fused)
        return fused


class TPC(nn.Module):
    """
    Temporal Pointwise Convolutional model for hourly LoS regression.

    This version expects:
        - x_values: hourly feature values
        - x_decay: hourly decay indicators
        - static: optional static features

    Inputs:
        x_values: [B, T, F]
        x_decay: [B, T, F]
        static: [B, S] or None

    Outputs:
        if return_sequence=True:
            y_hat: [B, T]
        else:
            y_hat: [B]

    Notes:
        - Initial channels are stacked as [value, decay] per feature.
        - Each TPC layer also feeds decay into the pointwise branch.
        - The default is sequence output because your task proposal is
          hourly remaining LoS prediction, not just a final single-time prediction.
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
    ) -> None:
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
        """
        Args:
            x_values: [B, T, F]
            x_decay: [B, T, F]
            static: [B, S] or None

        Returns:
            y_hat:
                [B, T] if return_sequence=True
                [B] if return_sequence=False
        """
        if x_values.ndim != 3:
            raise ValueError(
                f"Expected x_values to have shape [B, T, F], got {tuple(x_values.shape)}"
            )
        if x_decay.ndim != 3:
            raise ValueError(
                f"Expected x_decay to have shape [B, T, F], got {tuple(x_decay.shape)}"
            )
        if x_values.shape != x_decay.shape:
            raise ValueError(
                f"x_values and x_decay must have the same shape, got "
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
                f"but static=None was provided"
            )

        # Stack value and decay into per-feature channels.
        # x: [B, T, F, 2]
        x = torch.stack([x_values, x_decay], dim=-1)

        for layer in self.layers:
            x = layer(x, decay=x_decay, static=static)

        if self.return_sequence:
            # Predict a remaining LoS value at every hour.
            # [B, T, F, C] -> [B, T, F*C]
            all_x = x.reshape(bsz, seq_len, -1)

            if static is not None:
                static_rep = static.unsqueeze(1).expand(-1, seq_len, -1)
                all_x = torch.cat([all_x, static_rep], dim=-1)

            h = self.relu(self.final_fc1(all_x))
            y = self.final_fc2(h).squeeze(-1)  # [B, T]

            if self.positive_output:
                y = self.softplus(y)

            return y

        # Predict only from the last time step.
        last_x = x[:, -1, :, :].reshape(bsz, -1)  # [B, F*C]

        if static is not None:
            last_x = torch.cat([last_x, static], dim=-1)

        h = self.relu(self.final_fc1(last_x))
        y = self.final_fc2(h).squeeze(-1)  # [B]

        if self.positive_output:
            y = self.softplus(y)

        return y
