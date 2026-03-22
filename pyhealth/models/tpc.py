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
            [B, T, F, C_out]
        """
        bsz, seq_len, num_features, _ = x.shape
        assert num_features == self.num_features

        outputs = []
        left_pad = self.dilation * (self.kernel_size - 1)

        for feat_idx in range(self.num_features):
            feat_x = x[:, :, feat_idx, :].transpose(1, 2)  # [B, C_in, T]
            feat_x = F.pad(feat_x, (left_pad, 0))
            feat_y = self.feature_convs[feat_idx](feat_x)  # [B, C_out, T]
            feat_y = feat_y.transpose(1, 2)  # [B, T, C_out]
            outputs.append(feat_y.unsqueeze(2))  # [B, T, 1, C_out]

        y = torch.cat(outputs, dim=2)  # [B, T, F, C_out]
        y = self.dropout(y)
        return y


class PointwiseConvBlock(nn.Module):
    """
    Pointwise transformation applied at each time step.

    Input:
        x: [B, T, D_in]

    Output:
        y: [B, T, D_out]
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.dropout(y)
        return y


class TPCLayer(nn.Module):
    """
    One simplified TPC layer combining:

    - feature-wise causal temporal convolution
    - pointwise transformation over flattened current representation
    - static feature injection into the pointwise branch
    - concatenative skip-style fusion

    Input:
        x: [B, T, F, C_in]

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
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.in_channels = in_channels
        self.temporal_channels = temporal_channels
        self.pointwise_channels = pointwise_channels
        self.static_dim = static_dim
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
        self.pointwise = PointwiseConvBlock(
            input_dim=point_input_dim,
            output_dim=pointwise_channels,
            dropout=dropout,
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, F, C_in]
            static: [B, S] or None

        Returns:
            fused: [B, T, F, C_in + temporal_channels + pointwise_channels]
        """
        bsz, seq_len, num_features, in_channels = x.shape
        assert num_features == self.num_features
        assert in_channels == self.in_channels

        temp_out = self.temporal(x)  # [B, T, F, Y]

        flat_x = x.reshape(bsz, seq_len, num_features * in_channels)  # [B, T, F*C_in]

        if static is not None:
            static_rep = static.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, S]
            point_in = torch.cat([flat_x, static_rep], dim=-1)
        else:
            point_in = flat_x

        point_out = self.pointwise(point_in)  # [B, T, Z]
        point_broadcast = point_out.unsqueeze(2).expand(-1, -1, num_features, -1)

        fused = torch.cat([x, temp_out, point_broadcast], dim=-1)
        fused = self.activation(fused)
        return fused


class TPC(nn.Module):
    """
    Simplified Temporal Pointwise Convolutional model for hourly LoS regression.

    Current implementation characteristics:
        - feature-wise causal temporal convolutions with non-shared filters
        - pointwise branch conditioned on current representation + static features
        - concatenative skip-style fusion across layers
        - positive regression output via Softplus

    Input:
        time_series: [B, T, F]
        static: [B, S]

    Output:
        y_hat: [B]
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        temporal_channels: int = 8,
        pointwise_channels: int = 8,
        num_layers: int = 3,
        kernel_size: int = 3,
        fc_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.temporal_channels = temporal_channels
        self.pointwise_channels = pointwise_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout

        layers = []
        in_channels = 1

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
        time_series: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            time_series: [B, T, F]
            static: [B, S] or None

        Returns:
            y_hat: [B]
        """
        x = time_series.unsqueeze(-1)  # [B, T, F, 1]

        for layer in self.layers:
            x = layer(x, static=static)

        last_x = x[:, -1, :, :].reshape(x.size(0), -1)  # [B, F*C]

        if static is not None:
            last_x = torch.cat([last_x, static], dim=-1)

        h = self.relu(self.final_fc1(last_x))
        y = self.softplus(self.final_fc2(h)).squeeze(-1)
        return y
