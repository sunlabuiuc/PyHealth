from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    """
    Feature-wise causal temporal convolution.

    Input:
        x: [B, T, F, C_in]
    Output:
        y: [B, T, F, C_out]
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

        # one Conv1d per feature, matching the paper's non-shared temporal filters
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
            # [B, T, C_in] -> [B, C_in, T]
            feat_x = x[:, :, feat_idx, :].transpose(1, 2)
            feat_x = nn.functional.pad(feat_x, (left_pad, 0))
            feat_y = self.feature_convs[feat_idx](feat_x)  # [B, C_out, T]
            feat_y = feat_y.transpose(1, 2)  # [B, T, C_out]
            outputs.append(feat_y.unsqueeze(2))  # [B, T, 1, C_out]

        y = torch.cat(outputs, dim=2)  # [B, T, F, C_out]
        y = self.dropout(y)
        return y


class PointwiseConvBlock(nn.Module):
    """
    Pointwise convolution across features/channels at each time step.

    Input:
        temporal_flat: [B, T, P]
    Output:
        point_out: [B, T, Z]
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
    One TPC layer combining temporal and pointwise convolutions.
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
        self.temporal_channels = temporal_channels
        self.pointwise_channels = pointwise_channels

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
            static: [B, S]
        Returns:
            [B, T, F + Z, temporal_channels + 1]
            This is a simplified approximation of the paper's concatenation logic.
        """
        bsz, seq_len, num_features, in_channels = x.shape

        temp_out = self.temporal(x)  # [B, T, F, Y]

        flat_x = x.reshape(bsz, seq_len, num_features * in_channels)  # [B, T, F*C_in]
        if static is not None:
            static_rep = static.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, S]
            point_in = torch.cat([flat_x, static_rep], dim=-1)
        else:
            point_in = flat_x

        point_out = self.pointwise(point_in)  # [B, T, Z]

        # Broadcast pointwise features across feature axis, then append as channels
        point_broadcast = point_out.unsqueeze(2).repeat(1, 1, num_features, 1)  # [B, T, F, Z]

        # Simplified fusion: keep original x as skip, plus temporal, plus pointwise broadcast
        fused = torch.cat([x, temp_out, point_broadcast], dim=-1)  # [B, T, F, C_in+Y+Z]
        fused = self.activation(fused)
        return fused


class TPC(nn.Module):
    """
    Minimal TPC implementation for hourly LoS regression.

    This starter version is intentionally simpler than the full paper:
    - no diagnosis encoder yet
    - no decay indicators yet
    - no multitask mortality head yet
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

        self.input_proj = nn.Linear(1, 1)  # placeholder for channelized input
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
            in_channels = in_channels + temporal_channels + pointwise_channels

        self.layers = nn.ModuleList(layers)
        self.final_fc1 = nn.Linear((input_dim * in_channels) + static_dim, fc_dim)
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
            static: [B, S]
        Returns:
            y_hat: [B]
        """
        x = time_series.unsqueeze(-1)  # [B, T, F, 1]

        for layer in self.layers:
            x = layer(x, static=static)

        # use last time step only
        last_x = x[:, -1, :, :].reshape(x.size(0), -1)  # [B, F*C]
        if static is not None:
            last_x = torch.cat([last_x, static], dim=-1)

        h = self.relu(self.final_fc1(last_x))
        y = self.softplus(self.final_fc2(h)).squeeze(-1)
        return y
