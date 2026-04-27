"""
Temporal Pointwise Convolution (TPC) model for hourly ICU remaining
length-of-stay (LoS) regression in PyHealth.

This module provides a dataset-backed PyHealth ``BaseModel`` implementation
of the Temporal Pointwise Convolution (TPC) architecture described in:

Rocheteau, E., Liò, P., and Hyland, S. (2021).
"Temporal Pointwise Convolutional Networks for Length of Stay Prediction
in the Intensive Care Unit."

Overview:
    TPC is designed for multivariate, irregularly sampled EHR time series.
    It combines two complementary operations at each layer:

    1. Temporal convolution:
       Feature-wise or shared causal convolutions over time, allowing
       each clinical variable to learn temporal dynamics.

    2. Pointwise convolution:
       Per-time-step feature mixing to capture cross-feature interactions
       without temporal leakage.

    In this PyHealth implementation, the task supplies:
        - ``time_series``: per-sample hourly history encoded as [T, 3F]
          in [value, mask, decay] order for each feature
        - ``static``: optional static feature vector
        - ``target_los_hours``: scalar regression target

    The model consumes task-processed batch fields via ``forward(**kwargs)``
    and returns the standard PyHealth output dictionary:
        - ``loss``
        - ``y_prob``
        - ``y_true``
        - ``logit``

Key Components:
    - ``TemporalConvBlock``: causal temporal convolution per feature
    - ``PointwiseConvBlock``: per-time-step feature interaction layer
    - ``TPCLayer``: combined temporal + pointwise block
    - ``TPC``: full stacked regression model

Implementation Notes:
    - The public model contract is now fully dataset-backed through
      ``BaseModel``.
    - The model predicts one scalar remaining LoS value per sample.
    - Nonlinearities use ``nn.Module`` variants to remain compatible with
      PyHealth interpretability expectations.
    - The input ``time_series`` field is internally split into value, mask,
      and decay channels from a [T, 3F] representation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


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
            shared_temporal: Whether to share one temporal convolution across
                all features.

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
                feat_y = self.shared_conv(feat_x)
            else:
                feat_y = self.feature_convs[feat_idx](feat_x)

            feat_y = feat_y.transpose(1, 2)  # [B, T, C_out]
            outputs.append(feat_y.unsqueeze(2))  # [B, T, 1, C_out]

        y = torch.cat(outputs, dim=2)  # [B, T, F, C_out]
        y = self.dropout(y)
        return y


class PointwiseConvBlock(nn.Module):
    """Pointwise transformation applied independently at each time step.

    This block is implemented as a linear layer operating on the flattened
    per-time-step representation. It is equivalent in spirit to a per-time-step
    1x1 convolution across the feature/channel dimension.

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
            shared_temporal: Whether temporal filters are shared across
                features.
            use_temporal: Whether to enable the temporal branch.
            use_pointwise: Whether to enable the pointwise branch.
            use_skip_connections: Whether to concatenate the prior input
                representation into the layer output.

        Raises:
            ValueError: If layer dimensions are invalid or both branches
                are disabled.
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
            raise ValueError(
                "At least one of use_temporal or use_pointwise must be True"
            )

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
            temp_out = self.temporal(x)
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

            point_in = torch.cat(point_parts, dim=-1)
            point_out = self.pointwise(point_in)
            point_broadcast = point_out.unsqueeze(2).expand(
                -1, -1, num_features, -1
            )
            parts_to_concat.append(point_broadcast)

        fused = torch.cat(parts_to_concat, dim=-1)
        fused = self.activation(fused)
        return fused


class TPC(BaseModel):
    """Temporal Pointwise Convolution model for scalar LoS regression.

    This is a true dataset-backed PyHealth ``BaseModel`` implementation.

    Expected task contract:
        - input field ``time_series`` containing per-sample history encoded as
          ``[T, 3F]`` with interleaved [value, mask, decay] channels
        - input field ``static`` containing optional static features
        - output field ``target_los_hours`` declared as ``"regression"``

    The model predicts one scalar remaining length-of-stay value for each
    sample using the full observed history in that sample.

    Args:
        dataset: PyHealth ``SampleDataset`` produced by ``dataset.set_task()``.
        input_dim: Number of base time-series features ``F`` before expansion
            into [value, mask, decay].
        static_dim: Static feature dimension.
        temporal_channels: Temporal branch output channels per layer.
        pointwise_channels: Pointwise branch output channels per layer.
        num_layers: Number of stacked TPC layers.
        kernel_size: Temporal convolution kernel size.
        fc_dim: Hidden dimension in the final regression head.
        dropout: Dropout probability.
        loss_name: Loss function name. Use ``"msle"`` for mean squared
            logarithmic error or ``"mse"`` for mean squared error.
        use_decay_in_pointwise: Whether to inject decay channels into the
            pointwise branch.
        positive_output: Whether to enforce non-negative predictions with
            ``Softplus``.
        shared_temporal: Whether temporal filters are shared across features.
        use_temporal: Whether to enable the temporal branch.
        use_pointwise: Whether to enable the pointwise branch.
        use_skip_connections: Whether to use concatenative skip connections.

    Notes:
        - This model follows the standard PyHealth ``forward(**kwargs)``
          contract.
        - The output head size is derived from
          ``BaseModel.get_output_size()``.
    """

    def __init__(
        self,
        dataset,
        input_dim: int,
        static_dim: int = 0,
        temporal_channels: int = 8,
        pointwise_channels: int = 8,
        num_layers: int = 3,
        kernel_size: int = 3,
        fc_dim: int = 32,
        dropout: float = 0.1,
        loss_name: str = "msle",
        use_decay_in_pointwise: bool = True,
        positive_output: bool = True,
        shared_temporal: bool = False,
        use_temporal: bool = True,
        use_pointwise: bool = True,
        use_skip_connections: bool = True,
    ) -> None:
        """Initialize the TPC model."""
        super().__init__(dataset=dataset)

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
        if loss_name not in {"msle", "mse"}:
            raise ValueError("loss_name must be one of {'msle', 'mse'}")

        self.label_key = self.label_keys[0]

        required_feature_keys = {"time_series", "static"}
        missing = required_feature_keys.difference(set(self.feature_keys))
        if missing:
            raise ValueError(
                "TPC requires task input_schema to contain the feature keys "
                f"{sorted(required_feature_keys)}; missing {sorted(missing)}"
            )

        self.input_dim = input_dim
        self.static_dim = static_dim
        self.temporal_channels = temporal_channels
        self.pointwise_channels = pointwise_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.loss_name = loss_name
        self.use_decay_in_pointwise = use_decay_in_pointwise
        self.positive_output = positive_output
        self.shared_temporal = shared_temporal
        self.use_temporal = use_temporal
        self.use_pointwise = use_pointwise
        self.use_skip_connections = use_skip_connections

        layers = []
        in_channels = 2  # value + decay

        for layer_idx in range(num_layers):
            layer = TPCLayer(
                num_features=input_dim,
                in_channels=in_channels,
                temporal_channels=temporal_channels,
                pointwise_channels=pointwise_channels,
                static_dim=static_dim,
                kernel_size=kernel_size,
                dilation=layer_idx + 1,
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
        self.final_fc2 = nn.Linear(fc_dim, self.get_output_size())

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def _unpack_feature_value(self, key: str, feature: Any) -> torch.Tensor:
        """Extract the processor 'value' tensor for a feature key.

        PyHealth passes either:
            - a raw tensor, or
            - a tuple aligned with the processor schema

        Args:
            key: Feature key in the task schema.
            feature: Batch feature value from ``kwargs``.

        Returns:
            The tensor corresponding to the processor's ``value`` field.

        Raises:
            ValueError: If the processor schema does not contain ``value``.
        """
        if isinstance(feature, torch.Tensor):
            return feature

        if not isinstance(feature, (tuple, list)):
            raise ValueError(
                f"Expected feature '{key}' to be a Tensor or tuple/list, "
                f"got {type(feature).__name__}"
            )

        schema = self.dataset.input_processors[key].schema()
        if "value" not in schema:
            raise ValueError(
                f"Processor schema for feature '{key}' does not contain 'value': "
                f"{schema}"
            )
        return feature[schema.index("value")]

    def _split_value_mask_decay(
        self,
        time_series: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a batch time series tensor into values, masks, and decay.

        Supported input layouts:
            - ``[B, T, 3F]`` with interleaved [value, mask, decay]
            - ``[B, T, F, 3]`` with final channel order [value, mask, decay]

        Args:
            time_series: Batch time series tensor.

        Returns:
            Tuple ``(values, masks, decay)`` each with shape ``[B, T, F]``.

        Raises:
            ValueError: If the tensor shape is incompatible.
        """
        if time_series.ndim == 3:
            batch_size, seq_len, feat_dim = time_series.shape
            if feat_dim % 3 != 0:
                raise ValueError(
                    "Expected time_series last dimension divisible by 3 for "
                    f"[value, mask, decay], got {feat_dim}"
                )
            num_features = feat_dim // 3
            if num_features != self.input_dim:
                raise ValueError(
                    f"Expected input_dim={self.input_dim}, got {num_features}"
                )

            values = []
            masks = []
            decay = []
            for feature_idx in range(num_features):
                base = feature_idx * 3
                values.append(time_series[:, :, base].unsqueeze(-1))
                masks.append(time_series[:, :, base + 1].unsqueeze(-1))
                decay.append(time_series[:, :, base + 2].unsqueeze(-1))

            return (
                torch.cat(values, dim=-1),
                torch.cat(masks, dim=-1),
                torch.cat(decay, dim=-1),
            )

        if time_series.ndim == 4:
            batch_size, seq_len, num_features, channels = time_series.shape
            if channels != 3:
                raise ValueError(
                    "Expected time_series channel dimension size 3 for "
                    f"[value, mask, decay], got {channels}"
                )
            if num_features != self.input_dim:
                raise ValueError(
                    f"Expected input_dim={self.input_dim}, got {num_features}"
                )
            values = time_series[:, :, :, 0]
            masks = time_series[:, :, :, 1]
            decay = time_series[:, :, :, 2]
            return values, masks, decay

        raise ValueError(
            "Expected time_series to have shape [B, T, 3F] or [B, T, F, 3], "
            f"got {tuple(time_series.shape)}"
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the TPC model forward under the PyHealth BaseModel contract.

        Expected kwargs:
            - ``time_series``: tensor or processor tuple containing the
              [value, mask, decay] history representation
            - ``static``: static feature tensor or processor tuple
            - ``target_los_hours``: regression targets

        Returns:
            Dictionary containing:
                - ``loss``: scalar regression loss
                - ``y_prob``: prepared output probabilities/identity values
                - ``y_true``: ground-truth labels
                - ``logit``: raw model outputs

        Raises:
            ValueError: If required inputs are missing or malformed.
        """
        if "time_series" not in kwargs:
            raise ValueError("Missing required batch field 'time_series'")
        if "static" not in kwargs:
            raise ValueError("Missing required batch field 'static'")
        if self.label_key not in kwargs:
            raise ValueError(
                f"Missing required label field '{self.label_key}' in batch"
            )

        time_series = self._unpack_feature_value("time_series", kwargs["time_series"])
        static = self._unpack_feature_value("static", kwargs["static"])

        if not isinstance(time_series, torch.Tensor):
            raise ValueError("'time_series' value must be a Tensor after unpacking")
        if not isinstance(static, torch.Tensor):
            raise ValueError("'static' value must be a Tensor after unpacking")

        time_series = time_series.float().to(self.device)
        static = static.float().to(self.device)

        if time_series.ndim not in {3, 4}:
            raise ValueError(
                "Expected time_series batch tensor to have 3 or 4 dimensions, "
                f"got {tuple(time_series.shape)}"
            )

        if static.ndim != 2:
            raise ValueError(
                f"Expected static to have shape [B, S], got {tuple(static.shape)}"
            )
        if static.shape[1] != self.static_dim:
            raise ValueError(
                f"Expected static_dim={self.static_dim}, got {static.shape[1]}"
            )

        x_values, _, x_decay = self._split_value_mask_decay(time_series)

        x = torch.stack([x_values, x_decay], dim=-1)  # [B, T, F, 2]

        for layer in self.layers:
            x = layer(x, decay=x_decay, static=static)

        batch_size, seq_len, _, _ = x.shape
        last_x = x[:, -1, :, :].reshape(batch_size, -1)

        if static is not None:
            last_x = torch.cat([last_x, static], dim=-1)

        hidden = self.relu(self.final_fc1(last_x))
        logits = self.final_fc2(hidden)

        if self.positive_output:
            logits = self.softplus(logits)

        y_true = kwargs[self.label_key].float().to(self.device)
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        elif y_true.ndim > 2:
            raise ValueError(
                f"Expected y_true to have shape [B] or [B, 1], got {tuple(y_true.shape)}"
            )

        if self.loss_name == "msle":
            loss = F.mse_loss(torch.log1p(logits), torch.log1p(y_true))
        elif self.loss_name == "mse":
            loss = F.mse_loss(logits, y_true)
        else:
            raise ValueError(
                f"Unsupported loss_name '{self.loss_name}'. Expected 'msle' or 'mse'."
            )

        return {
            "loss": loss,
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }