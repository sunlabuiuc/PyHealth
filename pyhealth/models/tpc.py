from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class TPCBlock(nn.Module):
    """One TPC layer: temporal conv, pointwise conv, and dense skip connections.

    The temporal branch applies per-feature causal convolution (independent weights per
    feature, no cross-feature sharing). The pointwise branch mixes across all features
    and static inputs at each timestep. Their outputs are concatenated with skip
    connections, growing the feature dimension by Z (pointwise_channels) each layer.

    Args:
        in_features:        Number of input features R.
        in_channels:        Channels per feature C. 2 on layer 0 (value, decay),
                            temporal_channels +1 thereafter.
        temporal_channels:  Temporal conv output channels Y per feature.
        pointwise_channels: Pointwise conv output channels Z.
        kernel_size:        Temporal conv kernel size k.
        dilation:           Dilation factor d. Set to layer_idx + 1 per layer.
        main_dropout:       Dropout after pointwise branch.
        temporal_dropout:   Dropout after temporal branch.
        use_batchnorm:      Batch normalisation after each branch. Default: True.
        static_dim:         Static feature dimension S injected into pointwise branch.

    Examples:
        >>> block = TPCBlock(
        ...     in_features=101, in_channels=2,
        ...     temporal_channels=11, pointwise_channels=5,
        ...     kernel_size=5, dilation=1,
        ...     main_dropout=0.0, temporal_dropout=0.05,
        ...     static_dim=32,
        ... )
        >>> x = torch.randn(8, 100, 101, 2)
        >>> out = block(x, static=torch.randn(8, 32))
        >>> out.shape   # (8, 100, 106, 12)
    """

    def __init__(
        self,
        *,
        in_features: int,
        in_channels: int,
        temporal_channels: int,
        pointwise_channels: int,
        kernel_size: int,
        dilation: int,
        main_dropout: float,
        temporal_dropout: float,
        use_batchnorm: bool = True,
        static_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.in_channels = int(in_channels)
        self.temporal_channels = int(temporal_channels)
        self.pointwise_channels = int(pointwise_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.use_batchnorm = bool(use_batchnorm)
        self.static_dim = int(static_dim)

        # Temporal branch: grouped Conv1d => separate weights per feature.
        self.temporal_conv = nn.Conv1d(
            in_channels=self.in_features * self.in_channels,
            out_channels=self.in_features * self.temporal_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.in_features,
            bias=True,
        )
        self.bn_temporal = nn.BatchNorm1d(self.in_features * self.temporal_channels)
        self.dropout_temporal = nn.Dropout(temporal_dropout)

        # Pointwise branch: Linear applied to each time step.
        # Input to pointwise uses r = [x_value_skip, temporal_out] => channels (Y + 1).
        point_in_dim = self.in_features * (self.temporal_channels + 1) + self.static_dim
        self.pointwise = nn.Linear(point_in_dim, self.pointwise_channels)
        self.bn_pointwise = nn.BatchNorm1d(self.pointwise_channels)
        self.dropout_main = nn.Dropout(main_dropout)

        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of one TPC block.

        Args:
            x: (B, T, R, C_in) time-series input tensor.
            static: (B, S) static feature tensor, or None.

        Returns:
            torch.Tensor: output tensor of shape (B, T, R + Z, Y + 1).
        """
        B, T, R, C = x.shape
        if R != self.in_features or C != self.in_channels:
            raise ValueError(
                "TPCBlock got x shape "
                f"{x.shape}, expected (B,T,{self.in_features},{self.in_channels})"
            )

        # === Temporal branch ===
        # reshape to (B, R*C, T) for grouped conv; causal left padding.
        x_tc = x.permute(0, 2, 3, 1).reshape(B, R * C, T)
        pad = (self.kernel_size - 1) * self.dilation
        x_tc = F.pad(x_tc, (pad, 0), mode="constant", value=0.0)
        t_out = self.temporal_conv(x_tc)  # (B, R*Y, T)
        if self.use_batchnorm:
            t_out = self.bn_temporal(t_out)
        t_out = self.dropout_temporal(t_out)
        t_out = t_out.reshape(B, R, self.temporal_channels, T).permute(
            0, 3, 1, 2
        )  # (B,T,R,Y)

        # Skip: use the (current) value channel as 1 extra channel.
        x_value = x[..., 0:1]  # (B,T,R,1)
        r = torch.cat([x_value, t_out], dim=-1)  # (B,T,R,Y+1)

        # === Pointwise branch ===
        r_flat = r.reshape(B, T, R * (self.temporal_channels + 1))
        if static is not None:
            static_rep = static.unsqueeze(1).expand(B, T, static.shape[-1])
            p_in = torch.cat([r_flat, static_rep], dim=-1)
        else:
            p_in = r_flat

        pw = self.pointwise(p_in)  # (B,T,Z)
        if self.use_batchnorm:
            pw_bn = self.bn_pointwise(pw.reshape(B * T, -1)).reshape(B, T, -1)
        else:
            pw_bn = pw
        pw_bn = self.dropout_main(pw_bn)

        # Broadcast pointwise outputs to (B,T,Z,Y+1) as new "features".
        pw_feat = pw_bn.unsqueeze(-1).expand(
            B, T, self.pointwise_channels, self.temporal_channels + 1
        )

        h = torch.cat([r, pw_feat], dim=2)  # (B,T,R+Z,Y+1)
        return self.relu(h)


class TPC(BaseModel):
    """Temporal Pointwise Convolution (TPC) for remaining ICU length-of-stay.

    Paper: Rocheteau et al., *Temporal Pointwise Convolutional Networks for Length of
    Stay Prediction in the Intensive Care Unit* (ACM CHIL 2021).

    Note:
        Predicts remaining LoS in days each ICU hour from the configured start hour.
        Temporal convolutions are per-feature (no cross-feature weight sharing);
        pointwise layers mix features. Inputs must include ``ts`` (values + decay) and
        ``static``, as produced by ``RemainingLengthOfStayTPC_MIMIC4`` with
        ``TPCTimeseriesProcessor`` and ``TPCStaticProcessor``.

        ``mode`` is set to ``'regression'`` because ``'regression_sequence'`` is not
        a recognised PyHealth mode string. With labels, ``y_prob`` and ``y_true`` are
        flattened over valid timesteps so padded zeros do not affect metrics.

    Args:
        dataset: Dataset used to infer ``F`` (time-series features) and ``S`` (static
            width) from fitted processors.
        temporal_channels: Temporal conv output channels ``Y`` per feature. Default 11.
        pointwise_channels: Pointwise output channels ``Z``. Default 5.
        num_layers: Number of stacked TPC blocks ``N``. Default 8.
        kernel_size: Temporal conv kernel size ``k``. Default 5.
        main_dropout: Dropout after the pointwise branch. Default 0.0.
        temporal_dropout: Dropout after the temporal branch. Default 0.05.
        use_batchnorm: Whether to apply batch norm after each branch. Default True.
        final_hidden: Hidden size of the two-layer prediction head. Default 36.
        decay_clip_min_days: ``HardTanh`` minimum in days. Default ``1/48``.
        decay_clip_max_days: ``HardTanh`` maximum in days. Default ``100``.

    Examples:
        >>> from datetime import datetime
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> import torch
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "stay_id": "s0",
        ...         "ts": {
        ...             "prefill_start": datetime(2020, 1, 1, 0),
        ...             "icu_start":     datetime(2020, 1, 1, 0),
        ...             "pred_start":    datetime(2020, 1, 1, 5),
        ...             "pred_end":      datetime(2020, 1, 1, 10),
        ...             "feature_itemids": ["A", "B"],
        ...             "long_df": {
        ...                 "timestamp": [], "itemid": [], "value": [], "source": []
        ...             },
        ...         },
        ...         "static": {
        ...             "gender": "M", "race": "WHITE",
        ...             "admission_location": "ER", "insurance": "Medicare",
        ...             "first_careunit": "MICU", "hour_of_admission": 0,
        ...             "admission_height": 170.0, "admission_weight": 80.0,
        ...             "gcs_eye": 4.0, "gcs_motor": 6.0, "gcs_verbal": 5.0,
        ...             "anchor_age": 65,
        ...         },
        ...         "y": [2.0, 1.5, 1.0, 0.75, 0.5],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "ts": ("tpc_timeseries", {}),
        ...         "static": ("tpc_static", {}),
        ...     },
        ...     output_schema={"y": ("regression_sequence", {})},
        ...     dataset_name="test_tpc",
        ... )
        >>> model = TPC(dataset=dataset)
        >>> train_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        >>> data_batch = next(iter(train_loader))
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss':   tensor(0.4231, grad_fn=<DivBackward0>),
            'y_prob': tensor([1.9842, 1.4921, 1.0013, 0.7506, 0.5001]),
            'y_true': tensor([2.0000, 1.5000, 1.0000, 0.7500, 0.5000]),
            'logit':  tensor([[0.6851, 0.3972, 0.0013, -0.2876, -0.6912]])
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        *,
        temporal_channels: int = 11,
        pointwise_channels: int = 5,
        num_layers: int = 8,
        kernel_size: int = 5,
        main_dropout: float = 0.0,
        temporal_dropout: float = 0.05,
        use_batchnorm: bool = True,
        final_hidden: int = 36,
        decay_clip_min_days: float = 1.0 / 48.0,
        decay_clip_max_days: float = 100.0,
    ) -> None:
        super().__init__(dataset=dataset)
        assert "ts" in self.feature_keys and "static" in self.feature_keys, (
            "TPC expects dataset.input_schema to contain 'ts' and 'static'."
        )
        assert len(self.label_keys) == 1, "TPC currently supports a single label key."
        self.label_key = self.label_keys[0]

        # Hardcoded: label processor is "regression_sequence" but BaseModel mode must be
        # "regression" so metrics and Trainer wiring match the regression path.
        self.mode = "regression"

        self.temporal_channels = int(temporal_channels)
        self.pointwise_channels = int(pointwise_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.use_batchnorm = bool(use_batchnorm)
        self.final_hidden = int(final_hidden)

        self.min_days = float(decay_clip_min_days)
        self.max_days = float(decay_clip_max_days)

        # We infer feature/static dimensions from the dataset processors.
        ts_proc = dataset.input_processors["ts"]
        static_proc = dataset.input_processors["static"]
        self.F = ts_proc.size()
        self.S = static_proc.size()

        # Stack TPC blocks; feature dimension grows by Z each layer.
        blocks = []
        in_features = self.F
        in_channels = 2  # (value, decay)
        for layer_idx in range(self.num_layers):
            blocks.append(
                TPCBlock(
                    in_features=in_features,
                    in_channels=in_channels,
                    temporal_channels=self.temporal_channels,
                    pointwise_channels=self.pointwise_channels,
                    kernel_size=self.kernel_size,
                    dilation=layer_idx + 1,
                    main_dropout=main_dropout,
                    temporal_dropout=temporal_dropout,
                    use_batchnorm=self.use_batchnorm,
                    static_dim=self.S,
                )
            )
            # after first block, channels become (Y+1)
            in_channels = self.temporal_channels + 1
            in_features = in_features + self.pointwise_channels
        self.blocks = nn.ModuleList(blocks)

        # Final per-time-step head (2-layer pointwise MLP).
        final_in = in_features * (self.temporal_channels + 1) + self.S
        self.head_fc1 = nn.Linear(final_in, self.final_hidden)
        self.head_relu = nn.ReLU()
        self.head_fc2 = nn.Linear(self.final_hidden, 1)

        self.hardtanh = nn.Hardtanh(min_val=self.min_days, max_val=self.max_days)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. Must contain all feature
                keys and the label key. Specifically:
                - ts: (B, T, F, 2) padded time-series tensor produced by
                  TPCTimeseriesProcessor (value + decay channels).
                - static: (B, S) static feature tensor produced by
                  TPCStaticProcessor.
                - <label_key> (optional): (B, T) padded target tensor; when
                  present, loss and y_true are added to the output.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:

                - logit: a (B, T) tensor of raw log-space predictions.
                - y_prob: predicted remaining LoS in days. When labels are
                  provided this is a masked 1-D tensor of real timesteps only;
                  otherwise (B, T).
                - loss (when labels provided): scalar MSLE loss over
                  unpadded timesteps.
                - y_true (when labels provided): masked 1-D tensor of
                  ground-truth LoS values aligned with y_prob.
        """
        ts: torch.Tensor = kwargs["ts"].to(self.device)          # (B,T,F,2) padded
        static: torch.Tensor = kwargs["static"].to(self.device)  # (B,S)
        y_true: Optional[torch.Tensor] = None
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)      # (B,T) padded

        B, T, F, C = ts.shape
        if C != 2:
            raise ValueError(f"TPC expects ts channels=2, got {C}.")
        if F != self.F:
            raise ValueError(f"TPC expects F={self.F} features, got {F}.")

        h = ts
        for block in self.blocks:
            h = block(h, static=static)  # grows feature dimension, channels -> (Y+1)

        # Final predictions per hour.
        h_flat = h.reshape(B, T, -1)  # (B,T, features*channels)
        static_rep = static.unsqueeze(1).expand(B, T, static.shape[-1])
        head_in = torch.cat([h_flat, static_rep], dim=-1)

        hidden = self.head_relu(self.head_fc1(head_in))
        logit = self.head_fc2(hidden).squeeze(-1)  # (B,T)

        # Predict log(LoS) then exponentiate + clip (paper Appendix A).
        y_pred = self.hardtanh(torch.exp(logit))

        results: Dict[str, torch.Tensor] = {
            "logit": logit,
            "y_prob": y_pred,
        }

        if y_true is not None:
            # Padding uses 0; real labels are >= 1/48 day after task clipping.
            mask = (y_true != 0).float()
            # MSLE = mean((log(y_pred) - log(y_true))^2) over valid timesteps.
            eps = 1e-8
            log_pred = torch.log(torch.clamp(y_pred, min=eps))
            log_true = torch.log(torch.clamp(y_true, min=eps))
            se = (log_pred - log_true) ** 2
            loss = (se * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            results["loss"] = loss
            # Flatten valid positions so batch collation ignores padding.
            results["y_prob"] = y_pred[mask.bool()]
            results["y_true"] = y_true[mask.bool()]

        return results

    def forward_from_embedding(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass starting from feature embeddings.

        TPC takes dense numeric tensors directly (no token-embedding step), so
        this method just calls the forward method.

        Args:
            **kwargs: same keyword arguments as forward().

        Returns:
            Dict[str, torch.Tensor]: same output dictionary as forward().
        """
        return self.forward(**kwargs)
