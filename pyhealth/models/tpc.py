"""Temporal Pointwise Convolution (TPC) model for ICU length-of-stay prediction.

Paper:
    Rocheteau, E., Liò, P., & Hyland, S. (2021). Temporal Pointwise
    Convolutional Networks for Length of Stay Prediction in the Intensive Care
    Unit. ACM CHIL 2021. https://doi.org/10.1145/3450439.3451860
    (arXiv:2007.09483)

Original code: https://github.com/EmmaRocheteau/TPC-LoS-prediction
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

# Prediction bounds from the paper (Appendix A): 30 minutes to 100 days.
MIN_LOS_DAYS = 1.0 / 48.0
MAX_LOS_DAYS = 100.0


def compute_decay_indicators(
    observed: torch.Tensor, base: float = 0.75
) -> torch.Tensor:
    """Computes decay indicators from an observation mask.

    The indicator is ``base ** j`` where ``j`` is the number of hours since the
    feature was last recorded (Section 4.1 of the paper). Features that have
    never been observed get an indicator of 0.

    Args:
        observed: Float or bool tensor of shape ``(batch, time, features)``
            that is 1 where the feature was actually measured in that hour,
            before forward-filling.
        base: Decay base. The paper uses 0.75.

    Returns:
        Tensor of shape ``(batch, time, features)`` with values in ``[0, 1]``.

    Examples:
        >>> obs = torch.tensor([[[1.0], [0.0], [0.0]]])
        >>> compute_decay_indicators(obs).flatten()
        tensor([1.0000, 0.7500, 0.5625])
    """
    observed = observed.float()
    batch, steps, feats = observed.shape
    out = torch.zeros_like(observed)
    since = torch.full((batch, feats), float("inf"), device=observed.device)
    for t in range(steps):
        since = torch.where(
            observed[:, t] > 0, torch.zeros_like(since), since + 1.0
        )
        out[:, t] = torch.where(
            torch.isinf(since), torch.zeros_like(since), base**since
        )
    return out


def msle_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = MIN_LOS_DAYS
) -> torch.Tensor:
    """Mean squared logarithmic error, ``mean((log y_hat - log y) ** 2)``.

    Args:
        y_pred: Predicted LoS in days, any shape.
        y_true: True LoS in days, same shape as ``y_pred``.
        eps: Lower clamp applied before taking logs for stability.

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(
        torch.log(y_pred.clamp_min(eps)), torch.log(y_true.clamp_min(eps))
    )


class TPCLayer(nn.Module):
    """One Temporal Pointwise Convolution layer (Eqs. 1-5 of the paper).

    The layer runs two branches in parallel:

    * a temporal branch: causal, dilated 1-D convolution applied to each
      feature with its own weights (implemented as a grouped convolution);
    * a pointwise branch: a 1x1 convolution that mixes all features at each
      time step.

    Skip connections re-attach the raw values and the earlier pointwise
    outputs, so the feature axis grows by ``pointwise_channels`` per layer.

    Args:
        layer_idx: 1-indexed position of this layer. It sets the dilation
            (``d = layer_idx``) and the input feature count.
        num_features: Number of raw time-series features ``F``.
        num_static: Number of static features ``S`` (0 for none).
        temporal_channels: Temporal output channels per feature ``Y``.
        pointwise_channels: Pointwise output features ``Z``.
        kernel_size: Temporal kernel size ``k``.
        temporal_dropout: Dropout after the temporal branch.
        main_dropout: Dropout after the pointwise branch.

    Examples:
        >>> layer = TPCLayer(1, num_features=4, num_static=2)
        >>> h = torch.randn(3, 4, 2, 10)  # (batch, features, channels, time)
        >>> x = torch.randn(3, 4, 10)
        >>> decay = torch.rand(3, 4, 10)
        >>> s = torch.randn(3, 2)
        >>> h_next, z = layer(h, x, decay, s, [])
        >>> h_next.shape  # features grow from 4 to 4 + Z, channels become Y+1
        torch.Size([3, 17, 13, 10])
    """

    def __init__(
        self,
        layer_idx: int,
        num_features: int,
        num_static: int = 0,
        temporal_channels: int = 12,
        pointwise_channels: int = 13,
        kernel_size: int = 4,
        temporal_dropout: float = 0.05,
        main_dropout: float = 0.45,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dilation = layer_idx
        self.kernel_size = kernel_size
        self.pad = self.dilation * (kernel_size - 1)
        self.Y = temporal_channels
        self.Z = pointwise_channels
        self.F = num_features
        self.S = num_static

        # R^n features, C^n channels entering this layer.
        self.R = num_features + pointwise_channels * (layer_idx - 1)
        self.C = 2 if layer_idx == 1 else temporal_channels + 1

        # Temporal branch: groups=R gives every feature its own filter.
        self.temporal = nn.Conv1d(
            self.R * self.C,
            self.R * self.Y,
            kernel_size,
            dilation=self.dilation,
            groups=self.R,
        )
        self.temporal_bn = nn.BatchNorm1d(self.R * self.Y)
        self.temporal_drop = nn.Dropout(temporal_dropout)

        # Pointwise branch input: flattened h, static, decay indicators.
        point_in = self.R * self.C + self.S + num_features
        self.pointwise = nn.Conv1d(point_in, pointwise_channels, 1)
        self.point_bn = nn.BatchNorm1d(pointwise_channels)
        self.point_drop = nn.Dropout(main_dropout)

    def forward(
        self,
        h: torch.Tensor,
        x_values: torch.Tensor,
        x_decay: torch.Tensor,
        static: Optional[torch.Tensor],
        prev_point: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies one TPC layer.

        Args:
            h: Layer input of shape ``(batch, R, C, time)``.
            x_values: Raw (forward-filled) values, ``(batch, F, time)``.
            x_decay: Decay indicators, ``(batch, F, time)``.
            static: Static features ``(batch, S)`` or None.
            prev_point: Pointwise outputs from earlier layers, each
                ``(batch, Z, time)``.

        Returns:
            A tuple ``(h_next, point_out)``. ``h_next`` has shape
            ``(batch, R + Z, Y + 1, time)`` and ``point_out`` has shape
            ``(batch, Z, time)``.
        """
        batch, _, _, steps = h.shape

        # Temporal branch with left padding so no future data leaks.
        flat = h.reshape(batch, self.R * self.C, steps)
        temp = self.temporal(F.pad(flat, (self.pad, 0)))
        temp = self.temporal_drop(self.temporal_bn(temp))
        temp = temp.view(batch, self.R, self.Y, steps)

        # Skip connections: raw values and earlier pointwise outputs.
        skip = torch.cat([x_values] + prev_point, dim=1).unsqueeze(2)
        r = torch.cat([temp, skip], dim=2)  # (batch, R, Y+1, time)

        # Pointwise branch.
        parts = [flat]
        if self.S > 0 and static is not None:
            parts.append(static.unsqueeze(-1).expand(-1, -1, steps))
        parts.append(x_decay)
        point = self.pointwise(torch.cat(parts, dim=1))
        point = self.point_drop(self.point_bn(point))

        broadcast = point.unsqueeze(2).expand(-1, -1, self.Y + 1, -1)
        h_next = F.relu(torch.cat([r, broadcast], dim=1))
        return h_next, point


class TPC(BaseModel):
    """Temporal Pointwise Convolution network for remaining LoS prediction.

    Implements the model of Rocheteau et al. (CHIL 2021). The network stacks
    :class:`TPCLayer` blocks, then applies two pointwise layers, an
    exponential and a HardTanh clip to produce a LoS in days.

    The wrapper expects dense, fixed-length hourly windows. ``forward``
    returns the prediction at the last time step so that it fits PyHealth's
    scalar ``"regression"`` label. Use :meth:`forward_sequence` to get the
    per-hour predictions the paper trains on.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` with a single
            ``"regression"`` label (remaining LoS in days).
        value_key: Input key holding forward-filled time series of shape
            ``(time, F)``.
        decay_key: Input key holding decay indicators ``(time, F)``. If None,
            zeros are used (the paper's "no decay" ablation).
        static_key: Optional input key with static features ``(S,)``.
        diag_key: Optional input key with multi-hot diagnoses ``(D,)``.
        temporal_channels: Temporal channels per feature ``Y``.
        pointwise_channels: Pointwise features per layer ``Z``.
        num_layers: Number of TPC layers ``N``.
        kernel_size: Temporal kernel size ``k``.
        temporal_dropout: Dropout in the temporal branch.
        main_dropout: Dropout in the pointwise parts of the model.
        diag_embedding_size: Size ``D*`` of the diagnosis embedding.
        final_fc_size: Width ``X`` of the first final pointwise layer.
        loss: ``"msle"`` (paper default) or ``"mse"``.

    Note:
        Defaults follow the paper's best eICU configuration. Smaller values
        are recommended for CPU runs and unit tests.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": f"p{i}", "visit_id": f"v{i}",
        ...      "ts": torch.rand(6, 3).tolist(),
        ...      "decay": torch.rand(6, 3).tolist(), "los": 2.0}
        ...     for i in range(4)
        ... ]
        >>> ds = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ts": "tensor", "decay": "tensor"},
        ...     output_schema={"los": "regression"},
        ...     dataset_name="tpc_doc",
        ... )
        >>> model = TPC(ds, value_key="ts", decay_key="decay",
        ...             num_layers=2, temporal_channels=4, pointwise_channels=4)
        >>> batch = next(iter(get_dataloader(ds, batch_size=2)))
        >>> out = model(**batch)
        >>> out["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        value_key: str = "time_series",
        decay_key: Optional[str] = "decay",
        static_key: Optional[str] = None,
        diag_key: Optional[str] = None,
        temporal_channels: int = 12,
        pointwise_channels: int = 13,
        num_layers: int = 9,
        kernel_size: int = 4,
        temporal_dropout: float = 0.05,
        main_dropout: float = 0.45,
        diag_embedding_size: int = 64,
        final_fc_size: int = 17,
        loss: str = "msle",
    ):
        super().__init__(dataset=dataset)
        assert len(self.label_keys) == 1, "TPC supports a single label key."
        if self.mode != "regression":
            raise ValueError(
                f"TPC needs a 'regression' label, got mode={self.mode!r}."
            )
        if loss not in ("msle", "mse"):
            raise ValueError("loss must be 'msle' or 'mse'.")
        for key in (value_key, decay_key, static_key, diag_key):
            if key is not None and key not in self.feature_keys:
                raise ValueError(f"'{key}' is not in the dataset input_schema.")

        self.label_key = self.label_keys[0]
        self.value_key = value_key
        self.decay_key = decay_key
        self.static_key = static_key
        self.diag_key = diag_key
        self.loss_type = loss

        procs = dataset.input_processors
        num_features = procs[value_key].size()
        num_static = procs[static_key].size() if static_key else 0
        num_diag = procs[diag_key].size() if diag_key else 0

        self.layers = nn.ModuleList(
            [
                TPCLayer(
                    layer_idx=n,
                    num_features=num_features,
                    num_static=num_static,
                    temporal_channels=temporal_channels,
                    pointwise_channels=pointwise_channels,
                    kernel_size=kernel_size,
                    temporal_dropout=temporal_dropout,
                    main_dropout=main_dropout,
                )
                for n in range(1, num_layers + 1)
            ]
        )

        # Diagnosis encoder (eICU only in the paper).
        self.diag_encoder = None
        diag_out = 0
        if diag_key:
            diag_out = diag_embedding_size
            self.diag_encoder = nn.Sequential(
                nn.Linear(num_diag, diag_out),
                nn.BatchNorm1d(diag_out),
                nn.Dropout(main_dropout),
                nn.ReLU(),
            )

        # Final two pointwise layers (Appendix A, Eq. 8).
        r_final = num_features + pointwise_channels * num_layers
        final_in = r_final * (temporal_channels + 1) + num_static + diag_out
        self.final1 = nn.Conv1d(final_in, final_fc_size, 1)
        self.final_bn = nn.BatchNorm1d(final_fc_size)
        self.final_drop = nn.Dropout(main_dropout)
        self.final2 = nn.Conv1d(final_fc_size, 1, 1)

    @staticmethod
    def _tensor(feature) -> torch.Tensor:
        """Extracts the value tensor from a processor output."""
        return feature[0] if isinstance(feature, (tuple, list)) else feature

    def forward_sequence(self, **kwargs) -> torch.Tensor:
        """Predicts remaining LoS at every hour.

        Args:
            **kwargs: Batch of features keyed like the dataset input schema.

        Returns:
            Tensor of shape ``(batch, time)`` with LoS in days, clipped to
            ``[1/48, 100]``.
        """
        x = self._tensor(kwargs[self.value_key]).float().to(self.device)
        x = x.permute(0, 2, 1)  # (batch, F, time)
        batch, _, steps = x.shape

        if self.decay_key:
            decay = self._tensor(kwargs[self.decay_key]).float().to(self.device)
            decay = decay.permute(0, 2, 1)
        else:
            decay = torch.zeros_like(x)

        static = None
        if self.static_key:
            static = self._tensor(kwargs[self.static_key]).float().to(self.device)

        # First layer input: values and decay as two channels per feature.
        h = torch.stack([x, decay], dim=2)  # (batch, F, 2, time)
        prev_point: List[torch.Tensor] = []
        for layer in self.layers:
            h, point = layer(h, x, decay, static, prev_point)
            prev_point.append(point)

        parts = [h.reshape(batch, -1, steps)]
        if static is not None:
            parts.append(static.unsqueeze(-1).expand(-1, -1, steps))
        if self.diag_encoder is not None:
            diag = self._tensor(kwargs[self.diag_key]).float().to(self.device)
            parts.append(self.diag_encoder(diag).unsqueeze(-1).expand(-1, -1, steps))

        z = self.final_drop(self.final_bn(self.final1(torch.cat(parts, dim=1))))
        z = self.final2(F.relu(z)).squeeze(1)  # (batch, time)
        return torch.clamp(torch.exp(z), MIN_LOS_DAYS, MAX_LOS_DAYS)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass using the prediction at the final time step.

        Args:
            **kwargs: Batch features and, optionally, the label key.

        Returns:
            Dictionary with ``logit`` and ``y_prob`` (both the predicted LoS
            in days, shape ``(batch, 1)``), plus ``loss`` and ``y_true`` when
            the label is present.
        """
        pred = self.forward_sequence(**kwargs)[:, -1:]  # (batch, 1)
        # Regression: the prediction itself is the output (no sigmoid/softmax).
        results = {"logit": pred, "y_prob": pred}
        if self.label_key in kwargs:
            y_true = self._tensor(kwargs[self.label_key]).float().to(self.device)
            y_true = y_true.view_as(pred)
            if self.loss_type == "msle":
                results["loss"] = msle_loss(pred, y_true)
            else:
                results["loss"] = F.mse_loss(pred, y_true)
            results["y_true"] = y_true
        return results