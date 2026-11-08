"""Temporal Pointwise Convolution (TPC) model for ICU length-of-stay prediction.

Paper:
    Rocheteau, E., Liò, P., & Hyland, S. (2021). Temporal Pointwise
    Convolutional Networks for Length of Stay Prediction in the Intensive Care
    Unit. ACM CHIL 2021. https://doi.org/10.1145/3450439.3451860
    (arXiv:2007.09483)

Original code: https://github.com/EmmaRocheteau/TPC-LoS-prediction
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

# Prediction bounds from the paper (Appendix A): 30 minutes to 100 days.
MIN_LOS_DAYS = 1.0 / 48.0
MAX_LOS_DAYS = 100.0
TensorInput = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]


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
    if not 0.0 < base <= 1.0:
        raise ValueError("base must be in (0, 1].")
    if observed.ndim != 3:
        raise ValueError("observed must have shape (batch, time, features).")
    observed = observed.float()
    batch, steps, feats = observed.shape
    out = torch.zeros_like(observed)
    since = torch.full((batch, feats), float("inf"), device=observed.device)
    for t in range(steps):
        since = torch.where(observed[:, t] > 0, torch.zeros_like(since), since + 1.0)
        out[:, t] = torch.where(
            torch.isinf(since), torch.zeros_like(since), base**since
        )
    return out


def prepare_time_series(
    values: torch.Tensor,
    observed: torch.Tensor,
    decay_base: float = 0.75,
    lab_indices: Sequence[int] | None = None,
    forward_fill_labs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepares hourly values and decay without reading future measurements.

    Unobserved values before the first measurement are zero. Vitals are
    forward-filled; labs are either forward-filled or zero at missing hours.
    Missing entries in ``values`` may contain NaNs; observed entries must be finite.

    Args:
        values: Raw values of shape ``(batch, time, features)``.
        observed: Measurement mask with the same shape, before any filling.
        decay_base: Base in ``(0, 1]`` for the decay indicators.
        lab_indices: Lab feature indices; None treats every feature as a lab.
        forward_fill_labs: Whether to forward-fill missing lab measurements.

    Returns:
        Filled values and decay indicators, each with the input shape.

    Raises:
        ValueError: If shapes, lab indices, or observed values are invalid.

    Examples:
        >>> values = torch.tensor([[[2.0], [float("nan")]]])
        >>> filled, decay = prepare_time_series(values, torch.isfinite(values))
        >>> filled.flatten()
        tensor([2., 2.])
        >>> decay.flatten()
        tensor([1.0000, 0.7500])
    """
    if values.ndim != 3 or values.shape != observed.shape:
        raise ValueError("values and observed must share (batch, time, features).")
    mask = observed.to(device=values.device, dtype=torch.bool)
    if not torch.isfinite(values[mask]).all():
        raise ValueError("Observed values must be finite.")
    labs = list(range(values.shape[-1])) if lab_indices is None else list(lab_indices)
    if any(not isinstance(i, int) or not 0 <= i < values.shape[-1] for i in labs):
        raise ValueError("lab_indices must contain valid feature indices.")
    fill = torch.ones(values.shape[-1], dtype=torch.bool, device=values.device)
    if not forward_fill_labs:
        fill[labs] = False
    previous = (
        torch.zeros_like(values[:, 0])
        if values.shape[1]
        else values.new_zeros((values.shape[0], values.shape[2]))
    )
    filled = torch.zeros_like(values)
    for t in range(values.shape[1]):
        previous = torch.where(mask[:, t], values[:, t], previous)
        filled[:, t] = torch.where(mask[:, t] | fill, previous, 0.0)
    return filled, compute_decay_indicators(mask, base=decay_base)


def _log_residual(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float
) -> torch.Tensor:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if y_pred.shape != y_true.shape:
        raise ValueError("Predictions and targets must have identical shapes.")
    return torch.log(y_pred.clamp_min(eps)) - torch.log(y_true.clamp_min(eps))


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

    Examples:
        >>> y = torch.tensor([1.0, 2.0])
        >>> msle_loss(y, y).item()
        0.0
    """
    return _log_residual(y_pred, y_true, eps).square().mean()


def asymmetric_msle_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    underprediction_weight: float = 2.0,
    eps: float = MIN_LOS_DAYS,
) -> torch.Tensor:
    """Computes MSLE with extra weight on underpredictions.

    Args:
        y_pred: Predicted remaining LoS in days.
        y_true: True remaining LoS, with the same shape.
        underprediction_weight: Finite weight at least 1 for negative log errors.
        eps: Positive lower bound before taking logs.

    Returns:
        Scalar weighted log-squared loss. Weight 1 recovers ordinary MSLE.

    Examples:
        >>> target = torch.tensor([2.0])
        >>> under = asymmetric_msle_loss(torch.tensor([1.0]), target)
        >>> over = asymmetric_msle_loss(torch.tensor([4.0]), target)
        >>> bool(torch.isclose(under, 2 * over))
        True
    """
    if not math.isfinite(underprediction_weight) or underprediction_weight < 1:
        raise ValueError("underprediction_weight must be finite and at least 1.")
    residual = _log_residual(y_pred, y_true, eps)
    weight = torch.where(residual < 0, underprediction_weight, 1.0)
    return (weight * residual.square()).mean()


def huber_log_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float = 1.0,
    eps: float = MIN_LOS_DAYS,
) -> torch.Tensor:
    """Computes Huber loss on log LoS, limiting extreme log-error influence.

    Args:
        y_pred: Predicted remaining LoS in days.
        y_true: True remaining LoS, with the same shape.
        delta: Finite positive quadratic-to-linear transition in log units.
        eps: Positive lower bound before taking logs.

    Returns:
        Scalar Huber log-loss; small errors have half the MSLE scale.

    Examples:
        >>> y = torch.tensor([1.0, 2.0])
        >>> huber_log_loss(y, y).item()
        0.0
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive.")
    residual = _log_residual(y_pred, y_true, eps)
    return F.huber_loss(residual, torch.zeros_like(residual), delta=delta)


class TPCLayer(nn.Module):
    """One Temporal Pointwise Convolution layer (Eqs. 1-5 of the paper).

    The layer runs two branches in parallel:

    * a temporal branch: causal, dilated 1-D convolution applied to each
      feature with its own weights (implemented as a grouped convolution);
    * a pointwise branch: a 1x1 convolution that mixes all features at each
      time step.

    Skip connections re-attach the raw values and the earlier pointwise
    outputs, so the feature axis grows by ``pointwise_channels`` per layer.

    As in the official code, the pointwise branch reads the previous layer's
    temporal output, the latest pointwise output, the raw values and decay
    indicators, optional time features, and the static features.

    Args:
        layer_idx: 1-indexed position of this layer. It sets the number of
            input features.
        num_features: Number of raw time-series features ``F``.
        num_static: Number of static features ``S`` (0 for none).
        num_time: Number of per-hour time features (e.g. time in ICU, hour of
            day) that only feed the pointwise branch (0 for none).
        temporal_channels: Temporal output channels per feature ``Y``.
        pointwise_channels: Pointwise output features ``Z``.
        kernel_size: Temporal kernel size ``k``.
        dilation: Temporal dilation ``d``. Left padding is ``d * (k - 1)``.
        temporal_dropout: Dropout after the temporal branch.
        main_dropout: Dropout after the pointwise branch.

    Examples:
        >>> layer = TPCLayer(1, num_features=4, num_static=2)
        >>> h = torch.randn(3, 4, 2, 10)  # (batch, features, channels, time)
        >>> x = torch.randn(3, 4, 10)
        >>> decay = torch.rand(3, 4, 10)
        >>> s = torch.randn(3, 2)
        >>> h_next, z, t = layer(h, x, decay, s, None, [], None)
        >>> h_next.shape  # features grow from 4 to 4 + Z, channels become Y+1
        torch.Size([3, 17, 13, 10])
    """

    def __init__(
        self,
        layer_idx: int,
        num_features: int,
        num_static: int = 0,
        num_time: int = 0,
        temporal_channels: int = 12,
        pointwise_channels: int = 13,
        kernel_size: int = 4,
        dilation: int | None = None,
        temporal_dropout: float = 0.05,
        main_dropout: float = 0.45,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.dilation = layer_idx if dilation is None else dilation
        self.kernel_size = kernel_size
        self.pad = self.dilation * (kernel_size - 1)
        self.Y = temporal_channels
        self.Z = pointwise_channels
        self.F = num_features
        self.S = num_static
        self.n_time = num_time

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

        # Pointwise input: previous temporal output, latest pointwise output,
        # raw values + decay (2F), time features and static features.
        point_in = 2 * num_features + num_time + num_static
        if layer_idx > 1:
            point_in += (self.R - self.Z) * self.Y + self.Z
        self.pointwise = nn.Conv1d(point_in, pointwise_channels, 1)
        self.point_bn = nn.BatchNorm1d(pointwise_channels)
        self.point_drop = nn.Dropout(main_dropout)

    def forward(
        self,
        h: torch.Tensor,
        x_values: torch.Tensor,
        x_decay: torch.Tensor,
        static: torch.Tensor | None,
        time_feats: torch.Tensor | None,
        prev_point: list[torch.Tensor],
        prev_temp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies one TPC layer.

        Args:
            h: Layer input of shape ``(batch, R, C, time)``.
            x_values: Raw (forward-filled) values, ``(batch, F, time)``.
            x_decay: Decay indicators, ``(batch, F, time)``.
            static: Static features ``(batch, S)`` or None.
            time_feats: Per-hour time features ``(batch, n_time, time)`` or
                None.
            prev_point: Pointwise outputs from earlier layers, each
                ``(batch, Z, time)``.
            prev_temp: Previous layer's temporal output
                ``(batch, R_prev * Y, time)``, or None for the first layer.

        Returns:
            A tuple ``(h_next, point_out, temp_out)``. ``h_next`` has shape
            ``(batch, R + Z, Y + 1, time)``, ``point_out`` is
            ``(batch, Z, time)`` and ``temp_out`` is ``(batch, R * Y, time)``.
        """
        batch, _, _, steps = h.shape

        # Temporal branch with left padding so no future data leaks.
        flat = h.reshape(batch, self.R * self.C, steps)
        temp = self.temporal(F.pad(flat, (self.pad, 0)))
        temp = self.temporal_drop(self.temporal_bn(temp))
        temp_out = temp
        temp = temp.view(batch, self.R, self.Y, steps)

        # Skip connections: raw values and earlier pointwise outputs. The skip
        # channel goes first, as in the official code.
        skip = torch.cat([x_values] + prev_point, dim=1).unsqueeze(2)
        r = torch.cat([skip, temp], dim=2)  # (batch, R, Y+1, time)

        # Pointwise branch.
        parts = [x_values, x_decay]
        if prev_temp is not None:
            parts.append(prev_temp)
        if prev_point:
            parts.append(prev_point[-1])
        if time_feats is not None and self.n_time > 0:
            parts.append(time_feats)
        if static is not None and self.S > 0:
            parts.append(static.unsqueeze(-1).expand(-1, -1, steps))
        point = self.pointwise(torch.cat(parts, dim=1))
        point = self.point_drop(self.point_bn(point))

        broadcast = point.unsqueeze(2).expand(-1, -1, self.Y + 1, -1)
        h_next = F.relu(torch.cat([r, broadcast], dim=1))
        return h_next, point, temp_out


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
        time_key: Optional input key with per-hour time features
            ``(time, n_time)`` such as time in ICU and hour of day. They only
            feed the pointwise branch, as in the official code.
        temporal_channels: Temporal channels per feature ``Y``.
        pointwise_channels: Pointwise features per layer ``Z``.
        num_layers: Number of TPC layers ``N``.
        kernel_size: Temporal kernel size ``k``.
        temporal_dropout: Dropout in the temporal branch.
        main_dropout: Dropout in the pointwise parts of the model.
        diag_embedding_size: Size ``D*`` of the diagnosis embedding.
        final_fc_size: Width ``X`` of the first final pointwise layer.
        loss: ``"msle"`` (paper default), ``"mse"``, ``"asymmetric_msle"``,
            or ``"huber_log"``.
        dilation_schedule: ``"official"`` uses the released code's schedule
            (``d = 1`` for layer 1, then ``d = (n - 1) * (k - 1)``).
            ``"paper"`` and ``"linear"`` use ``d = n``; ``"constant"`` uses
            ``d = 1``; ``"exponential"`` uses ``d = 2 ** (n - 1)``.
            Linear is an alias of paper, not a distinct architectural novelty.
        underprediction_weight: Weight for asymmetric MSLE (at least 1).
        huber_delta: Positive Huber transition in log units.
        observed_key: Optional key containing the pre-fill measurement mask.
            If set, values are prepared internally and decay is recomputed;
            set ``decay_key=None`` to omit a separate decay input.
        decay_base: Decay base for internal preparation, default 0.75.
        lab_indices: Lab indices for internal preparation; None means all.
        forward_fill_labs: Whether internal preparation forward-fills labs.

    Note:
        Defaults follow the paper's best eICU configuration. Smaller values
        are recommended for CPU runs and unit tests.

        Differences from the official implementation
        (github.com/EmmaRocheteau/TPC-LoS-prediction, ``models/tpc_model.py``):

        * Standard ``BatchNorm1d`` is used. The official code's custom
          ``MyBatchNorm1d`` keeps using batch statistics in eval mode.
        * ``forward`` predicts only the last hour of each window (PyHealth's
          regression label is a single number). The official code predicts
          every hour from hour 5 on; use :meth:`forward_sequence` for that.
        * Temporal and pointwise sizes are the same in every layer (the
          official code accepts one size per layer).
        * No mortality head or multitask loss, and no sequence-length mask in
          the loss because windows have a fixed length.

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
        decay_key: str | None = "decay",
        static_key: str | None = None,
        diag_key: str | None = None,
        time_key: str | None = None,
        temporal_channels: int = 12,
        pointwise_channels: int = 13,
        num_layers: int = 9,
        kernel_size: int = 4,
        temporal_dropout: float = 0.05,
        main_dropout: float = 0.45,
        diag_embedding_size: int = 64,
        final_fc_size: int = 17,
        loss: str = "msle",
        dilation_schedule: str = "official",
        underprediction_weight: float = 2.0,
        huber_delta: float = 1.0,
        observed_key: str | None = None,
        decay_base: float = 0.75,
        lab_indices: Sequence[int] | None = None,
        forward_fill_labs: bool = True,
    ) -> None:
        """Initializes the TPC layers and day-valued output head."""
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("TPC supports a single label key.")
        if self.mode != "regression":
            raise ValueError(f"TPC needs a 'regression' label, got mode={self.mode!r}.")
        if self.label_processor.normalization is not None:
            raise ValueError("TPC requires unnormalized remaining-LoS labels in days.")
        if loss not in ("msle", "mse", "asymmetric_msle", "huber_log"):
            raise ValueError(
                "Unknown loss; use msle, mse, asymmetric_msle, or huber_log."
            )
        if dilation_schedule not in (
            "official",
            "paper",
            "constant",
            "linear",
            "exponential",
        ):
            raise ValueError("Unknown dilation_schedule.")
        if min(num_layers, kernel_size, temporal_channels, pointwise_channels) < 1:
            raise ValueError("Layer count, kernel size, and channels must be positive.")
        if not math.isfinite(underprediction_weight) or underprediction_weight < 1:
            raise ValueError("underprediction_weight must be finite and at least 1.")
        if not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive.")
        if not 0 < decay_base <= 1:
            raise ValueError("decay_base must be in (0, 1].")
        if observed_key is None and (
            decay_base != 0.75 or lab_indices is not None or not forward_fill_labs
        ):
            raise ValueError("Preprocessing options require observed_key.")
        for key in (value_key, decay_key, static_key, diag_key, time_key, observed_key):
            if key is not None and key not in self.feature_keys:
                raise ValueError(f"'{key}' is not in the dataset input_schema.")

        self.label_key = self.label_keys[0]
        self.value_key = value_key
        self.decay_key = decay_key
        self.static_key = static_key
        self.diag_key = diag_key
        self.time_key = time_key
        self.loss_type = loss
        self.underprediction_weight = underprediction_weight
        self.huber_delta = huber_delta
        self.observed_key = observed_key
        self.decay_base = decay_base
        self.lab_indices = lab_indices
        self.forward_fill_labs = forward_fill_labs

        procs = dataset.input_processors
        num_features = procs[value_key].size()
        num_static = procs[static_key].size() if static_key else 0
        num_diag = procs[diag_key].size() if diag_key else 0
        num_time = procs[time_key].size() if time_key else 0

        def dilation(n: int) -> int:
            if dilation_schedule == "paper":
                return n
            if dilation_schedule == "linear":
                return n
            if dilation_schedule == "constant":
                return 1
            if dilation_schedule == "exponential":
                return 2 ** (n - 1)
            return 1 if n == 1 else (n - 1) * (kernel_size - 1)

        self.layers = nn.ModuleList(
            [
                TPCLayer(
                    layer_idx=n,
                    num_features=num_features,
                    num_static=num_static,
                    num_time=num_time,
                    temporal_channels=temporal_channels,
                    pointwise_channels=pointwise_channels,
                    kernel_size=kernel_size,
                    dilation=dilation(n),
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
    def _tensor(feature: TensorInput) -> torch.Tensor:
        """Extracts the value tensor from a processor output."""
        return feature[0] if isinstance(feature, (tuple, list)) else feature

    def forward_sequence(self, **kwargs: TensorInput) -> torch.Tensor:
        """Predicts remaining LoS at every hour.

        Args:
            **kwargs: Batch of features keyed like the dataset input schema.

        Returns:
            Tensor of shape ``(batch, time)`` with LoS in days, clipped to
            ``[1/48, 100]``.
        """
        x = self._tensor(kwargs[self.value_key]).float().to(self.device)
        prepared_decay = None
        if self.observed_key:
            observed = self._tensor(kwargs[self.observed_key]).to(self.device)
            x, prepared_decay = prepare_time_series(
                x, observed, self.decay_base, self.lab_indices, self.forward_fill_labs
            )
        x = x.permute(0, 2, 1)  # (batch, F, time)
        batch, _, steps = x.shape

        if prepared_decay is not None:
            decay = prepared_decay.permute(0, 2, 1)
        elif self.decay_key:
            decay = self._tensor(kwargs[self.decay_key]).float().to(self.device)
            decay = decay.permute(0, 2, 1)
        else:
            decay = torch.zeros_like(x)

        static = None
        if self.static_key:
            static = self._tensor(kwargs[self.static_key]).float().to(self.device)

        time_feats = None
        if self.time_key:
            time_feats = self._tensor(kwargs[self.time_key]).float().to(self.device)
            time_feats = time_feats.permute(0, 2, 1)  # (batch, n_time, time)

        # First layer input: values and decay as two channels per feature.
        h = torch.stack([x, decay], dim=2)  # (batch, F, 2, time)
        prev_point: list[torch.Tensor] = []
        prev_temp = None
        for layer in self.layers:
            h, point, prev_temp = layer(
                h, x, decay, static, time_feats, prev_point, prev_temp
            )
            prev_point.append(point)

        parts = [h.reshape(batch, -1, steps)]
        if static is not None:
            parts.append(static.unsqueeze(-1).expand(-1, -1, steps))
        if self.diag_encoder is not None:
            diag = self._tensor(kwargs[self.diag_key]).float().to(self.device)
            parts.append(self.diag_encoder(diag).unsqueeze(-1).expand(-1, -1, steps))

        z = self.final_drop(self.final_bn(self.final1(torch.cat(parts, dim=1))))
        z = self.final2(F.relu(z)).squeeze(1)  # (batch, time)
        return torch.exp(z.clamp(math.log(MIN_LOS_DAYS), math.log(MAX_LOS_DAYS))).clamp(
            MIN_LOS_DAYS, MAX_LOS_DAYS
        )

    def forward(self, **kwargs: TensorInput) -> dict[str, torch.Tensor]:
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
            elif self.loss_type == "asymmetric_msle":
                results["loss"] = asymmetric_msle_loss(
                    pred, y_true, self.underprediction_weight
                )
            elif self.loss_type == "huber_log":
                results["loss"] = huber_log_loss(pred, y_true, self.huber_delta)
            else:
                results["loss"] = F.mse_loss(pred, y_true)
            results["y_true"] = y_true
        return results
