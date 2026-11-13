"""Author: Tamjit Kumar, Yi Wang
NetID: tkumar70, ywang4715
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Temporal Pointwise Convolution (TPC) model and TPCBlock architecture
             for ICU length of stay prediction and multitask mortality prediction.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from pyhealth.models import BaseModel

if TYPE_CHECKING:
    from pyhealth.datasets import SampleDataset

logger = logging.getLogger(__name__)


class MyBatchNorm1d(_BatchNorm):
    """Batch Normalization layer from Rocheteau et al. (tpc_model.py:63-92).

    In variable-length padded ICU sequences with causal left-padding, standard
    nn.BatchNorm1d accumulates running_mean and running_var across mini-batches of
    widely varying sequence lengths T, which skews running statistics during
    model.eval(). MyBatchNorm1d normalizes using current batch statistics during
    evaluation while freezing running statistics updates. Matches the official
    implementation from EmmaRocheteau/TPC-LoS-prediction.

    Note on Evaluation Batch Statistics and Length Bucketing:
        Unlike standard nn.BatchNorm1d which uses frozen running statistics during
        model.eval(), MyBatchNorm1d normalizes using the current mini-batch statistics.
        In variable-length sequences, this means predictions depend on the batch
        composition and right-padded zero timesteps contribute to the statistics.
        Rocheteau et al. mitigated this by grouping stays of similar lengths using
        length-bucketed batch sampling (BucketBatchSampler). When using standard
        un-bucketed DataLoaders (such as PyHealth's default get_dataloader), standard
        nn.BatchNorm1d (batchnorm="default") is recommended to ensure deterministic,
        batch-independent evaluation predictions.

    Args:
        num_features (int): Number of feature channels (C).
        eps (float): Value added to denominator for numerical stability. Default: 1e-5.
        momentum (float | None): Value used for running_mean and running_var
            computation during training. Default: 0.1.
        affine (bool): Whether to learn learnable affine parameters. Default: True.
        track_running_stats (bool): Whether to track running mean and variance.
            Default: True.

    Examples:
        >>> import torch
        >>> from pyhealth.models.tpc import MyBatchNorm1d
        >>> bn = MyBatchNorm1d(num_features=16)
        >>> x = torch.randn(2, 16, 24)
        >>> out = bn(x)
        >>> out.shape
        torch.Size([2, 16, 24])
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        elif self.training:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0

        if (
            self.track_running_stats
            and self.training
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum

        # Flatten sequence length into batch dimension if B * T > 1
        use_batch_stats = (input.numel() // max(input.size(1), 1)) > 1
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=use_batch_stats,
            momentum=exponential_average_factor,
            eps=self.eps,
        )


def _create_batchnorm_layers(
    batchnorm: str,
    temp_channels: int,
    point_channels: int,
) -> tuple[nn.Module, nn.Module]:
    """Creates (temp_bn, point_bn) matching Rocheteau et al. (tpc_model.py:149-158)."""
    bn_mode = (batchnorm or "none").lower()
    if bn_mode == "mybatchnorm":
        return MyBatchNorm1d(temp_channels), MyBatchNorm1d(point_channels)
    if bn_mode == "default":
        return nn.BatchNorm1d(temp_channels), nn.BatchNorm1d(point_channels)
    if bn_mode == "pointonly":
        return nn.Identity(), MyBatchNorm1d(point_channels)
    if bn_mode == "temponly":
        return MyBatchNorm1d(temp_channels), nn.Identity()
    if bn_mode == "low_momentum":
        return (
            nn.BatchNorm1d(temp_channels, momentum=0.01),
            nn.BatchNorm1d(point_channels, momentum=0.01),
        )
    if bn_mode == "none":
        return nn.Identity(), nn.Identity()
    raise ValueError(
        f"Unknown batchnorm mode: '{batchnorm}'. Supported: "
        f"'mybatchnorm', 'default', 'pointonly', 'temponly', 'low_momentum', 'none'."
    )


def _align_target(
    y_true: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Standardizes target tensor shape to (batch_size, seq_len, 1)."""
    y = y_true.to(device=device, dtype=torch.float32)
    if y.dim() == 1:
        return y.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
    if y.dim() == 2:
        if y.shape[1] == seq_len:
            return y.unsqueeze(-1)
        if y.shape[1] == 1:
            return y.unsqueeze(-1).expand(batch_size, seq_len, 1)
        return y.view(batch_size, -1, 1)
    if y.dim() == 3:
        if y.shape[1] == 1 and y.shape[2] == seq_len:
            return y.transpose(1, 2)
        if y.shape[1] == 1 and seq_len > 1:
            return y.expand(batch_size, seq_len, -1)
        return y
    return y.view(batch_size, seq_len, 1)


def _align_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Standardizes mask tensor shape to (batch_size, seq_len, 1)."""
    if mask is None:
        return torch.ones(batch_size, seq_len, 1, dtype=torch.bool, device=device)
    m = mask.to(device=device, dtype=torch.bool)
    if m.dim() == 1:
        return m.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
    if m.dim() == 2:
        if m.shape[1] == seq_len:
            return m.unsqueeze(-1)
        return m.unsqueeze(-1).expand(batch_size, seq_len, 1)
    if m.dim() == 3:
        if m.shape[1] == 1 and m.shape[2] == seq_len:
            return m.transpose(1, 2)
        if m.shape[1] == seq_len and m.shape[2] == 1:
            return m
        if m.shape[1] == seq_len:
            return m[:, :, :1]
    return m.view(batch_size, seq_len, 1)


class TPCBlock(nn.Module):
    """A single Temporal Pointwise Convolution (TPC) block.

    Implements Equations (1)-(5) and Figure 3 from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit' (Rocheteau et al., ACM CHIL 2021).

    Args:
        layer_idx (int): 1-indexed layer number (n).
        num_features (int): Number of initial time-series features (F).
        point_dim (int): Pointwise output channels (Z).
        temp_dim (int): Temporal output channels per feature (Y).
        static_dim (int): Number of static features (S).
        kernel_size (int): Filter size (k).
        dropout (float): Dropout probability for pointwise convolutions. Default: 0.1.
        use_decay (bool): Whether decay indicators are used. Default: True.
        temp_dropout (float | None): Dropout probability for temporal convolutions.
            If None, defaults to `dropout`. Default: None.
        batchnorm (str): BatchNorm variant ('mybatchnorm', 'default', 'pointonly',
            'temponly', 'low_momentum', 'none'). Default: 'default'.
        time_dim (int): Number of non-convolved temporal context features. Default: 0.
        share_weights (bool): Whether to share temporal convolution weights across
            all input features (groups=1). Corresponds to 'Temp. only (WS)' ablation
            in Rocheteau et al. (Table 3). Default: False.
        use_skip (bool): Whether to include skip connections from raw input and
            preceding pointwise layers. Set to False for 'TPC (no skip)' ablation
            in Table 3. Default: True.

    Examples:
        >>> import torch
        >>> from pyhealth.models.tpc import TPCBlock
        >>> block = TPCBlock(
        ...     layer_idx=1,
        ...     num_features=12,
        ...     point_dim=16,
        ...     temp_dim=8,
        ...     static_dim=5,
        ... )
        >>> x_raw = torch.randn(2, 12, 24)
        >>> x_decay = torch.rand(2, 12, 24)
        >>> h = torch.stack([x_raw, x_decay], dim=2)
        >>> s_expanded = torch.randn(2, 5, 24)
        >>> h_next, point_out = block(
        ...     h, x_raw, x_decay, s_expanded, prev_pointwise=[]
        ... )
        >>> h_next.shape
        torch.Size([2, 28, 9, 24])
    """

    def __init__(
        self,
        layer_idx: int,
        num_features: int,
        point_dim: int,
        temp_dim: int,
        static_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_decay: bool = True,
        temp_dropout: float | None = None,
        batchnorm: str = "default",
        time_dim: int = 0,
        share_weights: bool = False,
        use_skip: bool = True,
    ):
        super().__init__()
        self.n = layer_idx
        self.k = kernel_size
        self.d = layer_idx  # Dilation increments by 1 with each layer (d = n)
        self.pad = self.d * (self.k - 1)  # Causal left-sided padding
        self.use_decay = use_decay
        self.share_weights = share_weights
        self.use_skip = use_skip

        if self.use_skip:
            self.R_n = num_features + point_dim * (self.n - 1)
        else:
            # Without skip connections, temporal convolutions operate across
            # the F dynamic features across all layers
            self.R_n = num_features

        if self.n == 1:
            self.C_n = 2 if self.use_decay else 1
        else:
            self.C_n = temp_dim + 1 if self.use_skip else temp_dim
        self.temp_dim = temp_dim
        self.point_dim = point_dim
        self.static_dim = static_dim
        self.time_dim = time_dim

        t_drop = dropout if temp_dropout is None else temp_dropout

        # 1. Temporal Convolution
        if self.share_weights:
            # Weight sharing across features (Section 3.2 & Table 3 'Temp. WS'):
            # A single filter (C_n -> temp_dim) is shared identically across all
            # R_n features
            self.temporal_conv = nn.Conv1d(
                in_channels=self.C_n,
                out_channels=self.temp_dim,
                kernel_size=self.k,
                dilation=self.d,
                groups=1,
                bias=True,
            )
        else:
            # Independent filter per feature across time (groups = R_n)
            self.temporal_conv = nn.Conv1d(
                in_channels=self.R_n * self.C_n,
                out_channels=self.R_n * self.temp_dim,
                kernel_size=self.k,
                dilation=self.d,
                groups=self.R_n,
                bias=True,
            )
        self.temp_bn, self.point_bn = _create_batchnorm_layers(
            batchnorm, self.R_n * self.temp_dim, self.point_dim
        )
        self.temp_dropout = nn.Dropout(t_drop)

        # 2. Pointwise Convolution (1x1 cross-feature interaction)
        decay_dim = num_features if self.use_decay else 0
        if self.use_skip or self.n == 1:
            self.P_n = (
                (self.R_n * self.C_n) + self.static_dim + self.time_dim + decay_dim
            )
        else:
            # Without skip connections, deeper pointwise layers process the
            # previous pointwise features alongside static context
            self.P_n = self.point_dim + self.static_dim + self.time_dim
        self.pointwise_conv = nn.Conv1d(
            in_channels=self.P_n,
            out_channels=self.point_dim,
            kernel_size=1,
            bias=True,
        )
        self.point_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        x_raw: torch.Tensor,
        x_decay: torch.Tensor | None,
        s_expanded: torch.Tensor | None,
        prev_pointwise: list[torch.Tensor],
        t_expanded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of a single TPC block."""
        B, _, _, T = h.shape
        h_flat = h.view(B, self.R_n * self.C_n, T)

        # --- A. Temporal Convolution Branch ---
        if self.share_weights:
            h_ws = h.reshape(B * self.R_n, self.C_n, T)
            h_pad = F.pad(h_ws, (self.pad, 0))
            temp_out = self.temporal_conv(h_pad).view(B, self.R_n * self.temp_dim, T)
        else:
            h_pad = F.pad(h_flat, (self.pad, 0))
            temp_out = self.temporal_conv(h_pad)
        temp_out = self.temp_dropout(self.temp_bn(temp_out))
        temp_out = temp_out.view(B, self.R_n, self.temp_dim, T)

        # --- B. Pointwise Convolution Branch ---
        if self.use_skip or self.n == 1:
            point_in_list = [h_flat]
            if s_expanded is not None and self.static_dim > 0:
                point_in_list.append(s_expanded)
            if t_expanded is not None and self.time_dim > 0:
                point_in_list.append(t_expanded)
            if self.use_decay and x_decay is not None:
                point_in_list.append(x_decay)
        else:
            prev_p = prev_pointwise[-1].squeeze(2)
            point_in_list = [prev_p]
            if s_expanded is not None and self.static_dim > 0:
                point_in_list.append(s_expanded)
            if t_expanded is not None and self.time_dim > 0:
                point_in_list.append(t_expanded)
        point_in = torch.cat(point_in_list, dim=1)

        point_out = self.pointwise_conv(point_in)
        point_out = self.point_dropout(self.point_bn(point_out))  # (B, Z, T)

        # --- C. Skip Connections (Eq. 4) & D. Feature Concatenation (Eq. 5) ---
        if not self.use_skip:
            # Without skip connections, maintain parallel temporal and pointwise streams
            # matching Rocheteau et al. (temp_pointwise_no_skip)
            h_next = F.relu(temp_out)
            return h_next, F.relu(point_out).unsqueeze(2)

        if self.n == 1:
            r_n = torch.cat([temp_out, x_raw.unsqueeze(2)], dim=2)  # (B, F, Y+1, T)
        else:
            skips = torch.cat([x_raw.unsqueeze(2)] + prev_pointwise, dim=1)
            r_n = torch.cat([temp_out, skips], dim=2)  # (B, R_n, Y+1, T)

        point_broadcast = point_out.unsqueeze(2).repeat(1, 1, self.temp_dim + 1, 1)
        h_next = F.relu(
            torch.cat([r_n, point_broadcast], dim=1)
        )  # (B, R_n + Z, Y+1, T)

        return h_next, point_out.unsqueeze(2)


class TPC(BaseModel):
    """Temporal Pointwise Convolution (TPC) model for ICU LoS prediction.

    Adapted from the architecture described in:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit'
    (Emma Rocheteau, Pietro Liò, & Stephanie Hyland, ACM CHIL 2021).

    Args:
        dataset: Optional PyHealth SampleDataset. If provided, input dimensions
            are inferred from the first sample.
        feature_keys: Optional list of feature keys (BaseModel compatibility).
        label_key: Target LoS key in batch kwargs. Default: 'los'.
        mode: Prediction mode ('regression' or 'binary'). Default: 'regression'.
        continuous_dim (int | None): Number of time series features (F).
            Inferred from dataset if None. Required if dataset is None.
        static_dim (int | None): Number of static baseline features (S).
            Inferred from dataset if None, or defaults to 0.
        time_dim (int | None): Number of non-convolved temporal context features
            (e.g., elapsed days and hour-of-day). Inferred from dataset if None,
            or defaults to 0.
        time_series_key (str): Key for continuous dynamic time-series in kwargs.
            Default: 'vitals'.
        decay_key (str | None): Key for exponential decay indicator tensor in kwargs.
            Default: 'vitals_decay'.
        static_key (str | None): Key for static baseline tensor in kwargs.
            Default: 'static'.
        time_key (str | None): Key for non-convolved temporal context tensor in kwargs.
            Default: 'time_features'.
        mask_key (str): Key for observation/validity mask in kwargs.
            Default: 'mask'.
        aux_label_key (str | None): Optional key for auxiliary classification target
            (e.g. in-hospital mortality). Default: None.
        aux_weight (float): Loss weight for auxiliary classification task (corresponds
            to alpha in Rocheteau et al., arXiv:2007.09483 Appendix C). Default: 100.0.
        n_layers (int): Number of stacked TPC blocks (N). Default: 3.
        temp_dim (int): Temporal output channels per feature (Y). Default: 8.
        point_dim (int): Pointwise output channels (Z). Default: 16.
        kernel_size (int): Temporal filter size (k). Default: 3.
        hidden_dim (int): Hidden dimension for prediction head. Default: 64.
        dropout (float): Dropout probability for pointwise layers. Default: 0.1.
        temp_dropout (float | None): Dropout probability for temporal convolution
            layers. If None, defaults to `dropout`. Default: None.
        batchnorm (str): BatchNorm variant to use across layers ('mybatchnorm',
            'default', 'pointonly', 'temponly', 'low_momentum', 'none').
            Default: 'default'.
        loss_fn (str): Loss function to use ('msle' or 'mse'). Default: 'msle'.
        min_time (int): Warmup time index before evaluation begins (t >= min_time).
            Default: 4 (corresponding to 5 hours elapsed since ICU admission).
        floor_days (float): Minimum threshold floor for LoS (in days, ~30 mins = 1/48).
            Default: 1/48.
        max_days (float): Maximum threshold cap for LoS in days. Default: 100.0.
        use_decay (bool): Whether to include exponential decay channels (0.75^j).
            Set to False for the paper's 'No Decay' ablation variant. Default: True.
        decay_rate (float): Base decay factor for exponential decay indicators (0.75^j).
            Default: 0.75.
        no_exp (bool): Whether to disable softplus exponential transformation.
            Default: False.
        seq_to_seq (bool): Whether to produce hourly predictions (B, T, 1) or pool
            to the final valid time step (B, 1) for stay-level tasks. Default: True.
        share_weights (bool): Whether to share temporal convolution weights across
            all input features (groups=1). Corresponds to 'Temp. only (WS)' ablation
            in Table 3. Default: False.
        use_skip (bool): Whether to include residual skip connections. Set to False
            for 'TPC (no skip)' ablation in Table 3. Default: True.

    Note on Cohort & Database Schema:
        The pipeline models multi-modal clinical EHR cohorts using PyHealth's
        supported core tables (patient, vitalPeriodic, vitalAperiodic, lab,
        and physicalExam).

    Note on Batch Normalization Variants:

        - 'default' (Recommended): Standard nn.BatchNorm1d across temporal and
          pointwise channels. Uses running statistics during evaluation, ensuring
          predictions are strictly deterministic and independent of mini-batch
          composition or padding.
        - 'mybatchnorm': Normalizes using current mini-batch statistics during
          evaluation, matching Rocheteau et al. Requires length-bucketed batch
          sampling (BucketBatchSampler) to prevent right-padding skew.

    Examples:
        >>> import torch
        >>> from pyhealth.models.tpc import TPC
        >>> model = TPC(dataset=None, continuous_dim=12, static_dim=5)
        >>> vitals = torch.randn(2, 24, 12)
        >>> vitals_decay = torch.rand(2, 24, 12)
        >>> static = torch.randn(2, 5)
        >>> y_true = torch.rand(2, 24, 1) + 0.1
        >>> out = model(
        ...     vitals=vitals, vitals_decay=vitals_decay, static=static, y_true=y_true
        ... )
        >>> out["y_prob"].shape
        torch.Size([2, 24, 1])
        >>> "loss" in out
        True
    """

    @property
    def mode(self) -> str:
        """Prediction mode ('regression' or 'binary')."""
        val = getattr(self, "_mode", None)
        return val if val is not None else "regression"

    @mode.setter
    def mode(self, value: str | None) -> None:
        self._mode = value

    @property
    def label_processors(self) -> dict[str, Any]:
        """Label processors by output key, defaulting to mode processor if dataset uses raw tensors."""
        dataset = self.__dict__.get("dataset")
        processors = getattr(dataset, "output_processors", None) or {}
        from pyhealth.processors import PROCESSOR_REGISTRY, LabelProcessor
        label_procs = {
            key: processor
            for key, processor in processors.items()
            if isinstance(processor, LabelProcessor)
        }
        if label_procs:
            return label_procs
        mode = getattr(self, "_mode", "regression")
        cls = PROCESSOR_REGISTRY.get(mode)
        if cls is not None and inspect.isclass(cls) and issubclass(cls, LabelProcessor):
            label_key = getattr(self, "label_key", "los")
            return {label_key: cls()}
        return {}

    def __init__(
        self,
        dataset: Any = None,
        feature_keys: list[str] | None = None,
        label_key: str | None = None,
        mode: str | None = None,
        continuous_dim: int | None = None,
        static_dim: int | None = None,
        time_dim: int | None = None,
        time_series_key: str = "vitals",
        decay_key: str | None = "vitals_decay",
        static_key: str | None = "static",
        time_key: str | None = "time_features",
        mask_key: str = "mask",
        aux_label_key: str | None = None,
        aux_weight: float = 100.0,
        n_layers: int = 3,
        temp_dim: int = 8,
        point_dim: int = 16,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temp_dropout: float | None = None,
        batchnorm: str = "default",
        loss_fn: str = "msle",
        min_time: int = 4,
        floor_days: float = 1.0 / 48.0,
        max_days: float = 100.0,
        use_decay: bool = True,
        decay_rate: float = 0.75,
        no_exp: bool = False,
        seq_to_seq: bool = True,
        share_weights: bool = False,
        use_skip: bool = True,
        **kwargs,
    ):
        super().__init__(dataset=cast("SampleDataset", dataset), **kwargs)
        self.seq_to_seq = seq_to_seq
        self.share_weights = share_weights
        self.use_skip = use_skip

        # Single-tensor time_series_key mapping from feature_keys
        if (
            feature_keys is not None
            and len(feature_keys) > 0
            and time_series_key == "vitals"
            and "vitals" not in feature_keys
        ):
            time_series_key = feature_keys[0]
        elif (
            time_series_key == "vitals"
            and getattr(self, "feature_keys", None)
            and "vitals" not in self.feature_keys
            and len(self.feature_keys) > 0
        ):
            time_series_key = self.feature_keys[0]

        self.time_series_key = time_series_key
        self.decay_key = decay_key if use_decay else None
        self.static_key = static_key
        self.time_key = time_key
        self.mask_key = mask_key

        # Infer input dimensions from dataset if provided and not explicitly given
        if dataset is not None:
            try:
                sample0 = dataset[0]
                if continuous_dim is None and self.time_series_key in sample0:
                    v = sample0[self.time_series_key]
                    continuous_dim = v.shape[-1] if hasattr(v, "shape") else len(v[0])
                if (
                    static_dim is None
                    and self.static_key
                    and self.static_key in sample0
                ):
                    s = sample0[self.static_key]
                    static_dim = s.shape[-1] if hasattr(s, "shape") else len(s)
                if time_dim is None and self.time_key and self.time_key in sample0:
                    tf = sample0[self.time_key]
                    time_dim = tf.shape[-1] if hasattr(tf, "shape") else len(tf[0])
                # Auto-fallback: if decay feature is absent from dataset, disable decay
                has_decay = (
                    self.decay_key is not None and self.decay_key in sample0
                ) or ("vitals_time_delta" in sample0)
                if use_decay and not has_decay:
                    logger.info(
                        "Decay key '%s' not found in dataset sample; "
                        "auto-disabling use_decay.",
                        self.decay_key,
                    )
                    use_decay = False
                    self.decay_key = None
            except (AttributeError, IndexError, KeyError, TypeError) as err:
                raise ValueError(
                    f"Failed to automatically infer dimensions from dataset: {err}. "
                    f"Please provide continuous_dim and static_dim explicitly."
                ) from err

        if continuous_dim is None:
            raise ValueError(
                "continuous_dim must be explicitly provided when dataset is None."
            )
        if static_dim is None:
            static_dim = 0
        if time_dim is None:
            time_dim = 0

        self.continuous_dim = continuous_dim
        self.static_dim = static_dim
        self.time_dim = time_dim
        self.use_decay = use_decay

        # Construct feature_keys list when not explicitly provided
        if feature_keys is not None and len(feature_keys) > 0:
            self.feature_keys = feature_keys
        else:
            f_keys = [time_series_key]
            if self.use_decay and self.decay_key:
                f_keys.append(self.decay_key)
            if self.static_key and self.static_dim > 0:
                f_keys.append(self.static_key)
            if self.time_key and self.time_dim > 0:
                f_keys.append(self.time_key)
            self.feature_keys = f_keys

        self.label_key = label_key or (
            self.label_keys[0] if getattr(self, "label_keys", None) else "los"
        )
        if not self.label_keys:
            self.label_keys = [self.label_key]
        self.mode = mode or getattr(self, "mode", None) or "regression"
        if self.mode not in ("regression", "binary"):
            raise ValueError(
                f"Unsupported mode: '{self.mode}'. "
                f"Choose from ('regression', 'binary')."
            )
        self.n_layers = n_layers
        self.temp_dim = temp_dim
        self.point_dim = point_dim
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm
        loss_fn_type = loss_fn.lower()
        if loss_fn_type not in ("msle", "mse"):
            raise ValueError(
                f"Unsupported loss_fn: '{loss_fn}'. Choose from ('msle', 'mse')."
            )
        self.loss_fn_type = loss_fn_type
        self.min_time = min_time
        self.floor_days = floor_days
        self.log_floor = math.log(floor_days)
        self.max_days = max_days
        self.max_log = math.log(max_days)
        self.use_decay = use_decay
        self.decay_rate = decay_rate
        self.no_exp = no_exp

        # Dedicated auxiliary classification task (e.g. in-hospital mortality)
        self.aux_label_key = aux_label_key
        self.has_aux_task = aux_label_key is not None
        self.aux_weight = aux_weight

        # Stacked TPC blocks
        self.blocks = nn.ModuleList(
            [
                TPCBlock(
                    layer_idx=n,
                    num_features=continuous_dim,
                    point_dim=point_dim,
                    temp_dim=temp_dim,
                    static_dim=static_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_decay=use_decay,
                    temp_dropout=temp_dropout,
                    batchnorm=batchnorm,
                    time_dim=time_dim,
                    share_weights=share_weights,
                    use_skip=use_skip,
                )
                for n in range(1, n_layers + 1)
            ]
        )

        # Final feature count after N layers
        if self.use_skip:
            final_features = continuous_dim + n_layers * point_dim
            final_channels = temp_dim + 1
            final_ts_dim = final_features * final_channels
        else:
            # Without skip connections, concatenate final pointwise (Z)
            # and final temporal features (F * Y) matching Rocheteau et al.
            final_ts_dim = (continuous_dim * temp_dim) + point_dim
        final_in_dim = final_ts_dim + static_dim + time_dim

        # Prediction head: Two pointwise layers (Section 3.5 & Appendix A)
        self.head_conv1 = nn.Conv1d(final_in_dim, hidden_dim, kernel_size=1)
        _, self.head_bn1 = _create_batchnorm_layers(batchnorm, hidden_dim, hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.head_fc = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        # Dedicated auxiliary classification head (e.g. mortality)
        if self.has_aux_task:
            self.aux_conv1 = nn.Conv1d(final_in_dim, hidden_dim, kernel_size=1)
            _, self.aux_bn1 = _create_batchnorm_layers(
                batchnorm, hidden_dim, hidden_dim
            )
            self.aux_dropout = nn.Dropout(dropout)
            self.aux_fc = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
        logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes loss for regression (MSLE/MSE) or binary classification (BCE).

        Conforms to Equation (6) in Rocheteau et al. when mode='regression'
        and seq_to_seq=True: loss is averaged over valid time steps within each
        stay first, and then averaged across valid stays in the batch.

        Args:
            y_pred: Predicted values tensor of shape (B, T, 1) or (B, 1).
            y_true: Ground truth target tensor of shape (B, T, 1), (B, 1), or (B,).
            mask: Optional boolean or binary mask tensor of shape (B, T) or (B, T, 1).
            logit: Optional unactivated logit tensor corresponding to y_pred.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        if self.mode == "binary":
            preds = logit if logit is not None else y_pred
            if preds.dim() == 3:
                B, T, _ = preds.shape
                target = _align_target(y_true, B, T, preds.device)
                valid_mask = _align_mask(mask, B, T, preds.device)
                if self.min_time > 0 and T > self.min_time:
                    warmup_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                    warmup_mask[:, self.min_time :, :] = True
                    eval_mask = valid_mask & warmup_mask
                    if eval_mask.sum() == 0 and valid_mask.sum() > 0:
                        eval_mask = valid_mask
                else:
                    eval_mask = valid_mask
                if eval_mask.sum() > 0:
                    return F.binary_cross_entropy_with_logits(
                        preds[eval_mask], target[eval_mask]
                    )
                return F.binary_cross_entropy_with_logits(preds, target)
            else:
                B = preds.shape[0]
                target = _align_target(y_true, B, 1, preds.device).view(B, -1)
                return F.binary_cross_entropy_with_logits(preds, target)

        # Mode is 'regression'
        if y_pred.dim() == 3:
            B, T, _ = y_pred.shape
            target = _align_target(y_true, B, T, y_pred.device)
            valid_mask = _align_mask(mask, B, T, y_pred.device)

            # Enforce evaluation starting at min_time (default: 4, corresponding to 5h)
            if self.min_time > 0 and T > self.min_time:
                warmup_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                warmup_mask[:, self.min_time :, :] = True
                eval_mask = valid_mask & warmup_mask
                if eval_mask.sum() == 0 and valid_mask.sum() > 0:
                    eval_mask = valid_mask
            else:
                eval_mask = valid_mask

            # Compute per-step error
            if self.loss_fn_type == "msle":
                log_pred = torch.log(torch.clamp(y_pred, min=self.floor_days))
                log_true = torch.log(torch.clamp(target, min=self.floor_days))
                err = (log_pred - log_true) ** 2
            else:
                err = (y_pred - target) ** 2

            # Average per stay over valid hours (Eq. 6), then average across stays
            m = eval_mask.float()
            n = m.sum(dim=(1, 2))  # (B,) valid hours per patient
            per_stay = (err * m).sum(dim=(1, 2)) / n.clamp(min=1.0)
            valid_stays = n > 0
            if valid_stays.any():
                return per_stay[valid_stays].mean()
            return per_stay.mean()
        else:
            # Stay-level regression
            B = y_pred.shape[0]
            target = _align_target(y_true, B, 1, y_pred.device).view(B, -1)
            if self.loss_fn_type == "msle":
                log_pred = torch.log(torch.clamp(y_pred, min=self.floor_days))
                log_true = torch.log(torch.clamp(target, min=self.floor_days))
                return F.mse_loss(log_pred, log_true)
            return F.mse_loss(y_pred, target)

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Forward propagation for TPC.

        Args:
            **kwargs: Inputs including dynamic continuous time-series,
                optional decay indicators, optional static features,
                optional temporal context features, optional mask, and target labels.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing 'y_prob', 'logit',
                and optionally 'loss', 'y_true', 'mask', 'aux_prob',
                'aux_logit', 'aux_loss'.
        """
        # 1. Extract and validate continuous dynamic time-series
        if self.time_series_key not in kwargs:
            raise KeyError(
                f"Missing required time-series feature key '{self.time_series_key}' "
                f"in kwargs. Available keys: {list(kwargs.keys())}"
            )
        raw_vitals = kwargs[self.time_series_key]
        if not isinstance(raw_vitals, torch.Tensor):
            raise TypeError(
                f"Expected '{self.time_series_key}' to be a torch.Tensor, but got "
                f"{type(raw_vitals).__name__}"
            )

        vitals_tensor = raw_vitals if raw_vitals.dim() == 3 else raw_vitals.unsqueeze(1)
        B, T, F_in = vitals_tensor.shape
        device = self.device
        vitals_tensor = vitals_tensor.to(device=device, dtype=torch.float32)

        if F_in != self.continuous_dim:
            raise ValueError(
                f"Expected {self.continuous_dim} continuous features for "
                f"'{self.time_series_key}', but got tensor with {F_in} channels."
            )

        # 2. Extract or validate decay indicators (0.75^j)
        x_raw = vitals_tensor.transpose(1, 2).float()
        if self.use_decay:
            raw_decay = kwargs.get(self.decay_key) if self.decay_key else None
            if isinstance(raw_decay, torch.Tensor):
                vitals_decay = raw_decay.to(device=device, dtype=torch.float32)
            elif "vitals_time_delta" in kwargs and isinstance(
                kwargs["vitals_time_delta"], torch.Tensor
            ):
                vitals_decay = torch.pow(
                    self.decay_rate,
                    torch.clamp(kwargs["vitals_time_delta"].to(device=device), min=0.0),
                )
            elif self.decay_key is not None and self.decay_key in kwargs:
                raise TypeError(
                    f"Expected '{self.decay_key}' to be a torch.Tensor, but got "
                    f"{type(kwargs[self.decay_key]).__name__}"
                )
            else:
                # Graceful auto-fallback for generic PyHealth batches lacking decay
                vitals_decay = torch.ones_like(vitals_tensor)

            if vitals_decay.shape != vitals_tensor.shape:
                raise ValueError(
                    f"Decay tensor shape {vitals_decay.shape} must match "
                    f"time-series shape {vitals_tensor.shape}."
                )
            x_decay = vitals_decay.transpose(1, 2).float()
            h = torch.stack([x_raw, x_decay], dim=2)  # (B, F, 2, T)
        else:
            x_decay = None
            h = x_raw.unsqueeze(2)  # (B, F, 1, T)

        # 3. Extract and validate static features
        s_expanded: torch.Tensor | None = None
        if self.static_dim > 0:
            raw_static = kwargs.get(self.static_key) if self.static_key else None
            if raw_static is None:
                raise KeyError(
                    f"Model configured with static_dim={self.static_dim}, but "
                    f"static key '{self.static_key}' was not found in kwargs."
                )
            if not isinstance(raw_static, torch.Tensor):
                raise TypeError(
                    f"Expected '{self.static_key}' to be a torch.Tensor, but got "
                    f"{type(raw_static).__name__}"
                )
            s_tensor = raw_static.to(device=device, dtype=torch.float32)
            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(0).expand(B, -1)
            elif s_tensor.shape[0] == 1 and B > 1:
                s_tensor = s_tensor.expand(B, -1)
            if s_tensor.shape[1] != self.static_dim:
                raise ValueError(
                    f"Expected static feature dimension {self.static_dim}, "
                    f"but got {s_tensor.shape[1]}."
                )
            s_expanded = s_tensor.unsqueeze(-1).expand(-1, -1, T)

        # 3b. Extract and validate temporal context features
        t_expanded: torch.Tensor | None = None
        if self.time_dim > 0:
            raw_time = kwargs.get(self.time_key) if self.time_key else None
            if raw_time is None:
                raise KeyError(
                    f"Model configured with time_dim={self.time_dim}, but "
                    f"temporal context key '{self.time_key}' was not found in kwargs."
                )
            if not isinstance(raw_time, torch.Tensor):
                raise TypeError(
                    f"Expected '{self.time_key}' to be a torch.Tensor, but got "
                    f"{type(raw_time).__name__}"
                )
            t_tensor = raw_time.to(device=device, dtype=torch.float32)
            if t_tensor.dim() == 2:
                t_tensor = t_tensor.unsqueeze(0).expand(B, -1, -1)
            if t_tensor.shape[-1] != self.time_dim:
                raise ValueError(
                    f"Expected temporal context dimension {self.time_dim}, "
                    f"but got {t_tensor.shape[-1]}."
                )
            t_expanded = t_tensor.transpose(1, 2)  # (B, time_dim, T)

        # 4. Pass through stacked TPC blocks
        prev_pointwise: list[torch.Tensor] = []
        for block in self.blocks:
            h, point_out_ch = block(
                h, x_raw, x_decay, s_expanded, prev_pointwise, t_expanded
            )
            prev_pointwise.append(point_out_ch)

        # 5. Prediction Head (Section 3.5 & Appendix A)
        if self.use_skip:
            final_R = self.continuous_dim + self.n_layers * self.point_dim
            final_C = self.temp_dim + 1
            h_final = h.view(B, final_R * final_C, T)
            final_in_list = [h_final]
        else:
            h_final = h.view(B, self.continuous_dim * self.temp_dim, T)
            point_final = prev_pointwise[-1].squeeze(2)
            final_in_list = [h_final, point_final]

        if s_expanded is not None and self.static_dim > 0:
            final_in_list.append(s_expanded)
        if t_expanded is not None and self.time_dim > 0:
            final_in_list.append(t_expanded)
        final_in = (
            torch.cat(final_in_list, dim=1) if len(final_in_list) > 1 else h_final
        )

        # Pointwise conv -> BatchNorm -> ReLU -> Dropout
        feat = self.head_dropout(F.relu(self.head_bn1(self.head_conv1(final_in))))
        los_out = self.head_fc(feat).transpose(1, 2)  # (B, T, 1)

        if self.mode == "binary":
            y_prob = torch.sigmoid(los_out)
        elif self.no_exp:
            y_prob = F.softplus(los_out) + self.floor_days
        else:
            # Smoothly bound log-predictions to (log_floor, max_log) before exp()
            bounded_log = (
                self.log_floor
                + F.softplus(los_out - self.log_floor)
                - F.softplus(los_out - self.max_log)
            )
            y_prob = torch.exp(bounded_log)

        raw_mask = kwargs.get(self.mask_key)
        mask_tensor = (
            raw_mask.to(device=device) if isinstance(raw_mask, torch.Tensor) else None
        )

        # Stay-level pooling for visit-level classification/regression tasks
        batch_idx = torch.arange(B, device=device)
        last_idx: torch.Tensor | None = None
        if not self.seq_to_seq:
            if mask_tensor is not None:
                v_mask = _align_mask(mask_tensor, B, T, device)
                lengths = v_mask.squeeze(-1).sum(dim=1).long()
                last_idx = (lengths - 1).clamp(min=0)
                los_out_eff = los_out[batch_idx, last_idx, :]  # (B, 1)
                y_prob_eff = y_prob[batch_idx, last_idx, :]  # (B, 1)
            else:
                los_out_eff = los_out[:, -1, :]  # (B, 1)
                y_prob_eff = y_prob[:, -1, :]  # (B, 1)
        else:
            los_out_eff = los_out
            y_prob_eff = y_prob

        result: dict[str, torch.Tensor] = {
            "y_prob": y_prob_eff,
            "logit": los_out_eff,
        }

        # 6. Expose evaluation mask for sequence-level tasks
        m_eval: torch.Tensor | None = None
        if self.seq_to_seq and mask_tensor is not None:
            v_mask = _align_mask(mask_tensor, B, T, device)
            if self.min_time > 0 and T > self.min_time:
                w_mask = torch.zeros_like(v_mask, dtype=torch.bool)
                w_mask[:, self.min_time :, :] = True
                m_eval = v_mask & w_mask
                if m_eval.sum() == 0 and v_mask.sum() > 0:
                    m_eval = v_mask
            else:
                m_eval = v_mask
            result["mask"] = m_eval

        # 7. Dedicated auxiliary classification head (e.g. in-hospital mortality)
        aux_out: torch.Tensor | None = None
        if self.has_aux_task and hasattr(self, "aux_fc"):
            aux_feat = self.aux_dropout(F.relu(self.aux_bn1(self.aux_conv1(final_in))))
            aux_out_seq = self.aux_fc(aux_feat).transpose(1, 2)  # (B, T, 1) logits
            aux_prob_seq = torch.sigmoid(aux_out_seq)
            if not self.seq_to_seq:
                if last_idx is not None:
                    aux_out_val = aux_out_seq[batch_idx, last_idx, :]  # (B, 1)
                    aux_prob_val = aux_prob_seq[batch_idx, last_idx, :]  # (B, 1)
                else:
                    aux_out_val = aux_out_seq[:, -1, :]  # (B, 1)
                    aux_prob_val = aux_prob_seq[:, -1, :]  # (B, 1)
            else:
                aux_out_val = aux_out_seq
                aux_prob_val = aux_prob_seq
            aux_out = aux_out_val
            result["aux_prob"] = aux_prob_val
            result["aux_logit"] = aux_out_val

        # 8. Primary Loss Calculation
        raw_y_true = kwargs.get(self.label_key)
        if raw_y_true is None and "y_true" in kwargs:
            raw_y_true = kwargs["y_true"]

        if isinstance(raw_y_true, torch.Tensor):
            y_true_tensor = raw_y_true.to(device=device, dtype=torch.float32)
            if not self.seq_to_seq:
                if y_true_tensor.dim() >= 2 and y_true_tensor.shape[1] == T:
                    if last_idx is not None:
                        y_true_tensor = y_true_tensor[batch_idx, last_idx].view(B, 1)
                    else:
                        y_true_tensor = y_true_tensor[:, -1].view(B, 1)
                elif y_true_tensor.dim() == 1 or (
                    y_true_tensor.dim() == 2 and y_true_tensor.shape[1] == 1
                ):
                    y_true_tensor = y_true_tensor.view(B, 1)
            result["y_true"] = y_true_tensor
            primary_loss = self._compute_loss(
                y_prob_eff,
                y_true_tensor,
                mask=mask_tensor,
                logit=los_out_eff,
            )
            result["loss"] = primary_loss

            if (
                self.has_aux_task
                and self.aux_label_key is not None
                and aux_out is not None
            ):
                raw_aux = kwargs.get(self.aux_label_key)
                if isinstance(raw_aux, torch.Tensor):
                    if not self.seq_to_seq:
                        aux_true = raw_aux.to(device=device, dtype=torch.float32).view(
                            B, 1
                        )
                        aux_loss = F.binary_cross_entropy_with_logits(aux_out, aux_true)
                    else:
                        aux_true = _align_target(raw_aux, B, T, device)
                        a_eval = (
                            m_eval
                            if m_eval is not None
                            else _align_mask(None, B, T, device)
                        )
                        if a_eval.sum() > 0:
                            aux_loss = F.binary_cross_entropy_with_logits(
                                aux_out[a_eval], aux_true[a_eval]
                            )
                        else:
                            aux_loss = F.binary_cross_entropy_with_logits(
                                aux_out, aux_true
                            )
                    result["aux_loss"] = aux_loss
                    result["loss"] = primary_loss + self.aux_weight * aux_loss

        return result


__all__ = ["TPC", "MyBatchNorm1d", "TPCBlock"]
