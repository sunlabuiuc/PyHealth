"""
Temporal Pointwise Convolution (TPC) Model for ICU Length-of-Stay Prediction

Contributors:
    - [TODO: Add your name(s)]
    - [TODO: Add your NetID(s)]

Paper:
    Title: Temporal Pointwise Convolutional Networks for Length of Stay 
           Prediction in the Intensive Care Unit
    Authors: Emma Rocheteau, Pietro Liò, Stephanie Hyland
    Conference: CHIL 2021 (Conference on Health, Inference, and Learning)
    Link: https://arxiv.org/abs/2007.09483
    
Description:
    Implementation of the TPC model which combines grouped temporal convolutions
    with pointwise (1x1) convolutions for irregularly sampled multivariate time
    series in ICU settings. The model predicts remaining length of stay at hourly
    intervals throughout ICU admission.
    
    Novel Extension: Monte Carlo Dropout uncertainty estimation for predictive
    confidence intervals (not in original paper).

Usage:
    >>> from pyhealth.models import TPC
    >>> from pyhealth.datasets import MIMIC4EHRDataset
    >>> from pyhealth.tasks import RemainingLOSMIMIC4
    >>> 
    >>> dataset = mimic4.set_task(RemainingLOSMIMIC4())
    >>> model = TPC(dataset=dataset, n_layers=3, use_msle=True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel


class MSLELoss(nn.Module):
    """Masked mean squared logarithmic error loss used in the TPC paper.

    Computes the mean MSLE over valid (non-zero) timesteps per sequence,
    then averages across the batch.

    Args:
        None
    """

    def __init__(self) -> None:
        super().__init__()
        self.squared_error = nn.MSELoss(reduction="none")

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        seq_length: torch.Tensor,
        sum_losses: bool = False,
    ) -> torch.Tensor:
        """Compute masked MSLE.

        Args:
            y_hat: Predicted values of shape (B, T).
            y: True values of shape (B, T).
            mask: Boolean mask of shape (B, T); True where loss should be computed.
            seq_length: Valid sequence lengths of shape (B,).
            sum_losses: If True, sum per-sequence losses instead of averaging.

        Returns:
            Scalar loss tensor.
        """
        mask = mask.bool()
        eps = 1e-8
        log_y_hat = torch.where(mask, torch.log(y_hat.clamp_min(eps)), torch.zeros_like(y_hat))
        log_y = torch.where(mask, torch.log(y.clamp_min(eps)), torch.zeros_like(y))
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1).float()
        return loss.mean()


class MaskedMSELoss(nn.Module):
    """Masked mean squared error loss.

    Args:
        None
    """

    def __init__(self) -> None:
        super().__init__()
        self.squared_error = nn.MSELoss(reduction="none")

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        seq_length: torch.Tensor,
        sum_losses: bool = False,
    ) -> torch.Tensor:
        """Compute masked MSE.

        Args:
            y_hat: Predicted values of shape (B, T).
            y: True values of shape (B, T).
            mask: Boolean mask of shape (B, T); True where loss should be computed.
            seq_length: Valid sequence lengths of shape (B,).
            sum_losses: If True, sum per-sequence losses instead of averaging.

        Returns:
            Scalar loss tensor.
        """
        mask = mask.bool()
        y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y = torch.where(mask, y, torch.zeros_like(y))
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1).float()
        return loss.mean()


class TPC(BaseModel):
    """Temporal Pointwise Convolution (TPC) model for ICU remaining length-of-stay.

    Adapted from the official TPC-LoS-prediction repository (Rocheteau et al., 2021)
    and rewritten to fit PyHealth's BaseModel interface and the RemainingLOSMIMIC4 task.

    The model applies grouped temporal convolutions (one group per clinical feature)
    in parallel with pointwise (1x1) convolutions that look across all features at
    each timestep. Skip connections are maintained throughout, and the architecture
    stacks ``n_layers`` of these paired conv blocks before a final linear regression
    head.

    Expected task input keys
    ------------------------
    - ``timeseries``: Float tensor of shape ``(B, 2F+2, T)`` with channel order
      ``[elapsed, values(F), decay(F), hour_of_day]`` — channels first, as produced
      by ``RemainingLOSMIMIC4``.
    - ``static``: Float tensor of shape ``(B, S)`` containing age, sex, etc.
    - ``conditions``: Long tensor of shape ``(B, L)`` containing padded ICD code
      indices from ``SequenceProcessor`` (PAD index = 0).
    - ``los``: Float tensor of shape ``(B, T)`` with remaining LoS in hours at each hour.

    Args:
        dataset (SampleDataset): Dataset used to train the model. Provides feature
            and label key lists as well as fitted input processors.
        timeseries_key (str): Key for the timeseries input. Default: ``"timeseries"``.
        static_key (Optional[str]): Key for static features, or ``None`` to omit.
            Default: ``"static"``.
        conditions_key (Optional[str]): Key for diagnosis code sequences, or ``None``
            to omit. Default: ``"conditions"``.
        n_layers (int): Number of TPC layers. Default: ``3``.
        kernel_size (int): Temporal convolution kernel size. Default: ``4``.
        temp_kernels (Optional[Sequence[int]]): Number of temporal kernels per layer.
            Length must equal ``n_layers``. Default: ``[8] * n_layers``.
        point_sizes (Optional[Sequence[int]]): Pointwise output size per layer.
            Length must equal ``n_layers``. Default: ``[14] * n_layers``.
        diagnosis_size (int): Output dimension of the diagnosis encoder. Default: ``64``.
        last_linear_size (int): Hidden size of the final regression MLP. Default: ``64``.
        main_dropout_rate (float): Dropout applied to pointwise outputs and the final
            head. Default: ``0.3``.
        temp_dropout_rate (float): Dropout applied to temporal conv outputs.
            Default: ``0.3``.
        time_before_pred (int): Minimum history hours before predictions are made.
            Predictions are generated for timesteps ``[time_before_pred, T)``.
            Default: ``5``.
        use_msle (bool): Use MSLE loss if ``True``, MSE otherwise. Default: ``True``.
        sum_losses (bool): Sum per-sequence losses instead of averaging. Default: ``False``.
        apply_exp (bool): Exponentiate raw predictions before clamping. Since the model
            learns log-space targets this recovers hours. Default: ``True``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> import torch
        >>> F = 4  # number of clinical features
        >>> T = 20
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "timeseries": torch.randn(2 * F + 2, T),
        ...         "static": torch.tensor([65.0, 1.0]),
        ...         "conditions": ["icd_A01", "icd_B02"],
        ...         "los": torch.rand(T) * 48,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "timeseries": torch.randn(2 * F + 2, T),
        ...         "static": torch.tensor([72.0, 0.0]),
        ...         "conditions": ["icd_A01"],
        ...         "los": torch.rand(T) * 24,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "timeseries": "tensor",
        ...         "static": "tensor",
        ...         "conditions": "sequence",
        ...     },
        ...     output_schema={"los": "tensor"},
        ...     dataset_name="tpc_demo",
        ... )
        >>> model = TPC(dataset=dataset, n_layers=2, temp_kernels=[4, 4], point_sizes=[8, 8])
        >>> batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> out.keys()
        dict_keys(['logit', 'y_prob', 'loss', 'y_true'])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        timeseries_key: str = "timeseries",
        static_key: Optional[str] = "static",
        conditions_key: Optional[str] = "conditions",
        n_layers: int = 3,
        kernel_size: int = 4,
        temp_kernels: Optional[Sequence[int]] = None,
        point_sizes: Optional[Sequence[int]] = None,
        diagnosis_size: int = 64,
        last_linear_size: int = 64,
        main_dropout_rate: float = 0.3,
        temp_dropout_rate: float = 0.3,
        time_before_pred: int = 5,
        use_msle: bool = True,
        sum_losses: bool = False,
        apply_exp: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)
        # TPC manages its own loss; prevent BaseModel from resolving mode
        # from the "tensor" output schema entry (which it cannot map to a mode string).
        self.mode = "regression"

        if len(self.label_keys) != 1:
            raise ValueError("TPC supports exactly one label key.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        self.label_key = self.label_keys[0]
        self.timeseries_key = timeseries_key
        # Only register static/conditions keys if they are actually in the dataset
        self.static_key = static_key if static_key in self.feature_keys else None
        self.conditions_key = (
            conditions_key if conditions_key in self.feature_keys else None
        )

        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.temp_kernels: List[int] = (
            list(temp_kernels) if temp_kernels is not None else [8] * n_layers
        )
        self.point_sizes: List[int] = (
            list(point_sizes) if point_sizes is not None else [14] * n_layers
        )
        if len(self.temp_kernels) != n_layers:
            raise ValueError("temp_kernels must have exactly n_layers entries.")
        if len(self.point_sizes) != n_layers:
            raise ValueError("point_sizes must have exactly n_layers entries.")

        self.diagnosis_size = diagnosis_size
        self.last_linear_size = last_linear_size
        self.main_dropout_rate = main_dropout_rate
        self.temp_dropout_rate = temp_dropout_rate
        self.time_before_pred = time_before_pred
        self.use_msle = use_msle
        self.sum_losses = sum_losses
        self.apply_exp = apply_exp

        self.relu = nn.ReLU()
        # Clamp predictions to a clinically plausible range (30 min – 100 hours)
        self.hardtanh = nn.Hardtanh(min_val=1.0 / 48.0, max_val=100.0)
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)

        self.loss_fn: nn.Module = MSLELoss() if use_msle else MaskedMSELoss()

        # ------------------------------------------------------------------ #
        # Infer channel counts from the first dataset sample
        # ------------------------------------------------------------------ #
        sample = dataset[0]
        if self.timeseries_key not in sample:
            raise KeyError(
                f"timeseries_key '{self.timeseries_key}' not found in dataset sample. "
                f"Available keys: {list(sample.keys())}"
            )
        ts_sample: torch.Tensor = sample[self.timeseries_key]
        if ts_sample.dim() != 2:
            raise ValueError(
                f"Each timeseries sample must be 2-D (channels, time) or (time, channels), "
                f"got shape {tuple(ts_sample.shape)}."
            )

        # The RemainingLOSMIMIC4 task outputs (2F+2, T) — channels first.
        # We treat the smaller dimension as the channel axis when it equals 2F+2.
        # This is safe because the number of channels (70 for the default feature
        # set) is always much smaller than typical ICU stay lengths.
        num_channels = min(ts_sample.shape)
        if (num_channels - 2) % 2 != 0 or num_channels < 4:
            raise ValueError(
                "timeseries channel dimension must equal 2F+2 with F >= 1. "
                f"Detected smallest dim = {num_channels}."
            )
        self.F: int = (num_channels - 2) // 2

        # ------------------------------------------------------------------ #
        # Static feature size
        # ------------------------------------------------------------------ #
        self.no_flat_features: int = 0
        if self.static_key is not None:
            static_sample: torch.Tensor = sample[self.static_key]
            self.no_flat_features = (
                1 if static_sample.dim() == 0 else int(static_sample.shape[-1])
            )

        # ------------------------------------------------------------------ #
        # Diagnosis encoder (optional)
        # ------------------------------------------------------------------ #
        # D=0 means conditions are not used; we skip the encoder entirely and
        # substitute a zero vector at forward time to avoid Linear(0, ...).
        self.D: int = 0
        self.diagnosis_encoder: Optional[nn.Linear] = None
        self.bn_diagnosis_encoder: Optional[nn.BatchNorm1d] = None

        if self.conditions_key is not None:
            self.D = self.dataset.input_processors[self.conditions_key].size()
            self.diagnosis_encoder = nn.Linear(self.D, self.diagnosis_size)
            self.bn_diagnosis_encoder = nn.BatchNorm1d(self.diagnosis_size)

        # ------------------------------------------------------------------ #
        # Build TPC layers and final regression head
        # ------------------------------------------------------------------ #
        self.bn_point_last_los = nn.BatchNorm1d(self.last_linear_size)
        self._init_tpc()
        self.point_final_los = nn.Linear(self.last_linear_size, 1)

    # ---------------------------------------------------------------------- #
    # Architecture construction helpers
    # ---------------------------------------------------------------------- #

    def _init_tpc(self) -> None:
        """Build per-layer metadata and instantiate all conv/linear modules."""
        self._layer_info: List[Dict[str, Any]] = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1
            padding = [(self.kernel_size - 1) * dilation, 0]
            self._layer_info.append(
                {
                    "temp_kernels": self.temp_kernels[i],
                    "point_size": self.point_sizes[i],
                    "dilation": dilation,
                    "padding": padding,
                    "stride": 1,
                }
            )

        self._create_temp_pointwise_layers()

        # Final projection input: concatenation of [static, diagnosis_enc, last next_X slice]
        input_size = (
            (self.F + self._Zt) * (1 + self._Y)
            + self.diagnosis_size
            + self.no_flat_features
        )
        self.point_last_los = nn.Linear(input_size, self.last_linear_size)

    def _create_temp_pointwise_layers(self) -> None:
        """Instantiate grouped temporal conv and pointwise linear for each layer.

        After this call the following attributes are set:
            _Y  : number of temporal kernels in the last layer (needed by _init_tpc)
            _Zt : total accumulated pointwise output channels (needed by _init_tpc)
        """
        self.layer_modules = nn.ModuleDict()
        Y = 0   # temporal kernels from the previous layer
        Z = 0   # pointwise output size of the previous layer
        Zt = 0  # cumulative sum of all pointwise output sizes so far

        for i in range(self.n_layers):
            # Input channels to the temporal conv:
            #   layer 0 : 2*F  (values + decay, interleaved by feature)
            #   layer i : (F + Zt) * (1 + Y)  (feature groups × kernel fan-out)
            temp_in = (self.F + Zt) * (1 + Y) if i > 0 else 2 * self.F

            # Output channels: one group of temp_kernels[i] filters per feature group
            temp_out = (self.F + Zt) * self.temp_kernels[i]

            # Input to pointwise linear:
            #   prev_temp  : (F + Zt - Z) * Y  (all-but-last-layer skip streams)
            #   prev_point :  Z                 (previous pointwise output)
            #   X_orig     : 2*F + 2            (raw channels from input)
            #   flat       : no_flat_features
            point_in = (
                (self.F + Zt - Z) * Y
                + Z
                + 2 * self.F
                + 2
                + self.no_flat_features
            )
            point_out = self.point_sizes[i]

            self.layer_modules[str(i)] = nn.ModuleDict(
                {
                    "temp": nn.Conv1d(
                        in_channels=temp_in,
                        out_channels=temp_out,
                        kernel_size=self.kernel_size,
                        stride=self._layer_info[i]["stride"],
                        dilation=self._layer_info[i]["dilation"],
                        groups=self.F + Zt,
                    ),
                    "bn_temp": nn.BatchNorm1d(temp_out),
                    "point": nn.Linear(point_in, point_out),
                    "bn_point": nn.BatchNorm1d(point_out),
                }
            )

            Y = self.temp_kernels[i]
            Z = point_out
            Zt += Z

        self._Y = Y
        self._Zt = Zt

    # ---------------------------------------------------------------------- #
    # Input preparation helpers
    # ---------------------------------------------------------------------- #

    def _normalize_timeseries(self, x: torch.Tensor) -> torch.Tensor:
        """Return timeseries in channels-first layout ``(B, 2F+2, T)``.

        The task always produces ``(2F+2, T)`` per sample, so after batching
        the input is ``(B, 2F+2, T)``. We identify the channel axis as the
        dimension whose size equals ``2F+2``; if neither does we raise.

        Args:
            x: Raw batched timeseries tensor.

        Returns:
            Tensor of shape ``(B, 2F+2, T)``.

        Raises:
            ValueError: If the tensor is not 3-D or the channel dimension cannot
                be identified.
        """
        x = x.to(self.device, dtype=torch.float32)
        if x.dim() != 3:
            raise ValueError(
                f"Expected a 3-D batched timeseries (B, C, T) or (B, T, C), "
                f"got shape {tuple(x.shape)}."
            )
        expected_c = 2 * self.F + 2
        if x.shape[1] == expected_c:
            return x  # already (B, C, T)
        if x.shape[2] == expected_c:
            return x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        raise ValueError(
            f"Cannot identify channel dimension of size {expected_c} in "
            f"timeseries shape {tuple(x.shape)}."
        )

    def _prepare_static(self, batch_size: int, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Return static features as ``(B, no_flat_features)`` or empty tensor.

        Args:
            batch_size: Number of samples in the batch.
            kwargs: Full forward-pass keyword arguments.

        Returns:
            Float tensor of shape ``(B, no_flat_features)``.
        """
        if self.static_key is None or self.no_flat_features == 0:
            return torch.zeros(batch_size, 0, device=self.device)
        flat = kwargs[self.static_key].to(self.device, dtype=torch.float32)
        if flat.dim() == 1:
            # Single feature scalar per sample — reshape to (B, 1)
            flat = flat.unsqueeze(-1)
        return flat

    def _prepare_diagnoses(self, batch_size: int, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Encode diagnosis codes into a multi-hot vector, then project.

        PAD index (0) contributions are zeroed out after scatter.
        If no conditions key is configured, returns a zero vector of size
        ``diagnosis_size``.

        Args:
            batch_size: Number of samples in the batch.
            kwargs: Full forward-pass keyword arguments.

        Returns:
            Float tensor of shape ``(B, diagnosis_size)``.
        """
        if self.conditions_key is None or self.D == 0 or self.diagnosis_encoder is None:
            return torch.zeros(batch_size, self.diagnosis_size, device=self.device)

        codes = kwargs[self.conditions_key].to(self.device)  # (B, L)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)  # handle degenerate single-sample edge case

        multi_hot = torch.zeros(batch_size, self.D, device=self.device)
        valid = codes >= 0  # mask out any -1 padding (PAD=0 handled below)
        safe_codes = codes.masked_fill(~valid, 0)
        multi_hot.scatter_add_(1, safe_codes, valid.float())
        multi_hot[:, 0] = 0.0  # zero the PAD index contribution

        diag_enc = self.relu(
            self.main_dropout(
                self.bn_diagnosis_encoder(self.diagnosis_encoder((multi_hot > 0).float()))
            )
        )
        return diag_enc

    # ---------------------------------------------------------------------- #
    # Core TPC block
    # ---------------------------------------------------------------------- #

    def _temp_pointwise(
        self,
        B: int,
        T: int,
        X: torch.Tensor,
        X_orig: torch.Tensor,
        repeat_flat: torch.Tensor,
        temp: nn.Conv1d,
        bn_temp: nn.BatchNorm1d,
        point: nn.Linear,
        bn_point: nn.BatchNorm1d,
        temp_kernels: int,
        padding: List[int],
        prev_temp: Optional[torch.Tensor],
        prev_point: Optional[torch.Tensor],
        point_skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One TPC layer: grouped temporal conv + pointwise linear + skip connections.

        The function mirrors Algorithm 1 in the paper. All tensors live on the
        model's device.

        Args:
            B: Batch size.
            T: Sequence length.
            X: Current feature tensor ``(B, C_in, T)`` fed to the temporal conv.
            X_orig: Flattened raw input ``(B*T, 2F+2)`` for pointwise concat.
            repeat_flat: Static features tiled over time ``(B*T, S)``.
            temp: Grouped ``Conv1d`` for this layer.
            bn_temp: ``BatchNorm1d`` applied after the temporal conv.
            point: ``Linear`` for the pointwise path.
            bn_point: ``BatchNorm1d`` applied after the pointwise linear.
            temp_kernels: Number of temporal kernels for this layer.
            padding: ``[left_pad, right_pad]`` for causal convolution.
            prev_temp: Temporal output from the previous layer ``(B*T, C_t)``
                or ``None`` for layer 0.
            prev_point: Pointwise output from the previous layer ``(B*T, Z)``
                or ``None`` for layer 0.
            point_skip: Accumulated skip-connection tensor ``(B, C_skip, T)``.

        Returns:
            A 4-tuple:
                - ``temp_out``: Temporal features for the next layer ``(B*T, C_skip * temp_kernels)``.
                - ``point_out``: Pointwise features for the next layer ``(B*T, point_size)``.
                - ``next_X``: Combined feature tensor for the next layer ``(B, C_next, T)``.
                - ``point_skip``: Updated skip-connection tensor ``(B, C_skip + point_size, T)``.
        """
        # ---- Temporal path ------------------------------------------------
        X_padded = F.pad(X, padding, "constant", 0)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # (B, C_out, T)
        # C_out = (F + Zt) * temp_kernels for this layer; derive feature-group
        # count directly from the conv output rather than from point_skip so
        # the reshape is always consistent regardless of update order.
        C_feat_groups = X_temp.shape[1] // temp_kernels  # = F + Zt for this layer

        # ---- Pointwise path -----------------------------------------------
        # Flatten to (B*T, ...) so Linear can process every timestep jointly.
        concat_parts: List[torch.Tensor] = []
        if prev_temp is not None:
            concat_parts.append(prev_temp)
        if prev_point is not None:
            concat_parts.append(prev_point)
        concat_parts.append(X_orig)
        if repeat_flat.shape[1] > 0:
            concat_parts.append(repeat_flat)
        X_concat = torch.cat(concat_parts, dim=1)  # (B*T, point_in)
        point_out = self.main_dropout(bn_point(point(X_concat)))  # (B*T, point_size)

        # ---- Update skip accumulator --------------------------------------
        # Must happen BEFORE the reshape below so point_skip matches C_feat_groups.
        if prev_point is not None:
            Z_prev = prev_point.shape[1]
            point_skip = torch.cat(
                [point_skip, prev_point.view(B, T, Z_prev).permute(0, 2, 1)],
                dim=1,
            )  # (B, C_feat_groups, T)

        # ---- Combine temporal and pointwise for next layer ----------------
        # Reshape temporal output to (B, C_feat_groups, temp_kernels, T) and
        # stack with the skip channel (B, C_feat_groups, 1, T) along kernel axis.
        temp_4d = X_temp.view(B, C_feat_groups, temp_kernels, T)  # (B, G, K, T)
        skip_4d = point_skip.unsqueeze(2)                          # (B, G, 1, T)
        temp_stack = torch.cat([skip_4d, temp_4d], dim=2)  # (B, G, 1+K, T)

        # Broadcast pointwise output over the kernel axis
        point_size = point_out.shape[1]
        point_4d = (
            point_out.view(B, T, point_size)
            .permute(0, 2, 1)          # (B, point_size, T)
            .unsqueeze(2)              # (B, point_size, 1, T)
            .expand(-1, -1, 1 + temp_kernels, -1)  # (B, point_size, 1+K, T)
        )

        combined = self.relu(
            torch.cat([temp_stack, point_4d], dim=1)
        )  # (B, G + point_size, 1+K, T)

        next_X = combined.view(B, (C_feat_groups + point_size) * (1 + temp_kernels), T)

        # Temporal output flattened for next-layer pointwise concat
        temp_out = (
            X_temp.permute(0, 2, 1)                   # (B, T, G*K)
            .contiguous()
            .view(B * T, C_feat_groups * temp_kernels) # (B*T, G*K)
        )

        return temp_out, point_out, next_X, point_skip

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        return_full_sequence: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the TPC model.

        Predictions are produced for timesteps ``[time_before_pred, T)``.

        Args:
            return_full_sequence: If ``True``, return the full ``(B, post_hist)``
                prediction tensor in ``y_prob`` / ``logit`` instead of the
                flat masked subset. Useful for sequence-level evaluation.
            **kwargs: Batch dictionary containing all feature and label tensors.

        Returns:
            Dictionary with the following keys:

            - ``loss``: Scalar training loss (present only when label is in kwargs).
            - ``y_prob``: Predicted LoS values.
            - ``y_true``: Ground-truth LoS values (present only when label is in kwargs).
            - ``logit``: Same as ``y_prob`` (regression task).

        Raises:
            ValueError: If the sequence length ``T <= time_before_pred``.
        """
        # ---- Prepare inputs -----------------------------------------------
        X = self._normalize_timeseries(kwargs[self.timeseries_key])  # (B, 2F+2, T)
        B, _, T = X.shape

        if T <= self.time_before_pred:
            raise ValueError(
                f"Sequence length T={T} must be greater than "
                f"time_before_pred={self.time_before_pred}."
            )

        flat = self._prepare_static(B, kwargs)            # (B, S)
        diagnoses_enc = self._prepare_diagnoses(B, kwargs)  # (B, diagnosis_size)

        # Raw per-timestep features, flattened for the pointwise path
        X_orig = X.permute(0, 2, 1).contiguous().view(B * T, 2 * self.F + 2)
        repeat_flat = flat.repeat_interleave(T, dim=0)   # (B*T, S)

        # Split elapsed and hour_of_day channels; keep only values + decay
        # X[:, 0, :] = elapsed,  X[:, 1:F+1, :] = values,
        # X[:, F+1:2F+1, :] = decay,  X[:, 2F+1, :] = hour_of_day
        values = X[:, 1 : self.F + 1, :]      # (B, F, T)
        decay = X[:, self.F + 1 : 2 * self.F + 1, :]  # (B, F, T)

        # Layer 0 temporal conv input: interleave values and decay per feature
        # to give each feature group its 2 input channels [value, decay]
        next_X = torch.stack([values, decay], dim=2).reshape(B, 2 * self.F, T)

        # Skip accumulator starts with the values channel (one channel per feature)
        point_skip = values  # (B, F, T)

        prev_temp: Optional[torch.Tensor] = None
        prev_point: Optional[torch.Tensor] = None

        # ---- TPC layers ---------------------------------------------------
        for i in range(self.n_layers):
            mods = self.layer_modules[str(i)]
            prev_temp, prev_point, next_X, point_skip = self._temp_pointwise(
                B=B,
                T=T,
                X=next_X,
                X_orig=X_orig,
                repeat_flat=repeat_flat,
                temp=mods["temp"],
                bn_temp=mods["bn_temp"],
                point=mods["point"],
                bn_point=mods["bn_point"],
                temp_kernels=self._layer_info[i]["temp_kernels"],
                padding=self._layer_info[i]["padding"],
                prev_temp=prev_temp,
                prev_point=prev_point,
                point_skip=point_skip,
            )

        # ---- Final regression head ----------------------------------------
        post_hist = T - self.time_before_pred  # number of prediction timesteps

        # Slice next_X to post-history timesteps and flatten for the linear
        ts_features = (
            next_X[:, :, self.time_before_pred :]   # (B, C, post_hist)
            .permute(0, 2, 1)                        # (B, post_hist, C)
            .contiguous()
            .view(B * post_hist, -1)                 # (B*post_hist, C)
        )

        combined_features = torch.cat(
            [
                flat.repeat_interleave(post_hist, dim=0),             # (B*post_hist, S)
                diagnoses_enc.repeat_interleave(post_hist, dim=0),    # (B*post_hist, diag_size)
                ts_features,                                           # (B*post_hist, C)
            ],
            dim=1,
        )

        last_hidden = self.relu(
            self.main_dropout(
                self.bn_point_last_los(self.point_last_los(combined_features))
            )
        )  # (B*post_hist, last_linear_size)

        raw_pred = self.point_final_los(last_hidden).view(B, post_hist)  # (B, post_hist)
        if self.apply_exp:
            raw_pred = torch.exp(raw_pred)
        los_pred = self.hardtanh(raw_pred)  # clamp to [1/48, 100] hours

        # ---- Build output dict --------------------------------------------
        output: Dict[str, torch.Tensor] = {
            "logit": los_pred if return_full_sequence else los_pred.reshape(-1),
            "y_prob": los_pred if return_full_sequence else los_pred.reshape(-1),
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device, dtype=torch.float32)
            if y_true.dim() == 3 and y_true.shape[-1] == 1:
                y_true = y_true.squeeze(-1)
            y_true_post = y_true[:, self.time_before_pred :]  # (B, post_hist)

            mask = y_true_post > 0
            seq_lengths = mask.sum(dim=1)
            loss = self.loss_fn(los_pred, y_true_post, mask, seq_lengths, self.sum_losses)

            output["loss"] = loss
            if return_full_sequence:
                output["y_true"] = y_true_post
                output["mask"] = mask
            else:
                flat_mask = mask.reshape(-1)
                output["y_true"] = los_pred.reshape(-1)[flat_mask]
                output["y_prob"] = los_pred.reshape(-1)[flat_mask]
                output["logit"] = los_pred.reshape(-1)[flat_mask]
                output["y_true"] = y_true_post.reshape(-1)[flat_mask]

        if kwargs.get("embed", False):
            # Return the mean-pooled combined feature as a patient embedding
            output["embed"] = combined_features.view(B, post_hist, -1).mean(dim=1)

        return output

    # ---------------------------------------------------------------------- #
    # Monte Carlo Dropout uncertainty estimation (ablation)
    # ---------------------------------------------------------------------- #

    def predict_with_uncertainty(
        self,
        mc_samples: int = 30,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Estimate predictive uncertainty via Monte Carlo Dropout.

        Dropout layers are kept active during inference to obtain a distribution
        over predictions. This implements the Bayesian approximation proposed by
        Gal & Ghahramani (2016) and constitutes the ablation extension in the
        project proposal.

        Args:
            mc_samples: Number of stochastic forward passes. Default: ``30``.
            **kwargs: Same keyword arguments as ``forward()``.

        Returns:
            Dictionary with keys:

            - ``mean``: Mean prediction ``(B, post_hist)``.
            - ``std``: Standard deviation ``(B, post_hist)``.
            - ``samples``: All sampled predictions ``(mc_samples, B, post_hist)``.
            - ``mask``: Valid-timestep mask (when label is present).
            - ``y_true``: Ground-truth labels (when label is present).

        Raises:
            ValueError: If ``mc_samples < 1``.
        """
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1.")

        was_training = self.training
        self.train()  # keep dropout active

        samples: List[torch.Tensor] = []
        mask: Optional[torch.Tensor] = None
        y_true: Optional[torch.Tensor] = None

        with torch.no_grad():
            for _ in range(mc_samples):
                out = self.forward(return_full_sequence=True, **kwargs)
                samples.append(out["y_prob"])
                if mask is None:
                    mask = out.get("mask")
                if y_true is None:
                    y_true = out.get("y_true")

        if not was_training:
            self.eval()

        stacked = torch.stack(samples, dim=0)  # (mc_samples, B, post_hist)
        result: Dict[str, torch.Tensor] = {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "samples": stacked,
        }
        if mask is not None:
            result["mask"] = mask
        if y_true is not None:
            result["y_true"] = y_true
        return result