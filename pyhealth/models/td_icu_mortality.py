"""Temporal-difference ICU mortality prediction model.

This module implements the model from Frost et al., "Robust Real-Time
Mortality Prediction in the Intensive Care Unit using Temporal Difference
Learning" (arXiv:2411.04285), adapted to the PyHealth framework, plus an
extension that uses Monte Carlo dropout to attach a per-patient confidence
estimate to each mortality prediction.

The approach frames ICU mortality prediction as a value-estimation problem:
at each observation step the model predicts the probability of death within
a fixed horizon (e.g. 28 days), and is trained with a temporal-difference
(TD) target derived from a lagged copy of itself (the "target network").
This yields predictions that are calibrated across horizons and robust to
long sparse observation streams.

Two classes are provided:

* ``CNNLSTMPredictor``: the underlying CNN + LSTM architecture that maps an
  event stream to a mortality probability. Can be used on its own for
  supervised training.
* ``TDICUMortalityModel``: a PyHealth ``BaseModel`` that wraps two
  ``CNNLSTMPredictor`` instances (online + target) and implements the TD
  training rule, including the terminal-state handling from the paper.
  At inference time, ``predict_with_confidence`` returns MC-dropout
  uncertainty alongside each mortality prediction.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel


# -----------------------------------------------------------------------------
# Helper modules
# -----------------------------------------------------------------------------


class MaxPool1D(nn.Module):
    """NaN-aware 1D max pool for irregular event streams.

    Windows that are entirely NaN remain NaN in the output. Windows that
    contain at least one real value output the max of those real values.
    Padded positions can therefore be tracked across pooling layers.

    Args:
        kernel_size: Pooling window size.
        stride: Pooling stride.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run NaN-aware max pooling.

        Args:
            x: Input tensor of shape ``[batch, channels, seq_len]``.

        Returns:
            Pooled tensor with NaN preserved in all-NaN windows.
        """
        neg_inf = torch.tensor(-np.inf, dtype=x.dtype, device=x.device)
        pos_nan = torch.tensor(np.nan, dtype=x.dtype, device=x.device)
        out = torch.where(torch.isnan(x), neg_inf, x)
        out = F.max_pool1d(out, kernel_size=self.kernel_size, stride=self.stride)
        return torch.where(torch.isinf(out), pos_nan, out)


class Transpose(nn.Module):
    """Swap two tensor dimensions inside an ``nn.Sequential`` pipeline.

    Args:
        dim1: First dimension to swap.
        dim2: Second dimension to swap.
    """

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose the configured dimensions.

        Args:
            x: Input tensor.

        Returns:
            Tensor with ``dim1`` and ``dim2`` swapped.
        """
        return x.transpose(self.dim1, self.dim2)


# -----------------------------------------------------------------------------
# CNN + LSTM backbone
# -----------------------------------------------------------------------------


class CNNLSTMPredictor(nn.Module):
    """CNN + LSTM encoder producing mortality predictions.

    The predictor embeds each component of an irregular event stream
    (timepoint, value, feature id, delta-time, delta-value), fuses them via
    summation + batchnorm, applies a small CNN stack for sequence-length
    reduction, a 2-layer LSTM for temporal modelling, and a dense head for
    binary classification.

    Args:
        n_features: Size of the feature vocabulary.
        features: Ordered list of feature names used to assemble the
            scaling buffers from ``scaling``.
        output_dim: Output dimensionality (always 1 for binary mortality).
        scaling: Dictionary of per-feature statistics with the structure
            ``{"mean": {...}, "std": {...}}`` where each inner dict contains
            per-feature 1-element tensors for ``values``, ``delta_time``,
            and ``delta_value``, plus scalar tensors for ``timepoints``.
        cnn_layers: Number of CNN blocks (each applies a conv + ReLU + pool).
        hidden_dim: Channel/embedding dimension of the encoder.
        dropout: Dropout applied between LSTM layers.
        batch_first: Whether the LSTM consumes batch-first tensors.
        dtype: Floating point dtype for model parameters.
        device: Device string, used to choose between packed vs masked LSTM.

    Attributes:
        embedding_net: ``nn.ModuleDict`` of five per-component embeddings.
        cnn: CNN stack that reduces sequence length.
        lstm: Two-layer LSTM with hidden size ``hidden_dim * 8``.
        dense: Binary prediction head.
    """

    def __init__(
        self,
        n_features: int,
        features: List[str],
        output_dim: int,
        scaling: Mapping[str, Any],
        cnn_layers: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device_name = device
        self.n_features = n_features
        self.features = features
        self.output_dim = output_dim
        self.cnn_layers = cnn_layers
        self.hidden_dim = hidden_dim

        self._register_scaling_buffers(scaling)
        self._build_embeddings()
        self._build_cnn()
        self._build_lstm(dropout=dropout, batch_first=batch_first)
        self._build_head()

        self.to(device)
        self.init_weights()

    # -- construction helpers -------------------------------------------------

    def _register_scaling_buffers(
        self,
        scaling: Mapping[str, Any],
    ) -> None:
        """Register per-feature mean/std tensors as buffers."""
        for name in ["values", "delta_time", "delta_value"]:
            mean_t = torch.cat([scaling["mean"][name][f] for f in self.features])
            std_t = torch.cat([scaling["std"][name][f] for f in self.features])
            self.register_buffer(f"mean_{name}", mean_t)
            self.register_buffer(f"std_{name}", std_t)
        self.register_buffer("mean_timepoints", scaling["mean"]["timepoints"])
        self.register_buffer("std_timepoints", scaling["std"]["timepoints"])

    def _build_embeddings(self) -> None:
        """Build per-component embedding networks."""
        interim = int(np.sqrt(self.hidden_dim))
        self.embedding_net = nn.ModuleDict()
        for name in ["time", "value", "feature", "delta_time", "delta_value"]:
            if name == "feature":
                self.embedding_net[name] = nn.Embedding(
                    self.n_features, self.hidden_dim, dtype=self.dtype
                )
            else:
                self.embedding_net[name] = nn.Sequential(
                    nn.Linear(1, interim, dtype=self.dtype),
                    nn.ReLU(),
                    nn.Linear(interim, self.hidden_dim, dtype=self.dtype),
                )
        self.embedding_norm = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(self.hidden_dim),
            Transpose(1, 2),
        )

    def _build_cnn(self) -> None:
        """Build the CNN stack used after embedding fusion."""
        layers: List[nn.Module] = [Transpose(1, 2)]
        for _ in range(self.cnn_layers):
            layers += [
                nn.Conv1d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=2,
                    stride=1,
                    padding=1,
                    dtype=self.dtype,
                ),
                nn.ReLU(),
                MaxPool1D(kernel_size=2, stride=2),
            ]
        layers.append(Transpose(1, 2))
        self.cnn = nn.Sequential(*layers)

    def _build_lstm(self, dropout: float, batch_first: bool) -> None:
        """Build the stacked LSTM."""
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim * 8,
            num_layers=2,
            dropout=dropout,
            batch_first=batch_first,
            dtype=self.dtype,
        )

    def _build_head(self) -> None:
        """Build the dense classification head."""
        self.dense = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 8, dtype=self.dtype),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim, dtype=self.dtype),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim, dtype=self.dtype),
            nn.Linear(self.hidden_dim, self.output_dim, dtype=self.dtype),
        )

    def init_weights(self) -> None:
        """Xavier-normal initialization for matrix-shaped parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    # -- inference helpers ----------------------------------------------------

    @torch.no_grad()
    def soft_update(
        self, new_model: "CNNLSTMPredictor", alpha: float = 0.99
    ) -> None:
        """Exponential-moving-average update from another predictor.

        Implements ``theta_self = alpha * theta_self + (1 - alpha) * theta_new``
        in-place, which is the target-network update rule from the paper
        (Appendix D, Eq. 7).

        Args:
            new_model: Source predictor whose weights to blend in.
            alpha: EMA coefficient in ``[0, 1]``. ``alpha=1`` leaves ``self``
                unchanged; ``alpha=0`` fully copies ``new_model`` into ``self``.
        """
        src = new_model.state_dict()
        tgt = self.state_dict()
        for k in tgt:
            tgt[k].copy_(alpha * tgt[k] + (1.0 - alpha) * src[k])

    def pack_sequences(
        self, src: torch.Tensor, mask: torch.Tensor
    ) -> torch.nn.utils.rnn.PackedSequence:
        """Pack a padded batch for efficient LSTM consumption.

        Args:
            src: Embedded sequence tensor ``[batch, seq_len, hidden]``.
            mask: Boolean padding mask ``[batch, seq_len]`` where ``True``
                marks padded positions.

        Returns:
            A ``PackedSequence`` usable by ``self.lstm``.
        """
        lengths = (~mask).sum(-1)
        if lengths.min() == 0:
            mask = mask.clone()
            mask[torch.where(lengths == 0)[0], 0] = False
            lengths = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        return pack_padded_sequence(
            src, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

    def get_mask_after_conv(self, mask: torch.Tensor) -> torch.Tensor:
        """Propagate a padding mask through the CNN pooling stack.

        Args:
            mask: Boolean mask ``[batch, seq_len]`` aligned with the input.

        Returns:
            Boolean mask aligned with the post-CNN sequence length.
        """
        pooled = mask.unsqueeze(1).float()
        for _ in range(self.cnn_layers):
            pooled = F.avg_pool1d(pooled, kernel_size=2, stride=2)
        return pooled.squeeze(1) == 1

    def standardise_inputs(
        self,
        timepoints: torch.Tensor,
        values: torch.Tensor,
        features: torch.Tensor,
        delta_time: torch.Tensor,
        delta_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply per-feature z-score standardization.

        For each of ``values``, ``delta_time`` and ``delta_value``, scatter
        the event's value into a feature-indexed long tensor, apply per-
        feature mean/std, then collapse back to a dense per-event tensor
        using ``torch.nansum``. ``timepoints`` use a single global scaler.

        Args:
            timepoints: Event timepoints ``[batch, seq_len]``.
            values: Event values ``[batch, seq_len]``.
            features: Feature ids ``[batch, seq_len]`` with ``-1`` for padding.
            delta_time: Time since previous measurement of the same feature.
            delta_value: Value change since previous measurement of the same
                feature.

        Returns:
            Tuple of standardized tensors in the order
            ``(timepoints, values, delta_time, delta_value, features)``
            where ``features`` has had its ``-1`` sentinels remapped to ``0``.
        """
        safe_time_std = torch.where(
            self.std_timepoints == 0,
            torch.ones_like(self.std_timepoints),
            self.std_timepoints,
        )
        standardised = [(timepoints - self.mean_timepoints) / safe_time_std]

        features = torch.where(features == -1, 0, features)
        long_shape = values.shape[:2] + (self.n_features,)

        base_nan = torch.full(
            long_shape,
            float("nan"),
            dtype=self.dtype,
            device=values.device,
        )

        stats = {
            "values": (self.mean_values, self.std_values),
            "delta_time": (self.mean_delta_time, self.std_delta_time),
            "delta_value": (self.mean_delta_value, self.std_delta_value),
        }
        for name, vector in [
            ("values", values),
            ("delta_time", delta_time),
            ("delta_value", delta_value),
        ]:
            missing = torch.isnan(vector)
            scattered = base_nan.clone()
            scattered.scatter_(-1, features.unsqueeze(-1), vector.unsqueeze(-1))

            mean_v, std_v = stats[name]
            safe_std = torch.where(std_v == 0, torch.ones_like(std_v), std_v)
            scattered = (scattered - mean_v) / safe_std
            collapsed = torch.nansum(scattered, -1).unsqueeze(-1)
            collapsed = torch.where(
                missing.unsqueeze(-1),
                torch.tensor(float("nan"), dtype=self.dtype, device=values.device),
                collapsed,
            )
            standardised.append(collapsed)

        standardised.append(features)
        return tuple(standardised)

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        timepoints: torch.Tensor,
        values: torch.Tensor,
        features: torch.Tensor,
        delta_time: torch.Tensor,
        delta_value: torch.Tensor,
        normalise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the CNN-LSTM forward pass.

        Args:
            timepoints: ``[batch, seq_len]`` hours since intime, NaN for pad.
            values: ``[batch, seq_len]`` measurement values, NaN for pad.
            features: ``[batch, seq_len]`` long tensor of feature ids, -1 for
                pad.
            delta_time: ``[batch, seq_len]`` hours since previous measurement
                of the same feature.
            delta_value: ``[batch, seq_len]`` change in value since previous
                measurement of the same feature.
            normalise: If ``True``, apply ``standardise_inputs`` first.

        Returns:
            Tuple ``(probs, logits)`` each of shape ``[batch, output_dim]``.
            ``probs = sigmoid(logits)``.
        """
        if normalise:
            (
                timepoints,
                values,
                delta_time,
                delta_value,
                features,
            ) = self.standardise_inputs(
                timepoints, values, features, delta_time, delta_value
            )

        timepoints = timepoints.squeeze(-1) if timepoints.ndim == 3 else timepoints
        values = values.squeeze(-1) if values.ndim == 3 else values
        delta_time = delta_time.squeeze(-1) if delta_time.ndim == 3 else delta_time
        delta_value = (
            delta_value.squeeze(-1) if delta_value.ndim == 3 else delta_value
        )

        # Sort descending by timepoint; NaNs (padding) go to the end.
        argsort_idx = torch.argsort(
            torch.where(torch.isnan(timepoints), -torch.inf, timepoints),
            dim=1,
            descending=True,
        )
        timepoints = torch.gather(timepoints, 1, argsort_idx)
        values = torch.gather(values, 1, argsort_idx)
        features = torch.gather(features, 1, argsort_idx)
        delta_time = torch.gather(delta_time, 1, argsort_idx)
        delta_value = torch.gather(delta_value, 1, argsort_idx)

        src_mask = torch.isnan(timepoints)
        timepoints = torch.where(src_mask, 0, timepoints)
        values = torch.where(src_mask, 0, values)
        features = torch.where(src_mask, 0, features)

        dt_mask = torch.isnan(delta_time)
        dv_mask = torch.isnan(delta_value)
        delta_time = torch.where(dt_mask, 0, delta_time)
        delta_value = torch.where(dv_mask, 0, delta_value)

        time_emb = self.embedding_net["time"](timepoints.unsqueeze(-1))
        value_emb = self.embedding_net["value"](values.unsqueeze(-1))
        feature_emb = self.embedding_net["feature"](features)
        dt_emb = self.embedding_net["delta_time"](delta_time.unsqueeze(-1))
        dv_emb = self.embedding_net["delta_value"](delta_value.unsqueeze(-1))

        dt_emb = torch.where(dt_mask.unsqueeze(-1), 0, dt_emb)
        dv_emb = torch.where(dv_mask.unsqueeze(-1), 0, dv_emb)

        embedded = time_emb + value_emb + feature_emb + dt_emb + dv_emb
        embedded = self.embedding_norm(embedded)
        embedded = self.cnn(embedded)

        src_mask = self.get_mask_after_conv(src_mask)

        if self.device_name == "cuda":
            packed = self.pack_sequences(embedded, src_mask)
            embedded = self.lstm(packed)[1][0][-1]
        else:
            embedded = embedded.clone()
            embedded[src_mask] = 0
            embedded = self.lstm(embedded)[1][0][-1]

        logits = self.dense(embedded)
        probs = torch.sigmoid(logits)
        return probs, logits


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class WeightedBCELoss(nn.Module):
    """Thin wrapper around ``BCEWithLogitsLoss`` with optional pos-weighting.

    Args:
        pos_weight: Optional positive-class weight tensor. When ``None``,
            a standard unweighted BCE is used. Only set this for supervised
            training, never for TD training: a weighted loss produces
            incorrect gradients when the target is a continuous bootstrapped
            value rather than a 0/1 label.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss from logits.

        Args:
            logits: Raw model output ``[batch, 1]``.
            targets: Target values in ``[0, 1]`` (can be continuous for TD).

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(logits, targets.float())


# -----------------------------------------------------------------------------
# PyHealth-facing TD model
# -----------------------------------------------------------------------------


class TDICUMortalityModel(BaseModel):
    """Temporal-difference ICU mortality prediction model.

    Implements the method of Frost et al. (2024), arXiv:2411.04285, for
    real-time mortality prediction in the ICU. The model holds two
    ``CNNLSTMPredictor`` instances:

    * ``online_net``: updated by gradient descent on a TD target.
    * ``target_net``: an EMA-lagged copy of the online net whose predictions
      form part of the TD target.

    At each training step, for a sample transition ``(s_t, s_{t+1})``, the
    training target is:

    * ``target = label`` if ``s_{t+1}`` does not exist (terminal transition);
    * ``target = gamma * target_net(s_{t+1})`` otherwise.

    The online net is regressed against this target with an unweighted binary
    cross-entropy loss. The target net is updated once per optimizer step via
    ``soft_update_target()``.

    The model can also be used in a purely supervised mode
    (``train_td=False``) for baseline comparisons; in this mode the real
    ``label_key`` is used as the target and ``pos_weight`` (if provided) is
    applied.

    Example:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> from pyhealth.models.td_icu_mortality import TDICUMortalityModel
        >>> dataset = SampleEHRDataset(samples=[], dataset_name="td_icu_demo")
        >>> scaling = build_scaling_dict(...)  # see ``examples/`` directory
        >>> feature_names = [f"feat_{i}" for i in range(128)]
        >>> model = TDICUMortalityModel(
        ...     dataset=dataset,
        ...     feature_keys=["timepoints", "values", "features",
        ...                   "delta_time", "delta_value"],
        ...     label_key="28_day_died",
        ...     n_features=128,
        ...     scaling=scaling,
        ...     features_vocab=feature_names,
        ... )
        >>> out = model(batch, targets=targets, train_td=True)
        >>> out["loss"].backward()
        >>> model.soft_update_target()

    Args:
        dataset: A ``SampleEHRDataset`` instance (can be empty; only used for
            metadata such as ``input_schema`` / ``output_schema``).
        feature_keys: The five keys in the batch dict that hold the event
            tuple components, in the order
            ``[timepoints, values, features, delta_time, delta_value]``.
        label_key: Key in the ``targets`` dict holding the real outcome
            (e.g. ``"28_day_died"``).
        mode: PyHealth task mode. Only ``"binary"`` is currently supported.
        n_features: Size of the feature vocabulary.
        hidden_dim: Hidden size for the encoder and LSTM.
        cnn_layers: Number of CNN pooling blocks.
        dropout: Dropout between LSTM layers.
        output_dim: Output dimensionality (must be 1 for binary mortality).
        scaling: Per-feature mean/std dictionary (see ``CNNLSTMPredictor``).
        features_vocab: Ordered list of feature names keyed by ``scaling``.
        td_alpha: Target-network EMA coefficient. Paper uses 0.99.
        gamma: Discount factor on the bootstrapped next-state value. Default
            1.0 matches the paper.
        pos_weight: Optional positive-class weight for supervised mode only.
        device: Device string passed to the underlying predictors.

    Raises:
        ValueError: If ``mode`` is not ``"binary"``.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "binary",
        n_features: int = 128,
        hidden_dim: int = 32,
        cnn_layers: int = 2,
        dropout: float = 0.5,
        output_dim: int = 1,
        scaling: Optional[Mapping[str, Any]] = None,
        features_vocab: Optional[List[str]] = None,
        td_alpha: float = 0.99,
        gamma: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> None:
        if mode != "binary":
            raise ValueError(
                f"TDICUMortalityModel only supports mode='binary', got {mode!r}"
            )
        if scaling is None or features_vocab is None:
            raise ValueError(
                "scaling and features_vocab are required - see the "
                "example script for how to build them from training data."
            )

        # Adapt the SampleEHRDataset for BaseModel's expectations.
        dataset.feature_keys = feature_keys
        dataset.label_key = label_key
        dataset.mode = mode
        if not hasattr(dataset, "input_schema"):
            dataset.input_schema = {k: "sequence" for k in feature_keys}
        if not hasattr(dataset, "output_schema"):
            dataset.output_schema = {label_key: mode}

        super().__init__(dataset)

        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode
        self.device_name = device
        self.td_alpha = td_alpha
        self.gamma = gamma

        self.online_net = CNNLSTMPredictor(
            n_features=n_features,
            features=features_vocab,
            output_dim=output_dim,
            scaling=scaling,
            cnn_layers=cnn_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
        )
        self.target_net = CNNLSTMPredictor(
            n_features=n_features,
            features=features_vocab,
            output_dim=output_dim,
            scaling=scaling,
            cnn_layers=cnn_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
        )
        self.target_net.load_state_dict(deepcopy(self.online_net.state_dict()))

        # pos_weight only ever applied in supervised mode. A continuous TD
        # target combined with pos_weight would silently yield wrong grads.
        self.supervised_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.td_loss = WeightedBCELoss(pos_weight=None)

        self.to(device)

    # -- PyHealth BaseModel abstract methods ---------------------------------

    def prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Format labels into the BCE-compatible shape.

        Args:
            labels: Tensor of shape ``[batch]`` or ``[batch, 1]``.

        Returns:
            Float tensor of shape ``[batch, 1]``.
        """
        return labels.float().view(-1, 1)

    def get_loss_function(self) -> nn.Module:
        """Return the default (supervised) loss function.

        The TD path uses ``self.td_loss`` internally. ``get_loss_function``
        exists to satisfy PyHealth's ``BaseModel`` contract and returns the
        supervised (optionally pos-weighted) loss used when ``train_td=False``.

        Returns:
            The supervised ``WeightedBCELoss`` instance.
        """
        return self.supervised_loss

    # -- TD-specific helpers --------------------------------------------------

    def predict_current(
        self, batch: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the online net on the current-state window.

        Args:
            batch: Mapping with the five feature keys listed in
                ``self.feature_keys``.

        Returns:
            Tuple ``(probs, logits)``.
        """
        return self.online_net(
            batch["timepoints"],
            batch["values"],
            batch["features"],
            batch["delta_time"],
            batch["delta_value"],
        )

    @torch.no_grad()
    def predict_next_target(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Run the target net on the next-state window.

        The target net is placed in eval mode so BatchNorm stats are frozen
        and LSTM dropout is inactive. The output carries no gradient.

        Args:
            batch: Mapping with the five ``next_*`` keys
                (``next_timepoints``, ``next_values``, ``next_features``,
                ``next_delta_time``, ``next_delta_value``).

        Returns:
            Detached probabilities of shape ``[batch, output_dim]``.
        """
        self.target_net.eval()
        probs, _ = self.target_net(
            batch["next_timepoints"],
            batch["next_values"],
            batch["next_features"],
            batch["next_delta_time"],
            batch["next_delta_value"],
        )
        return probs.detach()

    def compute_td_target(
        self,
        batch: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the TD target for each transition in the batch.

        At terminal transitions (``isterminal > 0.5``) the target is the real
        mortality label; at non-terminal transitions it is
        ``gamma * target_net(next_state)``.

        Args:
            batch: Mapping containing at minimum ``isterminal`` and the five
                ``next_*`` keys.
            targets: Mapping containing ``self.label_key``.

        Returns:
            Detached target tensor of shape ``[batch, output_dim]``.
        """
        next_probs = self.predict_next_target(batch)
        real_reward = targets[self.label_key].float().view_as(next_probs)
        is_terminal = (batch["isterminal"] > 0.5).view_as(next_probs)
        td_target = torch.where(is_terminal, real_reward, self.gamma * next_probs)
        return td_target.detach()

    def soft_update_target(self) -> None:
        """Update the target net with an EMA step from the online net.

        Uses ``self.td_alpha``. Call exactly once after each optimizer step.
        """
        self.target_net.soft_update(self.online_net, alpha=self.td_alpha)

    # -- Forward pass ---------------------------------------------------------

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        targets: Optional[Mapping[str, torch.Tensor]] = None,
        train_td: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run the model.

        Args:
            batch: Mapping with ``self.feature_keys`` and (for TD training)
                also ``isterminal`` and the five ``next_*`` keys.
            targets: Optional mapping with ``self.label_key``. If omitted,
                no loss is computed.
            train_td: Whether to use the temporal-difference loss. When
                ``True``, the TD target is computed via the target net;
                when ``False``, the real label is used directly.

        Returns:
            A dict with keys:

            * ``loss``: scalar loss tensor or ``None`` if ``targets is None``.
            * ``y_prob``: ``[batch, output_dim]`` probabilities from the
              online net.
            * ``y_true``: the supplied label tensor, or ``None``.
            * ``logit``: ``[batch, output_dim]`` raw model output.
        """
        probs, logits = self.predict_current(batch)

        out: Dict[str, Optional[torch.Tensor]] = {
            "loss": None,
            "y_prob": probs,
            "y_true": targets[self.label_key] if targets is not None else None,
            "logit": logits,
        }

        if targets is not None:
            if train_td:
                td_target = self.compute_td_target(batch, targets)
                out["loss"] = self.td_loss(logits, td_target)
            else:
                supervised_target = self.prepare_labels(targets[self.label_key])
                out["loss"] = self.supervised_loss(logits, supervised_target)

        return out

    # -- Monte Carlo dropout for uncertainty quantification ------------------

    @staticmethod
    def _enable_dropout(module: nn.Module) -> None:
        """Put only ``Dropout`` and ``LSTM`` layers into train mode.

        PyTorch's ``LSTM`` applies dropout between stacked layers only when
        the module is in train mode, so we flip both ``nn.Dropout*`` and
        ``nn.LSTM`` modules to train while leaving BatchNorm etc. in eval.

        Args:
            module: Any ``nn.Module`` whose dropout we want to activate.
        """
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.LSTM)):
                m.train()

    @torch.no_grad()
    def predict_with_confidence(
        self,
        batch: Mapping[str, torch.Tensor],
        n_mc_samples: int = 30,
        high_conf_threshold: float = 0.005,
        low_conf_threshold: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """Return mortality predictions with per-patient confidence.

        Runs the online network ``n_mc_samples`` times with dropout active
        (BatchNorm frozen in eval), then reports the MC mean, standard
        deviation, and a ~95% credible interval per sample. Two boolean
        flags classify each prediction as high / low confidence relative
        to user-tunable thresholds on the MC standard deviation.

        This is the paper's base predictor extended with Monte Carlo
        dropout, providing a per-patient uncertainty estimate alongside
        the point mortality probability. Empirically, higher MC standard
        deviation correlates with higher true mortality rate, so the
        confidence score also serves as an auxiliary clinical-triage
        signal.

        Example:
            >>> out = model.predict_with_confidence(batch, n_mc_samples=30)
            >>> mortality = out["mortality_prob"]        # shape [batch]
            >>> ci_lo     = out["ci_95_lower"]           # shape [batch]
            >>> ci_hi     = out["ci_95_upper"]           # shape [batch]
            >>> uncertain = out["is_low_confidence"]     # bool tensor

        Args:
            batch: Mapping with ``self.feature_keys`` (the five event
                tuple components). ``next_*`` keys are not required since
                only the online network is sampled.
            n_mc_samples: Number of stochastic forward passes. More passes
                tighten the uncertainty estimate at linear inference cost.
                30 is a reasonable default; 50-100 for research-grade.
            high_conf_threshold: Predictions with MC standard deviation
                below this value are flagged "high confidence".
            low_conf_threshold: Predictions with MC standard deviation
                above this value are flagged "low confidence".

        Returns:
            Dict with tensor entries all shaped ``[batch]``:

            * ``mortality_prob``: MC-averaged probability.
            * ``confidence_std``: standard deviation across MC samples.
            * ``ci_95_lower``: ``clip(mean - 2*std, 0, 1)``.
            * ``ci_95_upper``: ``clip(mean + 2*std, 0, 1)``.
            * ``is_high_confidence``: bool, std < ``high_conf_threshold``.
            * ``is_low_confidence``: bool, std > ``low_conf_threshold``.
        """
        self.online_net.eval()
        self._enable_dropout(self.online_net)

        probs_mc: List[torch.Tensor] = []
        for _ in range(n_mc_samples):
            probs, _ = self.online_net(
                batch["timepoints"],
                batch["values"],
                batch["features"],
                batch["delta_time"],
                batch["delta_value"],
            )
            probs_mc.append(probs.float().squeeze(-1))

        stacked = torch.stack(probs_mc, dim=1)  # [batch, n_mc_samples]
        mean_prob = stacked.mean(dim=1)
        std_prob = stacked.std(dim=1)

        ci_lower = (mean_prob - 2.0 * std_prob).clamp(0.0, 1.0)
        ci_upper = (mean_prob + 2.0 * std_prob).clamp(0.0, 1.0)

        return {
            "mortality_prob": mean_prob,
            "confidence_std": std_prob,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "is_high_confidence": std_prob < high_conf_threshold,
            "is_low_confidence": std_prob > low_conf_threshold,
        }
