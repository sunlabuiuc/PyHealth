"""Temporal-difference CNN-LSTM model for real-time ICU mortality prediction.

This module implements the network architecture of Frost et al., *Robust
Real-Time Mortality Prediction in the Intensive Care Unit using Temporal
Difference Learning* (arXiv:2411.04285), as a PyHealth ``BaseModel``.

Each sample is one ICU measurement-time *state*: up to ``L`` irregular events,
each described by five aligned channels (relative time, value, feature ID,
time since the previous same-feature event, and change since that event).
A TD-trained model also receives the matching ``next_*`` channels of the
selected successor state.

Implementation status (task list T21-T29): input validation, train-derived
scaling, the five-branch event embedding, the no-delta ablation switch,
chronological ordering, the CNN/LSTM encoder, the dense decoder, and the frozen
target predictor. The TD/supervised losses, the labeled ``forward()`` output,
and ``update_target()`` are tasks T30-T33.

Intentional, documented differences from the released reference code
(``utils/train/models.py``), approved in ``MODEL_API_DECISIONS.md``:

* ``delta_time`` and ``delta_value`` embeddings use independent missingness
  masks (the reference masks both with the ``delta_time`` mask).
* Padded events contribute a zero embedding and are excluded from the
  embedding BatchNorm statistics (the reference embeds padded positions as
  zero-valued inputs, giving non-zero embeddings that enter BatchNorm).
* The post-convolution mask follows each block's actual receptive field
  (the reference average-pools the mask and ignores the convolution's
  one-step shift). The encoder also canonicalizes right-padding and reserves
  room for boundary responses so predictions do not depend on padding length.
* One length-aware packed LSTM path is used on CPU and CUDA (the reference
  uses packing on CUDA only).
* The target predictor is always in eval mode (the reference runs it in
  train mode, with dropout and batch-statistics BatchNorm).
"""

import copy
import math
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

STATE_KEYS: Tuple[str, ...] = (
    "timepoints",
    "values",
    "features",
    "delta_time",
    "delta_value",
)
"""Required per-state channels, in a fixed order."""

SCALAR_KEYS: Tuple[str, ...] = ("timepoints", "values", "delta_time", "delta_value")
"""Channels holding real-valued (scalable) inputs."""

FEATURE_SCALED_KEYS: Tuple[str, ...] = ("values", "delta_time", "delta_value")
"""Channels normalized with per-feature statistics."""

DELTA_KEYS: Tuple[str, ...] = ("delta_time", "delta_value")
"""Channels removed by the ``use_delta_embeddings=False`` ablation."""

EMBEDDING_NAMES: Dict[str, str] = {
    "timepoints": "time",
    "values": "value",
    "delta_time": "delta_time",
    "delta_value": "delta_value",
}
"""Scalar channel -> embedding-network name (the reference code's names).

``nn.ModuleDict`` cannot hold a module named ``values`` (it shadows the
``ModuleDict.values()`` method), so the networks are keyed by these names.
"""

NEXT_PREFIX = "next_"
OBJECTIVES: Tuple[str, ...] = ("td", "supervised")
PADDING_FEATURE_ID = -1


# ---------------------------------------------------------------------------
# T21: configuration validation
# ---------------------------------------------------------------------------
def _is_real_number(value: Any) -> bool:
    """Returns True for finite ints/floats that are not booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_config(
    n_features: Any,
    hidden_dim: Any,
    dropout: Any,
    use_delta_embeddings: Any,
    objective: Any,
    target_alpha: Any,
) -> None:
    """Validates constructor hyperparameters (policy 9).

    Raises:
        ValueError: If any hyperparameter is outside its allowed range.
        TypeError: If any hyperparameter has the wrong type.
    """
    for name, value in (("n_features", n_features), ("hidden_dim", hidden_dim)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if not _is_real_number(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout!r}")
    if not _is_real_number(target_alpha) or not 0.0 <= target_alpha <= 1.0:
        raise ValueError(
            f"target_alpha must satisfy 0 <= target_alpha <= 1, got {target_alpha!r}"
        )
    if not isinstance(use_delta_embeddings, bool):
        raise TypeError("use_delta_embeddings must be a bool")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")


# ---------------------------------------------------------------------------
# T23: masked embedding normalization
# ---------------------------------------------------------------------------
class _MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm over channels that ignores padded sequence positions.

    Statistics are computed over valid ``(batch, position)`` cells only, and
    padded positions are returned as zeros. Buffers and parameters have the
    same names as :class:`torch.nn.BatchNorm1d`, so the target-network soft
    update treats them identically.
    """

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes ``x`` using valid positions only.

        Args:
            x: Embeddings of shape ``(B, C, L)``.
            mask: Boolean validity mask of shape ``(B, L)``.

        Returns:
            Normalized tensor of shape ``(B, C, L)``, zero at padded positions.

        Raises:
            ValueError: In training mode, if fewer than two valid positions
                are available to estimate the batch variance.
        """
        mask_f = mask.unsqueeze(1).to(x.dtype)
        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            count = int(mask.sum())
            if count < 2:
                raise ValueError(
                    "Embedding BatchNorm needs at least 2 valid events per "
                    f"training batch, got {count}"
                )
            n = float(count)
            mean = (x * mask_f).sum(dim=(0, 2)) / n
            centered = (x - mean[None, :, None]) * mask_f
            var = centered.pow(2).sum(dim=(0, 2)) / n
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        momentum = 1.0 / float(self.num_batches_tracked)
                    else:
                        momentum = self.momentum
                    unbiased_var = var * (n / (n - 1.0))
                    self.running_mean.mul_(1.0 - momentum).add_(momentum * mean)
                    self.running_var.mul_(1.0 - momentum).add_(
                        momentum * unbiased_var
                    )
        else:
            mean = self.running_mean
            var = self.running_var
        out = (x - mean[None, :, None]) * torch.rsqrt(var[None, :, None] + self.eps)
        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * mask_f


# ---------------------------------------------------------------------------
# T23 + T24: five-branch event embedding with the no-delta switch
# ---------------------------------------------------------------------------
class _EventEmbedding(nn.Module):
    """Embeds each event as the sum of feature-ID and scalar-channel embeddings.

    Args:
        n_features: Feature-vocabulary size.
        hidden_dim: Embedding width ``H``.
        use_delta_embeddings: If False (T24 ablation), the ``delta_time`` and
            ``delta_value`` branches are not constructed and contribute
            nothing; all other branches are unchanged.
    """

    def __init__(
        self, n_features: int, hidden_dim: int, use_delta_embeddings: bool
    ) -> None:
        super().__init__()
        self.use_delta_embeddings = use_delta_embeddings
        self.feature_embedding = nn.Embedding(n_features, hidden_dim)
        interim_dim = max(1, math.isqrt(hidden_dim))
        self.scalar_keys: Tuple[str, ...] = tuple(
            key for key in SCALAR_KEYS if use_delta_embeddings or key not in DELTA_KEYS
        )
        self.scalar_embeddings = nn.ModuleDict(
            {
                EMBEDDING_NAMES[key]: nn.Sequential(
                    nn.Linear(1, interim_dim),
                    nn.ReLU(),
                    nn.Linear(interim_dim, hidden_dim),
                )
                for key in self.scalar_keys
            }
        )
        self.norm = _MaskedBatchNorm1d(hidden_dim)

    def forward(
        self, state: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Computes normalized event embeddings.

        Args:
            state: Scaled, chronologically sorted channels, each ``(B, L)``;
                missing scalars are NaN and padded feature IDs are -1.
            masks: Boolean ``(B, L)`` masks ``event``, ``delta_time`` and
                ``delta_value``.

        Returns:
            Tensor of shape ``(B, H, L)`` with zeros at padded positions.
        """
        event_mask = masks["event"]
        # T25: substitute a safe index only at padded positions, so the
        # embedding is never indexed with -1; the result is masked below.
        safe_ids = torch.where(
            event_mask, state["features"], torch.zeros_like(state["features"])
        )
        total = self.feature_embedding(safe_ids) * event_mask.unsqueeze(-1)
        for key in self.scalar_keys:
            net = self.scalar_embeddings[EMBEDDING_NAMES[key]]
            mask = masks[key] if key in DELTA_KEYS else event_mask
            x = torch.where(mask, state[key], torch.zeros_like(state[key]))
            # Mask after the MLP: its bias makes a zero input non-zero.
            total = total + net(x.unsqueeze(-1)) * mask.unsqueeze(-1)
        return self.norm(total.transpose(1, 2), event_mask)


# ---------------------------------------------------------------------------
# T26 + T27: CNN blocks with mask propagation and a packed LSTM
# ---------------------------------------------------------------------------
class _SequenceEncoder(nn.Module):
    """Two Conv1d/ReLU/MaxPool blocks followed by a two-layer LSTM.

    Args:
        hidden_dim: Channel width ``H`` of the event embeddings.
        dropout: Dropout between the two LSTM layers.
        n_blocks: Number of convolution blocks (2 in the paper).
    """

    def __init__(self, hidden_dim: int, dropout: float, n_blocks: int = 2) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=1, padding=1)
                for _ in range(n_blocks)
            ]
        )
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(n_blocks)])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=8 * hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

    @staticmethod
    def conv_mask(mask: torch.Tensor) -> torch.Tensor:
        """Propagates validity through ``Conv1d(kernel_size=2, padding=1)``.

        Output position ``j`` reads inputs ``j - 1`` and ``j``; it is valid if
        either is valid. Shape ``(B, L) -> (B, L + 1)``.
        """
        pooled = F.max_pool1d(
            mask.unsqueeze(1).float(), kernel_size=2, stride=1, padding=1
        )
        return pooled.squeeze(1) > 0.5

    @staticmethod
    def pool_mask(mask: torch.Tensor) -> torch.Tensor:
        """Propagates validity through ``MaxPool1d(2, 2)``.

        A pooled cell is valid if any of its inputs is valid.
        Shape ``(B, L) -> (B, L // 2)``.
        """
        pooled = F.max_pool1d(mask.unsqueeze(1).float(), kernel_size=2, stride=2)
        return pooled.squeeze(1) > 0.5

    @classmethod
    def propagate_mask(cls, mask: torch.Tensor) -> torch.Tensor:
        """Propagates validity through one full convolution block."""
        return cls.pool_mask(cls.conv_mask(mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of event sequences.

        Args:
            x: Event embeddings of shape ``(B, H, L)``, any ``L >= 1``.
            mask: Boolean ``(B, L)`` mask whose valid events form a prefix.

        Returns:
            Final hidden state of the last LSTM layer, shape ``(B, 8H)``.

        Raises:
            RuntimeError: If a row has no valid position (callers must never
                encode an all-padded state).
        """
        # we discard arbitrary caller-supplied trailing padding so its length
        # cannot change the convolution or pooling boundary for valid events
        lengths = mask.sum(dim=1)
        if bool((lengths == 0).any()):
            raise RuntimeError("Cannot encode a state with no valid events")
        valid_length = int(lengths.max().item())
        x = x[..., :valid_length]
        mask = mask[:, :valid_length]

        # each kernel-2/pool-2 block may consume a right-boundary response
        # we reserve enough zeros for all blocks, independent of input width
        boundary_padding = 2 ** len(self.convs) - 1
        x = F.pad(x, (0, boundary_padding))
        mask = F.pad(mask, (0, boundary_padding), value=False)

        for conv, activation in zip(self.convs, self.activations):
            x = x * mask.unsqueeze(1)
            conv_valid = self.conv_mask(mask)
            x = activation(conv(x)) * conv_valid.unsqueeze(1)
            x = self.pool(x)
            mask = self.pool_mask(conv_valid)
        lengths = mask.sum(dim=1)
        if bool((lengths == 0).any()):
            raise RuntimeError("Cannot encode a state with no valid events")
        packed = pack_padded_sequence(
            x.transpose(1, 2),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]


# ---------------------------------------------------------------------------
# T28: full predictor (embedding -> encoder -> decoder)
# ---------------------------------------------------------------------------
class _TDPredictor(nn.Module):
    """One mortality predictor; shared class for online and target networks.

    Args:
        n_features: Feature-vocabulary size.
        hidden_dim: Embedding width ``H``; the LSTM uses ``8H`` units.
        dropout: Dropout between LSTM layers.
        use_delta_embeddings: Whether the two delta branches are used.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        dropout: float,
        use_delta_embeddings: bool,
    ) -> None:
        super().__init__()
        self.embedding = _EventEmbedding(n_features, hidden_dim, use_delta_embeddings)
        self.encoder = _SequenceEncoder(hidden_dim, dropout)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(8 * hidden_dim),
            nn.Linear(8 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-normal initialization of matrices, as in the reference."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

    def forward(
        self, state: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Predicts one mortality logit per state.

        Args:
            state: Scaled, sorted channels, each ``(B, L)``.
            masks: Boolean ``(B, L)`` masks from ``TDCNNLSTM``.

        Returns:
            Logits of shape ``(B, 1)``.

        Raises:
            ValueError: In training mode with ``B < 2`` (decoder BatchNorm).
        """
        batch_size = state["values"].shape[0]
        if self.training and batch_size < 2:
            raise ValueError(
                "Training-mode TDCNNLSTM needs a batch size of at least 2 "
                "(decoder BatchNorm); use eval() for single-state inference"
            )
        embedded = self.embedding(state, masks)
        hidden = self.encoder(embedded, masks["event"])
        return self.decoder(hidden)


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------
class TDCNNLSTM(BaseModel):
    """CNN-LSTM mortality predictor trained with temporal-difference learning.

    The model predicts one 28-day mortality risk per ICU state. In TD mode it
    keeps a frozen target copy of the predictor that scores the successor
    state; the target is updated only through ``update_target()`` after each
    optimizer step (task T33).

    Args:
        dataset: PyHealth ``SampleDataset`` with ``output_schema`` exactly
            ``{"label": "binary"}`` (observed 28-day mortality) and the five
            current-state channels in ``input_schema``.
        n_features: Size of the feature vocabulary. Valid feature IDs are
            ``0 .. n_features - 1``; ``-1`` marks padding.
        hidden_dim: Embedding width ``H``. The LSTM has ``8 * H`` units.
        dropout: Dropout between LSTM layers, ``0 <= dropout < 1``.
        use_delta_embeddings: If False, omit the ``delta_time`` and
            ``delta_value`` embedding contributions (ablation). The delta
            inputs are still required and validated.
        objective: ``"td"`` (online and target predictors) or
            ``"supervised"`` (online predictor only).
        target_alpha: Soft-update coefficient used by ``update_target()``.
        scaling: Training-split statistics ``{"timepoints", "values",
            "delta_time", "delta_value"}``, each ``{"mean": ..., "std": ...}``.
            ``timepoints`` statistics are scalars; the others have shape
            ``(n_features,)``. ``None`` gives identity scaling and is intended
            for synthetic tests only.

    Raises:
        ValueError: For invalid hyperparameters, scaling, or dataset schema.
        TypeError: For hyperparameters or scaling of the wrong type.

    Examples:
        >>> model = TDCNNLSTM(dataset, n_features=4, hidden_dim=8)
        >>> _ = model.eval()
        >>> out = model(**current_state_batch)  # five (B, L) channels
        >>> tuple(out["y_prob"].shape) == (batch_size, 1)
        True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_features: int,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        use_delta_embeddings: bool = True,
        objective: str = "td",
        target_alpha: float = 0.99,
        scaling: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        _check_config(
            n_features,
            hidden_dim,
            dropout,
            use_delta_embeddings,
            objective,
            target_alpha,
        )
        super().__init__(dataset=dataset)
        self._check_dataset_schema(dataset)

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.use_delta_embeddings = use_delta_embeddings
        self.objective = objective
        self.target_alpha = float(target_alpha)
        self.label_key = self.label_keys[0]

        self._register_scaling(scaling)

        self.online = _TDPredictor(
            n_features, hidden_dim, self.dropout, use_delta_embeddings
        )
        # T29: the target starts as an exact, frozen copy of the online model.
        self.target: Optional[_TDPredictor] = None
        if objective == "td":
            target = copy.deepcopy(self.online)
            target.requires_grad_(False)
            target.eval()
            self.target = target

    # ------------------------------------------------------------------
    # T21: dataset schema, canonicalization and validation
    # ------------------------------------------------------------------
    def _check_dataset_schema(self, dataset: Optional[SampleDataset]) -> None:
        """Checks the single binary label and the current-state inputs."""
        if dataset is None:
            raise ValueError("TDCNNLSTM requires a SampleDataset")
        if self.label_keys != ["label"]:
            raise ValueError(
                "TDCNNLSTM expects output_schema {'label': 'binary'} "
                f"(observed 28-day mortality); got labels {self.label_keys}"
            )
        if self.mode != "binary":
            raise ValueError(f"TDCNNLSTM supports binary labels only; got {self.mode}")
        missing = [key for key in STATE_KEYS if key not in dataset.input_schema]
        if missing:
            raise ValueError(f"dataset.input_schema is missing {missing}")

    def _canonicalize_state(
        self,
        batch: Mapping[str, Any],
        prefix: str = "",
        expected_shape: Optional[torch.Size] = None,
    ) -> Dict[str, torch.Tensor]:
        """Collects one state's five channels as ``(B, L)`` tensors.

        Accepts ``(B, L)`` or ``(B, L, 1)`` inputs, removing only a trailing
        singleton dimension. Feature IDs become ``int64``; all other
        channels become ``float32``. Tensors are moved to the model device.

        Args:
            batch: Keyword batch passed to ``forward()``.
            prefix: ``""`` for the current state, ``"next_"`` for the successor.
            expected_shape: Optional current-state ``(B, L)`` shape, required
                to match when validating a paired successor state.

        Returns:
            Mapping from unprefixed channel name to a ``(B, L)`` tensor.

        Raises:
            ValueError: For missing channels, bad shapes or bad dtypes.
        """
        missing = [prefix + key for key in STATE_KEYS if prefix + key not in batch]
        if missing:
            raise ValueError(f"Missing required input(s): {missing}")
        state: Dict[str, torch.Tensor] = {}
        shape: Optional[torch.Size] = None
        for key in STATE_KEYS:
            name = prefix + key
            x = batch[name]
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.dim() == 3:
                if x.shape[-1] != 1:
                    raise ValueError(
                        f"{name} must have shape (B, L) or (B, L, 1); "
                        f"got {tuple(x.shape)}"
                    )
                x = x.squeeze(-1)
            if x.dim() != 2:
                raise ValueError(
                    f"{name} must have shape (B, L) or (B, L, 1); got {tuple(x.shape)}"
                )
            if shape is None:
                shape = x.shape
            elif x.shape != shape:
                raise ValueError(
                    f"{name} has shape {tuple(x.shape)}, but {prefix}timepoints "
                    f"has shape {tuple(shape)}; all channels must match"
                )
            if x.dtype == torch.bool or x.is_complex():
                raise ValueError(f"{name} has unsupported dtype {x.dtype}")
            if key == "features":
                if x.is_floating_point():
                    if not bool(torch.isfinite(x).all()) or not torch.equal(
                        x, torch.round(x)
                    ):
                        raise ValueError(f"{name} must contain integer feature IDs")
                x = x.to(device=self.device, dtype=torch.long)
            else:
                x = x.to(device=self.device, dtype=torch.float32)
            state[key] = x
        assert shape is not None
        if shape[0] == 0 or shape[1] == 0:
            raise ValueError(f"{prefix}state batch is empty: shape {tuple(shape)}")
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(
                f"{prefix}state shape {tuple(shape)} does not match current "
                f"state shape {tuple(expected_shape)}; paired states must "
                "have identical batch and sequence dimensions"
            )
        return state

    def _validate_state(
        self,
        state: Dict[str, torch.Tensor],
        allow_empty: bool,
        prefix: str = "",
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Validates sentinels and builds masks (INPUT_CONTRACT section 2).

        Args:
            state: Output of ``_canonicalize_state``.
            allow_empty: Whether all-padded rows are allowed (successor
                states only).
            prefix: Name prefix used in error messages.

        Returns:
            ``(masks, has_events)``: boolean ``(B, L)`` masks ``event``,
            ``delta_time`` and ``delta_value``, and a ``(B,)`` flag that is
            True for rows with at least one valid event.

        Raises:
            ValueError: For infinite values, half-padded events, non-sentinel
                padding, invalid feature IDs, negative valid times, negative
                delta times, or (if not ``allow_empty``) all-padded rows.
        """
        timepoints = state["timepoints"]
        values = state["values"]
        features = state["features"]
        delta_time = state["delta_time"]
        delta_value = state["delta_value"]

        for key in SCALAR_KEYS:
            if bool(torch.isinf(state[key]).any()):
                raise ValueError(f"{prefix}{key} contains infinite values")

        finite_time = torch.isfinite(timepoints)
        finite_value = torch.isfinite(values)
        half_padded = finite_time ^ finite_value
        if bool(half_padded.any()):
            raise ValueError(
                f"{int(half_padded.sum())} {prefix}event(s) have only one of "
                "timepoints/values; an event is either fully valid or padded"
            )
        event_mask = finite_time & finite_value
        padded = ~event_mask
        finite_delta_time = torch.isfinite(delta_time)
        finite_delta_value = torch.isfinite(delta_value)

        if bool((features[padded] != PADDING_FEATURE_ID).any()):
            raise ValueError(
                f"Padded {prefix}events must have feature ID {PADDING_FEATURE_ID}"
            )
        if bool((padded & (finite_delta_time | finite_delta_value)).any()):
            raise ValueError(f"Padded {prefix}events must have missing (NaN) deltas")

        valid_ids = features[event_mask]
        if bool(((valid_ids < 0) | (valid_ids >= self.n_features)).any()):
            raise ValueError(
                f"Valid {prefix}feature IDs must lie in [0, {self.n_features - 1}]"
            )
        if bool((timepoints[event_mask] < 0).any()):
            raise ValueError(f"Valid {prefix}timepoints must be >= 0 minutes")
        if bool((delta_time[finite_delta_time] < 0).any()):
            raise ValueError(f"Finite {prefix}delta_time values must be >= 0")

        has_events = event_mask.any(dim=1)
        if not allow_empty and not bool(has_events.all()):
            rows = torch.nonzero(~has_events).flatten().tolist()
            raise ValueError(
                f"{prefix}state is all padding at batch row(s) {rows[:10]}; "
                "an empty current state cannot be predicted"
            )
        masks = {
            "event": event_mask,
            "delta_time": event_mask & finite_delta_time,
            "delta_value": event_mask & finite_delta_value,
        }
        return masks, has_events

    # ------------------------------------------------------------------
    # T22: train-derived, feature-specific scaling
    # ------------------------------------------------------------------
    def _register_scaling(
        self, scaling: Optional[Mapping[str, Mapping[str, Any]]]
    ) -> None:
        """Validates scaling statistics and registers them as buffers.

        Registers ``{key}_mean``, ``{key}_std`` (effective, with zero or
        undefined standard deviations replaced by 1.0) and
        ``{key}_std_replaced`` for every key in ``SCALAR_KEYS``. All buffers
        are persistent, so they are saved in ``state_dict`` and follow
        ``.to(device)``.

        Raises:
            TypeError: If ``scaling`` or an entry is not a mapping.
            ValueError: For missing/unexpected keys, wrong shapes, non-finite
                means, or infinite/negative standard deviations.
        """
        expected_shapes = {
            "timepoints": (),
            "values": (self.n_features,),
            "delta_time": (self.n_features,),
            "delta_value": (self.n_features,),
        }
        if scaling is not None and not isinstance(scaling, Mapping):
            raise TypeError("scaling must be a mapping or None")
        if scaling is not None and set(scaling) != set(expected_shapes):
            raise ValueError(
                f"scaling keys must be exactly {sorted(expected_shapes)}; "
                f"got {sorted(scaling)}"
            )
        for key, shape in expected_shapes.items():
            if scaling is None:
                mean = torch.zeros(shape, dtype=torch.float32)
                std = torch.ones(shape, dtype=torch.float32)
            else:
                entry = scaling[key]
                if not isinstance(entry, Mapping):
                    raise TypeError(f"scaling[{key!r}] must be a mapping")
                if set(entry) != {"mean", "std"}:
                    raise ValueError(
                        f"scaling[{key!r}] keys must be exactly ['mean', 'std']; "
                        f"got {sorted(entry)}"
                    )
                mean = torch.as_tensor(entry["mean"], dtype=torch.float32).clone()
                std = torch.as_tensor(entry["std"], dtype=torch.float32).clone()
                for stat_name, stat in (("mean", mean), ("std", std)):
                    if tuple(stat.shape) != shape:
                        raise ValueError(
                            f"scaling[{key!r}][{stat_name!r}] must have shape "
                            f"{shape}; got {tuple(stat.shape)}"
                        )
                if not bool(torch.isfinite(mean).all()):
                    raise ValueError(f"scaling[{key!r}]['mean'] must be finite")
                if bool(torch.isinf(std).any()):
                    raise ValueError(f"scaling[{key!r}]['std'] must not be infinite")
                if bool((std < 0).any()):
                    raise ValueError(f"scaling[{key!r}]['std'] must be >= 0")
            replaced = torch.isnan(std) | (std == 0)
            std = torch.where(replaced, torch.ones_like(std), std)
            self.register_buffer(f"{key}_mean", mean)
            self.register_buffer(f"{key}_std", std)
            self.register_buffer(f"{key}_std_replaced", replaced)

    def _scale_state(
        self, state: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Applies shared time scaling and per-feature scalar scaling.

        Missing entries are NaN before scaling and remain NaN afterwards.

        Args:
            state: Validated ``(B, L)`` channels.
            masks: Masks from ``_validate_state``.

        Returns:
            New mapping with scaled scalar channels and unchanged feature IDs.
        """
        safe_ids = torch.where(
            masks["event"], state["features"], torch.zeros_like(state["features"])
        )
        scaled = dict(state)
        scaled["timepoints"] = (
            state["timepoints"] - self.timepoints_mean
        ) / self.timepoints_std
        for key in FEATURE_SCALED_KEYS:
            mean = getattr(self, f"{key}_mean")[safe_ids]
            std = getattr(self, f"{key}_std")[safe_ids]
            scaled[key] = (state[key] - mean) / std
        return scaled

    # ------------------------------------------------------------------
    # T25: chronological ordering
    # ------------------------------------------------------------------
    @staticmethod
    def _sort_state(
        state: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Orders events oldest to newest and moves padding to the end.

        ``timepoints`` are minutes *before* the state marker, so the order is
        descending time. The sort is stable: events with equal times keep
        their supplied order. All channels and masks are reordered together.

        Returns:
            ``(sorted_state, sorted_masks)``; valid events form a prefix.
        """
        sort_key = torch.where(
            masks["event"],
            state["timepoints"],
            torch.full_like(state["timepoints"], float("-inf")),
        )
        order = torch.argsort(sort_key, dim=1, descending=True, stable=True)
        sorted_state = {k: torch.gather(v, 1, order) for k, v in state.items()}
        sorted_masks = {k: torch.gather(m, 1, order) for k, m in masks.items()}
        return sorted_state, sorted_masks

    def _prepare_state(
        self,
        batch: Mapping[str, Any],
        prefix: str = "",
        allow_empty: bool = False,
        expected_shape: Optional[torch.Size] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Canonicalizes, validates, sorts and scales one state.

        Args:
            batch: Keyword batch passed to ``forward()``.
            prefix: ``""`` for the current state, ``"next_"`` for the successor.
            allow_empty: Whether all-padded rows are allowed.
            expected_shape: Expected ``(B, L)`` shape for a paired successor.

        Returns:
            ``(state, masks, has_events)`` ready for ``_TDPredictor``.
        """
        state = self._canonicalize_state(batch, prefix, expected_shape)
        masks, has_events = self._validate_state(state, allow_empty, prefix)
        state, masks = self._sort_state(state, masks)
        return self._scale_state(state, masks), masks, has_events

    # ------------------------------------------------------------------
    # T29: target predictor behavior
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "TDCNNLSTM":
        """Sets the online predictor's mode; the target always stays in eval.

        Args:
            mode: True for training mode, False for evaluation mode.

        Returns:
            This model.
        """
        super().train(mode)
        if self.target is not None:
            self.target.eval()
        return self

    @property
    def n_online_trainable_parameters(self) -> int:
        """Number of trainable online parameters (excludes the target).

        This is the count used by the reference learning-rate formula
        ``lr = 1 / sqrt(n_trainable_params)``.
        """
        return sum(p.numel() for p in self.online.parameters() if p.requires_grad)

    @torch.no_grad()
    def _target_probability(
        self,
        batch: Mapping[str, Any],
        current_shape: Optional[torch.Size] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores successor states with the frozen target predictor.

        Rows whose successor is all padding are never passed to the target.

        Args:
            batch: Keyword batch containing the five ``next_*`` channels.
            current_shape: Current-state ``(B, L)`` shape to validate against.
                Inferred from the batch when all current channels are present.

        Returns:
            ``(probability, has_next)``: a detached ``(B, 1)`` tensor that is
            NaN for rows without a successor, and a ``(B,)`` boolean flag.

        Raises:
            RuntimeError: If the model has no target (supervised objective).
        """
        if self.target is None:
            raise RuntimeError("The supervised objective has no target predictor")
        if current_shape is None and all(key in batch for key in STATE_KEYS):
            current_shape = self._canonicalize_state(batch)["values"].shape
        state, masks, has_next = self._prepare_state(
            batch,
            prefix=NEXT_PREFIX,
            allow_empty=True,
            expected_shape=current_shape,
        )
        batch_size = has_next.shape[0]
        probability = torch.full(
            (batch_size, 1), float("nan"), dtype=torch.float32, device=self.device
        )
        if bool(has_next.any()):
            sub_state = {k: v[has_next] for k, v in state.items()}
            sub_masks = {k: m[has_next] for k, m in masks.items()}
            probability[has_next] = torch.sigmoid(self.target(sub_state, sub_masks))
        return probability, has_next

    # ------------------------------------------------------------------
    # Forward (provisional until T30-T32)
    # ------------------------------------------------------------------
    def forward(self, **batch: Any) -> Dict[str, torch.Tensor]:
        """Predicts 28-day mortality risk for the current states.

        This is the unlabeled-inference path. Loss computation and the
        labeled output (``loss``, ``y_true``) are added in tasks T30-T32.

        Args:
            **batch: The five current-state channels, each ``(B, L)`` or
                ``(B, L, 1)``. Optional ``next_*`` channels are validated as
                a complete, shape-matched successor state but not scored until
                T30-T32. Unrelated metadata (e.g. ``patient_id``) is ignored.

        Returns:
            Dictionary with ``logit`` and ``y_prob``, each of shape ``(B, 1)``.

        Raises:
            NotImplementedError: If the batch contains the label (T30-T32).
        """
        if self.label_key in batch:
            raise NotImplementedError(
                "Labeled forward (loss, y_true) is implemented in T30-T32"
            )
        state, masks, _ = self._prepare_state(batch)

        # we validate supplied successors even on the provisional inference
        # path so a mismatched paired batch cannot silently pass T21
        if any(NEXT_PREFIX + key in batch for key in STATE_KEYS):
            self._prepare_state(
                batch,
                prefix=NEXT_PREFIX,
                allow_empty=True,
                expected_shape=state["values"].shape,
            )

        logit = self.online(state, masks)
        return {"logit": logit, "y_prob": torch.sigmoid(logit)}
