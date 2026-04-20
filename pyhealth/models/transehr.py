"""Dual-stream TransEHR-style encoder for PyHealth."""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.unified_embedding import SinusoidalTimeEmbedding
from pyhealth.processors import (
    SequenceProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)
from pyhealth.processors.multi_hot_processor import MultiHotProcessor
from pyhealth.processors.nested_sequence_processor import NestedSequenceProcessor
from pyhealth.processors.temporal_timeseries_processor import (
    TemporalTimeseriesProcessor,
)

EventKind = Literal["sequence", "nested_sequence", "multi_hot"]

# Batch values accepted from :class:`~torch.utils.data.DataLoader` collate fns.
ForwardKwarg = Union[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    dict[str, Any],
    list[Any],
    bool,
    int,
    float,
]


class TransEHR(BaseModel):
    """TransEHR model.

    This model is a PyHealth-friendly implementation of a simplified
    TransEHR-style architecture for irregular EHR time series. It uses
    one Transformer encoder for multivariate observations and another
    Transformer encoder for clinical events, then fuses the pooled
    representations for downstream prediction.

    The current implementation focuses on the supervised backbone only.
    It does not include the full self-supervised pretraining pipeline
    from the original paper, such as THP-based forecasting or the
    auxiliary pretraining heads.

    The model supports three kinds of event inputs:
        - sequence: shape (batch, seq_len)
        - nested_sequence: shape (batch, n_visits, n_codes)
        - multi_hot: shape (batch, n_events)

    It also supports:
        - multivariate time series with optional timestamps and masks
        - optional static features
        - an ablation flag ``use_event_stream`` to disable the event branch

    Args:
        dataset: The sample dataset used to initialize the model.
        feature_keys: Mapping from TransEHR input roles to dataset columns.
            Required keys are ``"multivariate"`` and ``"events"``.
            An optional ``"static"`` key can also be provided.
        label_key: Name of the output label column. If not given, the model
            uses the only label in ``dataset.output_schema``.
        mode: Prediction mode. If given, it must match the label processor.
        embedding_dim: Embedding dimension for event and time embeddings.
            Default is 128.
        hidden_dim: Hidden dimension used by both Transformer encoders.
            Default is 128.
        num_heads: Number of attention heads. Default is 4.
        dropout: Dropout rate. Default is 0.1.
        num_encoder_layers: Number of Transformer layers in each encoder.
            Default is 2.
        max_event_len: Maximum event sequence length for positional embeddings.
            Default is 512.
        max_ts_len: Maximum time-series length for positional embeddings.
            Default is 2048.
        multivariate_input_dim: Optional manual override for the multivariate
            feature dimension. If not given, it is inferred from the processor.
        use_event_stream: Whether to use the event encoder branch. Default is
            True. Setting it to False provides a simple structural ablation.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import TransEHR
        >>>
        >>> t0 = datetime(2020, 1, 1, 0, 0)
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "multivariate": (
        ...             [t0, t0 + timedelta(hours=1)],
        ...             np.array([[1.0, 0.0], [0.5, 0.2]], dtype=np.float32),
        ...         ),
        ...         "events": ["LAB_A", "MED_X"],
        ...         "static": [60.0, 1.0],
        ...         "label": 0,
        ...     }
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "multivariate": "temporal_timeseries",
        ...         "events": "sequence",
        ...         "static": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="demo_transehr",
        ... )
        >>> model = TransEHR(
        ...     dataset=dataset,
        ...     feature_keys={
        ...         "multivariate": "multivariate",
        ...         "events": "events",
        ...         "static": "static",
        ...     },
        ...     label_key="label",
        ... )

    Raises:
        KeyError: If ``feature_keys`` is missing ``"multivariate"`` or
            ``"events"``, or if a mapped column is absent from the dataset.
        ValueError: If ``dataset.output_schema`` is empty, multiple labels exist
            but ``label_key`` is omitted, ``mode`` disagrees with the label
            processor, ``hidden_dim`` is not divisible by ``num_heads``, or a
            processor cannot infer feature widths.
        TypeError: If the events or static column uses an unsupported processor.
    """

    _REQUIRED_FK = ("multivariate", "events")

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: Mapping[str, str],
        label_key: Optional[str] = None,
        mode: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_encoder_layers: int = 2,
        max_event_len: int = 512,
        max_ts_len: int = 2048,
        multivariate_input_dim: Optional[int] = None,
        use_event_stream: bool = True,
    ) -> None:
        """Build encoder stacks and the prediction head.

        Argument descriptions and runnable examples are documented on
        :class:`TransEHR`.

        Raises:
            KeyError: If ``feature_keys`` or column names are invalid.
            ValueError: If labels or hyperparameters are inconsistent.
            TypeError: If a configured column uses an unsupported processor.
        """
        # resolve label first
        output_keys = list(dataset.output_schema.keys())
        if not output_keys:
            raise ValueError("dataset.output_schema must define at least one label.")
        if label_key is not None:
            if label_key not in dataset.output_schema:
                raise KeyError(f"label_key {label_key!r} not in dataset.output_schema.")
            resolved_label_key = label_key
        elif len(output_keys) == 1:
            resolved_label_key = output_keys[0]
        else:
            raise ValueError(
                "Dataset has multiple outputs; pass label_key explicitly "
                "for TransEHR."
            )

        for k in self._REQUIRED_FK:
            if k not in feature_keys:
                raise KeyError(
                    f"feature_keys must contain {k!r}; got keys {tuple(feature_keys)}."
                )
        fk_map: Dict[str, str] = dict(feature_keys)
        _mv_col = fk_map["multivariate"]
        _ev_col = fk_map["events"]
        _st_col: Optional[str] = fk_map.get("static")

        if _mv_col not in dataset.input_processors:
            raise KeyError(f"Unknown multivariate column {_mv_col!r}.")
        if _ev_col not in dataset.input_processors:
            raise KeyError(f"Unknown events column {_ev_col!r}.")
        if _st_col is not None and _st_col not in dataset.input_processors:
            raise KeyError(f"Unknown static column {_st_col!r}.")

        # initialize BaseModel
        super().__init__(dataset=dataset)

        self.trans_feature_key_map: Dict[str, str] = fk_map
        self._mv_col = _mv_col
        self._ev_col = _ev_col
        self._st_col = _st_col

        # use a single prediction head for the model
        self.label_key = resolved_label_key
        self.label_keys = [self.label_key]

        resolved_mode = self._resolve_mode(dataset.output_schema[self.label_key])
        if mode is not None and mode.lower() != resolved_mode:
            raise ValueError(
                f"mode={mode!r} conflicts with dataset output_schema "
                f"for {self.label_key!r} (resolved {resolved_mode!r})."
            )
        self.mode = resolved_mode

        # keep only the columns used by this model
        used_cols = [_mv_col, _ev_col]
        if _st_col is not None:
            used_cols.append(_st_col)
        self.feature_keys = used_cols

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self._max_event_len = max_event_len
        self.use_event_stream = use_event_stream

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self._n_mv_features = multivariate_input_dim or self._infer_multivariate_dim()
        ev_proc = dataset.input_processors[self._ev_col]
        self._event_kind: EventKind = self._infer_event_kind(ev_proc)

        self.static_dim: Optional[int] = None
        if self._st_col is not None:
            self.static_dim = self._infer_static_dim(
                dataset.input_processors[self._st_col], self._st_col
            )

        # event branch
        self.event_type_embed: Optional[nn.Embedding] = None
        self.event_in_proj: Optional[nn.Linear] = None
        self.event_bag_proj: Optional[nn.Linear] = None

        if self._event_kind in ("sequence", "nested_sequence"):
            num_event_types = int(ev_proc.vocab_size())
            self.event_type_embed = nn.Embedding(
                num_event_types, embedding_dim, padding_idx=0
            )
            self.event_in_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        else:
            bag_dim = int(ev_proc.size())
            self.event_bag_proj = nn.Linear(bag_dim, hidden_dim)

        self.event_pos_embed = nn.Embedding(max_event_len, hidden_dim)
        self.register_buffer("_event_pos_ids", torch.arange(max_event_len))

        self.value_proj = nn.Linear(self._n_mv_features, hidden_dim)
        time_sin_dim = embedding_dim if embedding_dim % 2 == 0 else embedding_dim + 1
        self.time_encode = SinusoidalTimeEmbedding(time_sin_dim)
        self.time_proj = nn.Linear(time_sin_dim, hidden_dim)
        self.ts_pos_embed = nn.Embedding(max_ts_len, hidden_dim)
        self.register_buffer("_ts_pos_ids", torch.arange(max_ts_len))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ts_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_encoder_layers,
        )
        self.event_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.ts_in_norm = nn.LayerNorm(hidden_dim)
        self.event_in_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.static_proj: Optional[nn.Linear] = None
        if self.static_dim is not None:
            self.static_proj = nn.Linear(self.static_dim, hidden_dim)

        n_branches = 1 + (1 if self.use_event_stream else 0) + (
            1 if self.static_dim is not None else 0
        )
        fuse_dim = hidden_dim * n_branches
        output_size = self.get_output_size()
        self.classifier = nn.Linear(fuse_dim, output_size)

    @staticmethod
    def _infer_event_kind(ev_proc: Any) -> EventKind:
        """Infer which event representation is used by the dataset processor.

        Args:
            ev_proc: Fitted processor for the events column.

        Returns:
            One of ``"sequence"``, ``"nested_sequence"``, or ``"multi_hot"``.

        Raises:
            TypeError: If ``ev_proc`` is not a supported event processor.
        """
        if isinstance(ev_proc, SequenceProcessor):
            return "sequence"
        if isinstance(ev_proc, NestedSequenceProcessor):
            return "nested_sequence"
        if isinstance(ev_proc, MultiHotProcessor):
            return "multi_hot"
        raise TypeError(
            "TransEHR events column must use SequenceProcessor, "
            "NestedSequenceProcessor, or MultiHotProcessor; "
            f"got {type(ev_proc).__name__}."
        )

    def _tensor_processor_input_dim(self, proc: TensorProcessor, field: str) -> int:
        """Infer the last feature dimension of a tensor field from one sample.

        Args:
            proc: Fitted ``TensorProcessor`` for ``field``.
            field: Dataset column name whose processed tensors define the width.

        Returns:
            Number of features along the last axis (``>= 1``).

        Raises:
            ValueError: If the dataset yields no sample containing ``field``.
        """
        for sample in self.dataset:
            if field in sample:
                t = proc.process(sample[field])
                return int(t.shape[-1]) if t.dim() > 0 else 1
        raise ValueError(
            f"Could not infer TensorProcessor width for field {field!r}: empty dataset?"
        )

    def _infer_static_dim(self, st_proc: Any, col: str) -> int:
        """Infer the input dimension of the static feature column.

        Args:
            st_proc: Fitted processor for the static column.
            col: Column name in ``dataset.input_processors``.

        Returns:
            Static feature width (multi-hot vocabulary size or tensor width).

        Raises:
            TypeError: If ``st_proc`` is neither ``TensorProcessor`` nor
                ``MultiHotProcessor``.
        """
        if isinstance(st_proc, TensorProcessor):
            return self._tensor_processor_input_dim(st_proc, col)
        if isinstance(st_proc, MultiHotProcessor):
            return int(st_proc.size())
        raise TypeError(
            "Optional static column must use TensorProcessor or MultiHotProcessor; "
            f"got {type(st_proc).__name__}."
        )

    def _infer_multivariate_dim(self) -> int:
        """Infer the number of multivariate features from the input processor.

        Returns:
            Feature count ``F`` for the multivariate column.

        Raises:
            ValueError: If a time-series processor has no ``n_features`` after fit.
            TypeError: If the multivariate processor is not supported.
        """
        proc = self.dataset.input_processors[self._mv_col]
        if isinstance(proc, (TimeseriesProcessor, TemporalTimeseriesProcessor)):
            nf = getattr(proc, "n_features", None)
            if nf is None:
                raise ValueError(
                    f"Processor {type(proc).__name__} for {self._mv_col!r} has no "
                    "n_features; call dataset.fit() or pass multivariate_input_dim=…."
                )
            return int(nf)
        if isinstance(proc, TensorProcessor):
            return self._tensor_processor_input_dim(proc, self._mv_col)
        raise TypeError(
            f"Unsupported multivariate processor {type(proc).__name__}. "
            "Use TimeseriesProcessor, TemporalTimeseriesProcessor, or TensorProcessor, "
            "or pass multivariate_input_dim explicitly."
        )

    def _embed_multivariate(
        self,
        values: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the multivariate stream and return its padding mask.

        Args:
            values: Float tensor of shape ``(B, T, F)`` or ``(B, T)`` (expanded
                to ``(B, T, 1)``).
            time: Optional float tensor ``(B, T)`` of aligned timestamps.
            mask: Optional mask broadcastable to timesteps; zeros mark padding.

        Returns:
            Tuple ``(h, pad_bool)`` where ``h`` is ``(B, T, hidden_dim)`` and
            ``pad_bool`` is ``(B, T)`` with ``True`` at padded positions for
            :class:`torch.nn.TransformerEncoder`.
        """
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        b, t, _f = values.shape
        device = values.device
        x = self.value_proj(values.float())

        if time is not None:
            t_emb = self.time_proj(self.time_encode(time.float().to(device)))
            if t_emb.dim() == 2:
                t_emb = t_emb.unsqueeze(1).expand(-1, t, -1)
        else:
            pos = self._ts_pos_ids[:t].to(device)
            t_emb = self.ts_pos_embed(pos).unsqueeze(0).expand(b, -1, -1)

        h = self.ts_in_norm(x + t_emb)
        h = self.dropout(h)

        if mask is None:
            mask = (values.abs().sum(dim=-1) > 0).float()
        if mask.dim() == 3:
            mask = (mask.sum(dim=-1) > 0).float()
        pad = mask <= 0
        return h, pad

    def _time_bias(
        self,
        time: Optional[torch.Tensor],
        length: int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build time or positional embeddings for a sequence.

        Args:
            time: Optional per-step times ``(B, length)`` or broadcastable.
            length: Sequence length ``L``.
            batch: Batch size ``B``.
            device: Device for new tensors.

        Returns:
            Tensor of shape ``(B, L, hidden_dim)``.

        Raises:
            ValueError: If ``time`` is 2-D but its time dimension does not match
                ``length``.
        """
        if time is not None:
            t = time.float().to(device)
            if t.dim() == 1:
                t = t.unsqueeze(1).expand(-1, length)
            elif t.shape[1] != length:
                raise ValueError(
                    f"time tensor must have shape[1]={length}, "
                    f"got {tuple(t.shape)}."
                )
            return self.time_proj(self.time_encode(t))
        pos = self._event_pos_ids[:length].to(device)
        return self.event_pos_embed(pos).unsqueeze(0).expand(batch, -1, -1)

    def _embed_events_sequence(
        self,
        token_ids: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a flat event-code sequence and return its padding mask.

        Args:
            token_ids: Long tensor ``(B, L)`` of code indices (``0`` = pad).
            time: Optional ``(B, L)`` float times.
            mask: Optional float mask; zeros mark padding.

        Returns:
            Tuple ``(h, pad_bool)`` for the event Transformer.
        """
        assert self.event_type_embed is not None and self.event_in_proj is not None
        b, l = token_ids.shape
        device = token_ids.device
        e = self.event_type_embed(token_ids.long())
        h = self.event_in_proj(e)
        t_emb = self._time_bias(time, l, b, device)
        h = self.event_in_norm(h + t_emb)
        h = self.dropout(h)
        if mask is None:
            mask = (token_ids != 0).float()
        pad = mask <= 0
        return h, pad

    def _embed_events_nested(
        self,
        token_ids: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed visit-level events by mean-pooling code embeddings per visit.

        Args:
            token_ids: Long tensor ``(B, V, C)`` (visit × codes per visit).
            time: Optional ``(B, V)`` visit-level times.
            mask: Optional visit validity mask ``(B, V)``.

        Returns:
            Tuple ``(h, pad_bool)`` for the visit-level event Transformer.

        Raises:
            ValueError: If ``token_ids`` is not 3-dimensional.
        """
        assert self.event_type_embed is not None and self.event_in_proj is not None
        if token_ids.dim() != 3:
            raise ValueError(
                f"nested_sequence expects (B, V, C); got {tuple(token_ids.shape)}."
            )
        b, v, c = token_ids.shape
        device = token_ids.device
        if v > self._max_event_len:
            token_ids = token_ids[:, : self._max_event_len].contiguous()
            v = token_ids.shape[1]
            if time is not None and time.dim() == 2:
                time = time[:, : self._max_event_len].contiguous()
            if mask is not None and mask.dim() == 2:
                mask = mask[:, : self._max_event_len].contiguous()

        inner = token_ids.long()
        emb = self.event_type_embed(inner)
        inner_valid = (inner != 0).float().unsqueeze(-1)
        denom = inner_valid.sum(dim=2).clamp(min=1.0)
        pooled = (emb * inner_valid).sum(dim=2) / denom
        h = self.event_in_proj(pooled)
        t_emb = self._time_bias(time, v, b, device)
        h = self.event_in_norm(h + t_emb)
        h = self.dropout(h)

        if mask is None:
            visit_active = (token_ids.abs().sum(dim=-1) > 0).float()
        else:
            visit_active = mask.float()
        pad = visit_active <= 0
        return h, pad

    def _embed_events_multihot(
        self,
        indicators: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a multi-hot event vector as a single-step event sequence.

        Args:
            indicators: Float tensor ``(B, K)`` of event indicators.
            time: Optional global time per row ``(B,)`` or ``(B, 1)``.

        Returns:
            Tuple ``(h, pad_bool)`` with ``h`` of shape ``(B, 1, hidden_dim)``.
        """
        assert self.event_bag_proj is not None
        if indicators.dim() == 1:
            indicators = indicators.unsqueeze(0)
        b, _k = indicators.shape
        device = indicators.device
        h = self.event_bag_proj(indicators.float()).unsqueeze(1)
        if time is not None:
            t = time.float().to(device)
            if t.dim() == 2:
                t = t.squeeze(-1)
            t_emb = self.time_proj(self.time_encode(t))
            if t_emb.dim() == 2:
                t_emb = t_emb.unsqueeze(1)
        else:
            t_emb = self.event_pos_embed(
                torch.zeros(1, dtype=torch.long, device=device)
            ).view(1, 1, -1).expand(b, 1, -1)
        h = self.event_in_norm(h + t_emb)
        h = self.dropout(h)
        pad = torch.zeros(b, 1, dtype=torch.bool, device=device)
        return h, pad

    def _embed_events(
        self,
        value: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to the correct event embedding path.

        Args:
            value: Event tensor; layout depends on ``self._event_kind``.
            time: Optional aligned time tensor.
            mask: Optional padding mask for discrete event sequences.

        Returns:
            Tuple ``(h, pad_bool)`` ready for ``self.event_encoder``.
        """
        if self._event_kind == "sequence":
            tok = value
            if tok.dim() == 1:
                tok = tok.unsqueeze(1)
            return self._embed_events_sequence(tok, time=time, mask=mask)
        if self._event_kind == "nested_sequence":
            return self._embed_events_nested(value, time=time, mask=mask)
        return self._embed_events_multihot(value, time=time)

    @staticmethod
    def _masked_mean(seq: torch.Tensor, pad_bool: torch.Tensor) -> torch.Tensor:
        """Mean-pool a sequence while ignoring padded positions.

        Args:
            seq: Tensor ``(B, L, D)``.
            pad_bool: Boolean mask ``(B, L)``, ``True`` at padded steps.

        Returns:
            Pooled tensor of shape ``(B, D)``.
        """
        valid = (~pad_bool).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (seq * valid).sum(dim=1) / denom

    def _unpack_field(
        self, raw: Any, col: str
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert one batch field into ``value``, ``time``, and ``mask`` tensors.

        Args:
            raw: Collated field: tensor, tuple, dict, or ``list[dict]`` from the
                default dataloader collate.
            col: Dataset column name (used to read ``schema()``).

        Returns:
            ``(value, time, mask)`` with entries possibly ``None`` when absent.

        Raises:
            TypeError: If ``raw`` has an unsupported container type.
        """
        device = self.device
        proc = self.dataset.input_processors[col]
        schema = proc.schema()

        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            collated: dict[str, torch.Tensor] = {}
            for subkey in raw[0]:
                tensors = [d[subkey] for d in raw if subkey in d]
                if not tensors:
                    continue
                if isinstance(tensors[0], torch.Tensor):
                    collated[subkey] = (
                        torch.stack(tensors)
                        if all(t.shape == tensors[0].shape for t in tensors)
                        else pad_sequence(tensors, batch_first=True)
                    )
            raw = collated

        if isinstance(raw, dict):
            value = raw["value"].to(device)
            time = raw.get("time")
            if isinstance(time, torch.Tensor):
                time = time.to(device)
            mask = raw.get("mask")
            if isinstance(mask, torch.Tensor):
                mask = mask.to(device)
            return value, time, mask

        if isinstance(raw, torch.Tensor):
            return raw.to(device), None, None

        if isinstance(raw, (tuple, list)):
            parts = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in raw
            )
            if len(parts) == 2:
                a, b = parts
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    if b.dim() > a.dim():
                        return b, a, None
                    if a.shape == b.shape:
                        return a, None, b
            value = parts[schema.index("value")].to(device)
            time_t: Optional[torch.Tensor] = None
            if "time" in schema:
                t_i = schema.index("time")
                if isinstance(parts[t_i], torch.Tensor):
                    time_t = parts[t_i].to(device)
            mask_t: Optional[torch.Tensor] = None
            if "mask" in schema:
                m_i = schema.index("mask")
                if isinstance(parts[m_i], torch.Tensor):
                    mask_t = parts[m_i].to(device)
            return value, time_t, mask_t

        raise TypeError(f"Unsupported feature container for {col!r}: {type(raw)}.")

    def forward(self, **kwargs: ForwardKwarg) -> Dict[str, torch.Tensor]:
        """Supervised forward: encode, pool, fuse, and compute task outputs.

        Args:
            **kwargs: Batch dict from the task dataloader. Must include the
                dataset columns mapped by ``feature_keys`` (``multivariate``,
                ``events`` when ``use_event_stream`` is ``True``, and ``static``
                if configured), plus the label column ``self.label_key``. Optional
                key ``embed`` (``bool``): if truthy, the returned dict includes an
                ``"embed"`` tensor (fused representation before the classifier).

        Returns:
            Dictionary with keys ``logit``, ``y_prob``, ``loss``, ``y_true``
            (loss and y_true require labels in the batch). May include ``embed``.

        Raises:
            KeyError: If a required input column is missing from ``kwargs``.
            TypeError: If a field cannot be unpacked by :meth:`_unpack_field`.
        """
        mv_raw = kwargs[self._mv_col]

        mv_val, mv_time, mv_mask = self._unpack_field(mv_raw, self._mv_col)

        ts_h, ts_pad = self._embed_multivariate(mv_val, time=mv_time, mask=mv_mask)

        ts_z = self.ts_encoder(ts_h, src_key_padding_mask=ts_pad)

        ts_pool = self._masked_mean(ts_z, ts_pad)
        parts: list[torch.Tensor] = [ts_pool]

        if self.use_event_stream:
            ev_raw = kwargs[self._ev_col]
            ev_val, ev_time, ev_mask = self._unpack_field(ev_raw, self._ev_col)
            ev_h, ev_pad = self._embed_events(ev_val, time=ev_time, mask=ev_mask)
            ev_z = self.event_encoder(ev_h, src_key_padding_mask=ev_pad)
            ev_pool = self._masked_mean(ev_z, ev_pad)
            parts.append(ev_pool)

        if self._st_col is not None and self.static_proj is not None:
            st = kwargs[self._st_col]
            st_t, _, _ = self._unpack_field(st, self._st_col)
            if st_t.dim() == 1:
                st_t = st_t.unsqueeze(0)
            parts.append(self.static_proj(st_t.float()))

        fused = torch.cat(parts, dim=-1)
        logits = self.classifier(fused)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            out["embed"] = fused
        return out

    def forward_from_embedding(
        self,
        **kwargs: ForwardKwarg,
    ) -> Dict[str, torch.Tensor]:
        """Alias for :meth:`forward` for interpretability tooling compatibility.

        TransEHR does not use a separate :class:`~pyhealth.models.EmbeddingModel`;
        embeddings are produced inside this module from raw batch fields. Per
        :class:`~pyhealth.models.base_model.BaseModel` guidance, callers that
        already match the dataloader layout (including optional smoothed tensors
        injected into ``kwargs``) may use this entry point interchangeably with
        :meth:`forward`.

        Args:
            **kwargs: Same contract as :meth:`forward`.

        Returns:
            Same dictionary as :meth:`forward`.
        """
        return self.forward(**kwargs)

    def get_embedding_model(self) -> nn.Module | None:
        """Return a standalone embedding submodule when one exists.

        Returns:
            ``None`` because this model inlines embedding and encoders in one
            :class:`torch.nn.Module` tree.
        """
        return None
