"""UnifiedMultimodalEmbeddingModel, temporally aligned multimodal embedding.

Takes K temporal features ( dict outputs from ``TemporalFeatureProcessor``
subclasses ), embeds each event with a modality-specific encoder, then
interleaves all events on a shared timeline by sorting on timestamp and adding
sinusoidal time embeddings + learned modality-type embeddings.

Output shape: ``(B, S_total, E')``, a single sequence of events usable by
any downstream sequence model (Transformer, Mamba, RNN, …).

IMAGE encoding delegates to :class:`PatchEmbedding` from
:mod:`pyhealth.models.embedding.vision` (Josh's model), pooling patch tokens
to a single per-image vector via global mean pooling.

TEXT encoding uses a pretrained BERT tokenizer model directly, extracting the
[CLS] token per note, the same BERT-based approach as
:class:`TextEmbeddingModel` (Rian's model).

Unimodal model reuse via ``field_embeddings``::

    vision_model = VisionEmbeddingModel(dataset, embedding_dim=128)
    text_model   = TextEmbeddingModel(embedding_dim=128)
    unified = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,
        embedding_dim=128,
        field_embeddings={
            "chest_xray": vision_model,   # reuses trained backbone
            "notes":      text_model,     # reuses BERT + projection
        },
    )

Quickstart::

    from pyhealth.models.embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.datasets.collate import collate_temporal
    model = UnifiedMultimodalEmbeddingModel(dataset, embedding_dim=128)
    # inside forward:
    #   inputs = {field: {"value": Tensor, "time": Tensor, ...}, ...}
    out = model(inputs)
    # out["sequence"]: (B, S_total, 128)
    # out["mask"]:     (B, S_total)     , 1 = real event, 0 = padding
    # out["time"]:     (B, S_total)     , hours from first event
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from ...processors.base_processor import ModalityType, TemporalFeatureProcessor
from .base import BaseEmbeddingModel
from .vision import PatchEmbedding


# ── Helpers ───────────────────────────────────────────────────────────────────


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous sinusoidal embedding for scalar time values (in hours).

    Identical in spirit to the positional encoding in "Attention is All You
    Need" but operating on real-valued timestamps rather than integer positions.

    Args:
        dim: Output embedding dimension (must be even).
        max_hours: Maximum expected time value in hours.  Values are normalised
            to ``[0, 2π]`` before the sin/cos projection.  Default 720 (30 days).

    Shape:
        Input:  ``(*, )``  float tensor of times in hours
        Output: ``(*, dim)``
    """

    def __init__(self, dim: int, max_hours: float = 720.0):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim
        self.max_hours = max_hours
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)  # (dim//2,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """:param t: ``(...,)`` float, times in hours."""
        t_norm = t / self.max_hours * 2 * math.pi  # (...,)
        args = t_norm.unsqueeze(-1) * self.freqs  # (..., dim//2)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (..., dim)


class _MeanPool(nn.Module):
    """Pool a sequence of patch embeddings to a single vector via global mean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_patches, E) -> (B, E)
        return x.mean(dim=1)


# ── Main model ───────────────────────────────────────────────────────────────


class UnifiedMultimodalEmbeddingModel(nn.Module, BaseEmbeddingModel):
    """Embed heterogeneous temporal features into a single aligned sequence.

    **All** input processors must be ``TemporalFeatureProcessor`` subclasses.
    Non-temporal processors (e.g. ``SequenceProcessor``, ``MultiHotProcessor``)
    are rejected with a clear error, use :class:`EmbeddingModel` for those fields.

    Modality routing:

    - **CODE**: ``nn.Embedding`` lookup.
    - **TEXT**: Pretrained BERT (same approach as :class:`TextEmbeddingModel`),
      CLS token extracted per note.
    - **IMAGE**: :class:`PatchEmbedding` (from :class:`VisionEmbeddingModel`)
      followed by global mean pooling to produce one vector per image event.
    - **NUMERIC / SIGNAL**: ``nn.Linear`` projection.

    Unimodal model reuse:

    Pass pre-built :class:`EmbeddingModel`, :class:`VisionEmbeddingModel`, or
    :class:`TextEmbeddingModel` instances via ``field_embeddings`` to reuse
    their trained encoder weights instead of building new ones from scratch.
    The core encoder module is extracted from each pre-built model:

    - ``EmbeddingModel`` → ``embedding_layers[field_name]`` (``nn.Embedding`` /
      ``nn.Linear``)
    - ``VisionEmbeddingModel`` → ``embedding_layers[field_name]`` backbone +
      global mean pooling
    - ``TextEmbeddingModel`` → ``transformer`` (BERT) + ``fc`` (projection)

    Algorithm
    ---------
    For each temporal field:

    1. Route ``inputs[field]["value"]`` through a modality-specific encoder →
       ``(B, N_i, E')`` per-event embeddings.
    2. Retrieve ``inputs[field]["time"]`` → ``(B, N_i)`` timestamps (hours).
    3. (Optional) Retrieve ``inputs[field]["mask"]`` → ``(B, N_i, L)`` or
       ``(B, N_i)`` attention mask; reduced to event-level ``(B, N_i)`` if
       token-level.

    Then:

    4. Concatenate across all fields → ``(B, S_total, E')``.
    5. Sort events along dim=1 by timestamp (ascending).
    6. Add ``SinusoidalTimeEmbedding(time)`` + ``type_embedding(modality_idx)``.
    7. Return ``{"sequence", "time", "mask", "type_ids"}``.

    Args:
        processors: ``dict[field_name, TemporalFeatureProcessor]``, the
            processors for each temporal field in the dataset.  Pass
            ``dataset.input_processors`` directly.
        embedding_dim: Shared embedding dimension ``E'``.
        time_embedding: ``"sinusoidal"`` (default) or ``"learned"``.
        max_time_hours: Normalisation constant for the time embedding.
            Defaults to 720 h (30 days).
        image_size: Image size (H=W) assumed for IMAGE fields when using
            PatchEmbedding. Defaults to 224.
        image_channels: Number of input channels for IMAGE fields. Defaults to 3.
        patch_size: Patch size for IMAGE PatchEmbedding encoder. Defaults to 16.
        field_embeddings: Optional mapping of field names to pre-built unimodal
            embedding models.  Supported types:

            - :class:`EmbeddingModel` (codes / numeric) — extracts
              ``embedding_layers[field_name]``.
            - :class:`VisionEmbeddingModel` — extracts the backbone layer and
              wraps it with global mean pooling.
            - :class:`TextEmbeddingModel` — reuses ``transformer`` and ``fc``
              for BERT-based CLS extraction.

            Fields not present in this dict fall back to the default
            internally-built encoders.

    Example::

        model = UnifiedMultimodalEmbeddingModel(
            processors=dataset.input_processors,
            embedding_dim=128,
        )
        # inputs: {field: {"value": Tensor, "time": Tensor, "mask": Tensor}}
        out = model(inputs)
        seq = out["sequence"]   # (B, S_total, 128)
        mask = out["mask"]      # (B, S_total)  float, 1=valid 0=pad

        # With pre-built unimodal models:
        vision = VisionEmbeddingModel(dataset, embedding_dim=128)
        model = UnifiedMultimodalEmbeddingModel(
            processors=dataset.input_processors,
            embedding_dim=128,
            field_embeddings={"chest_xray": vision},
        )
    """

    def __init__(
        self,
        processors: dict[str, Any],
        embedding_dim: int = 128,
        time_embedding: str = "sinusoidal",
        max_time_hours: float = 720.0,
        image_size: int = 224,
        image_channels: int = 3,
        patch_size: int = 16,
        field_embeddings: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        _field_embeddings = field_embeddings or {}

        self.encoders: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        self.modality_types: dict[str, ModalityType] = {}
        self._shared_text_field_by_model: dict[str, str] = {}
        self._text_canonical: dict[str, str] = {}  # field → first field sharing the same tokenizer

        for field_name, processor in processors.items():
            if not isinstance(processor, TemporalFeatureProcessor):
                raise TypeError(
                    f"UnifiedMultimodalEmbeddingModel requires every input processor "
                    f"to be a TemporalFeatureProcessor subclass, but '{field_name}' "
                    f"uses {type(processor).__name__}.  For non-temporal fields use "
                    f"EmbeddingModel."
                )

            m = processor.modality()
            self.modality_types[field_name] = m
            pre_built = _field_embeddings.get(field_name)

            if m == ModalityType.CODE:
                self.encoders[field_name] = self._build_code_encoder(
                    field_name, processor, pre_built, embedding_dim
                )

            elif m == ModalityType.TEXT:
                self._build_text_encoder(
                    field_name, processor, pre_built, embedding_dim
                )

            elif m == ModalityType.IMAGE:
                self.encoders[field_name] = self._build_image_encoder(
                    field_name,
                    processor,
                    pre_built,
                    embedding_dim,
                    image_size,
                    image_channels,
                    patch_size,
                )

            elif m in (ModalityType.NUMERIC, ModalityType.SIGNAL):
                self.encoders[field_name] = self._build_numeric_encoder(
                    field_name, processor, pre_built, embedding_dim
                )

            else:
                raise NotImplementedError(
                    f"No encoder implemented for modality {m!r} (field '{field_name}')."
                )

        # Shared type embedding, one vector per unique modality in this dataset
        unique_modalities = sorted(set(self.modality_types.values()))
        self._modality_to_idx: dict[ModalityType, int] = {
            mod: i for i, mod in enumerate(unique_modalities)
        }
        self.type_embedding = nn.Embedding(len(unique_modalities), embedding_dim)
        self._warned_nested_code_flatten = False

        # Time embedding
        if time_embedding == "sinusoidal":
            self.time_embed = SinusoidalTimeEmbedding(embedding_dim, max_time_hours)
        else:
            raise NotImplementedError(
                "Only 'sinusoidal' time embedding is implemented."
            )

    # ── Encoder builders ──────────────────────────────────────────────────────

    def _build_code_encoder(
        self,
        field_name: str,
        processor: TemporalFeatureProcessor,
        pre_built: Any,
        embedding_dim: int,
    ) -> nn.Module:
        """Build CODE encoder: nn.Embedding, optionally from a pre-built EmbeddingModel."""
        if (
            pre_built is not None
            and hasattr(pre_built, "embedding_layers")
            and field_name in pre_built.embedding_layers
        ):
            layer = pre_built.embedding_layers[field_name]
            pre_dim = getattr(pre_built, "embedding_dim", embedding_dim)
            if pre_dim != embedding_dim:
                return nn.Sequential(layer, nn.Linear(pre_dim, embedding_dim))
            return layer

        vocab_size = processor.value_dim()
        return nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def _build_text_encoder(
        self,
        field_name: str,
        processor: TemporalFeatureProcessor,
        pre_built: Any,
        embedding_dim: int,
    ) -> None:
        """Build TEXT encoder: BERT + projection, optionally from TextEmbeddingModel."""

        def _set_projection(
            pre_dim: int, proj_source: Optional[nn.Module] = None
        ) -> None:
            if pre_dim != embedding_dim:
                if proj_source is not None:
                    self.projections[field_name] = nn.Sequential(
                        proj_source,
                        nn.Linear(pre_dim, embedding_dim),
                    )
                else:
                    self.projections[field_name] = nn.Linear(pre_dim, embedding_dim)
            elif proj_source is not None:
                self.projections[field_name] = proj_source

        if (
            pre_built is not None
            and hasattr(pre_built, "transformer")
            and hasattr(pre_built, "fc")
        ):
            self.encoders[field_name] = pre_built.transformer
            pre_dim = getattr(pre_built, "embedding_dim", embedding_dim)
            _set_projection(pre_dim, pre_built.fc)
            return

        if processor.is_token():
            from transformers import AutoModel

            tokenizer_model = getattr(processor, "tokenizer_model", None)
            if not tokenizer_model:
                raise ValueError(
                    f"TEXT processor '{field_name}' is token-based but does not "
                    "define tokenizer_model."
                )

            shared_field = self._shared_text_field_by_model.get(tokenizer_model)
            if shared_field is not None:
                # Second+ field with same tokenizer: reuse existing encoder, do NOT
                # register under a new key (avoids duplicate parameter registration).
                self._text_canonical[field_name] = shared_field
                shared_encoder = self.encoders[shared_field]
            else:
                shared_encoder = AutoModel.from_pretrained(tokenizer_model)
                self._shared_text_field_by_model[tokenizer_model] = field_name
                self.encoders[field_name] = shared_encoder

            hidden = shared_encoder.config.hidden_size
            _set_projection(hidden)
        else:
            raise ValueError(
                f"TEXT processor '{field_name}' must either supply a pre-built "
                f"TextEmbeddingModel via field_embeddings or use a tokenizer "
                f"(set tokenizer_model=...) to be used with "
                f"UnifiedMultimodalEmbeddingModel."
            )

    def _build_image_encoder(
        self,
        field_name: str,
        processor: TemporalFeatureProcessor,
        pre_built: Any,
        embedding_dim: int,
        image_size: int,
        image_channels: int,
        patch_size: int,
    ) -> nn.Module:
        """Build IMAGE encoder: backbone + mean pool, optionally from VisionEmbeddingModel."""
        if (
            pre_built is not None
            and hasattr(pre_built, "embedding_layers")
            and field_name in pre_built.embedding_layers
        ):
            backbone = pre_built.embedding_layers[field_name]
            pre_dim = getattr(pre_built, "embedding_dim", embedding_dim)
            if pre_dim != embedding_dim:
                return nn.Sequential(
                    backbone, _MeanPool(), nn.Linear(pre_dim, embedding_dim)
                )
            return nn.Sequential(backbone, _MeanPool())

        _image_size = getattr(processor, "image_size", image_size)
        _in_channels = getattr(processor, "in_channels", image_channels)
        return nn.Sequential(
            PatchEmbedding(_image_size, patch_size, _in_channels, embedding_dim),
            _MeanPool(),
        )

    def _build_numeric_encoder(
        self,
        field_name: str,
        processor: TemporalFeatureProcessor,
        pre_built: Any,
        embedding_dim: int,
    ) -> nn.Module:
        """Build NUMERIC/SIGNAL encoder: nn.Linear, optionally from EmbeddingModel."""
        if (
            pre_built is not None
            and hasattr(pre_built, "embedding_layers")
            and field_name in pre_built.embedding_layers
        ):
            layer = pre_built.embedding_layers[field_name]
            pre_dim = getattr(pre_built, "embedding_dim", embedding_dim)
            if pre_dim != embedding_dim:
                return nn.Sequential(layer, nn.Linear(pre_dim, embedding_dim))
            return layer

        in_features = processor.value_dim()
        return nn.Linear(in_features, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Encode and temporally align all temporal features.

        Args:
            inputs: ``{field_name: {"value": Tensor, "time": Tensor,
                        "mask": Tensor (optional)}}``
               , one dict per temporal feature, exactly as produced by
                  ``collate_temporal``.

        Returns:
            A dict with keys:

            * ``"sequence"``, ``(B, S_total, E')``  temporally-sorted events
            * ``"time"``    , ``(B, S_total)``       timestamps (hours)
            * ``"mask"``    , ``(B, S_total)``       1=real event, 0=padding
            * ``"type_ids"``, ``(B, S_total)``       modality index per event
        """
        all_embeddings: list[torch.Tensor] = []
        all_times: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_types: list[torch.Tensor] = []

        for field_name, feat_dict in inputs.items():
            value = feat_dict["value"]  # (B, N_i, ...) or (B, S, F)
            time = feat_dict["time"]  # (B, N_i)
            mask = feat_dict.get("mask")

            if time is None:
                # Fallback: treat every event as occurring at t=0
                time = torch.zeros(value.shape[:2], device=value.device)

            modality = self.modality_types[field_name]
            encoder_key = self._text_canonical.get(field_name, field_name)
            encoder = self.encoders[encoder_key]

            # ── Encode ────────────────────────────────────────────────────
            if modality == ModalityType.CODE:
                # CODE values may be either:
                # - flat indices: (B, S)
                # - nested indices: (B, S, C) where C is codes-per-event
                # For nested indices, flatten to (B, S*C, E') so code-level
                # detail is preserved, and expand time/mask to match.
                if value.dim() == 2:
                    emb = encoder(value)  # (B, S, E')
                elif value.dim() == 3:
                    bsz, seq_len, per_event_codes = value.shape
                    token_emb = encoder(value.long())  # (B, S, C, E')
                    emb = token_emb.reshape(bsz, seq_len * per_event_codes, -1)

                    if not self._warned_nested_code_flatten:
                        warnings.warn(
                            (
                                "UnifiedMultimodalEmbeddingModel detected "
                                f"nested CODE input for '{field_name}' with "
                                f"shape={tuple(value.shape)}. Flattening to "
                                f"(B, S*C, E) and repeating time along C."
                            ),
                            stacklevel=2,
                        )
                        self._warned_nested_code_flatten = True

                    if time is not None:
                        time = (
                            time.unsqueeze(-1)
                            .expand(-1, -1, per_event_codes)
                            .reshape(bsz, seq_len * per_event_codes)
                        )

                    if mask is not None:
                        if mask.dim() == 2:
                            mask = (
                                mask.unsqueeze(-1)
                                .expand(-1, -1, per_event_codes)
                                .reshape(bsz, seq_len * per_event_codes)
                            )
                        elif mask.dim() == 3:
                            mask = mask.reshape(bsz, seq_len * per_event_codes)
                else:
                    raise ValueError(
                        f"Unsupported CODE value rank for '{field_name}': "
                        f"shape={tuple(value.shape)}"
                    )

            elif modality == ModalityType.TEXT:
                b, n, l = value.shape
                flat_ids = value.view(b * n, l)
                flat_mask = mask.view(b * n, l) if mask is not None else None
                out = encoder(input_ids=flat_ids, attention_mask=flat_mask)
                cls_emb = out.last_hidden_state[:, 0, :]  # (B*N, H)
                if field_name in self.projections:
                    cls_emb = self.projections[field_name](cls_emb)
                emb = cls_emb.view(b, n, -1)  # (B, N, E')

            elif modality == ModalityType.IMAGE:
                # encoder = Sequential(PatchEmbedding, _MeanPool) → (B*N, E')
                b, n, c, h, w = value.shape
                flat_imgs = value.view(b * n, c, h, w)
                img_emb = encoder(flat_imgs)  # (B*N, E')
                emb = img_emb.view(b, n, -1)  # (B, N, E')

            else:  # NUMERIC / SIGNAL
                emb = encoder(value)  # (B, T, E')

            # ── Build event-level validity mask ───────────────────────────
            if mask is None:
                event_mask = torch.ones(emb.shape[:2], device=emb.device)
            else:
                if mask.dim() > time.dim():
                    # token-level (B, N, L) → event-level (B, N)
                    event_mask = (mask.sum(dim=-1) > 0).float()
                else:
                    event_mask = mask.float()

            # ── Modality type indices ─────────────────────────────────────
            type_idx = self._modality_to_idx[modality]
            type_ids = torch.full(
                emb.shape[:2], type_idx, dtype=torch.long, device=emb.device
            )

            all_embeddings.append(emb)
            all_times.append(time)
            all_masks.append(event_mask)
            all_types.append(type_ids)

        # ── Concatenate across all fields ─────────────────────────────────
        cat_emb = torch.cat(all_embeddings, dim=1)  # (B, S_total, E')
        cat_time = torch.cat(all_times, dim=1)  # (B, S_total)
        cat_mask = torch.cat(all_masks, dim=1)  # (B, S_total)
        cat_types = torch.cat(all_types, dim=1)  # (B, S_total)

        # ── Sort by time ──────────────────────────────────────────────────
        sort_idx = cat_time.argsort(dim=1)
        cat_emb = cat_emb.gather(1, sort_idx.unsqueeze(-1).expand_as(cat_emb))
        cat_time = cat_time.gather(1, sort_idx)
        cat_mask = cat_mask.gather(1, sort_idx)
        cat_types = cat_types.gather(1, sort_idx)

        # ── Add time + type embeddings ────────────────────────────────────
        time_emb = self.time_embed(cat_time)  # (B, S_total, E')
        type_emb = self.type_embedding(cat_types)  # (B, S_total, E')
        final = cat_emb + time_emb + type_emb  # (B, S_total, E')

        return {
            "sequence": final,  # (B, S_total, E')
            "time": cat_time,  # (B, S_total)
            "mask": cat_mask,  # (B, S_total)
            "type_ids": cat_types,  # (B, S_total)
        }
