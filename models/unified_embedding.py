"""UnifiedMultimodalEmbeddingModel — temporally aligned multimodal embedding.

Takes K temporal features ( dict outputs from ``TemporalFeatureProcessor``
subclasses ), embeds each event with a modality-specific encoder, then
interleaves all events on a shared timeline by sorting on timestamp and adding
sinusoidal time embeddings + learned modality-type embeddings.

Output shape: ``(B, S_total, E')`` — a single sequence of events usable by
any downstream sequence model (Transformer, Mamba, RNN, …).

Quickstart::

    from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
    from pyhealth.datasets.collate import collate_temporal
    model = UnifiedMultimodalEmbeddingModel(dataset, embedding_dim=128)
    # inside forward:
    #   inputs = {field: {"value": Tensor, "time": Tensor, ...}, ...}
    out = model(inputs)
    # out["sequence"]: (B, S_total, 128)
    # out["mask"]:     (B, S_total)      — 1 = real event, 0 = padding
    # out["time"]:     (B, S_total)      — hours from first event
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from pyhealth.processors.base_processor import ModalityType, TemporalFeatureProcessor


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
        t_norm = t / self.max_hours * 2 * math.pi           # (...,)
        args = t_norm.unsqueeze(-1) * self.freqs             # (..., dim//2)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (..., dim)


def _build_image_encoder(embedding_dim: int) -> nn.Module:
    """Lightweight 5-layer CNN encoder: C × H × W → embedding_dim.

    Uses ``torchvision.models.resnet18`` pre-trained backbone, strips the
    final FC layer, and adds a projection to ``embedding_dim``.  Falls back to
    a toy Conv-pool-flatten network if torchvision is not installed.
    """
    try:
        import torchvision.models as tv

        backbone = tv.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, embedding_dim)
        return backbone
    except ImportError:
        # Minimal fallback: single conv → global avg pool → linear
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
        )


# ── Main model ───────────────────────────────────────────────────────────────


class UnifiedMultimodalEmbeddingModel(nn.Module):
    """Embed heterogeneous temporal features into a single aligned sequence.

    **All** input processors must be ``TemporalFeatureProcessor`` subclasses.
    Non-temporal processors (e.g. ``SequenceProcessor``, ``MultiHotProcessor``)
    are rejected with a clear error — use the existing ``EmbeddingModel`` for
    those fields.

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
        processors: ``dict[field_name, TemporalFeatureProcessor]`` — the
            processors for each temporal field in the dataset.  Pass
            ``dataset.input_processors`` directly.
        embedding_dim: Shared embedding dimension ``E'``.
        time_embedding: ``"sinusoidal"`` (default) or ``"learned"``.
        max_time_hours: Normalisation constant for the time embedding.
            Defaults to 720 h (30 days).

    Example::

        model = UnifiedMultimodalEmbeddingModel(
            processors=dataset.input_processors,
            embedding_dim=128,
        )
        # inputs: {field: {"value": Tensor, "time": Tensor, "mask": Tensor}}
        out = model(inputs)
        seq = out["sequence"]   # (B, S_total, 128)
        mask = out["mask"]      # (B, S_total)  float, 1=valid 0=pad
    """

    def __init__(
        self,
        processors: dict[str, Any],
        embedding_dim: int = 128,
        time_embedding: str = "sinusoidal",
        max_time_hours: float = 720.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoders: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        self.modality_types: dict[str, ModalityType] = {}

        for field_name, processor in processors.items():
            if not isinstance(processor, TemporalFeatureProcessor):
                raise TypeError(
                    f"UnifiedMultimodalEmbeddingModel requires every input processor "
                    f"to be a TemporalFeatureProcessor subclass, but '{field_name}' "
                    f"uses {type(processor).__name__}.  For non-temporal fields use "
                    f"the existing EmbeddingModel."
                )

            m = processor.modality()
            self.modality_types[field_name] = m

            if m == ModalityType.CODE:
                vocab_size = processor.value_dim()
                self.encoders[field_name] = nn.Embedding(
                    vocab_size, embedding_dim, padding_idx=0
                )

            elif m == ModalityType.TEXT:
                if processor.is_token():
                    from transformers import AutoModel

                    bert = AutoModel.from_pretrained(processor.tokenizer_model)
                    self.encoders[field_name] = bert
                    hidden = bert.config.hidden_size
                    if hidden != embedding_dim:
                        self.projections[field_name] = nn.Linear(hidden, embedding_dim)
                else:
                    raise ValueError(
                        f"TEXT processor '{field_name}' must use a tokenizer "
                        f"(set tokenizer_model=...) to be used with "
                        f"UnifiedMultimodalEmbeddingModel."
                    )

            elif m == ModalityType.IMAGE:
                self.encoders[field_name] = _build_image_encoder(embedding_dim)

            elif m in (ModalityType.NUMERIC, ModalityType.SIGNAL):
                in_features = processor.value_dim()
                self.encoders[field_name] = nn.Linear(in_features, embedding_dim)

            else:
                raise NotImplementedError(
                    f"No encoder implemented for modality {m!r} (field '{field_name}')."
                )

        # Shared type embedding — one vector per unique modality in this dataset
        unique_modalities = sorted(set(self.modality_types.values()))
        self._modality_to_idx: dict[ModalityType, int] = {
            mod: i for i, mod in enumerate(unique_modalities)
        }
        self.type_embedding = nn.Embedding(len(unique_modalities), embedding_dim)

        # Time embedding
        if time_embedding == "sinusoidal":
            self.time_embed = SinusoidalTimeEmbedding(embedding_dim, max_time_hours)
        else:
            raise NotImplementedError("Only 'sinusoidal' time embedding is implemented.")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Encode and temporally align all temporal features.

        Args:
            inputs: ``{field_name: {"value": Tensor, "time": Tensor,
                        "mask": Tensor (optional)}}``
                — one dict per temporal feature, exactly as produced by
                  ``collate_temporal``.

        Returns:
            A dict with keys:

            * ``"sequence"`` — ``(B, S_total, E')``  temporally-sorted events
            * ``"time"``     — ``(B, S_total)``       timestamps (hours)
            * ``"mask"``     — ``(B, S_total)``       1=real event, 0=padding
            * ``"type_ids"`` — ``(B, S_total)``       modality index per event
        """
        all_embeddings: list[torch.Tensor] = []
        all_times:      list[torch.Tensor] = []
        all_masks:      list[torch.Tensor] = []
        all_types:      list[torch.Tensor] = []

        for field_name, feat_dict in inputs.items():
            value = feat_dict["value"]   # (B, N_i, ...) or (B, S, F)
            time  = feat_dict["time"]    # (B, N_i)
            mask  = feat_dict.get("mask")

            if time is None:
                # Fallback: treat every event as occurring at t=0
                time = torch.zeros(value.shape[:2], device=value.device)

            modality = self.modality_types[field_name]
            encoder  = self.encoders[field_name]

            # ── Encode ────────────────────────────────────────────────────
            if modality == ModalityType.CODE:
                emb = encoder(value)                           # (B, S, E')

            elif modality == ModalityType.TEXT:
                b, n, l = value.shape
                flat_ids  = value.view(b * n, l)
                flat_mask = mask.view(b * n, l) if mask is not None else None
                out       = encoder(input_ids=flat_ids, attention_mask=flat_mask)
                cls_emb   = out.last_hidden_state[:, 0, :]    # (B*N, H)
                if field_name in self.projections:
                    cls_emb = self.projections[field_name](cls_emb)
                emb = cls_emb.view(b, n, -1)                  # (B, N, E')

            elif modality == ModalityType.IMAGE:
                b, n, c, h, w = value.shape
                flat_imgs = value.view(b * n, c, h, w)
                img_emb   = encoder(flat_imgs)                 # (B*N, E')
                emb       = img_emb.view(b, n, -1)

            else:  # NUMERIC / SIGNAL
                emb = encoder(value)                           # (B, T, E')

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
            type_idx  = self._modality_to_idx[modality]
            type_ids  = torch.full(
                emb.shape[:2], type_idx, dtype=torch.long, device=emb.device
            )

            all_embeddings.append(emb)
            all_times.append(time)
            all_masks.append(event_mask)
            all_types.append(type_ids)

        # ── Concatenate across all fields ─────────────────────────────────
        cat_emb   = torch.cat(all_embeddings, dim=1)   # (B, S_total, E')
        cat_time  = torch.cat(all_times,      dim=1)   # (B, S_total)
        cat_mask  = torch.cat(all_masks,      dim=1)   # (B, S_total)
        cat_types = torch.cat(all_types,      dim=1)   # (B, S_total)

        # ── Sort by time ──────────────────────────────────────────────────
        sort_idx   = cat_time.argsort(dim=1)
        cat_emb    = cat_emb.gather(
            1, sort_idx.unsqueeze(-1).expand_as(cat_emb)
        )
        cat_time   = cat_time.gather(1, sort_idx)
        cat_mask   = cat_mask.gather(1, sort_idx)
        cat_types  = cat_types.gather(1, sort_idx)

        # ── Add time + type embeddings ────────────────────────────────────
        time_emb = self.time_embed(cat_time)             # (B, S_total, E')
        type_emb = self.type_embedding(cat_types)        # (B, S_total, E')
        final    = cat_emb + time_emb + type_emb         # (B, S_total, E')

        return {
            "sequence": final,      # (B, S_total, E')
            "time":     cat_time,   # (B, S_total)
            "mask":     cat_mask,   # (B, S_total)
            "type_ids": cat_types,  # (B, S_total)
        }
