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
import warnings
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn

from pyhealth.processors.base_processor import ModalityType, TemporalFeatureProcessor
from pyhealth.models.vision_embedding import PatchEmbedding


# ── Helpers ───────────────────────────────────────────────────────────────────


class SinusoidalTimeEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding for times in hours.

    Wavelengths are spaced geometrically from ``min_hours`` (within-stay
    resolution) to ``max_hours`` (longitudinal span). The previous encoding
    mapped ``t / 720 * 2π``, so every frequency wrapped every 30 days: a later
    stay at +9606h produced the same embedding as +6h even after the task
    stopped resetting the clock.

    Examples:
        >>> embedding = SinusoidalTimeEmbedding(dim=2, max_hours=24.0)
        >>> output = embedding(torch.tensor([0.0, 6.0]))
        >>> output.shape
        torch.Size([2, 2])
        >>> torch.isfinite(output).all().item()
        True

    Args:
        dim: Output embedding dimension (must be even).
        max_hours: Longest wavelength in hours. Default 87600 (10 years).
        min_hours: Shortest wavelength in hours. Default 1.0.

    Shape:
        Input:  ``(*, )``  float tensor of times in hours
        Output: ``(*, dim)``
    """

    def __init__(
        self,
        dim: int,
        max_hours: float = 87600.0,
        min_hours: float = 1.0,
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        if min_hours <= 0 or max_hours <= min_hours:
            raise ValueError(
                f"need 0 < min_hours < max_hours, got min={min_hours}, max={max_hours}"
            )
        self.dim = dim
        self.max_hours = float(max_hours)
        self.min_hours = float(min_hours)
        half = dim // 2
        periods = torch.exp(
            torch.linspace(
                math.log(self.min_hours),
                math.log(self.max_hours),
                half,
                dtype=torch.float32,
            )
        )
        self.register_buffer("freqs", 2 * math.pi / periods)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """:param t: ``(...,)`` float, times in hours."""
        args = t.unsqueeze(-1).to(dtype=self.freqs.dtype) * self.freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _MeanPool(nn.Module):
    """Pool a sequence of patch embeddings to a single vector via global mean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_patches, E) -> (B, E)
        return x.mean(dim=1)


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

    Unimodal model reuse via ``field_embeddings``::

        vision_model = VisionEmbeddingModel(dataset, embedding_dim=128)
        unified = UnifiedMultimodalEmbeddingModel(
            processors=dataset.input_processors,
            embedding_dim=128,
            field_embeddings={"chest_xray": vision_model},
        )

    Args:
        processors: ``dict[field_name, TemporalFeatureProcessor]`` — the
            processors for each temporal field in the dataset.  Pass
            ``dataset.input_processors`` directly.
        embedding_dim: Shared embedding dimension ``E'``.
        time_embedding: ``"sinusoidal"`` (default) or ``"learned"``.
        max_time_hours: Longest wavelength of the time embedding, in hours.
            Defaults to 87600 (10 years). Shortest wavelength is 1 hour.
        image_size: Image size (H=W) assumed for IMAGE fields when using
            PatchEmbedding. Defaults to 224.
        image_channels: Number of input channels for IMAGE fields. Defaults to 3.
        patch_size: Patch size for IMAGE PatchEmbedding encoder. Defaults to 16.
        image_pool: Pooling strategy applied to IMAGE patch tokens to produce
            one vector per image event. Only ``"mean"`` (global mean pooling)
            is currently implemented. Defaults to ``"mean"``.
        field_embeddings: Optional mapping of field names to pre-built unimodal
            embedding models (``EmbeddingModel``, ``VisionEmbeddingModel``, or
            ``TextEmbedding``). The core encoder module is extracted from each
            pre-built model; fields not present here use internally-built
            encoders. Defaults to None.
        freeze_text_encoder: If True, freeze TEXT encoder weights
            (``requires_grad=False``) so notes encode deterministically.
            Defaults to False.
        normalize_content: If True, apply parameter-free layer norm to the
            content term so one modality's raw magnitude cannot drown out the
            time/type terms. Defaults to True. This CHANGES outputs versus the
            previous unnormalized sum (content + time + type): the content
            term is rescaled to unit variance per event before the time and
            type terms are added. The default is True because raw feature
            magnitude is an accident of feature scaling, not a modelling
            decision.
        numeric_standardizers: Optional mapping of NUMERIC/SIGNAL field names
            to modules called as ``standardizer(value, observed)`` BEFORE the
            projection, where ``observed`` comes from the sibling
            ``{field}_mask`` field. Defaults to None. A ``{field}_mask`` input
            is skipped as a modality ONLY when ``field`` has a registered
            entry here; a field merely named ``x_mask`` with no standardizer
            stays a normal modality.

    Example::

        model = UnifiedMultimodalEmbeddingModel(
            processors=dataset.input_processors,
            embedding_dim=128,
        )
        # inputs: {field: {"value": Tensor, "time": Tensor, "mask": Tensor}}
        out = model(inputs)
        seq = out["sequence"]   # (B, S_total, 128)
        mask = out["mask"]      # (B, S_total)  float, 1=valid 0=pad

    Minimal example:

        >>> import torch
        >>> from pyhealth.processors import StageNetProcessor
        >>> from pyhealth.models.unified_embedding import (
        ...     UnifiedMultimodalEmbeddingModel,
        ... )
        >>> proc = StageNetProcessor()
        >>> proc.fit([{"codes": ([0.0], ["A"])}], "codes")
        >>> model = UnifiedMultimodalEmbeddingModel({"codes": proc}, embedding_dim=8)
        >>> out = model({"codes": {
        ...     "value": torch.zeros(1, 1, dtype=torch.long),
        ...     "time": torch.zeros(1, 1),
        ... }})
        >>> out["sequence"].shape
        torch.Size([1, 1, 8])
    """

    def __init__(
        self,
        processors: dict[str, Any],
        embedding_dim: int = 128,
        time_embedding: str = "sinusoidal",
        max_time_hours: float = 87600.0,
        image_size: int = 224,
        image_channels: int = 3,
        patch_size: int = 16,
        image_pool: str = "mean",
        field_embeddings: dict[str, Any] | None = None,
        freeze_text_encoder: bool = False,
        normalize_content: bool = True,
        numeric_standardizers: dict[str, Any] | None = None,
    ):
        super().__init__()
        if image_pool != "mean":
            raise NotImplementedError(
                f"Only image_pool='mean' is implemented, got {image_pool!r}."
            )
        self._embedding_dim = embedding_dim
        self._freeze_text_encoder = freeze_text_encoder
        self._frozen_text_fields: set[str] = set()
        self.image_pool = image_pool
        self.normalize_content = normalize_content
        # Statistics live in buffers, so they travel in state_dict. A checkpoint
        # therefore applies at inference the same transform it trained under.
        self.numeric_standardizers = nn.ModuleDict(numeric_standardizers or {})
        _field_embeddings = field_embeddings or {}

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
            pre_built = _field_embeddings.get(field_name)

            if m == ModalityType.CODE:
                self.encoders[field_name] = self._build_code_encoder(
                    field_name, processor, pre_built, embedding_dim
                )

            elif m == ModalityType.TEXT:
                self._build_text_encoder(
                    field_name, processor, pre_built, embedding_dim,
                    freeze=freeze_text_encoder,
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
                    image_pool,
                )

            elif m in (ModalityType.NUMERIC, ModalityType.SIGNAL):
                self.encoders[field_name] = self._build_numeric_encoder(
                    field_name, processor, pre_built, embedding_dim
                )

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
        self._warned_nested_code_flatten = False

        # Time embedding
        if time_embedding == "sinusoidal":
            self.time_embed = SinusoidalTimeEmbedding(embedding_dim, max_time_hours)
        else:
            raise NotImplementedError("Only 'sinusoidal' time embedding is implemented.")

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
        freeze: bool = False,
    ) -> None:
        """Build TEXT encoder: BERT + projection, optionally from TextEmbedding."""

        def _set_projection(
            pre_dim: int, proj_source: nn.Module | None = None
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
            if freeze:
                for p in pre_built.transformer.parameters():
                    p.requires_grad = False
                self._frozen_text_fields.add(field_name)
            pre_dim = getattr(pre_built, "embedding_dim", embedding_dim)
            _set_projection(pre_dim, pre_built.fc)
            return

        if processor.is_token():
            from transformers import AutoModel

            bert = AutoModel.from_pretrained(processor.tokenizer_model)
            if freeze:
                for p in bert.parameters():
                    p.requires_grad = False
                self._frozen_text_fields.add(field_name)
            self.encoders[field_name] = bert
            hidden = bert.config.hidden_size
            if hidden != embedding_dim:
                self.projections[field_name] = nn.Linear(hidden, embedding_dim)
        else:
            raise ValueError(
                f"TEXT processor '{field_name}' must either supply a pre-built "
                f"TextEmbedding via field_embeddings or use a tokenizer "
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
        image_pool: str,
    ) -> nn.Module:
        """Build IMAGE encoder: backbone + pool, optionally from VisionEmbeddingModel."""
        pool_layers: dict[str, nn.Module] = {"mean": _MeanPool()}
        pool_layer = pool_layers[image_pool]

        if (
            pre_built is not None
            and hasattr(pre_built, "embedding_layers")
            and field_name in pre_built.embedding_layers
        ):
            backbone = pre_built.embedding_layers[field_name]
            pre_dim = getattr(pre_built, "embedding_dim", embedding_dim)
            if pre_dim != embedding_dim:
                return nn.Sequential(
                    backbone, pool_layer, nn.Linear(pre_dim, embedding_dim)
                )
            return nn.Sequential(backbone, pool_layer)

        _image_size = getattr(processor, "image_size", image_size)
        _in_channels = getattr(processor, "in_channels", image_channels)
        return nn.Sequential(
            PatchEmbedding(_image_size, patch_size, _in_channels, embedding_dim),
            pool_layer,
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

    def train(self, mode: bool = True) -> UnifiedMultimodalEmbeddingModel:
        """Keep a frozen text encoder in eval mode.

        ``nn.Module.train()`` would enable dropout inside the encoder. Its
        output would then change between passes even though every weight has
        ``requires_grad=False``. That makes cached embeddings incorrect, and
        it also makes a frozen encoder give a different answer to the same input.
        """
        super().train(mode)
        for field_name in self._frozen_text_fields:
            self.encoders[field_name].eval()
        return self

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
            * ``"token_emb"`` — ``(B, S_total, E')`` content-only event embedding
              (before time/type are added); the target for masked modeling.
        """
        all_embeddings: list[torch.Tensor] = []
        all_times:      list[torch.Tensor] = []
        all_masks:      list[torch.Tensor] = []
        all_types:      list[torch.Tensor] = []

        for field_name, feat_dict in inputs.items():
            if field_name.endswith("_mask") and field_name[: -len("_mask")] in self.numeric_standardizers:
                # Observation-flag sibling consumed by the standardiser; not a
                # modality of its own. Encoding it would duplicate every lab
                # timestamp with a 0/1 vector. Skipped ONLY when the base
                # field registered a standardiser: a field merely named
                # ``x_mask`` with no standardiser stays a normal modality.
                continue
            value = feat_dict["value"]  # (B, N_i, ...) or (B, S, F)
            time = feat_dict["time"]  # (B, N_i)
            # Three different masks meet here and must not be conflated.
            #   mask      token level, from the processor schema; this is the
            #             attention mask a text encoder needs.
            #   pad_mask  event level, from the collator; which slots are real
            #             events rather than batch padding.
            #   {field}_mask  a separate FIELD meaning "was this value
            #             observed", consumed by the standardiser below.
            mask = feat_dict.get("mask")

            if time is None:
                # Fallback: treat every event as occurring at t=0
                time = torch.zeros(value.shape[:2], device=value.device)

            modality = self.modality_types[field_name]
            encoder  = self.encoders[field_name]

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
                flat_ids  = value.view(b * n, l)
                flat_mask = mask.view(b * n, l) if mask is not None else None
                out       = encoder(input_ids=flat_ids, attention_mask=flat_mask)
                cls_emb   = out.last_hidden_state[:, 0, :]    # (B*N, H)
                if field_name in self.projections:
                    cls_emb = self.projections[field_name](cls_emb)
                # cls_emb.shape[-1] instead of -1: when b*n == 0 (no note
                # slots in this batch), the -1 is ambiguous for a 0-element
                # tensor and torch raises.
                emb = cls_emb.view(b, n, cls_emb.shape[-1])  # (B, N, E')

            elif modality == ModalityType.IMAGE:
                b, n, c, h, w = value.shape
                flat_imgs = value.view(b * n, c, h, w)
                if b * n == 0:
                    # No image slots in this batch. reshape(0, -1) is
                    # ambiguous since -1 can be any value for 0 elements.
                    img_emb = value.new_zeros((0, self._embedding_dim))
                else:
                    img_emb = encoder(flat_imgs)               # (B*N, E')
                # Explicit dim instead of -1: same 0-element ambiguity.
                emb = img_emb.view(b, n, img_emb.shape[-1])

            else:  # NUMERIC / SIGNAL
                # Standardise BEFORE the projection. The projection mixes the
                # features, so a transform after it cannot correct a feature
                # whose physical unit gives it 300 times the magnitude of
                # another.
                # nn.ModuleDict has no .get(), so the ternary stays.
                standardizer = (
                    self.numeric_standardizers[field_name]  # noqa: SIM401
                    if field_name in self.numeric_standardizers
                    else None
                )
                if standardizer is not None:
                    # Observation flags live in the sibling ``{field}_mask``
                    # FIELD, not in this field's dict. Reading the padding mask
                    # here would tell the standardiser that every real event
                    # was measured, which is exactly the distinction the
                    # standardiser exists to preserve.
                    sibling = inputs.get(f"{field_name}_mask")
                    obs = sibling["value"] if isinstance(sibling, dict) else None
                    if obs is None:
                        raise ValueError(
                            f"The standardiser for {field_name!r} needs a paired "
                            f"{field_name}_mask field in the batch."
                        )
                    if obs.shape != value.shape:
                        raise ValueError(
                            f"{field_name}_mask has shape {tuple(obs.shape)}, "
                            f"which does not match {field_name} "
                            f"{tuple(value.shape)}."
                        )
                    value = standardizer(value, obs.bool())
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
        time_emb = self.time_embed(cat_time)  # (B, S_total, E')
        type_emb = self.type_embedding(cat_types)  # (B, S_total, E')
        if self.normalize_content:
            # Put the content term on the scale of the additive terms. Without
            # this the sum is decided by whichever modality has the larger
            # magnitude, which is an accident of feature scaling and not a
            # modelling decision. F.layer_norm without weight or bias adds NO
            # parameters, so an existing checkpoint still loads.
            cat_emb = F.layer_norm(cat_emb, (cat_emb.shape[-1],))
        final = cat_emb + time_emb + type_emb
        # Zero the padded slots so a consumer that ignores the mask, such as a
        # mean pool, still cannot pick them up.
        final = final * cat_mask.unsqueeze(-1).to(final.dtype)  # (B, S_total, E')

        return {
            "sequence": final,      # (B, S_total, E')
            "time":     cat_time,   # (B, S_total)
            "mask":     cat_mask,   # (B, S_total)
            "type_ids": cat_types,  # (B, S_total)
            # Per-event content embedding BEFORE time/type are added (same sort
            # order as ``sequence``).  Masked-modeling pretrainers should
            # reconstruct THIS rather than ``sequence``: the time/type
            # components are largely recoverable from event position, so
            # including them in the target dilutes the content signal.
            "token_emb": cat_emb,   # (B, S_total, E')
        }
