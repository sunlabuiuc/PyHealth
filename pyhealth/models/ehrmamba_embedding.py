"""
EHRMamba Section 2.2 + §2.1 Embeddings for MIMIC-IV.

This module contains only the embedding-side components of EHRMamba:
  - Token type constants and special token strings (shared with the task file
    via import).
  - LabQuantizer     — 5-bin lab tokenization (paper Appx. B).
  - TimeEmbeddingLayer  — learnable sinusoidal time/age embedding.
  - VisitEmbedding      — learned visit-segment embedding.
  - EHRMambaEmbedding   — full §2.2 / Eq. 1 embedding module.
  - EHRMambaEmbeddingAdapter — drop-in EmbeddingModel adapter for EHRMamba.

Implements the full 7-component token embedding scheme from the EHR Mamba paper (§2.2,
Equation 1) and Appendix C.2:

    E_{v,j} = E_concept + E_type + E_age + E_time + E_segment + E_visit_order + E_position

In practice (per Appx. C.2) concept, time, and age are first fused via concat+project:

    e_fused   = Tanh(W_proj · concat(e_code, e_time_delta, e_age))   ← Appx. C.2
    e_token   = e_fused + e_type + e_visit_order + e_visit_segment + e_position
    output    = LayerNorm(Dropout(e_token))

Sequence structure is built by MIMIC4EHRMambaTask:
  - One global [CLS] at the start of the FULL patient sequence.
  - Each visit is bracketed by [VS] (visit start) and [VE] (visit end) tokens.
  - [REG] register token inserted after each [VE] (§2.1, used as prediction anchor).
  - Inter-visit discrete time-interval tokens between [REG] and [VS]:
      [W0]–[W3] for gaps < 4 weeks, [M1]–[M12] for 1–12 months, [LT] for > 1 year.
  Token type vocabulary extended to 10 types to cover all structural tokens.

As per section §2.2 in the EHRMamba paper, the following subset of special tokens
receive zero-vectors for time, age, visit_order, and visit_segment embeddings:
  - [CLS], [VS], [VE], [REG], and time-interval tokens

Integration with pyhealth.models.ehrmamba.EHRMamba:
    Replace `self.embedding_model = EmbeddingModel(dataset, embedding_dim)` with
    `self.embedding_model = EHRMambaEmbeddingAdapter(
         EHRMambaEmbedding(vocab_size, embedding_dim))`.
    Before calling embedding_model(embedding_inputs), call
    `self.embedding_model.set_aux_inputs(type_ids, time_stamps, ages,
                                          visit_orders, visit_segments)`
    so the auxiliary tensors are cached for the forward pass.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Union
import os

import torch
import torch.nn as nn

from ..datasets import SampleDataset
from ..processors import (
    MultiHotProcessor,
    NestedFloatsProcessor,
    NestedSequenceProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
    DeepNestedSequenceProcessor,
    DeepNestedFloatsProcessor,
)
from .base_model import BaseModel

# ── MIMIC-IV token type IDs (paper §2.1 + §2.2 + Appx.B + Appx.E) ──────────
# Clinical types start at index 6; special tokens occupy indices 0–5.
MIMIC4_TOKEN_TYPES: Dict[str, int] = {
    "PAD":           0,   # padding
    "CLS":           1,   # global sequence start (one per patient)
    "VS":            2,   # visit start  [VS]
    "VE":            3,   # visit end    [VE]
    "REG":           4,   # register token [REG] — follows each [VE]
    "TIME_INTERVAL": 5,   # [W0]–[W3], [M1]–[M12], [LT] between visits
    "procedures_icd": 6,  # procedure (P) — ICD procedure code
    "prescriptions":  7,  # medication (M) — drug name
    "labevents":      8,  # lab result (L) — binned lab itemid
    "other":          9,  # reserved
}
NUM_TOKEN_TYPES: int = len(MIMIC4_TOKEN_TYPES)  # 10

# Threshold: type_ids <= SPECIAL_TYPE_MAX receive zero aux embeddings (§2.2)
SPECIAL_TYPE_MAX: int = 5  # PAD through TIME_INTERVAL are all structural/special

# ── Visit segment (paper §2.2) ────────────────────────────────────────────────
NUM_VISIT_SEGMENTS: int = 3   # 0=PAD/special, 1=segment_1, 2=segment_2
MAX_NUM_VISITS: int = 512      # cap on visit_order embedding table size

# ── Special token strings — reference only (defined and used in mimic4_ehr_mamba_task.py) ──
# The task constructs the full patient sequence in this order (paper §2.1 / Appx.E):
#
#   [CLS]                                ← one global start token (type_id 1)
#   [VS]  PR:xxx  RX:xxx  LB:xxx_bin2  [VE]  [REG]   ← visit 1 (segment 1)
#   [W2]                                 ← inter-visit time gap token (type_id 5)
#   [VS]  LB:xxx_bin0  PR:xxx          [VE]  [REG]   ← visit 2 (segment 2)
#   [M1]                                 ← inter-visit time gap token (type_id 5)
#   [VS]  …                             [VE]  [REG]   ← visit N (segment 1/2 alternating)
#
# SequenceProcessor converts these strings → integer vocab indices before this
# module receives them.  The embedding model works with those integers only;
# it never references the strings below directly.
#
# _CLS_TOKEN = "[CLS]"   type_id 1  — one per patient, zeroed aux embeddings
# _VS_TOKEN  = "[VS]"    type_id 2  — visit start,     zeroed aux embeddings
# _VE_TOKEN  = "[VE]"    type_id 3  — visit end,       zeroed aux embeddings
# _REG_TOKEN = "[REG]"   type_id 4  — register token,  zeroed aux embeddings
#                        type_id 5  — time-interval ([W0]–[W3],[M1]–[M12],[LT])
# All type_ids 0–5 satisfy type_id <= SPECIAL_TYPE_MAX and receive zero vectors
# for time, age, visit_order, and visit_segment embeddings (paper §2.2).



# ── Primitive embedding components ───────────────────────────────────────────

class TimeEmbeddingLayer(nn.Module):
    """Learnable sinusoidal embedding for a scalar time feature (§2.2).

    Projects a (B, L) float tensor to (B, L, embedding_size) via:
        sin(t * w + φ)
    where w and φ are learned per-dimension parameters.

    Args:
        embedding_size: Output embedding dimensionality.
        is_time_delta:  If True, convert absolute stamps to inter-visit deltas
                        before embedding (used for time_stamps). Set False for
                        absolute values such as patient age.
    """

    def __init__(self, embedding_size: int, is_time_delta: bool = False) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta
        self.w = nn.Parameter(torch.empty(1, embedding_size))
        self.phi = nn.Parameter(torch.empty(1, embedding_size))
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_stamps: (B, L) float tensor.
        Returns:
            (B, L, embedding_size) sinusoidal embeddings.
        """
        if self.is_time_delta:
            # Replace absolute stamps with [0, Δt_1, Δt_2, …]
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0.0,
                 time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        t = time_stamps.float().unsqueeze(-1)          # (B, L, 1)
        return torch.sin(t * self.w + self.phi)         # (B, L, E)


class VisitEmbedding(nn.Module):
    """Learned embedding for visit segment type (inpatient / outpatient / …).

    Args:
        num_segments: Total number of segment categories (e.g. 3 for
            0=PAD/special, 1=segment_1, 2=segment_2).
        embedding_size: Output embedding dimensionality.
    """

    def __init__(self, num_segments: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visit_segments: (B, L) int tensor with values in [0, num_segments).
        Returns:
            (B, L, embedding_size).
        """
        return self.embedding(visit_segments)


# ── Full §2.2 embedding module ────────────────────────────────────────────────

class EHRMambaEmbedding(BaseModel):
    """EHR Mamba §2.2 embedding layer for MIMIC-IV.

    Fuses code, time-delta, and age embeddings then adds token-type,
    visit-order, and visit-segment embeddings:

        e_fused  = Tanh(W_proj · concat(e_code, e_time_delta, e_age))
        e_token  = e_fused + e_type + e_visit_order + e_visit_segment
        output   = LayerNorm(Dropout(e_token))

    Supports two calling patterns:

    **Explicit** (recommended):
        embeddings = emb(input_ids, token_type_ids, time_stamps,
                         ages, visit_orders, visit_segments)

    **Cached** (compatible with mamba_ssm MixerModel backbone):
        emb.set_aux_inputs(token_type_ids, time_stamps, ages,
                           visit_orders, visit_segments)
        embeddings = emb(input_ids)   # auxiliary inputs consumed from cache

    If neither auxiliary inputs nor cached values are available, the module
    falls back to bare word embeddings (useful for inference with new tokens).

    Args:
        vocab_size:           Number of entries in the code vocabulary.
        hidden_size:          Output (and hidden) embedding dimension.
        dataset:              Dataset object passed to BaseModel (default None)
        padding_idx:          Vocabulary index reserved for padding (default 0).
        type_vocab_size:      Number of token type categories (default 10).
        max_num_visits:       Size of visit-order embedding table (default 512).
        time_embeddings_size: Dimensionality of sinusoidal time/age embeddings
                              (default 32; concatenated before projection).
        num_visit_segments:        Number of visit segment categories (default 3:
                                   0=PAD/special, 1=segment_1, 2=segment_2).
        max_position_embeddings:   Size of absolute position embedding table (default 4096).
        layer_norm_eps:            LayerNorm epsilon (default 1e-12).
        hidden_dropout_prob:       Dropout probability (default 0.1).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dataset: Optional[SampleDataset] = None,
        padding_idx: int = 0,
        type_vocab_size: int = NUM_TOKEN_TYPES,
        max_num_visits: int = MAX_NUM_VISITS,
        time_embeddings_size: int = 32,
        num_visit_segments: int = NUM_VISIT_SEGMENTS,
        max_position_embeddings: int = 4096,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__(dataset=dataset)

        self.hidden_size = hidden_size
        self.time_embeddings_size = time_embeddings_size

        # Core code embedding (shared vocabulary: procedures + medications + specials)
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        # Token-type embedding: which MIMIC-IV table each token came from
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        # Visit-order embedding: chronological admission index
        self.visit_order_embeddings = nn.Embedding(max_num_visits, hidden_size)
        # Visit-segment embedding: alternating 1/2 per visit (paper §2.2)
        self.visit_segment_embeddings = VisitEmbedding(num_visit_segments, hidden_size)
        # Absolute positional embedding E_position(P) – 7th component of Eq. 1
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        # Sinusoidal time-delta embedding (inter-visit week gaps, Appx. C.2)
        self.time_embeddings = TimeEmbeddingLayer(
            time_embeddings_size, is_time_delta=True
        )
        # Sinusoidal absolute-age embedding (patient age in years, Appx. C.2)
        self.age_embeddings = TimeEmbeddingLayer(
            time_embeddings_size, is_time_delta=False
        )
        # Project [e_code ‖ e_time ‖ e_age] → hidden_size  (Appx. C.2)
        self.scale_back_concat_layer = nn.Linear(
            hidden_size + 2 * time_embeddings_size, hidden_size
        )
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Cache for the backbone-compatible calling pattern
        self._type_ids: Optional[torch.Tensor] = None
        self._time_stamps: Optional[torch.Tensor] = None
        self._ages: Optional[torch.Tensor] = None
        self._visit_orders: Optional[torch.Tensor] = None
        self._visit_segments: Optional[torch.Tensor] = None

    # ── Cached-input API (for mamba_ssm MixerModel compatibility) ────────────
    # Not used in pyhealth directly; provided for compatibility with the
    # mamba_ssm backbone and other downstream tasks outside of pyhealth.
    # Also used for testing the embedding module in isolation.
    def set_aux_inputs(
        self,
        token_type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> None:
        """Cache auxiliary clinical tensors before the backbone forward call.

        All tensors must be (B, L) and on the same device as the module.
        The cache is consumed (cleared) on the next forward() call.
        """
        self._type_ids = token_type_ids
        self._time_stamps = time_stamps
        self._ages = ages
        self._visit_orders = visit_orders
        self._visit_segments = visit_segments

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        time_stamps: Optional[torch.Tensor] = None,
        ages: Optional[torch.Tensor] = None,
        visit_orders: Optional[torch.Tensor] = None,
        visit_segments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce enriched EHR embeddings.

        Accepts auxiliary tensors either as explicit arguments or from the
        internal cache populated by set_aux_inputs().  Explicit arguments
        take precedence over cached values.

        Args:
            input_ids:       (B, L) int - code vocabulary indices.
            token_type_ids:  (B, L) int - MIMIC4_TOKEN_TYPES values.
            time_stamps:     (B, L) float - weeks since first admission (paper §2.2).
            ages:            (B, L) float - patient age in years.
            visit_orders:    (B, L) int - 0-based chronological visit index.
            visit_segments:  (B, L) int - alternating 1/2 per visit (paper §2.2).

        Returns:
            (B, L, hidden_size) enriched token embeddings.
        """
        # Resolve auxiliary inputs: prefer explicit args, then cache
        _type_ids     = token_type_ids  if token_type_ids  is not None else self._type_ids
        _time_stamps  = time_stamps     if time_stamps      is not None else self._time_stamps
        _ages         = ages            if ages             is not None else self._ages
        _visit_orders = visit_orders    if visit_orders     is not None else self._visit_orders
        _visit_segs   = visit_segments  if visit_segments   is not None else self._visit_segments

        # Clear cache (consumed once)
        self._type_ids = self._time_stamps = self._ages = None
        self._visit_orders = self._visit_segments = None

        B, L = input_ids.shape
        word_embeds = self.word_embeddings(input_ids)  # (B, L, H)

        # Absolute positional embedding E_position(P) – always applied (Eq. 1, 7th component)
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeds = self.position_embeddings(positions)                    # (B, L, H)

        if _type_ids is not None:
            time_embeds        = self.time_embeddings(_time_stamps)              # (B, L, T)
            age_embeds         = self.age_embeddings(_ages)                      # (B, L, T)
            token_type_embeds  = self.token_type_embeddings(_type_ids)           # (B, L, H)
            visit_order_embeds = self.visit_order_embeddings(_visit_orders)      # (B, L, H)
            visit_seg_embeds   = self.visit_segment_embeddings(_visit_segs)      # (B, L, H)

            # Paper §2.2: special tokens (PAD, CLS, VS, VE, REG, TIME_INTERVAL)
            # receive zero vectors for time, age, visit_order, visit_segment.
            special_mask = (_type_ids <= SPECIAL_TYPE_MAX).float().unsqueeze(-1) # (B, L, 1)
            time_embeds        = time_embeds        * (1.0 - special_mask)
            age_embeds         = age_embeds         * (1.0 - special_mask)
            visit_order_embeds = visit_order_embeds * (1.0 - special_mask)
            visit_seg_embeds   = visit_seg_embeds   * (1.0 - special_mask)

            # Appx. C.2 fusion: concat(concept, time, age) → project → hidden_size
            fused = torch.cat([word_embeds, time_embeds, age_embeds], dim=-1)   # (B, L, H+2T)
            fused = self.tanh(self.scale_back_concat_layer(fused))              # (B, L, H)

            # Eq. 1: additive combination of all components
            embeddings = (
                fused + token_type_embeds + visit_order_embeds
                + visit_seg_embeds + position_embeds
            )
        else:
            # Fallback: word + positional only (no clinical metadata available)
            embeddings = word_embeds + position_embeds

        return self.LayerNorm(self.dropout(embeddings))                          # (B, L, H)


# ── Adapter: bridges EHRMambaEmbedding → EmbeddingModel's dict interface ─────

class EHRMambaEmbeddingAdapter(nn.Module):
    """Drop-in replacement for pyhealth.models.embedding.EmbeddingModel inside EHRMamba.

    EmbeddingModel expects:
        forward(inputs: Dict[str, Tensor]) -> Dict[str, Tensor]

    EHRMambaEmbedding expects individual tensors.  This adapter holds the aux
    inputs (token_type_ids, time_stamps, ages, visit_orders, visit_segments)
    between the set_aux_inputs() call and the forward() call, matching the
    EmbeddingModel interface exactly.

    Usage inside pyhealth.models.ehrmamba.EHRMamba:

        # __init__ (replace EmbeddingModel):
        vocab_size = dataset.input_processors["input_ids"].vocab_size
        core_emb = EHRMambaEmbedding(vocab_size=vocab_size, hidden_size=embedding_dim)
        self.embedding_model = EHRMambaEmbeddingAdapter(core_emb)

        # forward (add before self.embedding_model(embedding_inputs)):
        if "token_type_ids" in kwargs:
            self.embedding_model.set_aux_inputs(
                token_type_ids = kwargs["token_type_ids"].to(self.device),
                time_stamps    = kwargs["time_stamps"].to(self.device),
                ages           = kwargs["ages"].to(self.device),
                visit_orders   = kwargs["visit_orders"].to(self.device),
                visit_segments = kwargs["visit_segments"].to(self.device),
            )
    """

    def __init__(self, embedding: EHRMambaEmbedding) -> None:
        """Wrap an EHRMambaEmbedding to expose the EmbeddingModel interface.

        Args:
            embedding: Configured :class:`EHRMambaEmbedding` instance that
                performs the full §2.2 / Eq. 1 token embedding.
        """
        super().__init__()
        self.embedding = embedding
        self._aux: Dict[str, torch.Tensor] = {}

    def set_aux_inputs(
        self,
        token_type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> None:
        """Store auxiliary tensors to be consumed on the next forward() call."""
        self._aux = {
            "token_type_ids": token_type_ids,
            "time_stamps":    time_stamps,
            "ages":           ages,
            "visit_orders":   visit_orders,
            "visit_segments": visit_segments,
        }

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Matches EmbeddingModel.forward(inputs) → Dict[str, Tensor].

        Args:
            inputs: {feature_key: (B, L) LongTensor of code vocab indices}.
                    With MIMIC4EHRMambaTask the only key is "input_ids".

        Returns:
            {feature_key: (B, L, hidden_size) FloatTensor}.
        """
        aux = self._aux
        self._aux = {}  # consume once

        result: Dict[str, torch.Tensor] = {}
        for key, input_ids in inputs.items():
            result[key] = self.embedding(
                input_ids,
                token_type_ids = aux.get("token_type_ids"),
                time_stamps    = aux.get("time_stamps"),
                ages           = aux.get("ages"),
                visit_orders   = aux.get("visit_orders"),
                visit_segments = aux.get("visit_segments"),
            )
        return result


