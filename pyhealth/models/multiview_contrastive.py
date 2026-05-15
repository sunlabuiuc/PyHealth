"""Multi-View Contrastive Learning for Domain Adaptation in Medical Time Series.

This module implements the multi-view contrastive learning framework from:

    Oh, Y.; and Bui, A. 2025. Multi-View Contrastive Learning for Robust
    Domain Adaptation in Medical Time Series Analysis. In Proceedings of the
    Sixth Conference on Health, Inference, and Learning, volume 287, 502-526.
    PMLR.

The model constructs three views of a raw time-series signal -- temporal,
derivative, and frequency -- encodes each with an independent backbone, fuses
them via hierarchical attention, and classifies the fused representation.

Two configuration axes are exposed for ablation studies:

* **encoder_type** (``"transformer"`` | ``"cnn"`` | ``"gru"``): the per-view
  backbone architecture.
* **fusion_type** (``"attention"`` | ``"concat"`` | ``"mean"``): how the three
  view embeddings are aggregated before the classifier head.

A **view_type** parameter (``"T"`` | ``"D"`` | ``"F"`` | ``"TD"`` | ``"TF"``
| ``"DF"`` | ``"ALL"``) selects which subset of views is active.
"""

import math
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of the same shape with positional information added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _InteractionLayer(nn.Module):
    """Cross-view multi-head attention interaction layer."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, ht: torch.Tensor, hd: torch.Tensor, hf: torch.Tensor
    ) -> tuple:
        """Apply cross-view attention.

        Args:
            ht: Temporal hidden states ``(B, L, D)``.
            hd: Derivative hidden states ``(B, L, D)``.
            hf: Frequency hidden states ``(B, L, D)``.

        Returns:
            Tuple of three tensors, each ``(B, L, D)``.
        """
        B, L, D = ht.size()
        h = torch.stack([ht, hd, hf], dim=2)  # (B, L, 3, D)
        h = h.permute(0, 2, 1, 3).contiguous().view(B * 3, L, D)
        attn_output, _ = self.multihead_attn(h, h, h)
        output = self.norm(h + attn_output)
        output = output.view(B, 3, L, D).permute(0, 2, 1, 3)
        return output[:, :, 0, :], output[:, :, 1, :], output[:, :, 2, :]


class _SelfAttentionFusion(nn.Module):
    """Learned self-attention fusion with residual connection."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """Apply self-attention over stacked view embeddings.

        Args:
            x: Tensor of shape ``(B, n_views, D)``.

        Returns:
            Tuple of (attended tensor ``(B, n_views, D)``, attention weights).
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, v), attn_w


# ---------------------------------------------------------------------------
# Per-view backbone factories
# ---------------------------------------------------------------------------


def _make_transformer_encoder(
    num_feature: int,
    num_embedding: int,
    num_hidden: int,
    num_head: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Build a single-view Transformer encoder branch."""
    input_proj = nn.Linear(num_feature, num_embedding)
    pos_enc = _PositionalEncoding(num_embedding, dropout)
    enc_layer = nn.TransformerEncoderLayer(
        d_model=num_embedding,
        dim_feedforward=num_hidden,
        nhead=num_head,
        dropout=dropout,
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(enc_layer, num_layers)
    return nn.ModuleDict(
        {"input_proj": input_proj, "pos_enc": pos_enc, "encoder": transformer}
    )


def _make_cnn_encoder(
    num_feature: int,
    num_embedding: int,
    num_hidden: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Build a single-view 1D-CNN encoder branch."""
    layers = []
    in_ch = num_feature
    for i in range(num_layers):
        out_ch = num_hidden if i < num_layers - 1 else num_embedding
        layers.extend(
            [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Dropout(dropout),
            ]
        )
        in_ch = out_ch
    # Adaptive pool to guarantee fixed-length output regardless of input L
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


def _make_gru_encoder(
    num_feature: int,
    num_embedding: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Build a single-view GRU encoder branch."""
    return nn.GRU(
        input_size=num_feature,
        hidden_size=num_embedding,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

# Valid view and encoder/fusion type literals
VIEW_TYPES = {"T", "D", "F", "TD", "TF", "DF", "ALL"}
ENCODER_TYPES = {"transformer", "cnn", "gru"}
FUSION_TYPES = {"attention", "concat", "mean"}


class MultiViewContrastive(BaseModel):
    """Multi-View Contrastive model for medical time-series classification.

    Constructs three views (temporal, derivative, frequency) of a raw
    signal, encodes each with an independent backbone, fuses them, and
    classifies.

    Paper: Oh & Bui (2025), *Multi-View Contrastive Learning for Robust
    Domain Adaptation in Medical Time Series Analysis*, CHIL 2025.

    Args:
        dataset: A ``SampleDataset`` produced by ``dataset.set_task()``.
        num_embedding: Embedding / hidden dimension for each view encoder.
            Default ``64``.
        num_hidden: Feed-forward hidden dimension (Transformer) or
            intermediate channels (CNN). Default ``128``.
        num_head: Number of attention heads. Default ``4``.
        num_layers: Number of encoder layers per view. Default ``3``.
        dropout: Dropout rate. Default ``0.2``.
        encoder_type: Backbone architecture per view. One of
            ``"transformer"`` (default), ``"cnn"``, ``"gru"``.
        view_type: Which views to use. One of ``"T"``, ``"D"``, ``"F"``,
            ``"TD"``, ``"TF"``, ``"DF"``, ``"ALL"`` (default).
        fusion_type: How to aggregate view embeddings. One of
            ``"attention"`` (default), ``"concat"``, ``"mean"``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import MultiViewContrastive
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "signal": np.random.randn(1, 178).astype(np.float32),
        ...      "label": 0},
        ...     {"patient_id": "p1", "visit_id": "v0",
        ...      "signal": np.random.randn(1, 178).astype(np.float32),
        ...      "label": 1},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="test",
        ... )
        >>> model = MultiViewContrastive(dataset=dataset)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> ret = model(**batch)
        >>> ret.keys()
        dict_keys(['loss', 'y_prob', 'y_true', 'logit'])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_embedding: int = 64,
        num_hidden: int = 128,
        num_head: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "transformer",
        view_type: str = "ALL",
        fusion_type: str = "attention",
    ):
        super().__init__(dataset=dataset)

        # Validate arguments
        assert encoder_type in ENCODER_TYPES, (
            f"encoder_type must be one of {ENCODER_TYPES}, got '{encoder_type}'"
        )
        assert view_type in VIEW_TYPES, (
            f"view_type must be one of {VIEW_TYPES}, got '{view_type}'"
        )
        assert fusion_type in FUSION_TYPES, (
            f"fusion_type must be one of {FUSION_TYPES}, got '{fusion_type}'"
        )
        assert len(self.label_keys) == 1, "Only one label key is supported."
        self.feature_keys = [
            k for k in self.feature_keys if k != "stft"
        ]
        assert len(self.feature_keys) == 1, "Only one feature key is supported."

        self.num_embedding = num_embedding
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.view_type = view_type
        self.fusion_type = fusion_type

        # Determine input shape
        num_feature = self._infer_num_features()
        self.num_feature = num_feature

        # Determine active views
        self._active_views = self._parse_view_type(view_type)
        n_views = len(self._active_views)

        # Build per-view encoder branches
        self.encoders = nn.ModuleDict()
        for v in self._active_views:
            self.encoders[v] = self._build_encoder(num_feature)

        # Interaction layer (only when all 3 views are active)
        self.interaction_layer = None
        if n_views == 3:
            self.interaction_layer = _InteractionLayer(num_embedding, num_head)

        # Per-view output projection
        proj_input = num_embedding * 2 if n_views == 3 else num_embedding
        self.output_projs = nn.ModuleDict()
        for v in self._active_views:
            self.output_projs[v] = nn.Sequential(
                nn.Linear(proj_input, num_hidden),
                nn.LayerNorm(num_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_hidden, num_hidden),
            )

        # Fusion + classifier
        if fusion_type == "attention":
            self.self_attention = _SelfAttentionFusion(num_hidden)
            classifier_input = n_views * num_hidden
        elif fusion_type == "concat":
            self.self_attention = None
            classifier_input = n_views * num_hidden
        else:  # mean
            self.self_attention = None
            classifier_input = num_hidden

        output_size = self.get_output_size()
        self.fc = nn.Linear(classifier_input, output_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_num_features(self) -> int:
        """Infer the number of input channels from the dataset."""
        for sample in self.dataset:
            sig = sample.get(self.feature_keys[0])
            if sig is None:
                continue
            if isinstance(sig, np.ndarray):
                sig = torch.from_numpy(sig)
            if sig.dim() == 1:
                return 1
            elif sig.dim() == 2:
                return sig.shape[0]  # (C, T)
            else:
                raise ValueError(f"Unexpected signal shape: {sig.shape}")
        raise ValueError("Could not infer num_features from dataset.")

    @staticmethod
    def _parse_view_type(view_type: str) -> list:
        """Return list of active view keys."""
        mapping = {
            "T": ["t"],
            "D": ["d"],
            "F": ["f"],
            "TD": ["t", "d"],
            "TF": ["t", "f"],
            "DF": ["d", "f"],
            "ALL": ["t", "d", "f"],
        }
        return mapping[view_type]

    def _build_encoder(self, num_feature: int) -> nn.Module:
        """Build one encoder branch based on encoder_type."""
        if self.encoder_type == "transformer":
            return _make_transformer_encoder(
                num_feature,
                self.num_embedding,
                self.num_hidden,
                self.num_head,
                self.num_layers,
                self.dropout,
            )
        elif self.encoder_type == "cnn":
            return _make_cnn_encoder(
                num_feature,
                self.num_embedding,
                self.num_hidden,
                self.num_layers,
                self.dropout,
            )
        else:
            return _make_gru_encoder(
                num_feature,
                self.num_embedding,
                self.num_layers,
                self.dropout,
            )

    # ------------------------------------------------------------------
    # View computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_views(
        x: torch.Tensor,
    ) -> tuple:
        """Compute temporal, derivative, and frequency views.

        Args:
            x: Raw signal tensor of shape ``(B, C, T)`` (channels first).

        Returns:
            Tuple of ``(x_temporal, x_derivative, x_frequency)``, each of
            shape ``(B, T, C)`` (sequence-first for encoders).
        """
        # (B, C, T) -> (B, T, C)
        xt = x.permute(0, 2, 1).contiguous()

        # Derivative: finite difference, padded to preserve length
        dx = torch.diff(xt, dim=1)
        dx = torch.cat([dx, dx[:, -1:, :]], dim=1)

        # Frequency: FFT magnitude
        xf = torch.abs(fft.fft(xt, dim=1))

        return xt, dx, xf

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_view(self, enc: nn.Module, x_view: torch.Tensor) -> torch.Tensor:
        """Run one view through its encoder and return hidden states.

        Args:
            enc: The encoder module for this view.
            x_view: ``(B, T, C)`` for transformer/gru or ``(B, C, T)`` for cnn.

        Returns:
            Hidden states of shape ``(B, T, D)`` for transformer,
            ``(B, 1, D)`` for cnn, or ``(B, T, D)`` for gru.
        """
        if self.encoder_type == "transformer":
            h = enc["input_proj"](x_view)
            h = enc["pos_enc"](h)
            h = enc["encoder"](h)
            return h
        elif self.encoder_type == "cnn":
            # (B, T, C) -> (B, C, T) for Conv1d
            h = x_view.permute(0, 2, 1).contiguous()
            h = enc(h)  # (B, D, 1) after adaptive pool
            h = h.permute(0, 2, 1)  # (B, 1, D)
            return h
        else:  # gru
            h, _ = enc(x_view)  # (B, T, D)
            return h

    def _encode_all_views(
        self, xt: torch.Tensor, dx: torch.Tensor, xf: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode all active views and return dict of hidden states."""
        view_inputs = {"t": xt, "d": dx, "f": xf}
        hiddens = {}
        for v in self._active_views:
            h = self._encode_view(self.encoders[v], view_inputs[v])
            hiddens[v] = h
        return hiddens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Must contain the signal feature key and the label key.

        Returns:
            Dictionary with keys ``loss``, ``y_prob``, ``y_true``, ``logit``.
        """
        x = kwargs[self.feature_keys[0]].to(self.device).float()

        # Ensure (B, C, T) shape
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Clean NaN/Inf
        x = torch.nan_to_num(x)

        # Compute views
        xt, dx, xf = self.compute_views(x)

        # Encode
        hiddens = self._encode_all_views(xt, dx, xf)

        # Interaction (only for ALL views)
        if self.interaction_layer is not None and len(self._active_views) == 3:
            ht_i, hd_i, hf_i = self.interaction_layer(
                hiddens["t"], hiddens["d"], hiddens["f"]
            )
            interaction = {"t": ht_i, "d": hd_i, "f": hf_i}
        else:
            interaction = None

        # Project each view to latent embedding
        embeddings = []
        for v in self._active_views:
            h_mean = hiddens[v].mean(dim=1)  # (B, D)
            if interaction is not None:
                h_i_mean = interaction[v].mean(dim=1)
                proj_input = torch.cat([h_mean, h_i_mean], dim=-1)
            else:
                proj_input = h_mean
            z = self.output_projs[v](proj_input)
            embeddings.append(z)

        # Fuse
        if self.fusion_type == "attention":
            stacked = torch.stack(embeddings, dim=1)  # (B, n_views, D)
            attn_out, _ = self.self_attention(stacked)
            fused = (attn_out + stacked).reshape(stacked.shape[0], -1)
        elif self.fusion_type == "concat":
            fused = torch.cat(embeddings, dim=-1)
        else:  # mean
            fused = torch.stack(embeddings, dim=0).mean(dim=0)

        # Classify
        logits = self.fc(fused)

        y_true = kwargs[self.label_keys[0]].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = fused
        return results

    # ------------------------------------------------------------------
    # Contrastive pre-training helper
    # ------------------------------------------------------------------

    def encode_views(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode raw signal into per-view latent embeddings.

        This is a convenience method for contrastive pre-training, where
        you need the per-view embeddings (not the fused logits).

        Args:
            x: Raw signal ``(B, C, T)``.

        Returns:
            Dict mapping active view keys to latent embeddings ``(B, D)``.
        """
        x = torch.nan_to_num(x.float())
        if x.dim() == 2:
            x = x.unsqueeze(1)

        xt, dx, xf = self.compute_views(x)
        hiddens = self._encode_all_views(xt, dx, xf)

        if self.interaction_layer is not None and len(self._active_views) == 3:
            ht_i, hd_i, hf_i = self.interaction_layer(
                hiddens["t"], hiddens["d"], hiddens["f"]
            )
            interaction = {"t": ht_i, "d": hd_i, "f": hf_i}
        else:
            interaction = None

        latents = {}
        for v in self._active_views:
            h_mean = hiddens[v].mean(dim=1)
            if interaction is not None:
                h_i_mean = interaction[v].mean(dim=1)
                proj_input = torch.cat([h_mean, h_i_mean], dim=-1)
            else:
                proj_input = h_mean
            latents[v] = self.output_projs[v](proj_input)
        return latents
