from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    """Transformer-based fusion for temporally aligned multimodal embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_modality_token: bool = False,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.use_modality_token = use_modality_token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_modality_token:
            self.modality_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse modality embeddings.

        Args:
            modality_embeddings: mapping modality->Tensor of shape [B, T, D]
            attention_mask: optional mask of shape [B, M] or [B, T, M]

        Returns:
            fused output tensor of shape [B, T, D]
        """
        if len(modality_embeddings) == 0:
            raise ValueError("modality_embeddings must contain at least one modality")

        modality_keys = sorted(modality_embeddings.keys())
        embeddings = [modality_embeddings[key] for key in modality_keys]

        B, T, D = embeddings[0].shape
        for emb in embeddings:
            if emb.shape != (B, T, D):
                raise ValueError("All modalities must have the same shape [B, T, D]")

        x = torch.stack(embeddings, dim=2)  # [B, T, M, D]

        if self.use_modality_token:
            token = self.modality_token.expand(B, T, -1, -1)
            x = torch.cat([x, token], dim=2)  # [B, T, M+1, D]

        B, T, M, D = x.shape
        x = x.reshape(B * T, M, D)  # [B*T, M, D]

        src_key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape != (B, len(modality_keys)):
                    raise ValueError("attention_mask must be [B, M] when 2D")
                if self.use_modality_token:
                    token_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, token_mask], dim=1)
                src_key_padding_mask = ~attention_mask.repeat_interleave(T, dim=0)
            elif attention_mask.ndim == 3:
                if attention_mask.shape != (B, T, len(modality_keys)):
                    raise ValueError("attention_mask must be [B, T, M] when 3D")
                if self.use_modality_token:
                    token_mask = torch.ones(B, T, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, token_mask], dim=2)
                src_key_padding_mask = ~(attention_mask.reshape(B * T, -1))
            else:
                raise ValueError("attention_mask must be 2D or 3D")

        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = out.mean(dim=1)
        out = out.view(B, T, D)
        return out
