"""PyHealth-compatible Labrador model for laboratory data.

Paper: Bellamy et al., "Labrador: Exploring the Limits of Masked Language
Modeling for Laboratory Data", ML4H 2024.
https://arxiv.org/abs/2312.11502
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class LabradorValueEmbedding(nn.Module):
    """Embedding for lab values with optional special-token substitution.

    This mirrors the TensorFlow implementation in this repository:
    1) project each scalar value to `hidden_dim`
    2) swap masked/null positions with dedicated learned embeddings
    3) fuse with code embedding and transform with MLP + LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_value_token = mask_value_token
        self.null_value_token = null_value_token

        self.value_projection = nn.Linear(1, hidden_dim)
        self.special_value_embedding = nn.Embedding(2, hidden_dim)
        self.fusion_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, lab_values: torch.Tensor, code_emb: torch.Tensor) -> torch.Tensor:
        values = lab_values.unsqueeze(-1).float()
        value_emb = self.value_projection(values)

        mask_positions = lab_values.eq(self.mask_value_token)
        null_positions = lab_values.eq(self.null_value_token)

        if mask_positions.any():
            mask_vec = self.special_value_embedding.weight[0].view(1, 1, -1)
            value_emb = torch.where(mask_positions.unsqueeze(-1), mask_vec, value_emb)
        if null_positions.any():
            null_vec = self.special_value_embedding.weight[1].view(1, 1, -1)
            value_emb = torch.where(null_positions.unsqueeze(-1), null_vec, value_emb)

        x = value_emb + code_emb
        x = self.fusion_mlp(x)
        x = self.fusion_activation(x)
        x = self.layer_norm(x)
        return x


class LabradorEmbedding(nn.Module):
    """Joint embedding of lab code and lab value.

    Labrador first computes:
    - categorical embedding from lab code
    - continuous embedding from lab value + code embedding
    Then concatenates both and projects to `hidden_dim`.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        padding_idx: int = 0,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
    ) -> None:
        super().__init__()
        self.code_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )
        self.value_embedding = LabradorValueEmbedding(
            hidden_dim=hidden_dim,
            mask_value_token=mask_value_token,
            null_value_token=null_value_token,
        )
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, lab_codes: torch.Tensor, lab_values: torch.Tensor) -> torch.Tensor:
        code_emb = self.code_embedding(lab_codes)
        value_emb = self.value_embedding(lab_values=lab_values, code_emb=code_emb)
        return self.output_projection(torch.cat([code_emb, value_emb], dim=-1))


class LabradorModel(BaseModel):
    """Labrador for downstream tasks in PyHealth (PyTorch version).

    Inputs are expected as two aligned tensors:
    - `lab_codes`: integer IDs
    - `lab_values`: normalized float values in [0, 1],
      plus optional sentinel values for masked/null tokens.
    """

    def __init__(
        self,
        dataset,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pad_code_token: int = 0,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
    ) -> None:
        super().__init__(dataset)
        self.pad_code_token = pad_code_token

        self.embedding = LabradorEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            padding_idx=padding_idx,
            mask_value_token=mask_value_token,
            null_value_token=null_value_token,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, self.get_output_size())

    def forward(
        self,
        lab_codes: torch.Tensor,
        lab_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = lab_codes.eq(self.pad_code_token)

        x = self.embedding(lab_codes=lab_codes.long(), lab_values=lab_values)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-8)

        logit = self.classifier(pooled)
        y_prob = self.prepare_y_prob(logit)
        output = {"logit": logit, "y_prob": y_prob}

        if label is not None:
            if self.mode in ["binary", "multilabel"]:
                label = label.float().view_as(logit)
            output["loss"] = self.get_loss_function()(logit, label)
            output["y_true"] = label

        return output
