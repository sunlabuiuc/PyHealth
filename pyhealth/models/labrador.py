"""PyHealth adaptation of Labrador for laboratory data.

Paper: Bellamy et al., "Labrador: Exploring the Limits of Masked Language
Modeling for Laboratory Data", ML4H 2024.
https://arxiv.org/abs/2312.11502

This implementation preserves key Labrador design ideas:
- joint modeling of lab codes and continuous values
- optional dual-head MLM prediction module
- optional classifier head for downstream PyHealth tasks

This is not a full reproduction of the original paper's end-to-end
pretraining + evaluation pipeline.
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
    Then applies a final hidden projection.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        mask_code_token: int = 1,
        pad_code_token: int = 0,
        padding_idx: int = 0,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
    ) -> None:
        super().__init__()
        self.mask_code_token = mask_code_token
        self.pad_code_token = pad_code_token
        self.code_embedding = nn.Embedding(
            num_embeddings=vocab_size + 2,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )
        self.value_embedding = LabradorValueEmbedding(
            hidden_dim=hidden_dim,
            mask_value_token=mask_value_token,
            null_value_token=null_value_token,
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lab_codes: torch.Tensor, lab_values: torch.Tensor) -> torch.Tensor:
        code_emb = self.code_embedding(lab_codes)
        x = self.value_embedding(lab_values=lab_values, code_emb=code_emb)
        return self.output_projection(x)


class LabradorMLMHead(nn.Module):
    """Two-head MLM prediction module from the original Labrador design.

    - Categorical head predicts lab code distribution at each position.
    - Continuous head predicts normalized lab value at each position.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        continuous_head_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.categorical_dense = nn.Linear(hidden_dim, hidden_dim)
        self.categorical_activation = nn.ReLU()
        self.categorical_head = nn.Linear(hidden_dim, vocab_size)

        self.continuous_dense = nn.Linear(hidden_dim + vocab_size, hidden_dim + vocab_size)
        self.continuous_activation = nn.ReLU()
        self.continuous_head = nn.Linear(hidden_dim + vocab_size, 1)

        if continuous_head_activation == "sigmoid":
            self.continuous_output_activation = nn.Sigmoid()
        elif continuous_head_activation == "linear":
            self.continuous_output_activation = nn.Identity()
        else:
            raise ValueError(
                "continuous_head_activation must be 'sigmoid' or 'linear', "
                f"got {continuous_head_activation}"
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cat_hidden = self.categorical_activation(self.categorical_dense(x))
        categorical_logits = self.categorical_head(cat_hidden)
        categorical_output = torch.softmax(categorical_logits, dim=-1)

        continuous_input = torch.cat([x, categorical_output], dim=-1)
        cont_hidden = self.continuous_activation(self.continuous_dense(continuous_input))
        continuous_output = self.continuous_output_activation(self.continuous_head(cont_hidden))

        return {
            "categorical_logits": categorical_logits,
            "categorical_output": categorical_output,
            "continuous_output": continuous_output,
        }


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
        mask_code_token: int = 1,
        pad_code_token: int = 0,
        mask_value_token: float = -2.0,
        null_value_token: float = -1.0,
        include_mlm_head: bool = True,
        include_classifier_head: bool = True,
        continuous_head_activation: str = "sigmoid",
        mlm_ignore_index: int = -1,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

        super().__init__(dataset)
        self.pad_code_token = pad_code_token
        self.include_mlm_head = include_mlm_head
        self.include_classifier_head = include_classifier_head
        self.mlm_ignore_index = mlm_ignore_index
        if hasattr(self, "label_keys") and self.label_keys:
            self.label_key = self.label_keys[0]

        self.embedding = LabradorEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            mask_code_token=mask_code_token,
            pad_code_token=pad_code_token,
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
        self.classifier = nn.Linear(hidden_dim, self.get_output_size()) if include_classifier_head else None
        self.mlm_head = LabradorMLMHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            continuous_head_activation=continuous_head_activation,
        )

    def _categorical_mlm_loss(
        self, categorical_logits: torch.Tensor, masked_lab_codes: torch.Tensor
    ) -> torch.Tensor:
        """Masked categorical MLM loss.

        Expected convention matches the repository TensorFlow code:
        - targets are 1-indexed lab code IDs
        - `mlm_ignore_index` positions are removed first
        - remaining targets are shifted to 0-index before CE
        """
        active = masked_lab_codes.ne(self.mlm_ignore_index)
        if not active.any():
            return torch.zeros((), device=categorical_logits.device)
        labels = masked_lab_codes[active].long() - 1
        return nn.functional.cross_entropy(
            categorical_logits[active], labels, reduction="mean"
        )

    def _continuous_mlm_loss(
        self, continuous_output: torch.Tensor, masked_lab_values: torch.Tensor
    ) -> torch.Tensor:
        continuous_output = continuous_output.squeeze(-1)
        active = masked_lab_values.ne(float(self.mlm_ignore_index))
        if not active.any():
            return torch.zeros((), device=continuous_output.device)
        return nn.functional.mse_loss(
            continuous_output[active], masked_lab_values[active].float(), reduction="mean"
        )

    def forward(
        self,
        lab_codes: torch.Tensor,
        lab_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        masked_lab_codes: Optional[torch.Tensor] = None,
        masked_lab_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = lab_codes.eq(self.pad_code_token)

        x = self.embedding(lab_codes=lab_codes.long(), lab_values=lab_values)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-8)

        output = {}
        if self.include_classifier_head:
            logit = self.classifier(pooled)
            y_prob = self.prepare_y_prob(logit)
            output.update({"logit": logit, "y_prob": y_prob})

        if self.include_mlm_head:
            mlm_outputs = self.mlm_head(x)
            output.update(
                {
                    "categorical_output": mlm_outputs["categorical_output"],
                    "continuous_output": mlm_outputs["continuous_output"],
                }
            )

            if masked_lab_codes is not None:
                output["categorical_mlm_loss"] = self._categorical_mlm_loss(
                    mlm_outputs["categorical_logits"], masked_lab_codes.long()
                )
            if masked_lab_values is not None:
                output["continuous_mlm_loss"] = self._continuous_mlm_loss(
                    mlm_outputs["continuous_output"], masked_lab_values
                )

        if label is not None:
            if not self.include_classifier_head:
                raise ValueError("label is provided, but include_classifier_head is False.")
            if self.mode in ["binary", "multilabel", "regression"]:
                label = label.float().view_as(logit)
            elif self.mode == "multiclass":
                label = label.long().view(-1)
            output["loss"] = self.get_loss_function()(logit, label)
            output["y_true"] = label

        return output
