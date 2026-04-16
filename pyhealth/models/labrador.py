"""Labrador model for laboratory data.

Paper: Bellamy et al., "Labrador: Exploring the Limits of Masked Language
Modeling for Laboratory Data", ML4H 2024.
https://arxiv.org/abs/2312.11502
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


class LabradorEmbedding(nn.Module):
    """Joint embedding of lab code (integer) and lab value (float in [0,1])."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.code_embedding = nn.Embedding(vocab_size, hidden_dim // 2)
        self.value_projection = nn.Linear(1, hidden_dim // 2)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lab_codes: torch.Tensor, lab_values: torch.Tensor) -> torch.Tensor:
        code_emb = self.code_embedding(lab_codes)
        val_emb = self.value_projection(lab_values.unsqueeze(-1))
        combined = torch.cat([code_emb, val_emb], dim=-1)
        return self.output_projection(combined)


class LabradorMLMHead(nn.Module):
    """MLM head that predicts masked categorical codes and continuous values."""

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.code_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "mlm_code_logit": self.code_head(hidden_states),
            "mlm_value_pred": self.value_head(hidden_states).squeeze(-1),
        }


class LabradorModel(BaseModel):
    """Labrador model with optional classification and MLM prediction heads."""

    def __init__(
        self,
        dataset,
        vocab_size: int = 532,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        include_classifier_head: bool = True,
        include_mlm_head: bool = False,
    ) -> None:
        super().__init__(dataset)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.include_classifier_head = include_classifier_head
        self.include_mlm_head = include_mlm_head

        self.embedding = LabradorEmbedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        if self.include_classifier_head:
            output_size = self.get_output_size()
            self.classifier = nn.Linear(hidden_dim, output_size)

        if self.include_mlm_head:
            self.mlm_head = LabradorMLMHead(hidden_dim=hidden_dim, vocab_size=vocab_size)

    def categorical_mlm_loss(
        self,
        logits: torch.Tensor,
        target_codes: torch.Tensor,
        mlm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy on masked code positions only."""
        if mlm_mask.sum() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(logits[mlm_mask], target_codes[mlm_mask])

    def continuous_mlm_loss(
        self,
        pred_values: torch.Tensor,
        target_values: torch.Tensor,
        mlm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE on masked value positions only."""
        if mlm_mask.sum() == 0:
            return pred_values.new_zeros(())
        return F.mse_loss(pred_values[mlm_mask], target_values[mlm_mask])

    def forward(
        self,
        lab_codes: torch.Tensor,
        lab_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        mlm_target_codes: Optional[torch.Tensor] = None,
        mlm_target_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        x = self.embedding(lab_codes, lab_values.float())
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        result: Dict[str, torch.Tensor] = {}

        if self.include_classifier_head:
            if padding_mask is not None:
                mask = (~padding_mask).unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)

            logit = self.classifier(pooled)
            y_prob = self.prepare_y_prob(logit)
            result.update({"logit": logit, "y_prob": y_prob})

            if label is not None:
                if self.mode in {"binary", "multilabel"}:
                    label = label.float().view_as(logit)
                result["loss"] = self.get_loss_function()(logit, label)
                result["y_true"] = label
        elif label is not None:
            raise ValueError("label is provided but include_classifier_head=False")

        if self.include_mlm_head:
            mlm_outputs = self.mlm_head(x)
            result.update(mlm_outputs)

            if (
                mlm_mask is not None
                and mlm_target_codes is not None
                and mlm_target_values is not None
            ):
                c_loss = self.categorical_mlm_loss(
                    mlm_outputs["mlm_code_logit"],
                    mlm_target_codes,
                    mlm_mask,
                )
                r_loss = self.continuous_mlm_loss(
                    mlm_outputs["mlm_value_pred"],
                    mlm_target_values,
                    mlm_mask,
                )
                result["mlm_loss"] = c_loss + r_loss

        return result
