"""Labrador model for laboratory data.

Paper: Bellamy et al., "Labrador: Exploring the Limits of Masked Language
Modeling for Laboratory Data", ML4H 2024.
https://arxiv.org/abs/2312.11502
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


class LabradorEmbedding(nn.Module):
    """Joint embedding of lab code (integer) and lab value (float in [0,1]).

    Labrador's key novelty: it concatenates a learned code embedding
    with a scalar value projected into embedding space, then projects
    the combined vector to hidden_dim.

    Args:
        vocab_size: Number of unique lab codes + special tokens.
        hidden_dim: Embedding dimension.
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.code_embedding = nn.Embedding(vocab_size, hidden_dim // 2)
        self.value_projection = nn.Linear(1, hidden_dim // 2)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        lab_codes: torch.Tensor,       # (B, seq_len) int
        lab_values: torch.Tensor,      # (B, seq_len) float, already in [0,1]
    ) -> torch.Tensor:
        code_emb = self.code_embedding(lab_codes)            # (B, L, H/2)
        val_emb = self.value_projection(
            lab_values.unsqueeze(-1)                         # (B, L, 1)
        )                                                    # (B, L, H/2)
        combined = torch.cat([code_emb, val_emb], dim=-1)   # (B, L, H)
        return self.output_projection(combined)              # (B, L, H)


class LabradorModel(BaseModel):
    """Labrador: masked language model for EHR laboratory data.

    Implements the Labrador architecture from Bellamy et al. (2024).
    The model jointly embeds lab test codes and continuous values,
    then applies a Transformer encoder for downstream classification.

    Args:
        dataset: PyHealth dataset object (provides output_size).
        vocab_size: Number of unique lab codes. Default: 532 (MIMIC-IV).
        hidden_dim: Transformer hidden dimension. Default: 128.
        num_heads: Number of attention heads. Default: 4.
        num_layers: Number of Transformer encoder layers. Default: 2.
        dropout: Dropout rate. Default: 0.1.
        max_seq_len: Maximum number of labs per bag. Default: 64.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "lab_codes": [1, 5, 10],
        ...         "lab_values": [0.3, 0.7, 0.5],
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "lab_codes": "sequence",
        ...         "lab_values": "sequence",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = LabradorModel(dataset=dataset)
        >>> # forward pass handled by PyHealth trainer
    """

    def __init__(
        self,
        dataset,
        vocab_size: int = 532,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__(dataset)
        self.hidden_dim = hidden_dim

        # Joint lab code + value embedding
        self.embedding = LabradorEmbedding(vocab_size, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        output_size = self.get_output_size()
        self.classifier = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lab_codes: torch.Tensor,          # (B, seq_len) int
        lab_values: torch.Tensor,         # (B, seq_len) float
        padding_mask: Optional[torch.Tensor] = None,  # (B, seq_len) bool
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            lab_codes: Integer tensor of lab test codes, shape (B, L).
            lab_values: Float tensor of normalized lab values in [0,1],
                shape (B, L).
            padding_mask: Boolean mask where True = padding, shape (B, L).
            labels: Ground truth labels, shape (B,).

        Returns:
            Dict with keys:
                - loss: scalar cross-entropy loss (if labels provided).
                - y_prob: predicted probabilities, shape (B, output_size).
                - y_true: ground truth labels, shape (B,).
                - logit: raw logits, shape (B, output_size).
        """
        # Embed: (B, L, H)
        x = self.embedding(lab_codes, lab_values)
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pool: mean over non-padding positions
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)  # (B, H)

        # Classify
        logit = self.classifier(x)                        # (B, output_size)
        y_prob = self.prepare_y_prob(logit)               # softmax/sigmoid

        result = {"logit": logit, "y_prob": y_prob}

        if labels is not None:
            if self.mode == "binary":
                labels = labels.float().view_as(logit)
            elif self.mode == "multilabel":
                labels = labels.float().view_as(logit)
        
            result["loss"] = self.get_loss_function()(logit, labels)
            result["y_true"] = labels
        
        return result
