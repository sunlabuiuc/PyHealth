"""Clinical-event embeddings for a compact EHRMAMBA reproduction.

The optional age, absolute-time, and visit-segment embeddings are deliberately
configurable. They are omitted when no temporal metadata is provided.
"""

from __future__ import annotations

import torch
from torch import nn


class CEHREmbeddings(nn.Module):
    """Combine code, age, time, and visit-segment embeddings.

    Args:
        vocab_size: Number of code IDs, including padding.
        hidden_dim: Width of each embedding.
        max_age: Largest supported integer age bucket.
        max_time: Largest supported integer time bucket.
        max_segments: Number of visit-segment IDs.
        dropout: Embedding dropout probability.
        padding_idx: Padding code ID.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_age: int = 120,
        max_time: int = 4096,
        max_segments: int = 32,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_dim, max_age, max_time, max_segments) < 1:
            raise ValueError("Embedding dimensions and bucket counts must be positive")
        self.padding_idx = padding_idx
        self.code = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.age = nn.Embedding(max_age + 1, hidden_dim)
        self.time = nn.Embedding(max_time + 1, hidden_dim)
        self.segment = nn.Embedding(max_segments, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ages: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
        segments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed events of shape (batch, sequence).

        Args:
            input_ids: Integer clinical-code IDs.
            attention_mask: True for valid events.
            ages: Optional nonnegative integer age buckets.
            times: Optional nonnegative integer time buckets.
            segments: Optional nonnegative integer visit IDs.

        Returns:
            Tensor with shape (batch, sequence, hidden_dim).
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        mask = (
            input_ids.ne(self.padding_idx)
            if attention_mask is None
            else (attention_mask.bool())
        )
        if mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        x = self.code(input_ids.long())
        for values, layer in (
            (ages, self.age),
            (times, self.time),
            (segments, self.segment),
        ):
            if values is not None:
                if values.shape != input_ids.shape:
                    raise ValueError("Temporal metadata must match input_ids")
                x = x + layer(values.long())
        x = self.dropout(self.norm(x))
        return x * mask.unsqueeze(-1).to(x.dtype)
