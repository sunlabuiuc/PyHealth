# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Vector Institute / Odyssey authors
#
# Derived from Odyssey (https://github.com/VectorInstitute/odyssey):
#   odyssey/models/embeddings.py — MambaEmbeddingsForCEHR, TimeEmbeddingLayer, VisitEmbedding
# Modifications: removed HuggingFace MambaConfig dependency; explicit constructor args.

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features (sinusoidal)."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta
        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> torch.Tensor:
        if self.is_time_delta:
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        next_input = time_stamps.unsqueeze(-1) * self.w + self.phi
        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Embedding layer for visit segments."""

    def __init__(self, visit_order_size: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(visit_order_size, embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> torch.Tensor:
        return self.embedding(visit_segments)


class MambaEmbeddingsForCEHR(nn.Module):
    """CEHR-style combined embeddings for Mamba (concept + type + time + age + visit)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = 0,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.visit_order_embeddings = nn.Embedding(max_num_visits, hidden_size)
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size, is_time_delta=True
        )
        self.age_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size, is_time_delta=False
        )
        self.visit_segment_embeddings = VisitEmbedding(
            visit_order_size=visit_order_size, embedding_size=hidden_size
        )
        self.scale_back_concat_layer = nn.Linear(
            hidden_size + 2 * time_embeddings_size, hidden_size
        )
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids_batch: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.word_embeddings(input_ids)
        time_stamps_embeds = self.time_embeddings(time_stamps)
        ages_embeds = self.age_embeddings(ages)
        visit_segments_embeds = self.visit_segment_embeddings(visit_segments)
        visit_order_embeds = self.visit_order_embeddings(visit_orders)
        token_type_embeds = self.token_type_embeddings(token_type_ids_batch)
        concat_in = torch.cat(
            (inputs_embeds, time_stamps_embeds, ages_embeds), dim=-1
        )
        h = self.tanh(self.scale_back_concat_layer(concat_in))
        embeddings = h + token_type_embeds + visit_order_embeds + visit_segments_embeds
        embeddings = self.dropout(embeddings)
        return self.LayerNorm(embeddings)
