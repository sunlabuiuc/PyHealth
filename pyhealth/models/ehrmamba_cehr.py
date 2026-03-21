"""EHRMamba with CEHR-style embeddings for single-stream FHIR token sequences."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel
from .cehr_embeddings import MambaEmbeddingsForCEHR
from .ehrmamba import MambaBlock
from .utils import get_last_visit


class EHRMambaCEHR(BaseModel):
    """Mamba backbone over CEHR embeddings (FHIR / MPF pipeline).

    Args:
        dataset: Fitted :class:`~pyhealth.datasets.SampleDataset` with MPF task schema.
        vocab_size: Concept embedding vocabulary size (typically ``task.vocab.vocab_size``).
        embedding_dim: Hidden size (``hidden_size`` in CEHR embeddings).
        num_layers: Number of :class:`~pyhealth.models.ehrmamba.MambaBlock` layers.
        pad_token_id: Padding id for masking (default 0).
        state_size: SSM state size per channel.
        conv_kernel: Causal conv kernel in each block.
        dropout: Dropout before classifier.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        vocab_size: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        pad_token_id: int = 0,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        type_vocab_size: int = 16,
        max_num_visits: int = 512,
        time_embeddings_size: int = 32,
        visit_segment_vocab: int = 3,
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        assert len(self.label_keys) == 1, "EHRMambaCEHR supports single label key only"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embeddings = MambaEmbeddingsForCEHR(
            vocab_size=vocab_size,
            hidden_size=embedding_dim,
            pad_token_id=pad_token_id,
            type_vocab_size=type_vocab_size,
            max_num_visits=max_num_visits,
            time_embeddings_size=time_embeddings_size,
            visit_order_size=visit_segment_vocab,
        )
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=embedding_dim,
                    state_size=state_size,
                    conv_kernel=conv_kernel,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = self.get_output_size()
        self.fc = nn.Linear(embedding_dim, out_dim)
        self._forecasting_head: Optional[nn.Module] = None

    def forward_forecasting(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """Optional next-token / forecasting head (extension point; not implemented)."""

        return None

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        concept_ids = kwargs["concept_ids"].to(self.device).long()
        token_type_ids = kwargs["token_type_ids"].to(self.device).long()
        time_stamps = kwargs["time_stamps"].to(self.device).float()
        ages = kwargs["ages"].to(self.device).float()
        visit_orders = kwargs["visit_orders"].to(self.device).long()
        visit_segments = kwargs["visit_segments"].to(self.device).long()

        x = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=token_type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )
        mask = concept_ids != self.pad_token_id
        for blk in self.blocks:
            x = blk(x)
        pooled = get_last_visit(x, mask)
        logits = self.fc(self.dropout(pooled))
        y_true = kwargs[self.label_key].to(self.device).float()
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
