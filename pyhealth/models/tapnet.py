from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


class TapNetLayer(nn.Module):
    """Time series attentional prototype layer.

    A lightweight adaptation of TapNet that encodes a sequence with a temporal
    convolution, summarizes it with attention, and then attends over a small
    set of learnable prototypes.

    Args:
        input_dim: input feature size.
        hidden_dim: hidden feature size after convolution.
        num_prototypes: number of learnable prototypes.
        kernel_size: temporal convolution kernel size.
        dropout: dropout rate applied after convolution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_prototypes: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.attn_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape [batch, seq_len, input_dim].
            mask: optional tensor of shape [batch, seq_len] with 1 for valid steps.

        Returns:
            summary: tensor of shape [batch, hidden_dim * 2].
            proto_attn: attention weights over prototypes [batch, num_prototypes].
        """
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.dropout(torch.relu(h))

        attn_logits = self.attn_proj(h).squeeze(-1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        seq_repr = torch.sum(attn_weights.unsqueeze(-1) * h, dim=1)

        proto_logits = torch.matmul(seq_repr, self.prototypes.t()) / math.sqrt(
            self.hidden_dim
        )
        proto_attn = torch.softmax(proto_logits, dim=-1)
        proto_repr = torch.matmul(proto_attn, self.prototypes)

        summary = torch.cat([seq_repr, proto_repr], dim=-1)
        return summary, proto_attn


class TapNet(BaseModel):
    """Time series attentional prototype network.

    Each feature stream is embedded, encoded with a TapNetLayer, and the
    resulting representations are concatenated for prediction.

    Args:
        dataset: the dataset to train the model.
        embedding_dim: dimension of input embeddings.
        hidden_dim: hidden dimension inside TapNet layers.
        num_prototypes: number of prototypes per feature stream.
        kernel_size: temporal convolution kernel size.
        dropout: dropout rate applied in TapNet layers.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_prototypes: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(dataset=dataset)
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if TapNet is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.encoders = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.encoders[feature_key] = TapNetLayer(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_prototypes=num_prototypes,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * hidden_dim * 2, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        embedded = self.embedding_model(kwargs)
        patient_emb = []
        proto_attn = {}

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            mask = None
            if x.dim() >= 3:
                mask = (x.sum(dim=-1) != 0).int()
            summary, pa = self.encoders[feature_key](x, mask)
            patient_emb.append(summary)
            proto_attn[feature_key] = pa

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        if kwargs.get("return_attn", False):
            results["prototype_attention"] = proto_attn
        return results
