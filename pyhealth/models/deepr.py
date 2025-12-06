# Author: Joshua Steier
# Description: Deepr model implementation for PyHealth 2.0

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class DeeprLayer(nn.Module):
    """Deepr convolution + max pooling over the sequence dimension.

    Args:
        feature_size: Input feature size (embedding_dim).
        window: Sliding window radius d; kernel_size = 2*d + 1.
        hidden_size: Number of convolution filters.
    """

    def __init__(self, feature_size: int, window: int = 1, hidden_size: int = 128):
        super(DeeprLayer, self).__init__()
        if window < 0:
            raise ValueError("window must be non-negative")

        kernel_size = 2 * window + 1
        self.conv = nn.Conv1d(
            in_channels=feature_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=window,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: Tensor of shape [batch size, seq len, feature size].
            mask: Optional tensor of shape [batch size, seq len], 1 valid / 0 pad.

        Returns:
            Tensor of shape [batch size, hidden size].
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D x [B, T, C], got {x.shape}")

        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"Expected 2-D mask [B, T], got {mask.shape}")
            mask = mask.to(device=x.device, dtype=x.dtype)
            x = x * mask.unsqueeze(-1)

        x = x.permute(0, 2, 1)  # [B, C, T]
        x = torch.relu(self.conv(x))
        x = x.max(dim=-1).values  # [B, hidden]
        return x


class Deepr(BaseModel):
    """Deepr model for PyHealth 2.0 datasets.

    The model embeds each feature via EmbeddingModel and applies a DeeprLayer
    (Conv1d + max pooling) per feature. Feature representations are concatenated
    and fed into a linear head.

    Args:
        dataset: SampleDataset with fitted input and output processors.
        embedding_dim: Size of the intermediate embedding space.
        hidden_dim: Number of convolution filters produced by DeeprLayer.
        window: Sliding window radius d (kernel_size = 2*d + 1).
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        window: int = 1,
    ):
        super(Deepr, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.window = window

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Learnable gap embedding used when flattening nested sequences (B, V, T, C)
        self.gap_embedding = nn.Parameter(torch.zeros(embedding_dim))

        self.deepr = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.deepr[feature_key] = DeeprLayer(
                feature_size=embedding_dim,
                window=window,
                hidden_size=hidden_dim,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * hidden_dim, output_size)

    @staticmethod
    def _extract_feature_tensor(feature: Any) -> Any:
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature[1]
        return feature

    @staticmethod
    def _ensure_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    def _flatten_nested_with_gap(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flattens [B, V, T, C] -> [B, L, C] and inserts a gap token between visits."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D x [B, V, T, C], got {x.shape}")

        bsz, num_visits, num_events, emb = x.shape
        valid_mask = (x.abs().sum(dim=-1) != 0)  # [B, V, T]

        gap = self.gap_embedding.to(device=x.device, dtype=x.dtype).view(1, emb)

        sequences = []
        for b in range(bsz):
            parts = []
            non_empty_visits = []
            for v in range(num_visits):
                m = valid_mask[b, v]
                if m.any():
                    non_empty_visits.append(x[b, v][m])  # [len_v, C]

            for i, visit_tokens in enumerate(non_empty_visits):
                parts.append(visit_tokens)
                if i < len(non_empty_visits) - 1:
                    parts.append(gap)

            if len(parts) == 0:
                seq = x.new_zeros((1, emb))
            else:
                seq = torch.cat(parts, dim=0)

            sequences.append(seq)

        max_len = max(seq.size(0) for seq in sequences)
        out = x.new_zeros((bsz, max_len, emb))
        out_mask = x.new_zeros((bsz, max_len), dtype=x.dtype)

        for i, seq in enumerate(sequences):
            out[i, : seq.size(0)] = seq
            out_mask[i, : seq.size(0)] = 1

        return out, out_mask

    def _prepare_sequence_and_mask(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            mask = (x.abs().sum(dim=-1) != 0).to(dtype=x.dtype)  # [B, T]
            return x, mask
        if x.dim() == 4:
            return self._flatten_nested_with_gap(x)
        raise ValueError(f"Deepr expects embedded features to be 3-D or 4-D, got {x.shape}")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        patient_emb = []

        embed_inputs = {
            feature_key: self._extract_feature_tensor(kwargs[feature_key])
            for feature_key in self.feature_keys
        }
        embedded = self.embedding_model(embed_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            else:
                x = x.to(self.device)

            x = x.float()
            x, mask = self._prepare_sequence_and_mask(x)
            pooled = self.deepr[feature_key](x, mask)
            patient_emb.append(pooled)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results