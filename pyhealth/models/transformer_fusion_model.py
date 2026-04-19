from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base_model import BaseModel
from .embedding import EmbeddingModel
from .fusion import TransformerFusion


class TransformerFusionModel(BaseModel):
    """Transformer fusion model for multimodal temporal datasets.

    This model embeds each input feature using :class:`~pyhealth.models.embedding.EmbeddingModel`,
    aligns feature embeddings across time, and applies a transformer-based fusion
    layer before a task-specific output head.

    Args:
        dataset: A :class:`~pyhealth.datasets.sample_dataset.SampleDataset`
            prepared with a task and fitted processors.
        embedding_dim: Dimension of the shared embedding space.
        num_heads: Number of attention heads in the fusion transformer.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability in transformer layers.
        use_modality_token: Whether to append a learned modality token per time
            step prior to fusion.
    """

    def __init__(
        self,
        dataset,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_modality_token: bool = False,
    ):
        super().__init__(dataset=dataset)

        self.embedding_dim = embedding_dim
        self.embedding_model = EmbeddingModel(dataset, embedding_dim=embedding_dim)
        self.fusion = TransformerFusion(
            hidden_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_modality_token=use_modality_token,
        )

        self.output_head = nn.Linear(embedding_dim, self.get_output_size())
        self.label_key = self.label_keys[0]

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute model outputs for a single batch.

        Args:
            **kwargs: Keyword arguments for each input feature and label.

        Returns:
            A dictionary containing ``logit``, ``y_prob``, ``y_true``, and ``loss``.
        """
        inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for key in self.feature_keys:
            if key not in kwargs:
                raise KeyError(f"Missing feature `{key}` for model forward pass")

            value = kwargs[key]
            if isinstance(value, tuple) and len(value) == 2:
                tensor, mask = value
                inputs[key] = tensor
                masks[key] = mask
            else:
                inputs[key] = value

        embedded = self.embedding_model(inputs, masks=masks if masks else None)

        # Align modality embeddings along the time dimension.
        modality_embeddings: Dict[str, torch.Tensor] = {}
        max_length = 0
        batch_size = None

        for name, tensor in embedded.items():
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            elif tensor.dim() != 3:
                raise ValueError(
                    f"Embedded feature `{name}` must be 2D or 3D, got {tensor.dim()}D"
                )

            if batch_size is None:
                batch_size = tensor.shape[0]

            max_length = max(max_length, tensor.shape[1])
            modality_embeddings[name] = tensor

        assert batch_size is not None, "No modalities were embedded"

        for name, tensor in modality_embeddings.items():
            if tensor.shape[1] < max_length:
                padding = tensor.new_zeros(
                    (batch_size, max_length - tensor.shape[1], tensor.shape[2])
                )
                modality_embeddings[name] = torch.cat([tensor, padding], dim=1)

        fused = self.fusion(modality_embeddings)
        pooled = fused.mean(dim=1)
        logits = self.output_head(pooled)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        return {
            "loss": loss,
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }
