"""Phase 6: a multilabel MLP trained on fixed clinical text embeddings."""

import torch
from torch import nn

from .base_model import BaseModel


class KeyClass(BaseModel):
    """Four-linear-layer classifier with PyHealth Trainer output conventions.

    Use tensor processors for embeddings AND dense soft targets; a multilabel
    token processor would reinterpret probability values as category tokens.
    category_ids fixes output order independently of categories seen in a batch.
    Frozen embeddings are supplied separately, allowing encoder injection and
    caching. Encoder fine-tuning and self-training are subsequent experiments.
    """

    def __init__(
        self,
        dataset,
        input_dim,
        category_ids,
        hidden_dims=(256, 128, 64),
        dropout=0.5,
        feature_key="embedding",
        label_key="weak_labels",
    ):
        super().__init__(dataset)
        self.category_ids = tuple(category_ids)
        if len(self.category_ids) < 2 or len(set(self.category_ids)) != len(
            self.category_ids
        ):
            raise ValueError("Provide at least two unique category IDs in output order")
        if input_dim < 1 or not hidden_dims or any(d < 1 for d in hidden_dims):
            raise ValueError("Layer dimensions must be positive")
        self.mode = "multilabel"
        self.feature_key, self.label_key = feature_key, label_key
        self.feature_keys, self.label_keys = [feature_key], [label_key]
        layers = []
        for width in hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, width), nn.LeakyReLU(), nn.Dropout(dropout)]
            )
            input_dim = width
        layers.append(nn.Linear(input_dim, len(self.category_ids)))
        self.network = nn.Sequential(*layers)

    def forward(self, **kwargs):
        embedding = torch.as_tensor(
            kwargs[self.feature_key], dtype=torch.float32, device=self.device
        )
        if embedding.ndim != 2:
            raise ValueError("Embeddings must have shape [batch, dimension]")
        logits = self.network(embedding)
        result = {"logit": logits, "y_prob": torch.sigmoid(logits)}
        if self.label_key in kwargs:
            targets = torch.as_tensor(
                kwargs[self.label_key], dtype=logits.dtype, device=self.device
            )
            if (
                targets.shape != logits.shape
                or not torch.isfinite(targets).all()
                or ((targets < 0) | (targets > 1)).any()
            ):
                raise ValueError(
                    "Targets must match logits and contain probabilities in [0, 1]"
                )
            result.update(
                y_true=targets,
                loss=nn.functional.binary_cross_entropy_with_logits(logits, targets),
            )
        return result
