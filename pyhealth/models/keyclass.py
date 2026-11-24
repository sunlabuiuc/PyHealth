# Author: Amin Torabi
# Paper title: Classifying Unstructured Clinical Notes via Automatic Weak Supervision
# Paper link: https://arxiv.org/abs/2206.12088v2
# Description: Multilabel MLP trained on fixed clinical text embeddings.

import torch
from torch import nn

from pyhealth.processors import LabelProcessor, MultiLabelProcessor

from .base_model import BaseModel


class KeyClass(BaseModel):
    """Four-linear-layer classifier with PyHealth Trainer output conventions.

    Use tensor processors for embeddings AND dense soft targets; a multilabel
    token processor would reinterpret probability values as category tokens.
    category_ids fixes output order independently of categories seen in a batch.
    Frozen embeddings are supplied separately, allowing encoder injection and
    caching. Encoder fine-tuning is not included.

    Inputs are standardized per dimension with stored statistics (identity until
    fit_input_standardization is called). Mean-pooled BlueBERT vectors share one
    dominant direction, and the MLP trained on them raw collapsed to near-constant
    scores. The statistics are buffers, so saved models scale new notes the same
    way. The paper and the authors' code feed raw embeddings; this is a documented
    deviation. An optional sample_weight batch field scales each note's loss by its
    weak-label confidence, as in the authors' noise-aware loss.
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
        self._target_processor = MultiLabelProcessor(labels=self.category_ids)
        self.feature_key, self.label_key = feature_key, label_key
        self.feature_keys, self.label_keys = [feature_key], [label_key]
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        layers = []
        for width in hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, width), nn.LeakyReLU(), nn.Dropout(dropout)]
            )
            input_dim = width
        layers.append(nn.Linear(input_dim, len(self.category_ids)))
        self.network = nn.Sequential(*layers)

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Describe model outputs without tokenizing dense soft targets.

        The dataset keeps its tensor processor; this processor supplies the
        multilabel mode, output size, loss, and metrics expected by BaseModel.
        """
        return {self.label_key: self._target_processor}

    def fit_input_standardization(self, embeddings):
        """Store per-dimension mean and standard deviation of training embeddings.

        Uses no labels. Constant dimensions keep a scale of 1.
        """
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != len(self.input_mean):
            raise ValueError("Embeddings must have shape [notes, input_dim]")
        if len(embeddings) < 2 or not torch.isfinite(embeddings).all():
            raise ValueError("Need at least two finite embeddings")
        scale = embeddings.std(dim=0, unbiased=False)
        with torch.no_grad():
            self.input_mean.copy_(embeddings.mean(dim=0))
            self.input_scale.copy_(
                torch.where(scale > 0, scale, torch.ones_like(scale))
            )
        return self

    def forward(self, **kwargs):
        embedding = torch.as_tensor(
            kwargs[self.feature_key], dtype=torch.float32, device=self.device
        )
        if embedding.ndim != 2:
            raise ValueError("Embeddings must have shape [batch, dimension]")
        logits = self.network((embedding - self.input_mean) / self.input_scale)
        result = {"logit": logits, "y_prob": self.label_processor.activation(logits)}
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
            weights = kwargs.get("sample_weight")
            if weights is None:
                loss = self.label_processor.loss(logits, targets)
            else:
                # Noise-aware loss as in the authors' code: scale each note's loss
                # by its weak-label confidence, then average over the batch.
                weights = torch.as_tensor(
                    weights, dtype=logits.dtype, device=self.device
                )
                if weights.shape != logits.shape[:1] or (weights < 0).any():
                    raise ValueError("sample_weight must be nonnegative, one per row")
                per_note = self.label_processor.loss(
                    logits, targets, reduction="none"
                ).mean(dim=1)
                loss = (per_note * weights).mean()
            result.update(y_true=targets, loss=loss)
        return result
