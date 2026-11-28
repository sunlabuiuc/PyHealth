# Author: Amin Torabi
# Paper title: Classifying Unstructured Clinical Notes via Automatic Weak Supervision
# Paper link: https://arxiv.org/abs/2206.12088v2
# Description: Multilabel MLP trained on fixed clinical text embeddings.

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import LabelProcessor, MultiLabelProcessor

from .base_model import BaseModel


class KeyClass(BaseModel):
    """KeyClass downstream classifier for weakly labeled clinical notes.

    Paper: Chufan Gao, Mononito Goswami, Jieshi Chen and Artur Dubrawski.
    Classifying Unstructured Clinical Notes via Automatic Weak Supervision.
    MLHC 2022. https://arxiv.org/abs/2206.12088v2

    A four-linear-layer MLP (LeakyReLU and dropout after each hidden layer) that
    maps frozen document embeddings to one score per category. It is trained
    with binary cross-entropy on the soft weak labels produced by
    :class:`pyhealth.weak_supervision.KeywordLabeler` and
    :func:`pyhealth.weak_supervision.aggregate_votes`, so no gold labels are
    needed. Embeddings are computed outside the model, which allows any encoder
    and caching; encoder fine-tuning is not included.

    Both the embedding and the soft-label fields must use the ``"tensor"``
    processor: a multilabel token processor would read probability values as
    category tokens. ``category_ids`` fixes the output order regardless of
    which categories appear in a batch.

    Inputs are standardized per dimension with stored statistics, which are the
    identity until :meth:`fit_input_standardization` is called. Mean-pooled
    BlueBERT vectors share one dominant direction, and the MLP trained on them
    raw collapsed to near-constant scores. The statistics are buffers, so a
    saved model scales new notes the same way. The paper and the authors' code
    feed raw embeddings; this is a documented deviation.

    The loss uses the soft weak labels, but the returned ``"y_true"`` is
    binary so that :class:`pyhealth.trainer.Trainer` can score it with the
    multilabel metrics. When the batch has gold labels under ``gold_key``,
    those are ``"y_true"``, so ``Trainer.evaluate`` reports agreement with the
    gold labels. Otherwise ``"y_true"`` marks each category whose weak label
    is at least ``1 / categories``, the paper's prediction threshold for scores
    that sum to one, and the metrics measure agreement with the weak labels.

    Args:
        dataset: The sample dataset the model is trained on.
        input_dim: Dimension of the document embeddings.
        category_ids: Category IDs in output order; at least two, unique.
        hidden_dims: Widths of the hidden layers.
        dropout: Dropout probability after each hidden layer.
        feature_key: Batch field holding the embeddings.
        label_key: Batch field holding the soft weak labels.
        gold_key: Optional batch field holding gold 0/1 labels of shape
            ``[batch, categories]`` in ``category_ids`` order. They are used
            only as ``"y_true"`` for evaluation, never in the loss.

    Attributes:
        network: The MLP mapping standardized embeddings to logits.
        input_mean: Per-dimension mean subtracted from inputs.
        input_scale: Per-dimension scale inputs are divided by.

    Raises:
        ValueError: If there are fewer than two or duplicate category IDs, or a
            layer dimension is not positive.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "embedding": [0.1, 0.2],
        ...      "weak_labels": [1.0, 0.0]},
        ...     {"patient_id": "p1", "embedding": [0.3, 0.1],
        ...      "weak_labels": [0.4, 0.6]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"embedding": "tensor"},
        ...     output_schema={"weak_labels": "tensor"},
        ... )
        >>> model = KeyClass(dataset, input_dim=2, category_ids=["cardiac", "resp"])
        >>> _ = model.fit_input_standardization([[0.1, 0.2], [0.3, 0.1]])
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_dim: int,
        category_ids: Sequence[Hashable],
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.5,
        feature_key: str = "embedding",
        label_key: str = "weak_labels",
        gold_key: str | None = None,
    ) -> None:
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
        self.gold_key = gold_key
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
        """Describes model outputs without tokenizing dense soft targets.

        The dataset keeps its tensor processor; this multilabel processor
        supplies the mode, output size, loss and metrics ``BaseModel`` expects.

        Returns:
            A mapping from ``label_key`` to the multilabel processor.
        """
        return {self.label_key: self._target_processor}

    def fit_input_standardization(
        self, embeddings: torch.Tensor | np.ndarray | Sequence[Sequence[float]]
    ) -> "KeyClass":
        """Stores the per-dimension mean and spread of training embeddings.

        Uses no labels. Constant dimensions keep a scale of 1.

        Args:
            embeddings: Training embeddings of shape ``[notes, input_dim]``.

        Returns:
            The model, for chaining.

        Raises:
            ValueError: If the shape is wrong, there are fewer than two
                embeddings, or values are not finite.
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

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Scores a batch and, when soft labels are given, computes the loss.

        Args:
            **kwargs: A batch. ``kwargs[feature_key]`` holds embeddings of shape
                ``[batch, input_dim]``. Optional ``kwargs[label_key]`` holds soft
                labels in [0, 1] of shape ``[batch, categories]``. Optional
                ``kwargs["sample_weight"]`` holds nonnegative per-note weights
                of shape ``[batch]``; each note's loss is scaled by its weight,
                as in the authors' noise-aware loss. Optional
                ``kwargs[gold_key]`` holds gold 0/1 labels used as ``"y_true"``.

        Returns:
            A dict with ``"logit"`` and ``"y_prob"`` (sigmoid scores), both of
            shape ``[batch, categories]``. When soft labels are given it also
            has the scalar ``"loss"`` and the binary ``"y_true"``: the gold
            labels if present, otherwise the weak labels thresholded at
            ``1 / categories``.

        Raises:
            ValueError: If embeddings, labels or weights have the wrong shape,
                soft labels are not finite probabilities, or gold labels are
                not 0/1.
        """
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
                weights = torch.as_tensor(
                    weights, dtype=logits.dtype, device=self.device
                )
                if weights.shape != logits.shape[:1] or (weights < 0).any():
                    raise ValueError("sample_weight must be nonnegative, one per row")
                per_note = self.label_processor.loss(
                    logits, targets, reduction="none"
                ).mean(dim=1)
                loss = (per_note * weights).mean()
            if self.gold_key is not None and self.gold_key in kwargs:
                y_true = torch.as_tensor(
                    kwargs[self.gold_key], dtype=logits.dtype, device=self.device
                )
                if (
                    y_true.shape != logits.shape
                    or not torch.isin(y_true, y_true.new_tensor([0.0, 1.0])).all()
                ):
                    raise ValueError("Gold labels must match logits and be 0 or 1")
            else:
                threshold = 1 / len(self.category_ids)
                y_true = (targets >= threshold).to(logits.dtype)
            result.update(y_true=y_true, loss=loss)
        return result
