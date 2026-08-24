"""Ratio-based splitting of a :class:`SampleKGDataset` into train/val/test folds."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .sample_kg_dataset import SampleKGDataset

__all__ = ["split"]

Fold = list[dict[str, Any]]


def split(
    dataset: SampleKGDataset,
    ratios: list[float] | tuple[float, float, float],
    seed: int | None = None,
) -> tuple[Fold, Fold, Fold]:
    """Split a KG sample dataset into three disjoint folds.

    The split is uniform over triples: each sample is assigned to exactly one
    fold, so the three folds partition the dataset. Training samples carry the
    task hyper-parameters needed by the negative sampler; validation and test
    samples are flagged so that the model switches to filtered ranking
    evaluation.

    Args:
        dataset: The dataset to split.
        ratios: Three non-negative floats summing to 1, in train/val/test
            order.
        seed: Seed of a local random generator. The global NumPy state is
            left untouched, which keeps the function reproducible without
            side effects.

    Returns:
        The train, validation and test folds, each a list of sample
        dictionaries.

    Raises:
        ValueError: If ``ratios`` is malformed. Validation of user input is
            raised rather than asserted, because ``assert`` statements are
            stripped under ``python -O`` and this check must survive
            optimised runs. The tolerance comparison guards the rare
            triplets -- about 0.9% of two-decimal ratios -- for which
            floating-point summation does not land exactly on 1.

    Examples:
        >>> import torch
        >>> from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import (
        ...     SampleKGDataset,
        ... )
        >>> samples = [
        ...     {
        ...         "triple": (i, i % 2, (i + 1) % 5),
        ...         "ground_truth_head": [i, (i + 1) % 5],
        ...         "ground_truth_tail": [(i + 1) % 5],
        ...         "subsampling_weight": torch.tensor([0.25]),
        ...     }
        ...     for i in range(10)
        ... ]
        >>> dataset = SampleKGDataset(
        ...     samples=samples, entity_num=5, relation_num=2, negative_sampling=4
        ... )
        >>> train, val, test = split(dataset, [0.6, 0.2, 0.2], seed=0)
        >>> len(train), len(val), len(test)
        (6, 2, 2)
        >>> train[0]["train"], train[0]["hyperparameters"]
        (True, {'negative_sampling': 4})
        >>> val[0]["train"]
        False
        >>> split(dataset, [0.5, 0.2, 0.2], seed=0)
        Traceback (most recent call last):
            ...
        ValueError: ratios must sum to 1.0, got 0.9
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be three non-negative floats, got {ratios!r}")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    n = len(dataset)
    index = rng.permutation(n)

    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))
    slices = (index[:n_train], index[n_train:n_val], index[n_val:])

    hyperparameters = dataset.task_spec_param
    train = [
        {**dataset[int(i)], "train": True, "hyperparameters": hyperparameters}
        for i in slices[0]
    ]
    val = [{**dataset[int(i)], "train": False} for i in slices[1]]
    test = [{**dataset[int(i)], "train": False} for i in slices[2]]
    return train, val, test
