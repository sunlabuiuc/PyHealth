"""Subgroup resampling (Idrissi et al., CLeaR 2022), a MEDFAIR baseline."""

import functools
from collections.abc import Hashable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _hashable(value: Any) -> Hashable:
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        value = value.tolist()
    if isinstance(value, list):
        return value[0] if len(value) == 1 else tuple(value)
    return value


class GroupBalancedSampler(Sampler[int]):
    """Draws indices with replacement so every group is equally likely.

    Each sample gets weight ``1 / |group|``. With ``labels`` the groups are
    the (group, label) cells, which also balances the classes within each
    group (MEDFAIR's subgroup x class resampling).

    The draws of a pass depend only on ``seed`` and the epoch, which
    advances after every pass and can be set with :meth:`set_epoch`.

    Args:
        groups: Group of each sample, e.g. its sex or age bin.
        labels: Optional label of each sample, to balance the (group, label)
            cells. Defaults to None.
        num_samples: Indices drawn per pass. Defaults to ``len(groups)``.
        seed: Seed of the draws. Defaults to 0.

    Raises:
        ValueError: If ``groups`` is empty, has missing values, or differs
            in length from ``labels``.

    Examples:
        >>> sampler = GroupBalancedSampler(["F", "F", "F", "M"], seed=0)
        >>> sampler.weights.tolist()
        [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 1.0]
        >>> len(sampler)
        4
    """

    def __init__(
        self,
        groups: Sequence[Hashable],
        labels: Sequence[Hashable] | None = None,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        groups = [_hashable(g) for g in groups]
        if not groups:
            raise ValueError("groups is empty.")
        if any(pd.isna(g) for g in groups if not isinstance(g, tuple)):
            raise ValueError("groups contain missing values.")
        keys: list = groups
        if labels is not None:
            if len(labels) != len(groups):
                raise ValueError(
                    f"groups and labels differ in length: "
                    f"{len(groups)} vs {len(labels)}."
                )
            keys = list(zip(groups, (_hashable(y) for y in labels)))
        codes: dict = {}
        cells = np.array([codes.setdefault(k, len(codes)) for k in keys])
        self.weights = torch.as_tensor(
            1.0 / np.bincount(cells)[cells], dtype=torch.double
        )
        self.num_samples = len(cells) if num_samples is None else int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch of the next pass.

        Examples:
            >>> GroupBalancedSampler(["F", "M"]).set_epoch(3)
        """
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        state = np.random.SeedSequence([self.seed, self.epoch]).generate_state(1)
        generator = torch.Generator().manual_seed(int(state[0]))
        self.epoch += 1
        draws = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        )
        return iter(draws.tolist())

    def __len__(self) -> int:
        return self.num_samples


class _IndexView(Dataset):
    """Random-access view of a (streaming) sample dataset."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


def _collate_fn(dataset: Any):
    # Imported here: pyhealth.datasets.imaging imports this package.
    from ..datasets.utils import collate_fn_dict_with_padding

    processors = {**dataset.input_processors, **dataset.output_processors}
    collators = {
        key: processor.collate
        for key, processor in processors.items()
        if callable(getattr(processor, "collate", None))
    }
    return functools.partial(collate_fn_dict_with_padding, collators=collators)


def get_group_balanced_dataloader(
    dataset: Any,
    batch_size: int,
    group_key: str | None = None,
    label_key: str | None = None,
    groups: Sequence[Hashable] | None = None,
    labels: Sequence[Hashable] | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    **loader_kwargs: Any,
) -> DataLoader:
    """Training loader that resamples a sample dataset by subgroup.

    A drop-in replacement for :func:`~pyhealth.datasets.get_dataloader` on the
    training split. Give the groups either as a sample field (``group_key``,
    read from every sample once, which decodes its inputs too) or directly
    (``groups``, in dataset order); likewise for the optional labels.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset`.
        batch_size: Samples per batch.
        group_key: Sample field holding the sensitive attribute.
        label_key: Sample field holding the label, to balance the
            (group, label) cells. Defaults to None.
        groups: Group of each sample, instead of ``group_key``.
        labels: Label of each sample, instead of ``label_key``.
        num_samples: Samples drawn per epoch. Defaults to ``len(dataset)``.
        seed: Seed of the draws. Default None draws one from torch's random
            generator, so ``torch.manual_seed`` controls it.
        **loader_kwargs: Forwarded to ``DataLoader``, e.g. ``num_workers``.

    Returns:
        A ``DataLoader`` whose ``sampler`` is a :class:`GroupBalancedSampler`.

    Raises:
        ValueError: Unless exactly one of ``group_key`` and ``groups`` is
            given, if both ``label_key`` and ``labels`` are, or if
            ``groups`` does not match the dataset length.
        KeyError: If a sample lacks ``group_key`` or ``label_key``.

    Examples:
        >>> loader = get_group_balanced_dataloader(  # doctest: +SKIP
        ...     train_ds, batch_size=64, group_key="sex", seed=0
        ... )
        >>> trainer.train(train_dataloader=loader, ...)  # doctest: +SKIP
    """
    if (group_key is None) == (groups is None):
        raise ValueError("Pass exactly one of group_key and groups.")
    if label_key is not None and labels is not None:
        raise ValueError("Pass at most one of label_key and labels.")
    keys = [k for k in (group_key, label_key) if k is not None]
    if keys:
        columns = {k: [] for k in keys}
        for i in range(len(dataset)):
            sample = dataset[i]
            for k in keys:
                columns[k].append(sample[k])
        groups = columns[group_key] if group_key is not None else groups
        labels = columns[label_key] if label_key is not None else labels
    if len(groups) != len(dataset):
        raise ValueError(
            f"Got {len(groups)} groups for a dataset of {len(dataset)} samples."
        )
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, ()).item())
    sampler = GroupBalancedSampler(
        groups, labels=labels, num_samples=num_samples, seed=seed
    )
    return DataLoader(
        _IndexView(dataset),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=_collate_fn(dataset),
        **loader_kwargs,
    )
