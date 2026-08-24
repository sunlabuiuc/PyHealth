"""Task-specific sample dataset for knowledge-graph embedding models.

This module deliberately does **not** build on
:class:`pyhealth.datasets.SampleDataset`. Since PyHealth 2.0,
``SampleDataset`` is a ``litdata.StreamingDataset`` whose contract is "a
directory containing ``schema.pkl`` plus optimized chunks". A knowledge
graph task produces an in-memory list of triple-level records and has no
feature schema, no processors and no patient/visit index: the two
abstractions are unrelated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import Dataset

KGSample = Mapping[str, Any]

__all__ = ["SampleKGDataset"]


class SampleKGDataset(Dataset):
    r"""In-memory dataset of knowledge-graph link-prediction samples.

    Each sample is a mapping with the following keys:

    ``triple``
        A positive triple :math:`(h, r, t)` given as integer indices,
        e.g. ``(0, 0, 2835)``.
    ``ground_truth_head``
        All entities :math:`h'` such that :math:`(h', r, t)` is observed
        in the graph. Used to filter false negatives when scoring the
        query :math:`(?, r, t)`.
    ``ground_truth_tail``
        All entities :math:`t'` such that :math:`(h, r, t')` is observed
        in the graph.
    ``subsampling_weight``
        The word2vec-style subsampling weight of the triple, a scalar
        tensor.

    Args:
        samples: The task samples, typically produced by
            ``link_prediction_fn``.
        dataset_name: Human-readable name of the source dataset.
        task_name: Human-readable name of the task.
        dev: Whether the samples come from a development subset.
        entity_num: Number of entities. Inferred from ``entity2id`` when
            omitted.
        relation_num: Number of relations. Inferred from ``relation2id``
            when omitted.
        entity2id: Mapping from surface entity identifier to integer
            index.
        relation2id: Mapping from surface relation identifier to integer
            index.
        **task_spec_param: Task hyper-parameters forwarded to the model
            at training time (e.g. ``negative_sampling=128``).

    Raises:
        ValueError: If the declared cardinalities contradict the provided
            vocabularies.

    Examples:
        >>> import torch
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
        ...     samples=samples,
        ...     dataset_name="toy",
        ...     task_name="link_prediction",
        ...     entity2id={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
        ...     relation2id={"treats": 0, "causes": 1},
        ...     negative_sampling=4,
        ... )
        >>> len(dataset)
        10
        >>> dataset[0]["triple"]
        (0, 0, 1)
        >>> dataset.entity_num, dataset.relation_num
        (5, 2)
        >>> dataset.id2entity[2]
        'c'
        >>> dataset.task_spec_param
        {'negative_sampling': 4}
    """

    def __init__(
        self,
        samples: Sequence[KGSample],
        dataset_name: str = "",
        task_name: str = "",
        dev: bool = False,
        entity_num: int = 0,
        relation_num: int = 0,
        entity2id: Mapping[Any, int] | None = None,
        relation2id: Mapping[Any, int] | None = None,
        **task_spec_param: Any,
    ) -> None:
        self.samples: list[KGSample] = list(samples)
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.dev = dev

        self.entity2id: dict[Any, int] = dict(entity2id or {})
        self.relation2id: dict[Any, int] = dict(relation2id or {})
        self.id2entity: dict[int, Any] = {v: k for k, v in self.entity2id.items()}
        self.id2relation: dict[int, Any] = {
            v: k for k, v in self.relation2id.items()
        }

        self.entity_num = entity_num or len(self.entity2id)
        self.relation_num = relation_num or len(self.relation2id)
        self._validate_cardinalities()

        # ``None`` rather than ``{}`` preserves the historical sentinel
        # used by the models.
        self.task_spec_param: dict[str, Any] | None = task_spec_param or None

    def _validate_cardinalities(self) -> None:
        if self.entity2id and self.entity_num != len(self.entity2id):
            raise ValueError(
                f"entity_num={self.entity_num} contradicts len(entity2id)="
                f"{len(self.entity2id)}"
            )
        if self.relation2id and self.relation_num != len(self.relation2id):
            raise ValueError(
                f"relation_num={self.relation_num} contradicts "
                f"len(relation2id)={len(self.relation2id)}"
            )

    @property
    def sample_size(self) -> int:
        """Number of samples. Kept as a property for backward compatibility."""
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> KGSample:
        return self.samples[index]

    def stat(self) -> str:
        """Return -- and print -- a human-readable summary of the dataset."""
        lines = [
            "",
            f"Statistics of sample KG dataset (dev={self.dev}):",
            f"\t- Dataset: {self.dataset_name}",
            f"\t- Task name: {self.task_name}",
            f"\t- Number of triples: {len(self.samples)}",
            f"\t- Number of entities: {self.entity_num}",
            f"\t- Number of relations: {self.relation_num}",
            f"\t- Task-specific hyperparameters: {self.task_spec_param}",
            "",
        ]
        report = "\n".join(lines)
        print(report)
        return report

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dataset_name={self.dataset_name!r}, "
            f"task_name={self.task_name!r}, size={len(self)}, "
            f"entity_num={self.entity_num}, relation_num={self.relation_num})"
        )
