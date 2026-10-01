"""Task-specific sample dataset for knowledge-graph embedding models.

``SampleKGDataset`` builds on :class:`pyhealth.datasets.InMemorySampleDataset`
so that KG triples and the variable-length ``ground_truth_head`` /
``ground_truth_tail`` filter sets are converted to pure PyTorch tensors ahead
of ``litdata``'s pickle-based caching (the "Tensor Trick"), instead of being
serialized as raw Python lists on every access.

``ground_truth_head`` and ``ground_truth_tail`` are padded independently, each
to its own field's observed maximum length, by the registered
``"kg_entity_list"`` processor
(:class:`~pyhealth.processors.kg_processor.KGProcessor`), which returns a
``{"value": Tensor, "mask": Tensor}`` pair. Because the padding value is not
a valid entity id on its own, any code that filters on these fields (see
:class:`~pyhealth.medcode.pretrained_embeddings.kg_emb.models.kg_base.KGEBaseModel`)
must use the mask to recover the true, unpadded entity list first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from pyhealth.datasets.sample_dataset import InMemorySampleDataset

KGSample = Mapping[str, Any]

__all__ = ["SampleKGDataset"]


class SampleKGDataset(InMemorySampleDataset):
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
        pad_token_id: Entity id used to pad ``ground_truth_head`` /
            ``ground_truth_tail`` to their fitted per-field max length.
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
        pad_token_id: int = 0,
        **task_spec_param: Any,
    ) -> None:
        input_schema = {
            "triple": ("tensor", {"dtype": torch.long}),
            "ground_truth_head": ("kg_entity_list", {"pad_token_id": pad_token_id}),
            "ground_truth_tail": ("kg_entity_list", {"pad_token_id": pad_token_id}),
        }
        super().__init__(
            samples=list(samples),
            input_schema=input_schema,
            output_schema={},
            dataset_name=dataset_name,
            task_name=task_name,
        )

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
        return len(self)

    def stat(self) -> str:
        """Print a human-readable summary and return it."""
        lines = [
            "",
            f"Statistics of sample KG dataset (dev={self.dev}):",
            f"\t- Dataset: {self.dataset_name}",
            f"\t- Task name: {self.task_name}",
            f"\t- Number of triples: {len(self)}",
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
