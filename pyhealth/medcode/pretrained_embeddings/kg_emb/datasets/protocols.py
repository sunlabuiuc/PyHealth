"""Structural contract between knowledge-graph datasets and embedding models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = ["KGDatasetProtocol"]


@runtime_checkable
class KGDatasetProtocol(Protocol):
    """Minimal capability a dataset must expose to parameterise a KGE model.

    Embedding models only need the cardinality of the entity and relation
    vocabularies. They never read the samples at construction time. Depending
    on this Protocol rather than on a concrete class keeps the model layer
    testable with lightweight doubles and free of import-time coupling to the
    dataset layer.

    Examples:
        >>> class _Toy:
        ...     entity_num = 2
        ...     relation_num = 1
        ...     task_spec_param = None
        >>> isinstance(_Toy(), KGDatasetProtocol)
        True
    """

    entity_num: int
    relation_num: int
    task_spec_param: Mapping[str, Any] | None
