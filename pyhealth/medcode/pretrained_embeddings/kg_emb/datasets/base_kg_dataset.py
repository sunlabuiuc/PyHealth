"""Knowledge graphs on the standard PyHealth dataset backend.

A knowledge graph is one ``triples`` table, declared in a YAML config with
``patient_id: null`` and ``timestamp: null`` and the attributes ``head``,
``relation`` and ``tail``. The loader then numbers the rows, so each triple
is its own record (the same pattern as ``ClinVarDataset``): a
:class:`~pyhealth.datasets.PatientSplit` passed to ``set_task`` splits
triples.

Entity and relation ids are global. They are assigned on the full graph,
names sorted, ids ``0..n-1``, so every split shares one id space, as in the
transductive setting of Bordes et al. (2013) and Sun et al. (2019). Knowing
that an entity exists does not reveal any of its edges.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from pathlib import Path

import polars as pl

from pyhealth.datasets import BaseDataset
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

__all__ = [
    "BaseKGDataset",
    "entity_vocabulary",
    "index_triples",
    "relation_vocabulary",
    "triples_frame",
]

TABLE = "triples"
FIELDS = ("head", "relation", "tail")


def triples_frame(global_event_df: pl.LazyFrame) -> pl.LazyFrame:
    """Selects the triples of a ``BaseKGDataset`` event frame.

    Args:
        global_event_df: The dataset's ``global_event_df``.

    Returns:
        A lazy frame with the string columns ``patient_id``, ``head``,
        ``relation`` and ``tail``, one row per triple.

    Examples:
        >>> import polars as pl
        >>> events = pl.LazyFrame({
        ...     "patient_id": ["0"], "event_type": ["triples"],
        ...     "triples/head": ["a"], "triples/relation": ["r"],
        ...     "triples/tail": ["b"],
        ... })
        >>> triples_frame(events).collect().row(0)
        ('0', 'a', 'r', 'b')
    """
    return global_event_df.filter(pl.col("event_type") == TABLE).select(
        "patient_id", *(pl.col(f"{TABLE}/{f}").alias(f) for f in FIELDS)
    )


def _count_incomplete(triples: pl.LazyFrame) -> int:
    """Counts the triples with a missing head, relation or tail.

    The loader reads every column as a string and turns empty fields into
    nulls, so a malformed row shows up here.
    """
    return (
        triples.filter(pl.any_horizontal(pl.col(f).is_null() for f in FIELDS))
        .select(pl.len())
        .collect()
        .item()
    )


def entity_vocabulary(triples: pl.LazyFrame) -> pl.LazyFrame:
    """Every entity, as a head or a tail, with its id.

    Names are sorted before numbering, so the ids depend only on the set of
    entities, not on the order of the rows.

    Args:
        triples: Output of :func:`triples_frame`.

    Returns:
        A lazy frame with the columns ``name`` (string) and ``id`` (Int64).

    Examples:
        >>> import polars as pl
        >>> triples = pl.LazyFrame(
        ...     {"head": ["c", "a"], "relation": ["r", "r"], "tail": ["a", "b"]}
        ... )
        >>> entity_vocabulary(triples).collect().rows()
        [('a', 0), ('b', 1), ('c', 2)]
    """
    names = pl.concat(
        [
            triples.select(pl.col("head").alias("name")),
            triples.select(pl.col("tail").alias("name")),
        ]
    )
    return (
        names.unique()
        .sort("name")
        .with_row_index("id")
        .select("name", pl.col("id").cast(pl.Int64))
    )


def relation_vocabulary(triples: pl.LazyFrame) -> pl.LazyFrame:
    """Every relation with its id, numbered like :func:`entity_vocabulary`.

    Examples:
        >>> import polars as pl
        >>> triples = pl.LazyFrame(
        ...     {"head": ["a", "b"], "relation": ["s", "r"], "tail": ["b", "a"]}
        ... )
        >>> relation_vocabulary(triples).collect().rows()
        [('r', 0), ('s', 1)]
    """
    return (
        triples.select(pl.col("relation").alias("name"))
        .unique()
        .sort("name")
        .with_row_index("id")
        .select("name", pl.col("id").cast(pl.Int64))
    )


def index_triples(
    triples: pl.LazyFrame,
    entities: pl.LazyFrame,
    relations: pl.LazyFrame,
) -> pl.LazyFrame:
    """Replaces the names of each triple with their ids.

    Args:
        triples: Output of :func:`triples_frame`.
        entities: Output of :func:`entity_vocabulary` on the same graph.
        relations: Output of :func:`relation_vocabulary` on the same graph.

    Returns:
        ``triples`` with the Int64 columns ``head_id``, ``relation_id`` and
        ``tail_id`` added.

    Examples:
        >>> import polars as pl
        >>> triples = pl.LazyFrame(
        ...     {"head": ["b"], "relation": ["r"], "tail": ["a"]}
        ... )
        >>> indexed = index_triples(
        ...     triples, entity_vocabulary(triples), relation_vocabulary(triples)
        ... )
        >>> indexed.select("head_id", "relation_id", "tail_id").collect().row(0)
        (1, 0, 0)
    """
    return (
        triples.join(
            entities.rename({"name": "head", "id": "head_id"}), on="head", how="left"
        )
        .join(
            relations.rename({"name": "relation", "id": "relation_id"}),
            on="relation",
            how="left",
        )
        .join(
            entities.rename({"name": "tail", "id": "tail_id"}), on="tail", how="left"
        )
    )


def _uses_kg_triple(task: BaseTask) -> bool:
    spec = getattr(task, "input_schema", {}).get("triple")
    name = spec[0] if isinstance(spec, tuple) else spec
    return name == "kg_triple"


def _log_unseen_in_training(parts: tuple) -> None:
    """Logs, per held-out part, the triples whose entity or relation is not
    in any training triple.

    Ids are global, so such triples are kept and scored. An entity absent
    from the training triples is still drawn as a negative, so the data terms
    of the loss only train it to score low; a relation absent from them gets
    no gradient from those terms. Regularization, such as the default ``l3``
    of ComplEx and DistMult, still reaches both.
    """
    fitted = parts[0].input_processors["triple"]
    entities = {h for h, _ in fitted.true_tail} | {
        t for tails in fitted.true_tail.values() for t in tails
    }
    relations = {r for _, r in fitted.true_tail}
    for k, part in enumerate(parts[1:], start=1):
        n_entity = n_relation = 0
        # Sequential iteration: on a subset, litdata's part[i] costs O(i),
        # which made this pass quadratic in the size of the part.
        for sample in part:
            head, relation, tail = sample["triple"].tolist()
            n_entity += head not in entities or tail not in entities
            n_relation += relation not in relations
        level = logging.WARNING if n_entity or n_relation else logging.INFO
        logger.log(
            level,
            "Part %d: %d of %d triples involve an entity absent from the "
            "training triples, %d a relation absent from them.",
            k,
            n_entity,
            len(part),
            n_relation,
        )


class BaseKGDataset(BaseDataset):
    """A knowledge graph stored as one ``triples`` table.

    Each record of the table is one triple ``(head, relation, tail)``. Use
    ``set_task`` with a task and ``split=PatientSplit(...)`` to get one
    streaming :class:`~pyhealth.datasets.SampleDataset` per part.

    Args:
        root: Directory (or URL) holding the table file named in the config.
        config_path: YAML config with a ``triples`` table whose attributes
            include ``head``, ``relation`` and ``tail``.
        dataset_name: Name of the dataset. Defaults to the class name.
        cache_dir: Cache directory, as in :class:`~pyhealth.datasets.BaseDataset`.
        num_workers: Worker processes for loading and ``set_task``.
        dev: Keep 1000 triples, chosen by the dev filter of ``BaseDataset``:
            the first 1000 of the file when the table is read as a single
            partition, not necessarily otherwise.

    Raises:
        ValueError: If the config has no ``triples`` table with the
            ``head``, ``relation`` and ``tail`` attributes.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> root = Path(tempfile.mkdtemp())
        >>> _ = (root / "kg.tsv").write_text("head\\trelation\\ttail\\na\\tr\\tb\\n")
        >>> _ = (root / "kg.yaml").write_text(
        ...     'version: "1.0"\\n'
        ...     "tables:\\n"
        ...     "  triples:\\n"
        ...     "    file_path: kg.tsv\\n"
        ...     "    patient_id: null\\n"
        ...     "    timestamp: null\\n"
        ...     "    attributes: [head, relation, tail]\\n"
        ... )
        >>> ds = BaseKGDataset(  # doctest: +SKIP
        ...     root=str(root), config_path=root / "kg.yaml", cache_dir=root / "cache"
        ... )
        >>> ds.entity2id, ds.relation2id  # doctest: +SKIP
        ({'a': 0, 'b': 1}, {'r': 0})
    """

    def __init__(
        self,
        root: str,
        config_path: str | Path,
        dataset_name: str | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            tables=[TABLE],
            dataset_name=dataset_name,
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )
        self._check_config()
        self._entity2id: dict[str, int] | None = None
        self._relation2id: dict[str, int] | None = None

    def _check_config(self) -> None:
        table = self.config.tables.get(TABLE) if self.config else None
        if table is None:
            raise ValueError(f"A knowledge-graph config needs a '{TABLE}' table.")
        missing = [f for f in FIELDS if f not in table.attributes]
        if missing:
            raise ValueError(
                f"The '{TABLE}' table must have the attributes {list(FIELDS)}; "
                f"missing {missing}."
            )

    def _triples_frame(self) -> pl.LazyFrame:
        return triples_frame(self.global_event_df)

    def _load_vocabularies(self) -> None:
        triples = self._triples_frame()
        incomplete = _count_incomplete(triples)
        if incomplete:
            # Dropping them would change the graph; that decision is the
            # user's, so stop instead.
            raise ValueError(
                f"{incomplete} triple(s) have a missing head, relation or tail."
            )
        entities = entity_vocabulary(triples).collect()
        relations = relation_vocabulary(triples).collect()
        self._entity2id = dict(zip(entities["name"], entities["id"]))
        self._relation2id = dict(zip(relations["name"], relations["id"]))

    @property
    def entity2id(self) -> dict[str, int]:
        """Entity name to id, on the full graph."""
        if self._entity2id is None:
            self._load_vocabularies()
        return self._entity2id

    @property
    def relation2id(self) -> dict[str, int]:
        """Relation name to id, on the full graph."""
        if self._relation2id is None:
            self._load_vocabularies()
        return self._relation2id

    @property
    def id2entity(self) -> dict[int, str]:
        """Entity id to name."""
        return {i: name for name, i in self.entity2id.items()}

    @property
    def id2relation(self) -> dict[int, str]:
        """Relation id to name."""
        return {i: name for name, i in self.relation2id.items()}

    @property
    def num_entities(self) -> int:
        """Number of distinct entities in the graph."""
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        """Number of distinct relations in the graph."""
        return len(self.relation2id)

    def set_task(self, task: BaseTask | None = None, *args, **kwargs):
        """Builds the sample dataset(s) of ``task``.

        This is :meth:`pyhealth.datasets.BaseDataset.set_task`, with
        :class:`KGLinkPrediction` as the default task. For a task whose
        ``triple`` field uses the ``kg_triple`` processor, it also warns when
        ``split`` is missing, since that processor would then be fitted on
        every triple, and with ``split`` it logs, for each held-out part, how
        many triples involve an entity or a relation absent from the training
        triples.
        """
        if task is None:
            task = self.default_task

        bound = inspect.signature(BaseDataset.set_task).bind(
            self, task, *args, **kwargs
        ).arguments
        split = bound.get("split")
        # A supplied processor is used as it is, never refitted.
        fits_triples = _uses_kg_triple(task) and "triple" not in (
            bound.get("input_processors") or {}
        )
        if fits_triples and split is None:
            warnings.warn(
                "set_task without split= fits the kg_triple processor on every "
                "triple, so training negatives and subsampling weights depend on "
                "validation and test triples. Pass split=PatientSplit(...) to fit "
                "it on the training triples only.",
                UserWarning,
                stacklevel=2,
            )
        result = super().set_task(task, *args, **kwargs)
        if _uses_kg_triple(task) and isinstance(result, tuple):
            _log_unseen_in_training(result)
        return result

    @property
    def default_task(self) -> BaseTask:
        """:class:`KGLinkPrediction` on this graph."""
        from ..tasks.kg_link_prediction import KGLinkPrediction

        return KGLinkPrediction(
            num_entities=self.num_entities, num_relations=self.num_relations
        )
