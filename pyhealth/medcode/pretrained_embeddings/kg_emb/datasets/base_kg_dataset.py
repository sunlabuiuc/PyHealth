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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from pyhealth.datasets import BaseDataset, SampleDataset
from pyhealth.processors import PROCESSOR_REGISTRY, KGTripleProcessor
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

__all__ = [
    "BaseKGDataset",
    "KnownTriples",
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


@dataclass
class KnownTriples:
    """The known heads and tails of every pair, over the triples of a task.

    Filtered ranking (Bordes et al., 2013) ranks the true entity of a test
    triple against every other entity except those that also complete it,
    in any part: ``heads[(relation, tail)]`` and ``tails[(head, relation)]``
    hold them, as sorted lists of ids. :meth:`BaseKGDataset.set_task` builds
    them once, from the triples of all the parts it returns, and attaches
    them to every part as ``part.known_triples``. Their size therefore grows
    with the number of triples; a copy in each sample grew with the number
    of triples times the largest degree. One instance is shared by every
    part and by every model built from them, so treat it as read-only.
    Training never reads them: its filters come from the ``kg_triple``
    processor, fitted on the training part.

    Examples:
        >>> import polars as pl
        >>> ids = pl.DataFrame(
        ...     {"head_id": [0, 2, 0], "relation_id": [0, 0, 1], "tail_id": [1, 1, 2]}
        ... )
        >>> known = KnownTriples.from_ids(ids)
        >>> known.heads[(0, 1)], known.tails[(0, 0)]
        ([0, 2], [1])
    """

    heads: dict[tuple[int, int], list[int]]
    tails: dict[tuple[int, int], list[int]]

    @classmethod
    def from_ids(cls, ids: pl.DataFrame) -> KnownTriples:
        """Builds the sets from triples given as ``head_id``, ``relation_id``
        and ``tail_id`` columns; duplicate triples count once."""
        ids = ids.select("head_id", "relation_id", "tail_id").unique()
        heads = ids.group_by("relation_id", "tail_id").agg(pl.col("head_id").sort())
        tails = ids.group_by("head_id", "relation_id").agg(pl.col("tail_id").sort())
        return cls(
            heads={(r, t): h for r, t, h in heads.iter_rows()},
            tails={(h, r): t for h, r, t in tails.iter_rows()},
        )


def _stored_triples(
    parts: tuple[SampleDataset, ...], split, start: int = 0
) -> list[list[tuple[int, int, int]]]:
    """The triples of ``parts[start:]``, read by index from the stored samples.

    Iterating a part is not used. Under ``torch.distributed``, litdata
    streams to each rank only its share of the part, so filter sets built
    from an iteration would miss the triples of the other ranks. On a part
    whose samples are scattered, iteration also remaps the chunk once per
    run of consecutive indices, which made the reads grow faster than the
    number of triples. Index access on the unsplit dataset is neither
    sharded nor remapped. The sample indices of each part are those of
    ``split.npz``, from which ``BaseDataset.set_task`` builds the parts.

    Args:
        parts: The parts returned by ``BaseDataset.set_task``.
        split: The ``split`` passed to it, or None.
        start: Index of the first part to read.

    Returns:
        One list of ``(head, relation, tail)`` per part read, in sample order.
    """
    path = Path(parts[0].path)
    unsplit = SampleDataset(path=str(path))
    if split is None:
        # len() of a streaming dataset is the share of the current rank, so
        # count the stored samples instead.
        n_samples = sum(end - begin for begin, end in unsplit.region_of_interest)
        indices = [np.arange(n_samples)]
    else:
        with np.load(path / "split.npz") as stored:
            indices = [stored[f"arr_{k}"] for k in range(len(stored.files))]
    return [
        [tuple(unsplit[int(i)]["triple"].tolist()) for i in part]
        for part in indices[start:]
    ]


def _fitted_triples(processor: KGTripleProcessor) -> list[tuple[int, int, int]]:
    """The triples a ``kg_triple`` processor was fitted on."""
    return [(h, r, t) for (h, r), tails in processor.true_tail.items() for t in tails]


def _holds_the_first_part(processor: KGTripleProcessor) -> bool:
    """Whether a processor fitted by ``set_task`` holds exactly the triples
    of the first part, so that reading that part can be skipped.

    ``KGTripleProcessor.fit`` records every triple it is given, and
    ``set_task`` gives it the samples of the first part. A subclass may
    record other triples, so its first part is read.
    """
    return type(processor) is KGTripleProcessor


def _known_triples(triples: list[tuple[int, int, int]]) -> KnownTriples:
    """The :class:`KnownTriples` of ``triples``.

    ``set_task`` passes the triples of its parts rather than those of the
    raw graph, so that a task whose ``pre_filter`` drops triples, or numbers
    a subgraph, gets filter sets that match its own samples.
    """
    schema = {name: pl.Int64 for name in ("head_id", "relation_id", "tail_id")}
    return KnownTriples.from_ids(pl.DataFrame(triples, schema=schema, orient="row"))


def _uses_kg_triple(task: BaseTask) -> bool:
    """Whether the task's ``triple`` field uses a ``KGTripleProcessor``, in any
    form the schema accepts: a registered name or a class, alone or with a
    mapping of keyword arguments, or an instance. Any other spec, and a schema
    that is not a mapping, give False; ``BaseDataset.set_task`` then raises
    its own error on them."""
    schema = getattr(task, "input_schema", None)
    spec = schema.get("triple") if isinstance(schema, Mapping) else None
    if isinstance(spec, tuple):
        if len(spec) != 2 or not isinstance(spec[1], Mapping):
            return False
        spec = spec[0]
        if not (isinstance(spec, str) or inspect.isclass(spec)):
            return False
    if isinstance(spec, str):
        spec = PROCESSOR_REGISTRY.get(spec)
    if not inspect.isclass(spec):
        spec = type(spec)
    return issubclass(spec, KGTripleProcessor)


def _log_unseen_in_training(
    train: list[tuple[int, int, int]], held_out: list[list[tuple[int, int, int]]]
) -> None:
    """Logs, per held-out part, how many triples involve an entity or a
    relation absent from every training triple.

    Ids are global, so such triples are kept and scored. An entity absent
    from the training triples is still drawn as a negative, so the data terms
    of the loss only train it to score low; a relation absent from them gets
    no gradient from those terms. Regularization, such as the default ``l3``
    of ComplEx and DistMult, still reaches both.

    Args:
        train: The triples of the training part.
        held_out: The triples of each held-out part, in part order.
    """
    entities = {h for h, _, _ in train} | {t for _, _, t in train}
    relations = {r for _, r, _ in train}
    for k, triples in enumerate(held_out, start=1):
        n_entity = sum(h not in entities or t not in entities for h, _, t in triples)
        n_relation = sum(r not in relations for _, r, _ in triples)
        level = logging.WARNING if n_entity or n_relation else logging.INFO
        logger.log(
            level,
            "Part %d: %d of %d triples involve an entity absent from the "
            "training triples, %d a relation absent from them.",
            k,
            n_entity,
            len(triples),
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
        :class:`~pyhealth.medcode.pretrained_embeddings.kg_emb.tasks.KGLinkPrediction`
        as the default task. For a task whose
        ``triple`` field uses the ``kg_triple`` processor, it also warns when
        ``split`` is missing, since that processor would then be fitted on
        every triple. When the fitted ``triple`` processor is a
        :class:`~pyhealth.processors.KGTripleProcessor`, it logs, with
        ``split``, for each held-out part, how many triples involve an entity
        or a relation absent from the training triples, and every returned
        part carries, as ``known_triples``, the :class:`KnownTriples` of the
        triples of all the parts, which the models read to filter the
        ranking at evaluation. The triples of the held-out parts are read by
        index from the stored samples rather than streamed, and so are those
        of the first part when its processor was supplied or is a subclass,
        so under ``torch.distributed`` they are not limited to the share of
        the current rank.
        """
        if task is None:
            task = self.default_task

        bound = inspect.signature(BaseDataset.set_task).bind(
            self, task, *args, **kwargs
        ).arguments
        split = bound.get("split")
        # A supplied processor is used as it is, never refitted.
        supplied = "triple" in (bound.get("input_processors") or {})
        fits_triples = _uses_kg_triple(task) and not supplied
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
        parts = result if isinstance(result, tuple) else (result,)
        processor = parts[0].input_processors.get("triple")
        # Test the fitted processor, as the models do, rather than the schema
        # string: a processor class or instance in the schema gives the same
        # processor.
        if isinstance(processor, KGTripleProcessor):
            # The first part is usually the largest. A processor fitted here
            # by this class holds exactly its triples, which spares reading it
            # again; a supplied one may have been fitted on other triples.
            if not supplied and _holds_the_first_part(processor):
                first = _fitted_triples(processor)
                held_out = _stored_triples(parts, split, start=1)
            else:
                first, *held_out = _stored_triples(parts, split)
            if held_out:
                _log_unseen_in_training(first, held_out)
            known = _known_triples(first + [t for part in held_out for t in part])
            for part in parts:
                part.known_triples = known
        return result

    @property
    def default_task(self) -> BaseTask:
        """The default task:
        :class:`~pyhealth.medcode.pretrained_embeddings.kg_emb.tasks.KGLinkPrediction`
        on this graph."""
        from ..tasks.kg_link_prediction import KGLinkPrediction

        return KGLinkPrediction(
            num_entities=self.num_entities, num_relations=self.num_relations
        )
