"""Link prediction on a :class:`BaseKGDataset`, one sample per triple."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

from ..datasets.base_kg_dataset import (
    FIELDS,
    TABLE,
    _count_incomplete,
    entity_vocabulary,
    index_triples,
    relation_vocabulary,
    triples_frame,
)

logger = logging.getLogger(__name__)

__all__ = ["KGLinkPrediction"]


class KGLinkPrediction(BaseTask):
    """Link prediction: predict the tail of ``(h, r, ?)`` and the head of ``(?, r, t)``.

    Each triple of the graph gives one sample::

        {"patient_id": ..., "triple": (h, r, t),
         "ground_truth_head": [h' : (h', r, t) in the graph],
         "ground_truth_tail": [t' : (h, r, t') in the graph]}

    The ground-truth lists cover the whole graph. They are the filter sets of
    filtered ranking evaluation (Bordes et al., 2013; Sun et al., 2019):
    when a test triple is ranked, the other known true entities are removed
    from the candidates, whatever part they belong to. Training does not use
    them; it draws its negatives from the training triples only, through the
    ``kg_triple`` processor fitted on the training part.

    Because ``set_task`` builds samples patient by patient, and each triple
    is its own patient, these graph-wide lists are computed beforehand, in
    :meth:`pre_filter`, with Polars ``group_by`` and ``join``.

    Duplicate triples are removed in :meth:`pre_filter`, keeping the first in
    file order, and their number is logged: a duplicate could otherwise land
    in both the training and the test part.

    Args:
        num_entities: Number of entities of the graph
            (``dataset.num_entities``).
        num_relations: Number of relations of the graph
            (``dataset.num_relations``).

    Examples:
        >>> task = KGLinkPrediction(num_entities=4, num_relations=2)
        >>> sorted(task.input_schema)
        ['ground_truth_head', 'ground_truth_tail', 'triple']
        >>> task.input_schema["triple"]
        ('kg_triple', {'num_entities': 4, 'num_relations': 2})
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=0)
        ... )
    """

    task_name: str = "KGLinkPrediction"

    def __init__(self, num_entities: int, num_relations: int):
        for name, value in (("num_entities", num_entities), ("num_relations", num_relations)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.input_schema = {
            "triple": (
                "kg_triple",
                {"num_entities": num_entities, "num_relations": num_relations},
            ),
            "ground_truth_head": ("kg_entity_list", {"pad_token_id": 0}),
            "ground_truth_tail": ("kg_entity_list", {"pad_token_id": 0}),
        }
        self.output_schema = {}
        super().__init__()

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Indexes the triples, removes duplicates and adds the ground truths.

        Args:
            df: The dataset's ``global_event_df``.

        Returns:
            One event per distinct triple, with the attributes ``head_id``,
            ``relation_id``, ``tail_id``, ``ground_truth_head`` and
            ``ground_truth_tail``. The frame is computed here, once:
            ``set_task`` reads it again for every batch of records, which
            would otherwise repeat the graph-wide aggregations each time.

        Raises:
            ValueError: If a triple has a missing field, or if the graph
                does not have ``num_entities`` entities and
                ``num_relations`` relations.
        """
        triples = triples_frame(df)
        incomplete = _count_incomplete(triples)
        if incomplete:
            raise ValueError(
                f"{incomplete} triple(s) have a missing head, relation or tail."
            )

        entities = entity_vocabulary(triples).collect()
        relations = relation_vocabulary(triples).collect()
        if (len(entities), len(relations)) != (self.num_entities, self.num_relations):
            raise ValueError(
                f"The task expects {self.num_entities} entities and "
                f"{self.num_relations} relations; the graph has {len(entities)} "
                f"and {len(relations)}. Build it from this dataset's "
                "num_entities and num_relations."
            )

        indexed = (
            index_triples(triples, entities.lazy(), relations.lazy())
            .with_columns(row=pl.col("patient_id").cast(pl.Int64))
            .sort("row")
            .collect()
        )
        distinct = indexed.unique(subset=list(FIELDS), keep="first", maintain_order=True)
        n_duplicates = len(indexed) - len(distinct)
        if n_duplicates:
            logger.warning(
                "Removed %d duplicate triple(s), keeping the first occurrence of "
                "each; %d distinct triples remain.",
                n_duplicates,
                len(distinct),
            )

        ground_truth_head = distinct.group_by("relation_id", "tail_id").agg(
            pl.col("head_id").sort().alias("ground_truth_head")
        )
        ground_truth_tail = distinct.group_by("head_id", "relation_id").agg(
            pl.col("tail_id").sort().alias("ground_truth_tail")
        )
        samples = (
            distinct.join(ground_truth_head, on=["relation_id", "tail_id"], how="left")
            .join(ground_truth_tail, on=["head_id", "relation_id"], how="left")
            .select(
                "patient_id",
                pl.lit(TABLE).alias("event_type"),
                pl.lit(None, dtype=pl.Datetime("ms")).alias("timestamp"),
                *(
                    pl.col(c).alias(f"{TABLE}/{c}")
                    for c in (
                        "head_id",
                        "relation_id",
                        "tail_id",
                        "ground_truth_head",
                        "ground_truth_tail",
                    )
                ),
            )
            .sort("patient_id")
        )
        return samples.lazy()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Returns the sample of the record's triple.

        Args:
            patient: One record of the frame returned by :meth:`pre_filter`.
        """
        return [
            {
                "patient_id": patient.patient_id,
                "triple": (event.head_id, event.relation_id, event.tail_id),
                "ground_truth_head": list(event.ground_truth_head),
                "ground_truth_tail": list(event.ground_truth_tail),
            }
            for event in patient.get_events(event_type=TABLE)
        ]
