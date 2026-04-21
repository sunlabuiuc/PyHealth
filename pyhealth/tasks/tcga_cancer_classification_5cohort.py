"""5-cohort TCGA cancer-type classification task for BulkRNABert embeddings.

This task pairs a pre-computed :class:`~pyhealth.models.BulkRNABert` encoder
output (shape ``(embed_dim,)``) with an integer cancer-type label in
``{0, 1, 2, 3, 4}`` corresponding to the five cohorts
(BLCA, BRCA, GBM+LGG, LUAD, UCEC) used in the reference experiments.

The :meth:`TCGACancerClassification5Cohort.__call__` hook expects a
:class:`~pyhealth.data.Patient` whose ``rnaseq_embedding`` event carries
``cohort`` (string like ``"TCGA-BLCA"``) and ``embedding_json`` (JSON-
encoded list of floats), which is the format produced by
:class:`~pyhealth.datasets.TCGARNASeqEmbeddingDataset`. For the shortcut
factory :func:`~pyhealth.datasets.load_tcga_cancer_classification_5cohort`
this hook is not consulted because samples are assembled up-front; the
schemas here still apply and are read by
:class:`~pyhealth.datasets.SampleBuilder`.

Author: Yohei Shibata (NetID: yoheis2)
Paper: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
       (Gelard et al., PMLR 259, 2025)
Paper link: https://proceedings.mlr.press/v259/gelard25a.html
Description: BaseTask subclass mapping ``rnaseq_embedding`` events to integer
    cancer-type labels in ``{0..4}`` for 5 TCGA cohorts (BLCA, BRCA, GBM+LGG,
    LUAD, UCEC).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np

from .base_task import BaseTask

logger = logging.getLogger(__name__)


LABEL_MAP: Dict[str, int] = {
    "TCGA-BLCA": 0,
    "TCGA-BRCA": 1,
    "TCGA-GBM": 2,
    "TCGA-LGG": 2,
    "TCGA-LUAD": 3,
    "TCGA-UCEC": 4,
}

COHORT_NAMES: List[str] = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]


class TCGACancerClassification5Cohort(BaseTask):
    """PyHealth task for 5-way TCGA cancer-type classification.

    Attributes:
        task_name: Identifier string used by
            :class:`~pyhealth.datasets.SampleDataset`.
        input_schema: ``{"embedding": "tensor"}`` — a pre-computed
            BulkRNABert encoder output of shape ``(embed_dim,)``.
        output_schema: ``{"label": "multiclass"}`` — integer label in
            ``[0, 5)``.
    """

    task_name: str = "TCGACancerClassification5Cohort"
    input_schema: Dict[str, str] = {"embedding": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extract an ``(embedding, label)`` sample for one TCGA patient.

        Reads the single ``rnaseq_embedding`` event emitted by
        :class:`~pyhealth.datasets.TCGARNASeqEmbeddingDataset` and converts
        it to the schema declared on this task. Patients whose cohort tag
        is not in :data:`LABEL_MAP` or who lack the event (e.g. filtered
        out upstream) contribute zero samples.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with a
                ``rnaseq_embedding`` event carrying ``cohort`` and
                ``embedding_json`` attributes.

        Returns:
            Either an empty list (patient not in the 5-cohort set) or a
            single-entry list with the ``patient_id`` / ``embedding`` /
            ``label`` sample dict.
        """
        events = patient.get_events(event_type="rnaseq_embedding")
        if not events:
            return []
        if len(events) > 1:
            logger.warning(
                "Patient %s has %d rnaseq_embedding events; using the first.",
                patient.patient_id,
                len(events),
            )
        event = events[0]

        cohort = getattr(event, "cohort", None)
        if cohort not in LABEL_MAP:
            return []

        raw = getattr(event, "embedding_json", None)
        if raw is None:
            return []
        try:
            embedding = np.asarray(json.loads(raw), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Patient %s has malformed embedding_json (%s); skipping.",
                patient.patient_id,
                exc,
            )
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "embedding": embedding,
                "label": LABEL_MAP[cohort],
            }
        ]


__all__ = [
    "TCGACancerClassification5Cohort",
    "LABEL_MAP",
    "COHORT_NAMES",
]
