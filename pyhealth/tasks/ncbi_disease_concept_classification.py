"""Concept classification task for the NCBI Disease corpus."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ..data import Patient
from .base_task import BaseTask
from .ncbi_disease_recognition import NCBIDiseaseRecognition


class NCBIDiseaseConceptClassification(BaseTask):
    """Document-level multilabel concept classification for NCBI Disease.

    This task maps each document to the set of normalized disease concept IDs
    annotated in the selected text field. It is designed for PyHealth
    classification models such as :class:`~pyhealth.models.T5Classifier`.

    Args:
        split: Optional corpus split filter. Use ``"train"``, ``"dev"``,
            ``"test"``, or ``None`` to keep all documents.
        text_source: Which field to expose as the model input. Supported values
            are ``"full_text"``, ``"title"``, and ``"abstract"``.

    Examples:
        >>> from pyhealth.datasets import NCBIDiseaseDataset
        >>> from pyhealth.tasks import NCBIDiseaseConceptClassification
        >>> dataset = NCBIDiseaseDataset(root="/path/to/ncbi_disease")
        >>> samples = dataset.set_task(
        ...     NCBIDiseaseConceptClassification(split="train")
        ... )
        >>> sample = samples[0]
        >>> sample["text"]
        >>> sample["concept_ids"]
    """

    task_name: str = "NCBIDiseaseConceptClassification"
    input_schema: Dict[str, str] = {"text": "text"}
    output_schema: Dict[str, str] = {"concept_ids": "multilabel"}

    def __init__(
        self,
        split: Optional[Literal["train", "dev", "test"]] = None,
        text_source: Literal["full_text", "title", "abstract"] = "full_text",
    ) -> None:
        if split not in {None, "train", "dev", "test"}:
            raise ValueError("split must be one of None, 'train', 'dev', or 'test'")
        self.split = split
        self.text_source = text_source
        self._recognition_task = NCBIDiseaseRecognition(text_source=text_source)

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="documents")
        if len(events) == 0:
            return []

        event = events[0]
        event_split = str(getattr(event, "split", "") or "")
        if self.split is not None and event_split != self.split:
            return []

        title = str(getattr(event, "title", "") or "")
        abstract = str(getattr(event, "abstract", "") or "")
        entities = NCBIDiseaseRecognition._load_entities(
            str(getattr(event, "mentions_json", "") or "")
        )
        text, entities = self._recognition_task._select_text_and_entities(
            title, abstract, entities
        )

        concept_ids = sorted(
            {
                entity["concept_id"]
                for entity in entities
                if entity.get("concept_id") not in {None, "", "-1"}
            }
        )

        if not text.strip():
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "record_id": patient.patient_id,
                "document_id": str(getattr(event, "doc_id", "")),
                "split": event_split,
                "text": text,
                "title": title,
                "abstract": abstract,
                "concept_ids": concept_ids,
            }
        ]
