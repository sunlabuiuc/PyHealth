"""Text-to-text disease recognition task for the NCBI Disease corpus."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Tuple

from ..data import Patient
from .base_task import BaseTask


class NCBIDiseaseRecognition(BaseTask):
    """Document-level disease recognition task formatted for seq2seq T5 training.

    The task exposes a ``source_text`` prompt and a ``target_text`` BIO-tag
    sequence while preserving raw entity spans and concept identifiers in the
    sample for inspection and evaluation.

    Args:
        text_source: Which field to expose as the main text input.
            Supported values are ``"full_text"``, ``"title"``, and
            ``"abstract"``.

    Examples:
        >>> from pyhealth.datasets import NCBIDiseaseDataset
        >>> from pyhealth.tasks import NCBIDiseaseRecognition
        >>> dataset = NCBIDiseaseDataset(root="/path/to/ncbi_disease")
        >>> samples = dataset.set_task(NCBIDiseaseRecognition())
        >>> sample = samples[0]
        >>> tokens, tags = NCBIDiseaseRecognition.entities_to_bio_tags(
        ...     sample["text"], sample["entities"]
        ... )
    """

    task_name: str = "NCBIDiseaseRecognition"
    input_schema: Dict[str, str] = {"source_text": "text"}
    output_schema: Dict[str, str] = {"target_text": "text"}

    def __init__(
        self,
        text_source: Literal["full_text", "title", "abstract"] = "full_text",
    ) -> None:
        if text_source not in {"full_text", "title", "abstract"}:
            raise ValueError(
                "text_source must be one of 'full_text', 'title', or 'abstract'"
            )
        self.text_source = text_source

    @staticmethod
    def _load_entities(mentions_json: str) -> List[Dict[str, Any]]:
        if not mentions_json:
            return []
        entities = json.loads(mentions_json)
        return [
            {
                "text": entity["text"],
                "type": entity["type"],
                "concept_id": entity["concept_id"],
                "start": int(entity["start"]),
                "end": int(entity["end"]),
            }
            for entity in entities
        ]

    def _select_text_and_entities(
        self,
        title: str,
        abstract: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if self.text_source == "title":
            selected_text = title
            start_offset = 0
            end_offset = len(title)
        elif self.text_source == "abstract":
            selected_text = abstract
            start_offset = len(title) + 1 if title and abstract else 0
            end_offset = start_offset + len(abstract)
        else:
            selected_text = title if not abstract else f"{title} {abstract}"
            return selected_text, entities

        selected_entities: List[Dict[str, Any]] = []
        for entity in entities:
            if start_offset <= entity["start"] and entity["end"] <= end_offset:
                shifted = dict(entity)
                shifted["start"] = entity["start"] - start_offset
                shifted["end"] = entity["end"] - start_offset
                selected_entities.append(shifted)

        return selected_text, selected_entities

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Convert one NCBI Disease document into a PyHealth sample."""
        events = patient.get_events(event_type="documents")
        if len(events) == 0:
            return []

        event = events[0]
        title = str(getattr(event, "title", "") or "")
        abstract = str(getattr(event, "abstract", "") or "")
        entities = self._load_entities(str(getattr(event, "mentions_json", "") or ""))
        text, entities = self._select_text_and_entities(title, abstract, entities)
        _, bio_tags = self.entities_to_bio_tags(text, entities)

        concept_ids = sorted(
            {
                entity["concept_id"]
                for entity in entities
                if entity.get("concept_id") not in {None, "", "-1"}
            }
        )

        return [
            {
                "patient_id": patient.patient_id,
                "record_id": patient.patient_id,
                "document_id": str(getattr(event, "doc_id", "")),
                "split": str(getattr(event, "split", "")),
                "text": text,
                "source_text": f"ncbi disease: {text}",
                "target_text": " ".join(bio_tags),
                "title": title,
                "abstract": abstract,
                "entities": entities,
                "concept_ids": concept_ids,
            }
        ]

    @staticmethod
    def entities_to_bio_tags(
        text: str, entities: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Convert character spans into whitespace-token BIO tags.

        Args:
            text: Source text.
            entities: Entity dictionaries with ``start`` and ``end`` offsets.

        Returns:
            A tuple of ``(tokens, tags)``.
        """
        tokens: List[str] = []
        tags: List[str] = []
        entity_spans = sorted(
            [(int(entity["start"]), int(entity["end"])) for entity in entities]
        )

        for match in re.finditer(r"\S+", text):
            token = match.group(0)
            token_start = match.start()
            token_end = match.end()

            tag = "O"
            for entity_start, entity_end in entity_spans:
                if entity_start <= token_start and token_end <= entity_end:
                    tag = "B-Disease" if token_start == entity_start else "I-Disease"
                    break

            tokens.append(token)
            tags.append(tag)

        return tokens, tags
