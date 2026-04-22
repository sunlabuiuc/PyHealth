"""Hallmarks of Cancer (HOC) text-to-text classification task."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import polars as pl

from pyhealth.data import Patient

from .base_task import BaseTask

logger = logging.getLogger(__name__)


def _parse_labels_raw(raw: str) -> List[str]:
    """Parse label field from CSV: ``##``-separated list or JSON list string."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            logger.debug("labels JSON parse failed, falling back to split: %r", raw)
    return [p for p in raw.split("##") if p.strip()]


class HallmarksOfCancerSentenceClassification(BaseTask):
    """Sentence-level hallmark prediction formatted for seq2seq T5 training.

    Each PyHealth "patient" is one sentence row. The task exposes a
    ``source_text`` prompt and a ``target_text`` string containing the hallmark
    labels joined by ``" ; "``. Raw ``labels`` are preserved in the sample for
    downstream inspection and metric computation.

    Args:
        split: Which split to keep: ``train``, ``validation``, or ``test``.
            Rows are filtered via :meth:`pre_filter` on the ``hoc/split`` column.

    Attributes:
        task_name: Fixed task name string.
        input_schema: ``{"source_text": "text"}``.
        output_schema: ``{"target_text": "text"}``.

    Examples:
        >>> from pyhealth.datasets import HallmarksOfCancerDataset
        >>> from pyhealth.tasks import HallmarksOfCancerSentenceClassification
        >>> ds = HallmarksOfCancerDataset(root="/path/to/data")
        >>> task = HallmarksOfCancerSentenceClassification(split="train")
        >>> samples = ds.set_task(task)
    """

    task_name: str = "HallmarksOfCancerSentenceClassification"

    def __init__(self, split: str = "train") -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(
                f"split must be 'train', 'validation', or 'test', got {split!r}"
            )
        self.split = split
        self.input_schema: Dict[str, str] = {"source_text": "text"}
        self.output_schema: Dict[str, str] = {"target_text": "text"}

    @staticmethod
    def labels_to_target_text(labels: List[str]) -> str:
        """Serialize a label list into a deterministic target string."""
        normalized = sorted(label.strip() for label in labels if label.strip())
        if not normalized:
            return "none"
        return " ; ".join(normalized)

    @staticmethod
    def target_text_to_labels(target_text: str) -> List[str]:
        """Parse a generated hallmark string back into label names."""
        text = (target_text or "").strip()
        if not text:
            return []
        return [label.strip() for label in text.split(";") if label.strip()]

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keep only rows for the requested train/validation/test split."""
        return df.filter(pl.col("hoc/split") == self.split)

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="hoc")
        samples: List[Dict[str, Any]] = []
        for event in events:
            text = event["text"]
            if not isinstance(text, str) or not text.strip():
                continue
            raw_labels = event["labels"]
            if not isinstance(raw_labels, str):
                raw_labels = str(raw_labels)
            labels = _parse_labels_raw(raw_labels)
            target_text = self.labels_to_target_text(labels)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": patient.patient_id,
                    "text": text,
                    "source_text": f"hoc: {text}",
                    "target_text": target_text,
                    "labels": labels,
                }
            )
        return samples
