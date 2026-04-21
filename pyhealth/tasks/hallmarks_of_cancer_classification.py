"""Hallmarks of Cancer (HOC) sentence-level multi-label classification task."""

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
    """Multi-label sentence classification on the Hallmarks of Cancer corpus.

    Each PyHealth "patient" is one sentence row. Inputs are sentence strings;
    outputs are lists of hallmark class names (including ``none`` when no
    hallmark applies), matching the BigBio ``hallmarks_of_cancer_bigbio_text``
    schema.

    Args:
        split: Which split to keep: ``train``, ``validation``, or ``test``.
            Rows are filtered via :meth:`pre_filter` on the ``hoc/split`` column.

    Attributes:
        task_name: Fixed task name string.
        input_schema: ``{"text": "text"}``.
        output_schema: ``{"labels": "multilabel"}``.

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
        self.input_schema: Dict[str, str] = {"text": "text"}
        self.output_schema: Dict[str, str] = {"labels": "multilabel"}

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
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": patient.patient_id,
                    "text": text,
                    "labels": labels,
                }
            )
        return samples
