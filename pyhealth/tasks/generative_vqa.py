"""Generative medical visual question answering task definitions."""

from __future__ import annotations

from typing import Dict, List, Optional

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask


def _normalize_split(split_value: Optional[str]) -> str:
    if split_value is None:
        return "unspecified"
    split = str(split_value).strip().lower()
    if split in {"val", "valid", "dev"}:
        return "validation"
    if split in {"train", "training"}:
        return "train"
    if split in {"test", "testing"}:
        return "test"
    return split


class GenerativeMedicalVQA(BaseTask):
    """Task for generative medical VQA with raw text/image passthrough schema."""

    task_name: str = "GenerativeMedicalVQA"
    input_schema: Dict[str, str] = {
        "image_path": "raw",
        "question": "raw",
        "split": "raw",
        "question_id": "raw",
        "image_id": "raw",
        "dataset": "raw",
    }
    output_schema: Dict[str, str] = {"answer": "raw"}

    def __init__(self, split: Optional[str] = None) -> None:
        self.split = _normalize_split(split) if split else None

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.split is None:
            return df

        # Dataset events are prefixed by table names in BaseDataset.
        split_cols = [
            col
            for col in ("vqa_rad/split", "path_vqa/split")
            if col in df.collect_schema().names()
        ]
        if not split_cols:
            return df

        expr = None
        for col in split_cols:
            col_expr = pl.col(col).str.to_lowercase() == self.split
            expr = col_expr if expr is None else expr | col_expr
        return df.filter(expr)

    def __call__(self, patient: Patient) -> List[Dict]:
        samples: List[Dict] = []

        events = []
        for event_type in ("vqa_rad", "path_vqa"):
            events.extend(patient.get_events(event_type=event_type))

        for index, event in enumerate(events):
            event_split = _normalize_split(event["split"] if "split" in event else None)
            if self.split is not None and event_split != self.split:
                continue

            required_keys = ("path", "question", "answer")
            missing_keys = [key for key in required_keys if key not in event]
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(
                    "GenerativeMedicalVQA event is missing required key(s): "
                    f"{missing}"
                )

            question_id = (
                str(event["question_id"])
                if "question_id" in event
                else f"{patient.patient_id}_{index}"
            )
            image_id = str(event["image_id"] if "image_id" in event else patient.patient_id)
            dataset_name = str(event["dataset"] if "dataset" in event else event.event_type)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": question_id,
                    "image_path": str(event["path"]),
                    "question": str(event["question"]),
                    "answer": str(event["answer"]),
                    "split": event_split,
                    "question_id": question_id,
                    "image_id": image_id,
                    "dataset": dataset_name,
                }
            )

        return samples
