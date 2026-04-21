import json
from typing import Any, Dict, List, Optional, Sequence

import polars as pl

from .base_task import BaseTask


class ECGQASingleChooseTask(BaseTask):
    """Single-choose ECG-QA classification task."""

    # changed on purpose so PyHealth does not reuse the old cached task artifacts
    task_name: str = "ecgqa_single_choose_seq"

    input_schema: Dict[str, str] = {
        "question": "sequence",
    }

    output_schema: Dict[str, str] = {
        "label": "multiclass",
    }

    def __init__(
        self,
        question_types: Optional[Sequence[str]] = None,
        require_single_ecg: bool = True,
        drop_none_answers: bool = False,
    ) -> None:
        self.question_types = list(question_types or ["single-choose"])
        self.require_single_ecg = require_single_ecg
        self.drop_none_answers = drop_none_answers

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        filtered = df.filter(
            pl.col("qa/question_type").is_in(self.question_types)
        )
        if self.require_single_ecg:
            filtered = filtered.filter(pl.col("qa/n_ecgs") == 1)
        return filtered

    def __call__(self, patient) -> List[Dict[str, Any]]:
        rows = patient.data_source.to_dicts()
        samples: List[Dict[str, Any]] = []

        for row in rows:
            answers = json.loads(row["qa/answer_json"])
            label = answers[0] if len(answers) > 0 else ""

            if label == "":
                continue
            if self.drop_none_answers and label.strip().lower() == "none":
                continue

            samples.append(
                {
                    "patient_id": row["patient_id"],
                    "visit_id": str(row["qa/sample_id"]),
                    "record_id": int(row["qa/sample_id"]),
                    "question": row["qa/question"].lower().split(),
                    "label": label,
                }
            )

        return samples