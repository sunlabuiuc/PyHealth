import json
from typing import Any, Dict, List, Optional, Sequence

import polars as pl

from pyhealth.data import Patient

from .base_task import BaseTask


class ECGQASingleChooseTask(BaseTask):
    """Builds single-choose ECG-QA multiclass samples from QA events.

    This task converts per-patient ECG-QA rows into sample dictionaries that
    can be consumed by PyHealth models. The generated samples use tokenized
    question text as the feature and the first answer option as the label.

    Attributes:
        task_name: Unique task cache key used by ``dataset.set_task``.
        input_schema: Feature schema consumed by processors/models.
        output_schema: Label schema consumed by processors/models.
    """

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
        """Initializes task-level filtering and label handling options.

        Args:
            question_types: Allowed ECG-QA question types. Defaults to only
                ``"single-choose"`` when not provided.
            require_single_ecg: If ``True``, keeps only QA rows tied to a
                single ECG.
            drop_none_answers: If ``True``, drops rows with label text
                equal to ``"none"`` (case-insensitive).
        """
        self.question_types = list(question_types or ["single-choose"])
        self.require_single_ecg = require_single_ecg
        self.drop_none_answers = drop_none_answers

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Applies table-level filters before patient-wise task execution.

        Args:
            df: Input lazy dataframe containing ECG-QA event rows.

        Returns:
            Filtered lazy dataframe satisfying the configured constraints.
        """
        filtered = df.filter(
            pl.col("qa/question_type").is_in(self.question_types)
        )
        if self.require_single_ecg:
            filtered = filtered.filter(pl.col("qa/n_ecgs") == 1)
        return filtered

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Converts one patient's QA rows into model-ready sample dicts.

        Args:
            patient: Patient object whose ``data_source`` contains ECG-QA rows.

        Returns:
            A list of sample dictionaries with keys ``patient_id``,
            ``visit_id``, ``record_id``, ``question``, and ``label``.
        """
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
