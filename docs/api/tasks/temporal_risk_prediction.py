"""Temporal risk prediction task for PyHealth datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask


class TemporalMortalityMIMIC4(BaseTask):
    """Temporal mortality prediction task for MIMIC-IV style EHR data.

    This task creates one sample per admission. Each sample contains:
    1. a lightweight numeric feature vector summarizing patient history
    2. the admission year for temporal splitting
    3. a binary mortality label

    The task is designed to pair with temporal evaluation utilities that
    compare deployment-like temporal splits against random train/test splits.

    Attributes:
        task_name: Name of the task.
        input_schema: Schema for model inputs.
        output_schema: Schema for model outputs.
    """

    task_name: str = "TemporalMortalityMIMIC4"

    input_schema: Dict[str, str] = {
        "features": "tensor",
        "year": "tensor",
    }

    output_schema: Dict[str, str] = {
        "label": "binary",
    }

    def __init__(self, min_history_events: int = 1) -> None:
        """Initializes the task.

        Args:
            min_history_events: Minimum number of historical events required
                to emit a sample.
        """
        self.min_history_events = min_history_events

    def _safe_year(self, event: Any) -> Optional[int]:
        """Extracts the year from an event timestamp.

        Args:
            event: Event object with a timestamp attribute.

        Returns:
            The year if available, otherwise None.
        """
        timestamp = getattr(event, "timestamp", None)
        if timestamp is None:
            return None
        return int(timestamp.year)

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Processes one patient into temporal prediction samples.

        Args:
            patient: A PyHealth Patient object.

        Returns:
            A list of sample dictionaries.
        """
        samples: List[Dict[str, Any]] = []

        admissions = patient.get_events("admissions")
        for admission in admissions:
            year = self._safe_year(admission)
            if year is None:
                continue

            end_time = admission.timestamp

            diagnoses = patient.get_events("diagnoses_icd", end=end_time)
            procedures = patient.get_events("procedures_icd", end=end_time)
            prescriptions = patient.get_events("prescriptions", end=end_time)

            diagnosis_codes = [
                getattr(event, "icd_code", None)
                for event in diagnoses
                if getattr(event, "icd_code", None) is not None
            ]
            procedure_codes = [
                getattr(event, "icd_code", None)
                for event in procedures
                if getattr(event, "icd_code", None) is not None
            ]
            drug_names = [
                getattr(event, "drug", None)
                for event in prescriptions
                if getattr(event, "drug", None) is not None
            ]

            history_count = (
                len(diagnosis_codes) + len(procedure_codes) + len(drug_names)
            )
            if history_count < self.min_history_events:
                continue

            features = [
                float(len(set(diagnosis_codes))),
                float(len(set(procedure_codes))),
                float(len(set(drug_names))),
                float(history_count),
            ]

            label = int(
                getattr(admission, "hospital_expire_flag", "0") == "1"
            )

            samples.append(
                {
                    "features": features,
                    "year": [float(year)],
                    "label": label,
                }
            )

        return samples