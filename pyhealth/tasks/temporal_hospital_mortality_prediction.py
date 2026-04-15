"""
Author: Simona Bernfeld
Paper: Feature Robustness in Non-stationary Health Records:
Caveats to Deployable Model Performance in Common Clinical Machine Learning Tasks
Paper link: https://proceedings.mlr.press/v106/nestor19a.html

Description:
Task definition for admission-level in-hospital mortality prediction with temporal
metadata using MIMIC-III-style structured EHR data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask


class TemporalHospitalMortalityPredictionMIMIC3(BaseTask):
    """Admission-level mortality prediction task with temporal metadata.

    This task emits diagnosis, procedure, and drug features together with a
    normalized admission year and binary mortality label.

    Args:
        min_year: Minimum year for normalization.
        max_year: Maximum year for normalization.
        require_all_modalities: If True, requires conditions, procedures, and
            drugs to all be present.
    """

    task_name: str = "TemporalHospitalMortalityPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
        "admission_year": "tensor",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(
        self,
        min_year: int = 2001,
        max_year: int = 2012,
        require_all_modalities: bool = True,
    ) -> None:
        self.min_year = min_year
        self.max_year = max_year
        self.require_all_modalities = require_all_modalities

    def _normalize_year(self, year: int) -> float:
        """Normalizes a year into [0, 1]."""
        if self.max_year <= self.min_year:
            raise ValueError("max_year must be greater than min_year.")
        year = min(max(year, self.min_year), self.max_year)
        return (year - self.min_year) / float(self.max_year - self.min_year)

    def _clean_sequence(self, sequence: Optional[List[Any]]) -> List[str]:
        """Converts a feature list to a clean string sequence."""
        if sequence is None:
            return []
        return [
            str(item).strip()
            for item in sequence
            if item is not None and str(item).strip()
        ]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generates task samples from one patient.

        Args:
            patient: Patient object containing admissions and related events.

        Returns:
            A list of admission-level samples.
        """
        samples: List[Dict[str, Any]] = []

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        for admission in admissions:
            timestamp = getattr(admission, "timestamp", None)
            hadm_id = getattr(admission, "hadm_id", None)
            if timestamp is None or hadm_id is None:
                continue

            admission_year_raw = int(timestamp.year)
            admission_year = self._normalize_year(admission_year_raw)

            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", hadm_id)],
            )

            conditions = self._clean_sequence(
                [getattr(event, "icd9_code", None) for event in diagnoses]
            )
            procedures_list = self._clean_sequence(
                [getattr(event, "icd9_code", None) for event in procedures]
            )
            drugs = self._clean_sequence(
                [
                    getattr(event, "drug", None) or getattr(event, "drug_name", None)
                    for event in prescriptions
                ]
            )

            if self.require_all_modalities:
                if len(conditions) == 0 or len(procedures_list) == 0 or len(drugs) == 0:
                    continue
            else:
                if len(conditions) == 0 and len(procedures_list) == 0 and len(drugs) == 0:
                    continue

            mortality = int(
                getattr(admission, "hospital_expire_flag", 0) in [1, "1", True]
            )

            samples.append(
                {
                    "visit_id": str(hadm_id),
                    "hadm_id": hadm_id,
                    "patient_id": str(patient.patient_id),
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "admission_year": [float(admission_year)],
                    "admission_year_raw": admission_year_raw,
                    "mortality": mortality,
                }
            )

        return samples
