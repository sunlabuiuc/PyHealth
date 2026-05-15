from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class TemporalMortalityPredictionEICU(BaseTask):
    """Task for temporal mortality prediction using the eICU dataset.

    This task predicts whether the patient will die in the *next* hospital stay
    based on diagnoses, physical exams, and medications from the current ICU stay.

    It extends the standard eICU mortality task with temporal metadata:
    - ``discharge_year`` for coarse calendar-time grouping
    - ``stay_order`` for within-patient chronology
    - ``split_group`` (early/late) for simple temporal cohorting

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks.temporal_mortality import TemporalMortalityPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = TemporalMortalityPredictionEICU()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "TemporalMortalityPredictionEICU"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    @staticmethod
    def _normalize_year(value: Optional[Any]) -> int:
        """Converts the provided year value into an integer if possible."""
        if value is None:
            return -1
        try:
            year = int(value)
            # Basic sanity range for eICU years.
            if 1900 <= year <= 2100:
                return year
        except (TypeError, ValueError):
            pass
        return -1

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient into temporal mortality samples."""
        samples: List[Dict[str, Any]] = []

        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) <= 1:
            return []

        num_candidate_stays = len(patient_stays) - 1

        for i in range(num_candidate_stays):
            stay = patient_stays[i]
            next_stay = patient_stays[i + 1]

            discharge_status = getattr(next_stay, "hospitaldischargestatus", None)
            if discharge_status not in ["Alive", "Expired"]:
                mortality_label = 0
            else:
                mortality_label = 0 if discharge_status == "Alive" else 1

            stay_id = str(getattr(stay, "patientunitstayid", ""))

            diagnoses = patient.get_events(
                event_type="diagnosis",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            physical_exams = patient.get_events(
                event_type="physicalexam",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            medications = patient.get_events(
                event_type="medication",
                filters=[("patientunitstayid", "==", stay_id)],
            )

            conditions = [
                getattr(event, "icd9code", "")
                for event in diagnoses
                if getattr(event, "icd9code", None)
            ]
            procedures_list = [
                getattr(event, "physicalexampath", "")
                for event in physical_exams
                if getattr(event, "physicalexampath", None)
            ]
            drugs = [
                getattr(event, "drugname", "")
                for event in medications
                if getattr(event, "drugname", None)
            ]

            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            discharge_year = self._normalize_year(
                getattr(stay, "hospitaldischargeyear", None)
            )
            split_group = "early" if i < max(1, num_candidate_stays // 2) else "late"

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                    "discharge_year": discharge_year,
                    "stay_order": i,
                    "split_group": split_group,
                }
            )

        return samples