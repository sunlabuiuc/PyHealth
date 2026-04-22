from typing import Any, Dict, List

from .base_task import BaseTask


class SyntheticEHRGenerationTask(BaseTask):
    """Task for LLMSYN synthetic EHR generation evaluated via TSTR.

    Converts real MIMIC-III admissions into samples used to initialize
    LLMSYNModel. The model generates synthetic records per batch item and
    computes BCE loss against the real mortality label (TSTR framing).

    Each admission produces one sample: ICD-9 diagnosis codes as input,
    hospital mortality as the binary label.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import SyntheticEHRGenerationTask
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = SyntheticEHRGenerationTask()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "SyntheticEHRGeneration"
    input_schema: Dict[str, str] = {"conditions": "sequence"}
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into per-admission TSTR samples.

        Args:
            patient: A PyHealth Patient object from MIMIC3Dataset.

        Returns:
            List of sample dicts, one per admission, each with:
            patient_id, visit_id, conditions, mortality.
        """
        samples = []

        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        for admission in admissions:
            if admission.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(admission.hospital_expire_flag)

            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            conditions = [
                event.icd9_code for event in diagnoses if event.icd9_code
            ]

            if not conditions:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": admission.hadm_id,
                    "conditions": conditions,
                    "mortality": mortality_label,
                }
            )

        return samples
