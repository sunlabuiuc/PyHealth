from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .base_task import BaseTask


class ReadmissionNDaysMIMIC4(BaseTask):
    """Task for predicting N-day readmission using MIMIC-IV data."""

    task_name: str = "ReadmissionNDaysMIMIC4"
    input_schema: Dict[str, str] = {"conditions": "sequence", "procedures": "sequence", "drugs": "sequence"}
    output_schema: Dict[str, str] = {"readmission": "binary"}
    
    def __init__(self, n_days: int = 30):
        """Initialize the ReadmissionNDaysMIMIC4 task with a specific number of days.
        
        Args:
            n_days (int): Number of days for the readmission window. Defaults to 30.
        """
        super().__init__()
        self.n_days = n_days
        self.task_name = f"Readmission{n_days}DaysMIMIC4"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the readmission prediction task.

        Readmission prediction aims at predicting whether the patient will be
        readmitted into hospital within a specified time window based on the
        clinical information from the current visit (e.g., conditions and procedures).

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a dictionary
            with patient_id, visit_id, and other task-specific attributes as keys.

        Note that we define the task as a binary classification task.

        Examples:
            >>> from pyhealth.datasets import MIMIC4Dataset
            >>> mimic4_base = MIMIC4Dataset(
            ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
            ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            ...     code_mapping={"ICD10PROC": "CCSPROC"},
            ... )
            >>> from pyhealth.tasks import ReadmissionNDaysMIMIC4
            >>> task = ReadmissionNDaysMIMIC4(n_days=15)  # 15-day readmission task
            >>> mimic4_sample = mimic4_base.set_task(task)
            >>> mimic4_sample.samples[0]
            [{'patient_id': '103', 'admission_id': '130744', 'conditions': ['ICD9_42', 'ICD9_109', ...], 'procedures': ['ICD9_1'], 'drugs': ['Acetaminophen', ...], 'readmission': 0}]
        """
        samples = []

        demographics = patient.get_events(event_type="patients")
        assert len(demographics) == 1
        demographics = demographics[0]
        anchor_age = int(demographics["anchor_age"])
        
        if anchor_age < 18:
            return samples

        admissions = patient.get_events(event_type="admissions")

        for i in range(len(admissions)):
            admission = admissions[i]
            next_admission = admissions[i + 1] if i < len(admissions) - 1 else None

            # get time difference between current visit and next visit
            admission_dischtime = datetime.strptime(
                admission.dischtime, "%Y-%m-%d %H:%M:%S"
            )
            duration_hour = (
                (admission_dischtime - admission.timestamp).total_seconds() / 3600
            )
            if duration_hour <= 12:
                continue
            if next_admission is not None:
                time_diff_hour = (
                    (next_admission.timestamp - admission_dischtime).total_seconds() / 3600
                )
                if time_diff_hour <= 3:
                    continue
                # Use the configurable n_days parameter here
                readmission = 1 if time_diff_hour < self.n_days * 24 else 0
            else:
                readmission = 0

            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime
            )
            conditions = [
                f"{event.icd_version}_{event.icd_code}" for event in diagnoses_icd
            ]
            procedures = [
                f"{event.icd_version}_{event.icd_code}" for event in procedures_icd
            ]
            drugs = [f"{event.drug}" for event in prescriptions]

            # exclude: visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "admission_id": admission.hadm_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "readmission": readmission,
                }
            )

        return samples