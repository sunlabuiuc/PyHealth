from datetime import datetime
from typing import Any, Dict, List

import polars as pl

from .base_task import BaseTask


class Readmission30DaysMIMIC4(BaseTask):
    """
    Task for predicting 30-day readmission using MIMIC-IV data.

    This task processes patient data to predict whether a patient will be
    readmitted within 30 days after discharge. It uses sequences of
    conditions, procedures, and drugs as input features.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A sequence of condition codes.
            - procedures: A sequence of procedure codes.
            - drugs: A sequence of drug codes.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - readmission: A binary indicator of readmission within 30 days.
    """

    task_name: str = "Readmission30DaysMIMIC4"
    input_schema: Dict[str, str] = {"conditions": "sequence", "procedures": "sequence", "drugs": "sequence"}
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = []

        demographics = patient.get_events(event_type="patients")
        assert len(demographics) == 1
        demographics = demographics[0]
        anchor_age = int(demographics["anchor_age"])

        # exclude: patients under 18 years old
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
                readmission = 1 if time_diff_hour < 30 * 24 else 0
            else:
                readmission = 0

            # returning polars dataframe is much faster than returning list of events
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime,
                return_df=True
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime,
                return_df=True
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime,
                return_df=True
            )
            # convert to list of codes
            conditions = diagnoses_icd.select(
                pl.concat_str(["diagnoses_icd/icd_version", "diagnoses_icd/icd_code"], separator="_")
            ).to_series().to_list()
            procedures = procedures_icd.select(
                pl.concat_str(["procedures_icd/icd_version", "procedures_icd/icd_code"], separator="_")
            ).to_series().to_list()
            drugs = prescriptions.select(
                pl.concat_str(["prescriptions/drug"], separator="_")
            ).to_series().to_list()

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
