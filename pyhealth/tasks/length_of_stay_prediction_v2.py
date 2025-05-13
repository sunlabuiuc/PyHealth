from typing import Any, Dict, List
from datetime import datetime
from pyhealth.data import Patient

from .base_task import BaseTask

def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9


class MIMIC4LoSPredication(BaseTask):
    """A task for predicting length of the stays.

    This task predicts length of stays prediction
    It expects a patient visit level conditions, procedures and drugs as inputs
    and return corresponding los category label.

    Attributes:
        task_name (str): The name of the task, set to
            "MIMIC4LoSPredication".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains a "conditions", "procedures", and "drugs".
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains a single key "disease" with value "los_category".
    """

    task_name: str = "MIMIC4LoSPredication"
    input_schema: Dict[str, str] = {"conditions": "sequence", "procedures": "sequence", "drugs": "sequence"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = []

        for admission in patient.get_events('admissions'):

            admission_dischtime = datetime.strptime(
                admission.dischtime, "%Y-%m-%d %H:%M:%S"
            )

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

            los_days = (admission_dischtime - admission.timestamp).days
            los_category = categorize_los(los_days)

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "label": los_category,
                }
            )

        # no cohort selection
        return samples


if __name__ == "__main__":
    from pyhealth.datasets.mimic4 import MIMIC4EHRDataset
    from pyhealth.tasks import MIMIC4LoSPredication

    mimic4_ds = MIMIC4EHRDataset(
        # Argument 1: It specifies the data folder root.
        root="/srv/local/data/physionet.org/files/mimiciv/2.0",

        # Argument 2: The users need to input a list of raw table names (e.g., DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv).
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    )

    # Set LoS prediction task for MIMIC-IV
    mimic4_sample = mimic4_ds.set_task(MIMIC4LoSPredication())

    # Check sample output
    print(mimic4_sample.samples[0])
