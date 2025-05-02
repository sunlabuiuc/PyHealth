from pyhealth.data import Patient
from typing import Any, Dict, List
from .base_task import BaseTask
from datetime import datetime


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


class LengthOfStayGreaterThanXPredictionMIMIC3(BaseTask):
    """Processes a single patient for the length-of-stay prediction task.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import LengthOfStayPredictionMIMIC3
        >>> mimic3_sample = mimic3_base.set_task(LengthOfStayPredictionMIMIC3())
        >>> mimic3_sample.samples[0]
        {
            'visit_id': '142289',
            'patient_id': '40189',
            'conditions': tensor([ 1,  2,  3,  4,  5]),
            'procedures': tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            'drugs': tensor([ 1,  2,  3,  4,  5]),
            'label': tensor([1.])
        }
    """

    task_name: str = "LengthOfStayGreaterThanXPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, threshold: int):
        super().__init__()
        if not isinstance(threshold, int):
            raise ValueError("Threshold must be an integer")
        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0")
        self.threshold = threshold

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        samples = []

        for visit in patient.get_events(event_type="admissions"):
            try:
                # Check the type and convert if necessary
                if isinstance(visit.dischtime, str):
                    discharge_time = datetime.strptime(visit.dischtime, "%Y-%m-%d")
                else:
                    discharge_time = visit.dischtime
            except (ValueError, AttributeError):
                # If conversion fails, skip this visit
                print("Error parsing discharge time:", visit.dischtime)
                continue

            # Get clinical codes
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                start=visit.timestamp,
                end=discharge_time,  # Now using a datetime object
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                start=visit.timestamp,
                end=discharge_time,  # Now using a datetime object
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=visit.timestamp,
                end=discharge_time,  # Now using a datetime object
            )

            conditions = [event.icd9_code for event in diagnoses]
            procedures_list = [event.icd9_code for event in procedures]
            drugs = [event.drug for event in prescriptions]

            # exclude: visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            los_days = (discharge_time - visit.timestamp).days
            los_category = categorize_los(los_days)
            los_category = int(los_category > self.threshold)

            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "visit_id": visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "label": los_category,
                }
            )
        # no cohort selection
        return samples


def length_of_stay_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import length_of_stay_prediction_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(length_of_stay_prediction_mimic3_fn)
        >>> mimic3_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 4}]
    """
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
    # no cohort selection
    return samples


def length_of_stay_prediction_mimic4_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import length_of_stay_prediction_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(length_of_stay_prediction_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 2}]
    """
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
    # no cohort selection
    return samples


def length_of_stay_prediction_eicu_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import length_of_stay_prediction_eicu_fn
        >>> eicu_sample = eicu_base.set_task(length_of_stay_prediction_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 5}]
    """
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
    # no cohort selection
    return samples


def length_of_stay_prediction_omop_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import length_of_stay_prediction_omop_fn
        >>> omop_sample = omop_base.set_task(length_of_stay_prediction_eicu_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 7}]
    """
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category,
            }
        )
        # no cohort selection
    return samples


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=length_of_stay_prediction_mimic3_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=length_of_stay_prediction_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import eICUDataset

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=length_of_stay_prediction_eicu_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import OMOPDataset

    base_dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=length_of_stay_prediction_omop_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
