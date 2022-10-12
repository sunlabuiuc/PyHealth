from datetime import datetime

from pyhealth.data import Patient
from pyhealth.tasks.utils import (
    get_code_from_list_of_event,
)


def categorize_los(days: int):
    """Categorize length of stay into 10 categories.

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


def length_of_stay_prediction_mimic3_fn(patient: Patient):
    """
    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Process a single patient for the length-of-stay prediction task.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.
    """
    samples = []

    for visit in patient:

        conditions = get_code_from_list_of_event(
            visit.get_event_list(event_type="DIAGNOSES_ICD")
        )
        procedures = get_code_from_list_of_event(
            visit.get_event_list(event_type="PROCEDURES_ICD")
        )
        drugs = get_code_from_list_of_event(
            visit.get_event_list(event_type="PRESCRIPTIONS")
        )
        # exclude: visits without (condition and procedure) or drug code
        if len(conditions) + len(procedures) + len(drugs) == 0:
            continue

        encounter_time = datetime.strptime(visit.encounter_time, "%Y-%m-%d %H:%M:%S")
        discharge_time = datetime.strptime(visit.discharge_time, "%Y-%m-%d %H:%M:%S")
        los_days = (discharge_time - encounter_time).days
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
    """
    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Process a single patient for the length-of-stay prediction task.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.
    """
    samples = []

    for visit in patient:

        conditions = get_code_from_list_of_event(
            visit.get_event_list(event_type="diagnoses_icd")
        )
        procedures = get_code_from_list_of_event(
            visit.get_event_list(event_type="procedures_icd")
        )
        drugs = get_code_from_list_of_event(
            visit.get_event_list(event_type="prescriptions")
        )
        # exclude: visits without (condition and procedure) or drug code
        if len(conditions) + len(procedures) + len(drugs) == 0:
            continue

        encounter_time = datetime.strptime(visit.encounter_time, "%Y-%m-%d %H:%M:%S")
        discharge_time = datetime.strptime(visit.discharge_time, "%Y-%m-%d %H:%M:%S")
        los_days = (discharge_time - encounter_time).days
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
    """
    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Process a single patient for the length-of-stay prediction task.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.
    """

    samples = []
    # we will drop the last visit
    for visit in patient:

        conditions = get_code_from_list_of_event(
            visit.get_event_list(event_type="diagnosis")
        )
        procedures = get_code_from_list_of_event(
            visit.get_event_list(event_type="physicalExam")
        )
        drugs = get_code_from_list_of_event(
            visit.get_event_list(event_type="medication")
        )
        # exclude: visits without (condition and procedure) or drug code
        if len(conditions) + len(procedures) + len(drugs) == 0:
            continue

        encounter_time = datetime.strptime(visit.encounter_time, "%Y-%m-%d %H:%M:%S")
        discharge_time = datetime.strptime(visit.discharge_time, "%Y-%m-%d %H:%M:%S")
        los_days = (discharge_time - encounter_time).days
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
    """
    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Process a single patient for the length-of-stay prediction task.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.

    Note that we define the task as a multi-class classification task.
    """
    samples = []

    for visit in patient:

        conditions = get_code_from_list_of_event(
            visit.get_event_list(event_type="condition_occurrence")
        )
        procedures = get_code_from_list_of_event(
            visit.get_event_list(event_type="procedure_occurrence")
        )
        drugs = get_code_from_list_of_event(
            visit.get_event_list(event_type="drug_exposure")
        )
        # exclude: visits without (condition and procedure and drug code)
        if len(conditions) + len(procedures) + len(drugs) == 0:
            continue

        encounter_time = datetime.strptime(visit.encounter_time, "%Y-%m-%d %H:%M:%S")
        discharge_time = datetime.strptime(visit.discharge_time, "%Y-%m-%d %H:%M:%S")
        los_days = (discharge_time - encounter_time).days
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

    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"prescriptions": "ATC"},
        refresh_cache=False,
    )
    dataset.stat()
    dataset.set_task(task_fn=length_of_stay_prediction_mimic3_fn)
    dataset.stat()
