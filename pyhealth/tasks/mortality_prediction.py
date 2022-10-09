import sys

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
from pyhealth.data import Patient, Visit
from pyhealth.tasks.utils import (
    get_code_from_list_of_event,
    datetime_string_to_datetime,
)


def mortality_prediction_mimic3_fn(patient: Patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in the next hospital visit based on the \
        clinical information from current visit (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that we define the task as a binary classification task.
    """

    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = next_visit.discharge_status

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
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction_mimic4_fn(patient: Patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in the next hospital visit based on the \
        clinical information from current visit (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object
        
    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that we define the task as a binary classification task.
    """

    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = next_visit.discharge_status

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
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction_eicu_fn(patient: Patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in the next hospital visit based on the \
        clinical information from current visit (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object
        
    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that we define the task as a binary classification task.
    """

    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in ["Alive", "Expired"]:
            mortality_label = 0
        else:
            mortality_label = 0 if next_visit.discharge_status == "Alive" else 1

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
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction_omop_fn(patient: Patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in the next hospital visit based on the \
        clinical information from current visit (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object
        
    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that we define the task as a binary classification task.
    """

    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        mortality_label = next_visit.discharge_status

        conditions = get_code_from_list_of_event(
            visit.get_event_list(event_type="condition_occurrence")
        )
        procedures = get_code_from_list_of_event(
            visit.get_event_list(event_type="procedure_occurrence")
        )
        drugs = get_code_from_list_of_event(
            visit.get_event_list(event_type="drug_exposure")
        )
        # labs = get_code_from_list_of_event(
        #     visit.get_event_list(event_type="measurement")
        # )

        # exclude: visits without (condition and procedure and drug code)
        if len(conditions) + len(procedures) + len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction(dataset_name: str, patient: Patient):
    if dataset_name == "MIMIC-III":
        return mortality_prediction_mimic3_fn(patient)
    elif dataset_name == "MIMIC-IV":
        return mortality_prediction_mimic4_fn(patient)
    elif dataset_name == "eICU":
        return mortality_prediction_eicu_fn(patient)
    elif dataset_name == "OMOP":
        return mortality_prediction_omop_fn(patient)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, eICUDataset, OMOPDataset

    # mimic3dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # mimic3dataset.stat()
    # mimic3dataset.set_task(mortality_prediction)
    # mimic3dataset.stat()

    # mimic4dataset = MIMIC4Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    #     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    #     dev=True,
    #     code_mapping={"prescriptions": "ATC3"},
    #     refresh_cache=False,
    # )
    # mimic4dataset.stat()
    # mimic4dataset.set_task(task_fn=mortality_prediction)
    # mimic4dataset.stat()

    # eicudataset = eICUDataset(
    #     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
    #     tables=["diagnosis", "medication", "physicalExam"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # eicudataset.stat()
    # eicudataset.set_task(task_fn=mortality_prediction)
    # eicudataset.stat()

    omopdataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=[
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "measurement",
        ],
        dev=False,
        refresh_cache=True,
    )
    omopdataset.stat()
    omopdataset.set_task(task_fn=mortality_prediction)
    omopdataset.stat()
