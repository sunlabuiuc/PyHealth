import sys

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
from pyhealth.data import Patient, Visit


def mortality_prediction_mimic3_fn(patient: Patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in the
        next hospital visit based on the clinical information from current visit
        (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

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
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, and drug code
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
    Mortality prediction aims at predicting whether the patient will decease in the
        next hospital visit based on the clinical information from current visit
        (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

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
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, and drug code
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
    Mortality prediction aims at predicting whether the patient will decease in the
        next hospital visit based on the clinical information from current visit
        (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

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

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, and drug code
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
    Mortality prediction aims at predicting whether the patient will decease in the
        next hospital visit based on the clinical information from current visit
        (e.g., conditions and procedures).

    Process a single patient for the mortality prediction task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # labs = visit.get_code_list(table="measurement")

        # exclude: visits without condition, procedure, and drug code
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


if __name__ == "__main__":
    # from pyhealth.datasets import MIMIC3Dataset
    #
    # dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
    #     refresh_cache=False,
    # )
    # dataset.set_task(task_fn=mortality_prediction_mimic3_fn)
    # dataset.stat()
    # print(dataset.available_keys)

    # from pyhealth.datasets import MIMIC4Dataset
    #
    # dataset = MIMIC4Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    #     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    #     dev=True,
    #     code_mapping={"NDC": "ATC"},
    #     refresh_cache=False,
    # )
    # dataset.set_task(task_fn=mortality_prediction_mimic4_fn)
    # dataset.stat()
    # print(dataset.available_keys)

    # from pyhealth.datasets import eICUDataset
    #
    # dataset = eICUDataset(
    #     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
    #     tables=["diagnosis", "medication", "physicalExam"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # dataset.set_task(task_fn=mortality_prediction_eicu_fn)
    # dataset.stat()
    # print(dataset.available_keys)

    from pyhealth.datasets import OMOPDataset

    dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    dataset.set_task(task_fn=mortality_prediction_omop_fn)
    dataset.stat()
    print(dataset.available_keys)

