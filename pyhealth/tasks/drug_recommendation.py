import sys

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
from pyhealth.data import Patient, Visit
from pyhealth.tasks.utils import get_code_from_list_of_event


def drug_recommendation_mimic3_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health history  (e.g., conditions
    and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
    to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
    from the task dataset by returning an empty list.
    """

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
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
        if (len(conditions) + len(procedures)) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    return samples


def drug_recommendation_mimic4_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health history  (e.g., conditions
    and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
    to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
    from the task dataset by returning an empty list.
    """

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
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
        if (len(conditions) + len(procedures)) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    return samples


def drug_recommendation_eicu_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health history  (e.g., conditions
    and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
    to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
    from the task dataset by returning an empty list.
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
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
        if (len(conditions) + len(procedures)) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    return samples


def drug_recommendation_omop_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health history  (e.g., conditions
    and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
    to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
    from the task dataset by returning an empty list.
    """

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = get_code_from_list_of_event(
            visit.get_event_list(event_type="condition_occurrence")
        )
        procedures = get_code_from_list_of_event(
            visit.get_event_list(event_type="procedure_occurrence")
        )
        drugs = get_code_from_list_of_event(
            visit.get_event_list(event_type="drug_exposure")
        )
        # exclude: visits without (condition and procedure) or drug code
        if (len(conditions) + len(procedures)) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    return samples


def drug_recommendation(dataset_name: str, patient: Patient):
    if dataset_name == "MIMIC-III":
        return drug_recommendation_mimic3_fn(patient)
    elif dataset_name == "MIMIC-IV":
        return drug_recommendation_mimic4_fn(patient)
    elif dataset_name == "eICU":
        return drug_recommendation_eicu_fn(patient)
    elif dataset_name == "OMOP":
        return drug_recommendation_omop_fn(patient)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC4Dataset, eICUDataset, OMOPDataset

    # mimic4dataset = MIMIC4Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    #     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    #     dev=True,
    #     code_mapping={"prescriptions": "ATC3"},
    #     refresh_cache=False,
    # )
    # mimic4dataset.stat()
    # mimic4dataset.set_task(task_fn=drug_recommendation)
    # mimic4dataset.stat()

    eicudataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    eicudataset.stat()
    eicudataset.set_task(task_fn=drug_recommendation)
    eicudataset.stat()

    # omopdataset = OMOPDataset(
    #     root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
    #     tables=[
    #         "condition_occurrence",
    #         "procedure_occurrence",
    #         "drug_exposure",
    #         "measurement",
    #     ],
    #     dev=False,
    #     refresh_cache=False,
    # )
    # omopdataset.stat()
    # omopdataset.set_task(task_fn=drug_recommendation)
    # omopdataset.stat()
