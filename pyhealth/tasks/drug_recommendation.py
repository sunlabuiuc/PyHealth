import sys

from pyhealth.data import Patient, Visit

# TODO: remove this hack later
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")


def drug_recommendation_mimic3_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health
        history  (e.g., conditions and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with
        three visits may be converted to three samples ([visit 1], [visit 1, visit 2],
        [visit 1, visit 2, visit 3]). Patients can also be excluded from the task
        dataset by returning an empty list.
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": drugs,
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
        samples[i]["conditions"] = \
            samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = \
            samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    for i in range(len(samples)):
        samples[i]["drugs"][i] = []

    return samples


def drug_recommendation_mimic4_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health
        history  (e.g., conditions and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with
        three visits may be converted to three samples ([visit 1], [visit 1, visit 2],
        [visit 1, visit 2, visit 3]). Patients can also be excluded from the task
        dataset by returning an empty list.
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": drugs,
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
        samples[i]["conditions"] = \
            samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = \
            samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    for i in range(len(samples)):
        samples[i]["drugs"][i] = []

    return samples


def drug_recommendation_eicu_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health
        history  (e.g., conditions and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with
        three visits may be converted to three samples ([visit 1], [visit 1, visit 2],
        [visit 1, visit 2, visit 3]). Patients can also be excluded from the task
        dataset by returning an empty list.
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": drugs,
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
        samples[i]["conditions"] = \
            samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = \
            samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    for i in range(len(samples)):
        samples[i]["drugs"][i] = []

    return samples


def drug_recommendation_omop_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health
        history  (e.g., conditions and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with
        three visits may be converted to three samples ([visit 1], [visit 1, visit 2],
        [visit 1, visit 2, visit 3]). Patients can also be excluded from the task
        dataset by returning an empty list.
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": drugs,
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
        samples[i]["conditions"] = \
            samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = \
            samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    for i in range(len(samples)):
        samples[i]["drugs"][i] = []

    return samples


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
        refresh_cache=False,
    )
    dataset.set_task(task_fn=drug_recommendation_mimic3_fn)
    dataset.stat()
    print(dataset.available_keys)

    # from pyhealth.datasets import MIMIC4Dataset
    #
    # dataset = MIMIC4Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    #     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    #     dev=True,
    #     code_mapping={"NDC": "ATC"},
    #     refresh_cache=False,
    # )
    # dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
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
    # dataset.set_task(task_fn=drug_recommendation_eicu_fn)
    # dataset.stat()
    # print(dataset.available_keys)

    # from pyhealth.datasets import OMOPDataset
    #
    # dataset = OMOPDataset(
    #     root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
    #     tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # dataset.set_task(task_fn=drug_recommendation_omop_fn)
    # dataset.stat()
    # print(dataset.available_keys)
