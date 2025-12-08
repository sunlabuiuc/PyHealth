from datetime import datetime, timedelta
from typing import Dict, List

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


class LengthOfStayPredictionMIMIC3(BaseTask):
    """Task for predicting length of stay using MIMIC-III dataset.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A list of condition codes.
            - procedures: A list of procedure codes.
            - drugs: A list of drug codes.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - los: A multi-class label for length of stay category.

    Note that we define the task as a multi-class classification task with 10 categories.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import LengthOfStayPredictionMIMIC3
        >>> task = LengthOfStayPredictionMIMIC3()
        >>> mimic3_sample = mimic3_base.set_task(task)
        >>> mimic3_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'drugs': [['...']], 'los': 4}]
    """

    task_name: str = "LengthOfStayPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "list",
        "procedures": "list",
        "drugs": "list",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
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
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


def length_of_stay_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    This is a legacy function wrapper for backward compatibility.
    Please use LengthOfStayPredictionMIMIC3 class instead.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.
    """
    task = LengthOfStayPredictionMIMIC3()
    return task(patient)


class LengthOfStayPredictionMIMIC4(BaseTask):
    """Task for predicting length of stay using MIMIC-IV dataset.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A list of condition codes.
            - procedures: A list of procedure codes.
            - drugs: A list of drug codes.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - los: A multi-class label for length of stay category.

    Note that we define the task as a multi-class classification task with 10 categories.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import LengthOfStayPredictionMIMIC4
        >>> task = LengthOfStayPredictionMIMIC4()
        >>> mimic4_sample = mimic4_base.set_task(task)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'drugs': [['...']], 'los': 2}]
    """

    task_name: str = "LengthOfStayPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "list",
        "procedures": "list",
        "drugs": "list",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
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
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


def length_of_stay_prediction_mimic4_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    This is a legacy function wrapper for backward compatibility.
    Please use LengthOfStayPredictionMIMIC4 class instead.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.
    """
    task = LengthOfStayPredictionMIMIC4()
    return task(patient)


class LengthOfStayPredictioneICU(BaseTask):
    """Task for predicting length of stay using eICU dataset.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A list of condition codes.
            - procedures: A list of procedure codes.
            - drugs: A list of drug codes.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - los: A multi-class label for length of stay category.

    Note that we define the task as a multi-class classification task with 10 categories.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalExam"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import LengthOfStayPredictioneICU
        >>> task = LengthOfStayPredictioneICU()
        >>> eicu_sample = eicu_base.set_task(task)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'drugs': [['...']], 'los': 5}]
    """

    task_name: str = "LengthOfStayPredictioneICU"
    input_schema: Dict[str, str] = {
        "conditions": "list",
        "procedures": "list",
        "drugs": "list",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
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
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


def length_of_stay_prediction_eicu_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    This is a legacy function wrapper for backward compatibility.
    Please use LengthOfStayPredictioneICU class instead.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.
    """
    task = LengthOfStayPredictioneICU()
    return task(patient)


class LengthOfStayPredictionOMOP(BaseTask):
    """Task for predicting length of stay using OMOP dataset.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A list of condition codes.
            - procedures: A list of procedure codes.
            - drugs: A list of drug codes.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - los: A multi-class label for length of stay category.

    Note that we define the task as a multi-class classification task with 10 categories.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import LengthOfStayPredictionOMOP
        >>> task = LengthOfStayPredictionOMOP()
        >>> omop_sample = omop_base.set_task(task)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'drugs': [['...']], 'los': 7}]
    """

    task_name: str = "LengthOfStayPredictionOMOP"
    input_schema: Dict[str, str] = {
        "conditions": "list",
        "procedures": "list",
        "drugs": "list",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
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
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


def length_of_stay_prediction_omop_fn(patient: Patient):
    """Processes a single patient for the length-of-stay prediction task.

    This is a legacy function wrapper for backward compatibility.
    Please use LengthOfStayPredictionOMOP class instead.

    Args:
        patient: a Patient object.

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key.
    """
    task = LengthOfStayPredictionOMOP()
    return task(patient)


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
        refresh_cache=False,
    )
    task = LengthOfStayPredictionMIMIC3()
    sample_dataset = base_dataset.set_task(task_fn=task)
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
    task = LengthOfStayPredictionMIMIC4()
    sample_dataset = base_dataset.set_task(task_fn=task)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import eICUDataset

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    task = LengthOfStayPredictioneICU()
    sample_dataset = base_dataset.set_task(task_fn=task)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import OMOPDataset

    base_dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    task = LengthOfStayPredictionOMOP()
    sample_dataset = base_dataset.set_task(task_fn=task)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
