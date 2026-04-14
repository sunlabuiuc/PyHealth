from typing import Any, Dict, List

import polars as pl

from pyhealth.data import Patient, Visit
from .base_task import BaseTask


class DrugRecommendationMIMIC3(BaseTask):
    """Task for drug recommendation using MIMIC-III dataset.

    Drug recommendation aims at recommending a set of drugs given the
    patient health history (e.g., conditions and procedures). This task
    creates samples with cumulative history, where each visit includes
    all previous visit information.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - conditions: Nested list of diagnosis codes (history +
              current)
            - procedures: Nested list of procedure codes (history +
              current)
            - drugs_hist: Nested list of drug codes from history (current
              visit excluded)
        output_schema (Dict[str, str]): The schema for output data:
            - drugs: List of drugs to predict for current visit
    """

    task_name: str = "DrugRecommendationMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs_hist": "nested_sequence",
    }
    output_schema: Dict[str, str] = {"drugs": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create drug recommendation samples.

        Creates one sample per visit (after first visit) with cumulative history.
        Each sample includes all previous visits' conditions, procedures, and drugs.

        Args:
            patient: Patient object with get_events method

        Returns:
            List of samples, each with patient_id, visit_id, conditions history,
            procedures history, drugs history, and target drugs
        """
        samples = []

        # Get all admissions
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            # Need at least 2 visits for history-based prediction
            return []

        # Process each admission
        for i, admission in enumerate(admissions):
            # Get diagnosis codes using hadm_id
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            conditions = (
                diagnoses_icd.select(pl.col("diagnoses_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Get procedure codes using hadm_id
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            procedures = (
                procedures_icd.select(pl.col("procedures_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Get prescriptions using hadm_id
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            drugs = (
                prescriptions.select(pl.col("prescriptions/ndc")).to_series().to_list()
            )

            # ATC 3 level (first 4 characters)
            drugs = [drug[:4] for drug in drugs if drug]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "drugs_hist": drugs,
                }
            )

        # Exclude patients with less than 2 valid visits
        if len(samples) < 2:
            return []

        # Add cumulative history for first sample
        samples[0]["conditions"] = [samples[0]["conditions"]]
        samples[0]["procedures"] = [samples[0]["procedures"]]
        samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

        # Add cumulative history for subsequent samples
        for i in range(1, len(samples)):
            samples[i]["conditions"] = samples[i - 1]["conditions"] + [
                samples[i]["conditions"]
            ]
            samples[i]["procedures"] = samples[i - 1]["procedures"] + [
                samples[i]["procedures"]
            ]
            samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
                samples[i]["drugs_hist"]
            ]

        # Remove target drug from history (set current visit drugs_hist to empty)
        for i in range(len(samples)):
            samples[i]["drugs_hist"][i] = []

        return samples


class DrugRecommendationMIMIC4(BaseTask):
    """Task for drug recommendation using MIMIC-IV dataset.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history (e.g., conditions and procedures). This task creates samples with
    cumulative history, where each visit includes all previous visit information.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - conditions: Nested list of diagnosis codes (history + current)
            - procedures: Nested list of procedure codes (history + current)
            - drugs_hist: Nested list of drug codes from history (current visit excluded)
        output_schema (Dict[str, str]): The schema for output data:
            - drugs: List of drugs to predict for current visit

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import DrugRecommendationMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = DrugRecommendationMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "DrugRecommendationMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs_hist": "nested_sequence",
    }
    output_schema: Dict[str, str] = {"drugs": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create drug recommendation samples.

        Creates one sample per visit (after first visit) with cumulative history.
        Each sample includes all previous visits' conditions, procedures, and drugs.

        Args:
            patient: Patient object with get_events method

        Returns:
            List of samples, each with patient_id, visit_id, conditions history,
            procedures history, drugs history, and target drugs
        """
        samples = []

        # Get all admissions
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            # Need at least 2 visits for history-based prediction
            return []

        # Process each admission
        for i, admission in enumerate(admissions):
            # Get diagnosis codes using hadm_id
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            conditions = (
                diagnoses_icd.select(
                    pl.concat_str(
                        ["diagnoses_icd/icd_version", "diagnoses_icd/icd_code"],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            # Get procedure codes using hadm_id
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            procedures = (
                procedures_icd.select(
                    pl.concat_str(
                        ["procedures_icd/icd_version", "procedures_icd/icd_code"],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            # Get prescriptions using hadm_id
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            drugs = (
                prescriptions.select(pl.col("prescriptions/ndc")).to_series().to_list()
            )

            # ATC 3 level (first 4 characters)
            drugs = [drug[:4] for drug in drugs if drug]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "drugs_hist": drugs,
                }
            )

        # Exclude patients with less than 2 valid visits
        if len(samples) < 2:
            return []

        # Add cumulative history for first sample
        samples[0]["conditions"] = [samples[0]["conditions"]]
        samples[0]["procedures"] = [samples[0]["procedures"]]
        samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

        # Add cumulative history for subsequent samples
        for i in range(1, len(samples)):
            samples[i]["conditions"] = samples[i - 1]["conditions"] + [
                samples[i]["conditions"]
            ]
            samples[i]["procedures"] = samples[i - 1]["procedures"] + [
                samples[i]["procedures"]
            ]
            samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
                samples[i]["drugs_hist"]
            ]

        # Remove target drug from history (set current visit drugs_hist to empty)
        for i in range(len(samples)):
            samples[i]["drugs_hist"][i] = []

        return samples


def drug_recommendation_mimic3_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key, like this
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "conditions": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of drug in visit 1, list of drug in visit 2, ..., list of drug in visit (N-1)],
                "drugs": list of drug in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(drug_recommendation_mimic3_fn)
        >>> mimic3_sample.samples[0]
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
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
                "drugs_hist": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples


def drug_recommendation_mimic4_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "conditions": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of drug in visit 1, list of drug in visit 2, ..., list of drug in visit (N-1)],
                "drugs": list of drug in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(drug_recommendation_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
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
                "drugs_hist": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples


class DrugRecommendationEICU(BaseTask):
    """Task for drug recommendation using eICU dataset.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history (e.g., conditions and procedures). This task creates samples with
    cumulative history, where each visit includes all previous visit information.

    Features key-value pairs:
    - using diagnosis table as condition codes
    - using physicalexam table as procedure codes
    - using medication table as drug codes

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - conditions: Nested list of diagnosis codes (history + current)
            - procedures: Nested list of procedure codes (history + current)
            - drugs_hist: Nested list of drug codes from history (current visit excluded)
        output_schema (Dict[str, str]): The schema for output data:
            - drugs: List of drugs to predict for current visit

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import DrugRecommendationEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = DrugRecommendationEICU()
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "DrugRecommendationEICU"
    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs_hist": "nested_sequence",
    }
    output_schema: Dict[str, str] = {"drugs": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create drug recommendation samples.

        Creates one sample per visit (after first visit) with cumulative history.
        Each sample includes all previous visits' conditions, procedures, and drugs.

        Args:
            patient: Patient object with get_events method

        Returns:
            List of samples, each with patient_id, visit_id, conditions history,
            procedures history, drugs history, and target drugs
        """
        samples = []

        # Get all patient stays (each row in patient table is an ICU stay)
        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) < 2:
            # Need at least 2 visits for history-based prediction
            return []

        # Process each patient stay
        for stay in patient_stays:
            # Get the patientunitstayid for filtering
            stay_id = str(getattr(stay, "patientunitstayid", ""))

            # Get diagnosis codes using patientunitstayid-based filtering
            diagnoses = patient.get_events(
                event_type="diagnosis",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            conditions = [
                getattr(event, "icd9code", "") for event in diagnoses
                if getattr(event, "icd9code", None)
            ]

            # Get physical exam codes
            physical_exams = patient.get_events(
                event_type="physicalexam",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            procedures = [
                getattr(event, "physicalexampath", "") for event in physical_exams
                if getattr(event, "physicalexampath", None)
            ]

            # Get medication codes
            medications = patient.get_events(
                event_type="medication",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            drugs = [
                getattr(event, "drugname", "") for event in medications
                if getattr(event, "drugname", None)
            ]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "drugs_hist": drugs,
                }
            )

        # Exclude patients with less than 2 valid visits
        if len(samples) < 2:
            return []

        # Add cumulative history for first sample
        samples[0]["conditions"] = [samples[0]["conditions"]]
        samples[0]["procedures"] = [samples[0]["procedures"]]
        samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

        # Add cumulative history for subsequent samples
        for i in range(1, len(samples)):
            samples[i]["conditions"] = samples[i - 1]["conditions"] + [
                samples[i]["conditions"]
            ]
            samples[i]["procedures"] = samples[i - 1]["procedures"] + [
                samples[i]["procedures"]
            ]
            samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
                samples[i]["drugs_hist"]
            ]

        # Remove target drug from history (set current visit drugs_hist to empty)
        for i in range(len(samples)):
            samples[i]["drugs_hist"][i] = []

        return samples


def drug_recommendation_omop_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_omop_fn
        >>> omop_sample = omop_base.set_task(drug_recommendation_omop_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51'], ['98', '663', '58', '51']], 'procedures': [['1'], ['2', '3']], 'label': [['2', '3', '4'], ['0', '1', '4', '5']]}]
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
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


if __name__ == "__main__":
    # from pyhealth.datasets import MIMIC3Dataset
    # base_dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"ICD9CM": "CCSCM"},
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic3_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)
    # print(sample_dataset.samples[0])

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
    print(sample_dataset.samples[0])

    # from pyhealth.datasets import eICUDataset
    # from pyhealth.tasks import DrugRecommendationEICU

    # base_dataset = eICUDataset(
    #     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
    #     tables=["diagnosis", "medication", "physicalExam"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # task = DrugRecommendationEICU()
    # sample_dataset = base_dataset.set_task(task)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)

    # from pyhealth.datasets import OMOPDataset

    # base_dataset = OMOPDataset(
    #     root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
    #     tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_omop_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)
