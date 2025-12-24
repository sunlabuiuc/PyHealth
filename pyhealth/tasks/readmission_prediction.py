from datetime import datetime, timedelta
from typing import Dict, List

import polars as pl

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

class ReadmissionPredictionMIMIC3(BaseTask):
    """
    Readmission prediction on the MIMIC3 dataset.

    This task aims at predicting whether the patient will be readmitted into hospital within
    a specified number of days based on clinical information from the current visit.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
    """
    task_name: str = "ReadmissionPredictionMIMIC3"
    input_schema: Dict[str, str] = {"conditions": "sequence", "procedures": "sequence", "drugs": "sequence"}
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(self, window: timedelta=timedelta(days=15), exclude_minors: bool=True) -> None:
        """
        Initializes the task object.

        Args:
            window (timedelta): If two admissions are closer than this window, it is considered a readmission. Defaults to 15 days.
            exclude_minors (bool): Whether to exclude visits where the patient was under 18 years old. Defaults to True.
        """
        self.window = window
        self.exclude_minors = exclude_minors

    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generates binary classification data samples for a single patient.

        Visits with no conditions OR no procedures OR no drugs are excluded from the output but are still used to calculate readmission for prior visits.

        Args:
            patient (Patient): A patient object.

        Returns:
            List[Dict]: A list containing a dictionary for each patient visit with:
                - 'visit_id': MIMIC3 hadm_id.
                - 'patient_id': MIMIC3 subject_id.
                - 'conditions': MIMIC3 diagnoses_icd table ICD-9 codes.
                - 'procedures': MIMIC3 procedures_icd table ICD-9 codes.
                - 'drugs': MIMIC3 prescriptions table drug column entries.
                - 'readmission': binary label.
        """
        patients: List[Event] = patient.get_events(event_type="patients")
        assert len(patients) == 1

        if self.exclude_minors:
            try:
                dob = datetime.strptime(patients[0].dob, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dob = datetime.strptime(patients[0].dob, "%Y-%m-%d")

        admissions: List[Event] = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        samples = []
        for i in range(len(admissions) - 1): # Skip the last admission since we need a "next" admission
            if self.exclude_minors:
                age = admissions[i].timestamp.year - dob.year
                age = age-1 if ((admissions[i].timestamp.month, admissions[i].timestamp.day) < (dob.month, dob.day)) else age
                if age < 18:
                    continue

            filter = ("hadm_id", "==", admissions[i].hadm_id)

            diagnoses = patient.get_events(event_type="diagnoses_icd", filters=[filter])
            diagnoses = [event.icd9_code for event in diagnoses]
            if len(diagnoses) == 0:
                continue

            procedures = patient.get_events(event_type="procedures_icd", filters=[filter])
            procedures = [event.icd9_code for event in procedures]
            if len(procedures) == 0:
                continue

            prescriptions = patient.get_events(event_type="prescriptions", filters=[filter])
            prescriptions = [event.drug for event in prescriptions]
            if len(prescriptions) == 0:
                continue

            try:
                discharge_time = datetime.strptime(admissions[i].dischtime, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                discharge_time = datetime.strptime(admissions[i].dischtime, "%Y-%m-%d")

            readmission = int((admissions[i + 1].timestamp - discharge_time) < self.window)

            samples.append(
                {
                    "visit_id": admissions[i].hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": diagnoses,
                    "procedures": procedures,
                    "drugs": prescriptions,
                    "readmission": readmission,
                }
            )

        return samples


def readmission_prediction_mimic4_fn(patient: Patient, time_window=15):
    """Processes a single patient for the readmission prediction task.

    Readmission prediction aims at predicting whether the patient will be readmitted
    into hospital within time_window days based on the clinical information from
    current visit (e.g., conditions and procedures).

    Args:
        patient: a Patient object
        time_window: the time window threshold (gap < time_window means label=1 for
            the task)

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import readmission_prediction_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(readmission_prediction_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 0}]
    """
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


def readmission_prediction_eicu_fn(patient: Patient, time_window=5):
    """Processes a single patient for the readmission prediction task.

    Readmission prediction aims at predicting whether the patient will be readmitted
    into hospital within time_window days based on the clinical information from
    current visit (e.g., conditions and procedures).

    Features key-value pairs:
    - using diagnosis table (ICD9CM and ICD10CM) as condition codes
    - using physicalExam table as procedure codes
    - using medication table as drugs codes

    Args:
        patient: a Patient object
        time_window: the time window threshold (gap < time_window means label=1 for
            the task)

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalExam"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import readmission_prediction_eicu_fn
        >>> eicu_sample = eicu_base.set_task(readmission_prediction_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

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
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


def readmission_prediction_eicu_fn2(patient: Patient, time_window=5):
    """Processes a single patient for the readmission prediction task.

    Readmission prediction aims at predicting whether the patient will be readmitted
    into hospital within time_window days based on the clinical information from
    current visit (e.g., conditions and procedures).

    Similar to readmission_prediction_eicu_fn, but with different code mapping:
    - using admissionDx table and diagnosisString under diagnosis table as condition codes
    - using treatment table as procedure codes

    Args:
        patient: a Patient object
        time_window: the time window threshold (gap < time_window means label=1 for
            the task)

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "treatment", "admissionDx"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import readmission_prediction_eicu_fn2
        >>> eicu_sample = eicu_base.set_task(readmission_prediction_eicu_fn2)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        admissionDx = visit.get_code_list(table="admissionDx")
        diagnosisString = list(
            set(
                [
                    dx.attr_dict["diagnosisString"]
                    for dx in visit.get_event_list("diagnosis")
                ]
            )
        )
        treatment = visit.get_code_list(table="treatment")

        # exclude: visits without treatment, admissionDx, diagnosisString
        if len(admissionDx) * len(diagnosisString) * len(treatment) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": admissionDx + diagnosisString,
                "procedures": treatment,
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


def readmission_prediction_omop_fn(patient: Patient, time_window=15):
    """Processes a single patient for the readmission prediction task.

    Readmission prediction aims at predicting whether the patient will be readmitted
    into hospital within time_window days based on the clinical information from
    current visit (e.g., conditions and procedures).

    Args:
        patient: a Patient object
        time_window: the time window threshold (gap < time_window means label=1 for
            the task)

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import readmission_prediction_omop_fn
        >>> omop_sample = omop_base.set_task(readmission_prediction_eicu_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # labs = get_code_from_list_of_event(
        #     visit.get_event_list(table="measurement")
        # )

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=readmission_prediction_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import eICUDataset

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=readmission_prediction_eicu_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "admissionDx", "treatment"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=readmission_prediction_eicu_fn2)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import OMOPDataset

    base_dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=readmission_prediction_omop_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
