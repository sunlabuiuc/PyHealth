from datetime import datetime
from typing import Any, Dict, List

from pyhealth.data.data import Patient
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
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        samples = []

        # Get all admissions
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        # Process each admission
        for admission in admissions:
            # Get diagnosis codes using hadm_id
            diagnoses_events = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            conditions = [event.icd9_code for event in diagnoses_events]

            # Get procedure codes using hadm_id
            procedures_events = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures = [event.icd9_code for event in procedures_events]

            # Get prescriptions using hadm_id
            prescriptions_events = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            drugs = [event.ndc for event in prescriptions_events]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            # Calculate length of stay
            # admission.timestamp is the admit time (from the timestamp column)
            # admission.dischtime is the discharge time (from attributes)
            admit_time = admission.timestamp
            discharge_time = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
            los_days = (discharge_time - admit_time).days
            los_category = categorize_los(los_days)

            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


class LengthOfStayThresholdPredictionMIMIC3(BaseTask):
    """Task for predicting whether length of stay exceeded a certain number of days
    using the MIMIC-III dataset.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (e.g., conditions and procedures).

    Args:
        days: Threshold days

    Raises:
         TypeError:  if days is not an integer.
         ValueError: if days is not a positive integer.

    Attributes:
        task_name: The name of the task.
        input_schema: The schema for input data, which includes:
            - conditions: A list of condition codes.
            - procedures: A list of procedure codes.
            - drugs: A list of drug codes.
        output_schema: The schema for output data, which includes:
            - los: A binary class label for whether length of stay exceeded the given
                   number of days.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import LengthOfStayPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> task = LengthOfStayPredictionMIMIC3(3.0)
        >>> mimic3_sample = dataset.set_task(task)
    """
    task_name: str = "LengthOfStayThresholdPredictionMIMIC3"

    input_schema = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }

    output_schema: Dict[str, str] = {"los": "binary"}

    def __init__(self, days: float = 3, exclude_minors: bool = True):
        """
        Initializes the length-of-stay prediction task.

        Args:
            days: Threshold in days. LOS > days → label = 1.
            exclude_minors: Whether to exclude minor patients whose age is less than
            18. Defaults to True.
        """
        if not isinstance(days, (int, float)):
            raise TypeError("Days must be a number (int or float)")
        if days <= 0:
            raise ValueError("Days must be greater than 0")
    
        self.days = float(days)
        self.exclude_minors = exclude_minors

    def __call__(self, patient: Any) -> List[Dict]:
        """
        Generates binary length-of-stay (LOS) prediction samples for a single patient.

        Each admission is converted into one sample with a binary label indicating
        whether the length of stay exceeds a specified threshold (``self.days``).

        Visits with no conditions OR no procedures OR no drugs are excluded from
        the output.

        Args:
            patient: A patient object (expected to implement get_events())

        Returns:
            List[Dict]: A list containing a dictionary for each valid admission with:
                - 'visit_id': MIMIC3 hadm_id.
                - 'patient_id': MIMIC3 subject_id.
                - 'conditions': Diagnosis codes from the diagnoses_icd table.
                - 'procedures': Procedure codes from the procedures_icd table.
                - 'drugs': Drug codes from the prescriptions table.
                - 'los': Binary label where 1 indicates LOS > ``self.days`` and 0
                otherwise.

        Raises:
            ValueError: If date strings (e.g., date of birth or discharge time)
            cannot be parsed into datetime objects.
        """
        samples = []

        # Get all admissions
        admissions = patient.get_events(event_type = "admissions")
        if len(admissions) == 0:
            return []

        patients = patient.get_events(event_type = "patients")
        assert len(patients) == 1

        # check for minor (patients less than 18 years old) exclusion
        if self.exclude_minors:
            try:
                dob = datetime.strptime(patients[0].dob, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dob = datetime.strptime(patients[0].dob, "%Y-%m-%d")

        # Process each admission
        for admission in admissions:
            if self.exclude_minors:
                age = admission.timestamp.year - dob.year
                if (admission.timestamp.month, admission.timestamp.day) < (dob.month,
                                                                           dob.day):
                    # Patient's birthday has not yet occurred, adjust age
                    age -= 1
                if age < 18:
                    # Exclude minors
                    continue

            # Get diagnosis codes using hadm_id
            diagnoses_events = patient.get_events(
                event_type = "diagnoses_icd",
                filters = [("hadm_id", "==", admission.hadm_id)],
            )
            conditions = [event.icd9_code for event in diagnoses_events]

            # Get procedure codes using hadm_id
            procedures_events = patient.get_events(
                event_type = "procedures_icd",
                filters = [("hadm_id", "==", admission.hadm_id)],
            )
            procedures = [event.icd9_code for event in procedures_events]

            # Get prescriptions using hadm_id
            prescriptions_events = patient.get_events(
                event_type = "prescriptions",
                filters = [("hadm_id", "==", admission.hadm_id)],
            )
            drugs = [event.ndc for event in prescriptions_events]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            # Calculate length of stay
            # admission.timestamp is the admit time (from the timestamp column)
            # admission.dischtime is the discharge time (from attributes)
            admit_time = admission.timestamp
            discharge_time = datetime.strptime(admission.dischtime,
                                               "%Y-%m-%d %H:%M:%S")
            los_days = (discharge_time - admit_time).days

            # generate label
            label = int(los_days > self.days)

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "los": label,
                }
            )
        # no cohort selection
        return samples


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
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        samples = []

        # Get all admissions
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        # Process each admission
        for admission in admissions:
            # Get diagnosis codes using hadm_id
            diagnoses_events = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            # Combine icd_version and icd_code (e.g., "9_4011" or "10_I10")
            conditions = [
                f"{event.icd_version}_{event.icd_code}" for event in diagnoses_events
            ]

            # Get procedure codes using hadm_id
            procedures_events = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures = [
                f"{event.icd_version}_{event.icd_code}" for event in procedures_events
            ]

            # Get prescriptions using hadm_id
            prescriptions_events = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            drugs = [event.ndc for event in prescriptions_events]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            # Calculate length of stay
            # admission.timestamp is the admit time (from the timestamp column)
            # admission.dischtime is the discharge time (from attributes)
            admit_time = admission.timestamp
            discharge_time = datetime.strptime(admission.dischtime, "%Y-%m-%d %H:%M:%S")
            los_days = (discharge_time - admit_time).days
            los_category = categorize_los(los_days)

            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


class LengthOfStayPredictioneICU(BaseTask):
    """Task for predicting length of stay using eICU dataset.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current ICU stay based on the clinical information from the stay
    (e.g., diagnoses, physical exams, and medications).

    In eICU, timestamps are stored as minute-offsets from ICU admission.
    The ICU length of stay is computed directly from ``unitdischargeoffset``
    (minutes from ICU admission to ICU discharge).

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A list of diagnosis strings.
            - procedures: A list of physical exam values.
            - drugs: A list of drug names.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - los: A multi-class label for length of stay category.

    Note that we define the task as a multi-class classification task with 10 categories.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import LengthOfStayPredictioneICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = LengthOfStayPredictioneICU()
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "LengthOfStayPredictioneICU"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        samples = []

        # In the new BaseDataset, each row of the patient table is an ICU stay.
        # The patient table has timestamp=null in the YAML, so we use
        # get_events(event_type="patient") to iterate over ICU stays.
        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) == 0:
            return []

        for stay in patient_stays:
            stay_id = str(getattr(stay, "patientunitstayid", ""))
            if not stay_id:
                continue

            # --- Diagnoses ---
            # YAML: diagnosis table has attributes [patientunitstayid,
            #        diagnosisoffset, diagnosisstring, icd9code, diagnosispriority]
            diagnosis_events = patient.get_events(
                event_type="diagnosis",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            conditions = [
                getattr(event, "diagnosisstring", "")
                for event in diagnosis_events
                if getattr(event, "diagnosisstring", None)
            ]

            # --- Physical exams (used as "procedures") ---
            # YAML: physicalexam table has attributes [patientunitstayid,
            #        physicalexamvalue]
            physicalexam_events = patient.get_events(
                event_type="physicalexam",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            procedures = [
                getattr(event, "physicalexamvalue", "")
                for event in physicalexam_events
                if getattr(event, "physicalexamvalue", None)
            ]

            # --- Medications ---
            # YAML: medication table has attributes [patientunitstayid,
            #        drugstartoffset, drugstopoffset, drugname, ...]
            medication_events = patient.get_events(
                event_type="medication",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            drugs = [
                getattr(event, "drugname", "")
                for event in medication_events
                if getattr(event, "drugname", None)
            ]

            # Exclude stays without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            # --- Length of stay ---
            # unitdischargeoffset is the number of minutes from ICU admission
            # to ICU discharge. This directly gives us the ICU LOS.
            unit_discharge_offset = getattr(stay, "unitdischargeoffset", None)
            if unit_discharge_offset is None:
                continue

            try:
                los_minutes = int(unit_discharge_offset)
            except (ValueError, TypeError):
                continue

            los_days = los_minutes // (60 * 24)
            los_category = categorize_los(los_days)

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "los": los_category,
                }
            )
        # no cohort selection
        return samples


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
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"los": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        samples = []

        # Get all visit occurrences
        visit_occurrences = patient.get_events(event_type="visit_occurrence")
        if len(visit_occurrences) == 0:
            return []

        # Process each visit
        for visit in visit_occurrences:
            # Get condition codes
            condition_events = patient.get_events(
                event_type="condition_occurrence",
                filters=[("visit_occurrence_id", "==", visit.visit_occurrence_id)],
            )
            conditions = [event.condition_concept_id for event in condition_events]

            # Get procedure codes
            procedure_events = patient.get_events(
                event_type="procedure_occurrence",
                filters=[("visit_occurrence_id", "==", visit.visit_occurrence_id)],
            )
            procedures = [event.procedure_concept_id for event in procedure_events]

            # Get drug exposures
            drug_events = patient.get_events(
                event_type="drug_exposure",
                filters=[("visit_occurrence_id", "==", visit.visit_occurrence_id)],
            )
            drugs = [event.drug_concept_id for event in drug_events]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            # Calculate length of stay
            admit_time = datetime.strptime(
                visit.visit_start_datetime, "%Y-%m-%d %H:%M:%S"
            )
            discharge_time = datetime.strptime(
                visit.visit_end_datetime, "%Y-%m-%d %H:%M:%S"
            )
            los_days = (discharge_time - admit_time).days
            los_category = categorize_los(los_days)

            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "visit_id": visit.visit_occurrence_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "los": los_category,
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
    )
    task = LengthOfStayPredictionMIMIC3()
    sample_dataset = base_dataset.set_task(task)
    sample_dataset.stats()
    print(sample_dataset.samples[0] if sample_dataset.samples else "No samples")

    task = LengthOfStayThresholdPredictionMIMIC3(3)
    sample_dataset = base_dataset.set_task(task)
    sample_dataset.stats()
    print(sample_dataset.samples[0] if sample_dataset.samples else "No samples")

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
    )
    task = LengthOfStayPredictionMIMIC4()
    sample_dataset = base_dataset.set_task(task)
    sample_dataset.stats()
    print(sample_dataset.samples[0] if sample_dataset.samples else "No samples")

    from pyhealth.datasets import eICUDataset

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
    )
    task = LengthOfStayPredictioneICU()
    sample_dataset = base_dataset.set_task(task)
    sample_dataset.stats()
    print(sample_dataset.samples[0] if sample_dataset.samples else "No samples")

    from pyhealth.datasets import OMOPDataset

    base_dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
    )
    task = LengthOfStayPredictionOMOP()
    sample_dataset = base_dataset.set_task(task)
    sample_dataset.stats()
    print(sample_dataset.samples[0] if sample_dataset.samples else "No samples")
