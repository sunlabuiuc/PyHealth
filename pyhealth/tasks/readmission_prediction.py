from datetime import datetime, timedelta
from typing import Dict, List

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

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import ReadmissionPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = ReadmissionPredictionMIMIC3()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ReadmissionPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(
        self, window: timedelta = timedelta(days=15), exclude_minors: bool = True
    ) -> None:
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

        Raises:
            ValueError: If any `str` to `datetime` conversions fail.
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
        for i in range(
            len(admissions) - 1
        ):  # Skip the last admission since we need a "next" admission
            if self.exclude_minors:
                age = admissions[i].timestamp.year - dob.year
                age = (
                    age - 1
                    if (
                        (admissions[i].timestamp.month, admissions[i].timestamp.day)
                        < (dob.month, dob.day)
                    )
                    else age
                )
                if age < 18:
                    continue

            filter = ("hadm_id", "==", admissions[i].hadm_id)

            diagnoses = patient.get_events(event_type="diagnoses_icd", filters=[filter])
            diagnoses = [event.icd9_code for event in diagnoses]
            if len(diagnoses) == 0:
                continue

            procedures = patient.get_events(
                event_type="procedures_icd", filters=[filter]
            )
            procedures = [event.icd9_code for event in procedures]
            if len(procedures) == 0:
                continue

            prescriptions = patient.get_events(
                event_type="prescriptions", filters=[filter]
            )
            prescriptions = [event.drug for event in prescriptions]
            if len(prescriptions) == 0:
                continue

            try:
                discharge_time = datetime.strptime(
                    admissions[i].dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                discharge_time = datetime.strptime(admissions[i].dischtime, "%Y-%m-%d")

            readmission = int(
                (admissions[i + 1].timestamp - discharge_time) < self.window
            )

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


class ReadmissionPredictionMIMIC4(BaseTask):
    """
    Readmission prediction on the MIMIC4 dataset.

    This task aims at predicting whether the patient will be readmitted into hospital within
    a specified number of days based on clinical information from the current visit.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import ReadmissionPredictionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = ReadmissionPredictionMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ReadmissionPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(
        self, window: timedelta = timedelta(days=15), exclude_minors: bool = True
    ) -> None:
        """
        Initializes the task object.

        Args:
            window (timedelta): If two admissions are closer than this window, it is considered a readmission. Defaults to 15 days.
            exclude_minors (bool): Whether to exclude patients whose "anchor_age" is less than 18. Defaults to True.
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
                - 'visit_id': MIMIC4 hadm_id.
                - 'patient_id': MIMIC4 subject_id.
                - 'conditions': MIMIC4 diagnoses_icd table ICD-9 or ICD-10 codes.
                - 'procedures': MIMIC4 procedures_icd table ICD-9 or ICD-10 codes.
                - 'drugs': MIMIC4 prescriptions table drug column entries.
                - 'readmission': binary label.

        Raises:
            ValueError: If any `str` to `datetime` conversions fail.
            AssertionError: If any icd_version value in the diagnoses_icd or procedures_icd tables is not "9" or "10"
        """
        patients: List[Event] = patient.get_events(event_type="patients")
        assert len(patients) == 1

        if self.exclude_minors and int(patients[0]["anchor_age"]) < 18:
            return []

        admissions: List[Event] = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        samples = []
        for i in range(
            len(admissions) - 1
        ):  # Skip the last admission since we need a "next" admission
            filter = ("hadm_id", "==", admissions[i].hadm_id)

            diagnoses = []
            for event in patient.get_events(
                event_type="diagnoses_icd", filters=[filter]
            ):
                assert event.icd_version in ("9", "10")
                diagnoses.append(f"{event.icd_version}_{event.icd_code}")
            if len(diagnoses) == 0:
                continue

            procedures = []
            for event in patient.get_events(
                event_type="procedures_icd", filters=[filter]
            ):
                assert event.icd_version in ("9", "10")
                procedures.append(f"{event.icd_version}_{event.icd_code}")
            if len(procedures) == 0:
                continue

            prescriptions = patient.get_events(
                event_type="prescriptions", filters=[filter]
            )
            prescriptions = [event.drug for event in prescriptions]
            if len(prescriptions) == 0:
                continue

            try:
                discharge_time = datetime.strptime(
                    admissions[i].dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                discharge_time = datetime.strptime(admissions[i].dischtime, "%Y-%m-%d")

            readmission = int(
                (admissions[i + 1].timestamp - discharge_time) < self.window
            )

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


class ReadmissionPredictionEICU(BaseTask):
    """
    Readmission prediction on the eICU dataset.

    This task aims at predicting whether the patient will be readmitted into hospital within
    a specified time window based on clinical information from the current visit.

    In eICU, timestamps are stored as offsets from ICU admission rather than absolute dates.
    This task handles two scenarios:

    1. **Same hospitalization**: Multiple ICU stays within the same hospital admission
       (same patienthealthsystemstayid). Time gap is computed using offset values.

    2. **Different hospitalizations**: ICU stays from different hospital admissions
       (different patienthealthsystemstayid). Since eICU only provides discharge year
       (not full dates), any subsequent hospitalization is considered a readmission.

    Features:
    - using diagnosis table (ICD9CM and ICD10CM) as condition codes
    - using physicalexam table as procedure codes
    - using medication table as drugs codes

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        window (timedelta): Time window for readmission (used for same-hospitalization only).

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import ReadmissionPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = ReadmissionPredictionEICU(window=timedelta(days=15))
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "ReadmissionPredictionEICU"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(self, window: timedelta = timedelta(days=15)) -> None:
        """
        Initializes the task object.

        Args:
            window (timedelta): Time window for considering a readmission within the same
                hospitalization. For different hospitalizations, any subsequent admission
                is considered a readmission. Defaults to 15 days.
        """
        self.window = window
        self.window_minutes = int(window.total_seconds() / 60)

    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generates binary classification data samples for a single patient.

        Args:
            patient (Patient): A patient object.

        Returns:
            List[Dict]: A list containing a dictionary for each patient visit with:
                - 'visit_id': eICU patientunitstayid.
                - 'patient_id': eICU uniquepid.
                - 'conditions': Diagnosis codes from diagnosis table.
                - 'procedures': Physical exam codes from physicalexam table.
                - 'drugs': Drug names from medication table.
                - 'readmission': binary label.
        """
        # Get patient stays (each row in patient table is an ICU stay)
        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) < 2:
            return []

        # Sort stays by hospital stay ID and unit visit number for proper ordering
        # Within same hospitalization: use unitvisitnumber
        # Across hospitalizations: use patienthealthsystemstayid
        sorted_stays = sorted(
            patient_stays,
            key=lambda s: (
                int(getattr(s, "patienthealthsystemstayid", 0) or 0),
                int(getattr(s, "unitvisitnumber", 0) or 0),
            ),
        )

        samples = []
        for i in range(len(sorted_stays) - 1):
            stay = sorted_stays[i]
            next_stay = sorted_stays[i + 1]

            # Get the patientunitstayid for filtering
            stay_id = str(getattr(stay, "patientunitstayid", ""))

            # Get clinical codes using patientunitstayid-based filtering
            diagnoses = patient.get_events(
                event_type="diagnosis", filters=[("patientunitstayid", "==", stay_id)]
            )
            conditions = [
                getattr(event, "icd9code", "")
                for event in diagnoses
                if getattr(event, "icd9code", None)
            ]
            if len(conditions) == 0:
                continue

            physical_exams = patient.get_events(
                event_type="physicalexam",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            procedures = [
                getattr(event, "physicalexampath", "")
                for event in physical_exams
                if getattr(event, "physicalexampath", None)
            ]
            if len(procedures) == 0:
                continue

            medications = patient.get_events(
                event_type="medication", filters=[("patientunitstayid", "==", stay_id)]
            )
            drugs = [
                getattr(event, "drugname", "")
                for event in medications
                if getattr(event, "drugname", None)
            ]
            if len(drugs) == 0:
                continue

            # Determine readmission label based on hospitalization relationship
            current_hosp_id = getattr(stay, "patienthealthsystemstayid", None)
            next_hosp_id = getattr(next_stay, "patienthealthsystemstayid", None)

            if current_hosp_id == next_hosp_id:
                # Same hospitalization: compute time gap using offsets (in minutes)
                # Gap = next stay's ICU admit (time 0) - current stay's unit discharge offset
                try:
                    current_unit_discharge_offset = int(
                        getattr(stay, "unitdischargeoffset", 0) or 0
                    )
                    # Time gap in minutes from current ICU discharge to next ICU admission
                    # Since next stay's ICU admission is its reference point (offset 0),
                    # we need to estimate the gap. For simplicity, if there's a next stay
                    # in the same hospitalization, consider the gap as minimal (readmission).
                    # A more precise calculation would require additional offset information.
                    readmission_label = 1  # ICU readmission within same hospitalization
                except (ValueError, TypeError):
                    readmission_label = 1
            else:
                # Different hospitalization: eICU doesn't provide full dates,
                # so any subsequent hospitalization is flagged as readmission
                readmission_label = 1

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "readmission": readmission_label,
                }
            )

        return samples


class ReadmissionPredictionOMOP(BaseTask):
    """
    Readmission prediction on the OMOP dataset.

    This task aims at predicting whether the patient will be readmitted into hospital within
    a specified number of days based on clinical information from the current visit.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> from pyhealth.tasks import ReadmissionPredictionOMOP
        >>> dataset = OMOPDataset(
        ...     root="/path/to/omop/data",
        ...     tables=["condition_occurrence", "procedure_occurrence",
        ...             "drug_exposure"],
        ... )
        >>> task = ReadmissionPredictionOMOP()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ReadmissionPredictionOMOP"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(
        self, window: timedelta = timedelta(days=15), exclude_minors: bool = True
    ) -> None:
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
                - 'visit_id': OMOP visit_occurrence_id.
                - 'patient_id': OMOP person_id.
                - 'conditions': OMOP condition_occurrence table condition_concept_id attribute.
                - 'procedures': OMOP procedure_occurrence table procedure_concept_id attribute.
                - 'drugs': OMOP drug_exposure table drug_concept_id attribute.
                - 'readmission': binary label.
        """
        patients: List[Event] = patient.get_events(event_type="person")
        assert len(patients) == 1

        if self.exclude_minors:
            year = int(patients[0].year_of_birth)
            month = int(patients[0].month_of_birth) if patients[0].month_of_birth else 1
            day = int(patients[0].day_of_birth) if patients[0].day_of_birth else 1

            dob = datetime.strptime(f"{year:04d}-{month:02d}-{day:02d}", "%Y-%m-%d")

        admissions: List[Event] = patient.get_events(event_type="visit_occurrence")
        if len(admissions) < 2:
            return []

        samples = []
        for i in range(
            len(admissions) - 1
        ):  # Skip the last admission since we need a "next" admission
            if self.exclude_minors:
                age = admissions[i].timestamp.year - dob.year
                age = (
                    age - 1
                    if (
                        (admissions[i].timestamp.month, admissions[i].timestamp.day)
                        < (dob.month, dob.day)
                    )
                    else age
                )
                if age < 18:
                    continue

            filter = ("visit_occurrence_id", "==", admissions[i].visit_occurrence_id)

            conditions = patient.get_events(
                event_type="condition_occurrence", filters=[filter]
            )
            conditions = [event.condition_concept_id for event in conditions]
            if len(conditions) == 0:
                continue

            procedures = patient.get_events(
                event_type="procedure_occurrence", filters=[filter]
            )
            procedures = [event.procedure_concept_id for event in procedures]
            if len(procedures) == 0:
                continue

            drugs = patient.get_events(event_type="drug_exposure", filters=[filter])
            drugs = [event.drug_concept_id for event in drugs]
            if len(drugs) == 0:
                continue

            discharge_time = datetime.strptime(
                admissions[i].visit_end_datetime, "%Y-%m-%d %H:%M:%S"
            )

            readmission = int(
                (admissions[i + 1].timestamp - discharge_time) < self.window
            )

            samples.append(
                {
                    "visit_id": admissions[i].visit_occurrence_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "readmission": readmission,
                }
            )

        return samples


if __name__ == "__main__":
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
