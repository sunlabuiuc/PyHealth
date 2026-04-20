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
        self, window: timedelta = timedelta(days=15), exclude_minors: bool = True, **kwargs
    ) -> None:
        """Initializes the task object.

        Args:
            window: If two admissions are closer than this window, it is
                considered a readmission. Defaults to 15 days.
            exclude_minors: Whether to exclude visits where the patient
                was under 18 years old. Defaults to True.
            **kwargs: Passed to :class:`~pyhealth.tasks.BaseTask`, e.g.
                ``code_mapping``.
        """
        super().__init__(**kwargs)
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
                - 'drugs': MIMIC3 prescriptions table NDC (National Drug Code) entries.
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
        # Skip the last admission since we need a "next" admission
        for i in range(len(admissions) - 1):
            if self.exclude_minors:
                age = admissions[i].timestamp.year - dob.year
                if (admissions[i].timestamp.month, admissions[i].timestamp.day) < (dob.month, dob.day):
                    age -= 1
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
            prescriptions = [event.ndc for event in prescriptions if event.ndc]
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
        self, window: timedelta = timedelta(days=15), exclude_minors: bool = True, **kwargs
    ) -> None:
        """Initializes the task object.

        Args:
            window: If two admissions are closer than this window, it is
                considered a readmission. Defaults to 15 days.
            exclude_minors: Whether to exclude patients whose
                ``anchor_age`` is less than 18. Defaults to True.
            **kwargs: Passed to :class:`~pyhealth.tasks.BaseTask`, e.g.
                ``code_mapping``.
        """
        super().__init__(**kwargs)
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
                - 'drugs': MIMIC4 prescriptions table NDC (National Drug Code) entries.
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
        # Skip the last admission since we need a "next" admission
        for i in range(len(admissions) - 1):
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
            prescriptions = [event.ndc for event in prescriptions if event.ndc]
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


class ReadmissionPredictionEICU(BaseTask):
    """
    Readmission prediction on the eICU dataset.

    This task aims at predicting whether the patient will be readmitted into the ICU
    during the same hospital stay based on clinical information from the current ICU
    visit.

    Features:
    - using diagnosis table (ICD9CM and ICD10CM) as condition codes
    - using physicalexam table as procedure codes
    - using medication table as drugs codes

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        exclude_minors (bool): Whether to exclude patients whose age is less than 18.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import ReadmissionPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = ReadmissionPredictionEICU()
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "ReadmissionPredictionEICU"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(self, exclude_minors: bool = True, **kwargs) -> None:
        """Initializes the task object.

        Args:
            exclude_minors: Whether to exclude patients whose age is
                less than 18. Defaults to True.
            **kwargs: Passed to :class:`~pyhealth.tasks.BaseTask`, e.g.
                ``code_mapping``.
        """
        super().__init__(**kwargs)
        self.exclude_minors = exclude_minors

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

            if self.exclude_minors:
                try:
                    if int(stay.age) < 18:
                        continue
                except (ValueError, TypeError):
                    pass

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

            # If the current and next hospital ID are the same, the patient was readmitted to the ICU
            current_hosp_id = getattr(stay, "patienthealthsystemstayid", None)
            next_hosp_id = getattr(next_stay, "patienthealthsystemstayid", None)
            readmission = int(current_hosp_id == next_hosp_id)

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "readmission": readmission,
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
        self, window: timedelta = timedelta(days=15), exclude_minors: bool = True, **kwargs
    ) -> None:
        """Initializes the task object.

        Args:
            window: If two admissions are closer than this window, it is
                considered a readmission. Defaults to 15 days.
            exclude_minors: Whether to exclude visits where the patient
                was under 18 years old. Defaults to True.
            **kwargs: Passed to :class:`~pyhealth.tasks.BaseTask`, e.g.
                ``code_mapping``.
        """
        super().__init__(**kwargs)
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
        # Skip the last admission since we need a "next" admission
        for i in range(len(admissions) - 1):
            if self.exclude_minors:
                age = admissions[i].timestamp.year - dob.year
                if (admissions[i].timestamp.month, admissions[i].timestamp.day) < (dob.month, dob.day):
                    age -= 1
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
