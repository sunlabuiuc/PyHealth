from typing import Dict, List
from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

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

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import ReadmissionPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = ReadmissionPredictionEICU(exclude_minors=True)
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
                - 'readmission': binary label (1 if readmitted, 0 otherwise).
        """
        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) < 2:
            return []
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
                    age_str = str(getattr(stay, "age", "0")).replace(">", "").strip()
                    if int(age_str) < 18:
                        continue
                except (ValueError, TypeError):
                    pass
            stay_id = str(getattr(stay, "patientunitstayid", ""))
            diagnoses = patient.get_events(
                event_type = "diagnosis", filters = [("patientunitstayid", "==", stay_id)]
            )
            conditions = [
                getattr(event, "icd9code", "") for event in diagnoses
                if getattr(event, "icd9code", None)
            ]
            physical_exams = patient.get_events(
                event_type = "physicalexam", filters = [("patientunitstayid", "==", stay_id)]
            )
            procedures = [
                getattr(event, "physicalexampath", "") for event in physical_exams
                if getattr(event, "physicalexampath", None)
            ]
            medications = patient.get_events(
                event_type = "medication", filters = [("patientunitstayid", "==", stay_id)]
            )
            drugs = [
                getattr(event, "drugname", "") for event in medications
                if getattr(event, "drugname", None)
            ]
            if len(conditions) == 0 and len(procedures) == 0 and len(drugs) == 0:
                continue
            current_hosp_id = getattr(stay, "patienthealthsystemstayid", None)
            next_hosp_id = getattr(next_stay, "patienthealthsystemstayid", None)
            readmission = int(current_hosp_id == next_hosp_id and current_hosp_id is not None)
            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": [conditions],
                    "procedures": [procedures],
                    "drugs": [drugs],
                    "readmission": readmission,
                }
            )

        return samples