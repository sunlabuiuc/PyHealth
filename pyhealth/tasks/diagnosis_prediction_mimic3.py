from typing import List, Dict, Any
from pyhealth.tasks import BaseTask


class DiagnosisPredictionMIMIC3(BaseTask):
    """Diagnosis prediction task for MIMIC-III.

    This task combines patient admissions and diagnosis records into samples
    for multilabel diagnosis prediction. Each sample consists of:
        - patient_id: the id of patient
        - visit_id: the id of each visit for that patient
        - prev_diag: sequence of diagnosis of each visits
                    (each visit is a list of code indices)
        - label: set of diagnosis codes (multilabel)

    The prediction target is the diagnoses in the last visit,
    given all previous visits.

    Example:
    >>> from pyhealth.datasets import MIMIC3Dataset
    ... from pyhealth.tasks.diagnosis_prediction_mimic3 import DiagnosisPredictionMIMIC3
    >>> dataset = MIMIC3Dataset(
    ...             root="/path/to/mimic-iii/1.4",
    ...             tables=["DIAGNOSES_ICD"],
    ... )
    >>> task = DiagnosisPredictionMIMIC3()
    >>> dataset = dataset.set_task(task)
    >>> print(dataset[0].keys())
    dict_keys(['patient_id', 'visit_id', 'prev_diag', 'label'])
    """

    def __init__(self):
        super().__init__()

        self.task_name = "diagnosis_prediction_mimic3"
        self.input_schema = {"prev_diag": "nested_sequence"}
        self.output_schema = {"label": "multilabel"}

    def __call__(self, patient: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prediction samples for a single patient.

        Args:
            patient: A PyHealth Patient object containing
                    admissions and diagnosis events.

        Returns:
            List of sample dictionaries, each with:
                - "prev_diag":
                    List[List[str]] - past visits (list of diagnosis code lists)
                - "label": List[str] - diagnosis codes of the next visit
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []
        admissions = sorted(admissions, key=lambda e: e.timestamp)

        # Extract diagnoses per visit
        visits = []
        for admission in admissions:
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            codes = [getattr(event, "icd9_code") for event in diagnoses]
            if codes:
                visits.append((admission.hadm_id, codes))

        if len(visits) < 2:
            return []

        # Create samples: for each i, use visits[0..i] as input, visits[i+1] as label
        samples = []
        for i in range(len(visits) - 1):
            # Only create sample if the label visit has at least one diagnosis
            if not visits[i + 1][1]:
                continue
            # Past visits: all visits up to i (including empty ones)
            past_visits = [v[1] for v in visits[: i + 1]]
            samples.append({
                "patient_id": patient.patient_id,
                "visit_id": visits[i][0],
                "prev_diag": past_visits,
                "label": visits[i + 1][1],
            })
        return samples
