"""
PyHealth task for diagnosis prediction using the MIMIC-III
and MIMIC-IV datasets. The task aims to predict diagnosis
of next visit for a patient based on historical records.

Dataset Citation:
MIMIC-III:
Johnson, Alistair, et al. "MIMIC-III Clinical Database" (version 1.4).
PhysioNet (2016). RRID:SCR_007345. https://doi.org/10.13026/C2XW26

MIMIC-IV:
Johnson, Alistair, et al. "MIMIC-IV" (version 3.1).
PhysioNet (2024). RRID:SCR_007345.
https://doi.org/10.13026/kpb9-mt58

Paper Citation:
Yu, Leisheng, et al. "Self-Explaining Hypergraph Neural Networks
for Diagnosis Prediction." arXiv preprint arXiv:2502.10689 (2025).

"""

from typing import Any, Dict, List

from pyhealth.tasks import BaseTask


def _extract_visit_diagnoses(patient, admissions, code_attr: str):
    """Extract per-visit diagnosis codes from a patient's admissions.

    Args:
        patient: A Patient object that supports get_events method.
        admissions: List of admission events.
        code_attr: Attribute name on the event object for the ICD code

    Returns:
        A list of (hadm_id, codes) tuples for visits that have at
        least one diagnosis code.
    """
    visits = []
    for admission in admissions:
        diagnoses = patient.get_events(
            event_type="diagnoses_icd",
            filters=[("hadm_id", "==", admission.hadm_id)],
        )
        codes = [
            getattr(event, code_attr)
            for event in diagnoses
            if getattr(event, code_attr, None) is not None
        ]
        if codes:
            visits.append((admission.hadm_id, codes))
    return visits


class DiagnosisPredictionMIMIC3(BaseTask):
    """A PyHealth task class for diagnosis prediction using the MIMIC-III dataset,
    which is formulated as a multilabel classification problem.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict[str, str]): The schema for model input:
            - diagnoses_hist: Nested sequence of diagnosis codes across
              historical visits.
        output_schema (Dict[str, str]): The schema for model output:
            - diagnoses: Multilabel set of diagnosis codes for next visit.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import DiagnosisPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = DiagnosisPredictionMIMIC3()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "DiagnosisPredictionMIMIC3"
    input_schema: Dict[str, str] = {"diagnoses_hist": "nested_sequence"}
    output_schema: Dict[str, str] = {"diagnoses": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the diagnosis prediction task.

        Args:
            patient: A Patient object that supports get_events method.

        Returns:
            A list containing sample dictionaries, which contains patient_id,
            visit_id, diagnoses_hist, and diagnoses, or an empty list if the
            patient has fewer than two valid visits.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        visits = _extract_visit_diagnoses(patient, admissions, "icd9_code")

        if len(visits) < 2:
            return []

        samples = []
        history = []
        for t in range(len(visits)):
            visit_id, codes = visits[t]
            if t > 0:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit_id,
                        "diagnoses_hist": [h for h in history],
                        "diagnoses": codes,
                    }
                )
            history.append(codes)

        return samples


class DiagnosisPredictionMIMIC4(BaseTask):
    """A PyHealth task class for diagnosis prediction using the MIMIC-IV dataset,
    which is formulated as a multilabel classification problem.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict[str, str]): The schema for model input:
            - diagnoses_hist: Nested sequence of diagnosis codes across
              historical visits.
        output_schema (Dict[str, str]): The schema for model output:
            - diagnoses: Multilabel set of diagnosis codes for next visit.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import DiagnosisPredictionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/3.1",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = DiagnosisPredictionMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "DiagnosisPredictionMIMIC4"
    input_schema: Dict[str, str] = {"diagnoses_hist": "nested_sequence"}
    output_schema: Dict[str, str] = {"diagnoses": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the diagnosis prediction task.

        Args:
            patient: A Patient object that supports get_events method.

        Returns:
            A list containing sample dictionaries, which contains patient_id,
            visit_id, diagnoses_hist, and diagnoses, or an empty list if the
            patient has fewer than two valid visits.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        visits = _extract_visit_diagnoses(patient, admissions, "icd_code")
        if len(visits) < 2:
            return []

        samples = []
        history = []
        for t in range(len(visits)):
            visit_id, codes = visits[t]
            if t > 0:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit_id,
                        "diagnoses_hist": [h for h in history],
                        "diagnoses": codes,
                    }
                )
            history.append(codes)

        return samples
