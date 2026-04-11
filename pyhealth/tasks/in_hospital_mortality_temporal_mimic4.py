from typing import Any, Dict, List

from .base_task import BaseTask


class InHospitalMortalityTemporalMIMIC4(BaseTask):
    """In-ICU mortality prediction on MIMIC-IV with temporal (EMDOT-style) evaluation.

    Each sample is tagged with its admission year so callers can partition data
    chronologically to simulate real-world deployment conditions, following the
    EMDOT framework (Zhou et al., 2023). Supports both all-historical and
    sliding window training regimes.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data, which includes:
            - conditions: A sequence of diagnosis ICD codes.
            - procedures: A sequence of procedure ICD codes.
            - drugs: A sequence of prescribed drug names.
        output_schema (Dict[str, str]): The schema for output data, which includes:
            - mortality: A binary indicator of in-hospital mortality.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import InHospitalMortalityTemporalMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = InHospitalMortalityTemporalMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "InHospitalMortalityTemporalMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generates binary mortality samples tagged with admission year.

        Admissions with no conditions OR no procedures OR no drugs are excluded.
        Patients under 18 years old (anchor_age) are excluded.

        Args:
            patient (Any): A PyHealth Patient object.

        Returns:
            List[Dict[str, Any]]: A list of dicts, each containing:
                - 'patient_id': MIMIC-IV subject_id.
                - 'admission_id': MIMIC-IV hadm_id.
                - 'conditions': ICD codes from diagnoses_icd.
                - 'procedures': ICD codes from procedures_icd.
                - 'drugs': Drug names from prescriptions.
                - 'mortality': binary label (1 = died in hospital, 0 = survived).
                - 'admission_year': int year of admission for temporal splits.
        """
        demographics = patient.get_events(event_type="patients")
        assert len(demographics) == 1
        if int(demographics[0].anchor_age) < 18:
            return []

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        samples = []
        for admission in admissions:
            filter = ("hadm_id", "==", admission.hadm_id)

            conditions = []
            for event in patient.get_events(
                event_type="diagnoses_icd", filters=[filter]
            ):
                assert event.icd_version in ("9", "10")
                conditions.append(f"{event.icd_version}_{event.icd_code}")
            if len(conditions) == 0:
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
            drugs = [event.drug for event in prescriptions]
            if len(drugs) == 0:
                continue

            if admission.hospital_expire_flag is None:
                continue
            mortality = int(admission.hospital_expire_flag)

            samples.append({
                "patient_id": patient.patient_id,
                "admission_id": admission.hadm_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "mortality": mortality,
                "admission_year": admission.timestamp.year,
            })

        return samples