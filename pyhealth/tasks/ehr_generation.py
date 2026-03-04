"""Task function for PromptEHR synthetic EHR generation.

Provides task classes for training PromptEHR on MIMIC-III and MIMIC-IV datasets.
Demographics (age, gender) are extracted alongside visit codes because PromptEHR
conditions generation on patient-level continuous and categorical features.
"""

from datetime import datetime
from typing import Dict, List

import polars as pl

from pyhealth.tasks.base_task import BaseTask


class PromptEHRGenerationMIMIC3(BaseTask):
    """Task for PromptEHR synthetic data generation using MIMIC-III.

    PromptEHR is a BART-based seq2seq model that conditions generation on
    patient demographics (age, gender) via learned prompt vectors. This task
    extracts per-admission ICD-9 diagnosis codes grouped into a nested visit
    list, along with patient demographics for conditioning.

    Patients with fewer than 2 admissions containing diagnosis codes are
    excluded.

    Attributes:
        task_name (str): Unique task identifier.
        input_schema (dict): ``"visits"`` uses ``"nested_sequence"`` encoding
            (list of lists of code strings).
        output_schema (dict): Empty — generative task, no conditioning label.
        _icd_col (str): Polars column path for ICD codes in MIMIC-III.

    Examples:
        >>> fn = PromptEHRGenerationMIMIC3()
        >>> fn.task_name
        'PromptEHRGenerationMIMIC3'
    """

    task_name = "PromptEHRGenerationMIMIC3"
    input_schema = {"visits": "nested_sequence"}
    output_schema = {}
    _icd_col = "diagnoses_icd/icd9_code"

    def __call__(self, patient) -> List[Dict]:
        """Extract visit sequences and demographics for a single patient.

        Diagnosis codes are grouped per admission into a nested list. Age is
        computed as years between date-of-birth and the first admission date.
        Gender is encoded as 0 (male) or 1 (female). Defaults of
        ``age=60.0, gender=0`` are used when demographics are unavailable.

        Args:
            patient: A PyHealth Patient object with admissions and
                diagnoses_icd event data.

        Returns:
            list of dict: A single-element list, or empty list if fewer
            than 2 visits have diagnosis codes. Each dict contains:
                ``"patient_id"`` (str): patient identifier.
                ``"visits"`` (list of list of str): ICD codes per visit.
                ``"age"`` (float): patient age at first admission in years.
                ``"gender"`` (int): 0 for male, 1 for female.
        """
        admissions = list(patient.get_events(event_type="admissions"))
        if len(admissions) < 2:
            return []

        # --- Demographics ---
        age = 60.0
        gender = 0
        patients_df = patient.get_events(event_type="patients", return_df=True)
        if len(patients_df) > 0:
            if "patients/gender" in patients_df.columns:
                gender_val = patients_df["patients/gender"][0]
                if gender_val == "F":
                    gender = 1
            if "patients/dob" in patients_df.columns and admissions:
                dob_val = patients_df["patients/dob"][0]
                first_admit_ts = admissions[0].timestamp
                if dob_val is not None and first_admit_ts is not None:
                    # dob_val may be a date/datetime or a string
                    if hasattr(dob_val, "year"):
                        dob_dt = datetime(dob_val.year, dob_val.month, dob_val.day)
                    else:
                        dob_dt = datetime.strptime(str(dob_val)[:10], "%Y-%m-%d")
                    raw_age = (first_admit_ts - dob_dt).days / 365.25
                    # Clamp: MIMIC-III shifts >89-year-old DOBs far into the
                    # past; treat those as 90.
                    age = float(min(90.0, max(0.0, raw_age)))

        # --- Visit codes ---
        visits = []
        for adm in admissions:
            codes = (
                patient.get_events(
                    event_type="diagnoses_icd",
                    filters=[("hadm_id", "==", adm.hadm_id)],
                    return_df=True,
                )
                .select(pl.col(self._icd_col))
                .to_series()
                .drop_nulls()
                .to_list()
            )
            if codes:
                visits.append(codes)

        if len(visits) < 2:
            return []

        return [{
            "patient_id": patient.patient_id,
            "visits": visits,
            "age": age,
            "gender": gender,
        }]


class PromptEHRGenerationMIMIC4(PromptEHRGenerationMIMIC3):
    """Task for PromptEHR synthetic data generation using MIMIC-IV.

    Inherits all logic from :class:`PromptEHRGenerationMIMIC3`. Overrides only
    the task name and ICD code column to match the MIMIC-IV schema, where the
    column is ``icd_code`` (unversioned) rather than ``icd9_code``.

    Attributes:
        task_name (str): Unique task identifier.
        _icd_col (str): Polars column path for ICD codes in MIMIC-IV.

    Examples:
        >>> fn = PromptEHRGenerationMIMIC4()
        >>> fn.task_name
        'PromptEHRGenerationMIMIC4'
    """

    task_name = "PromptEHRGenerationMIMIC4"
    _icd_col = "diagnoses_icd/icd_code"


promptehr_generation_mimic3_fn = PromptEHRGenerationMIMIC3()
promptehr_generation_mimic4_fn = PromptEHRGenerationMIMIC4()
