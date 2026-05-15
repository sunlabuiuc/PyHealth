"""In-hospital mortality prediction task using synthetic demographic note templates.

This module implements a clinical text-based mortality prediction task for
MIMIC-III, reproducing the downstream evaluation setup from:

    Zhang et al. "Hurtful Words: Quantifying Biases in Clinical Contextual
    Word Embeddings." ACM CHIL 2020. https://arxiv.org/abs/2003.11515

Since clinical notes may be unavailable or restricted, this task generates
synthetic note templates populated with real patient demographics (gender, age)
drawn from the PATIENTS and ADMISSIONS tables. This mirrors the fill-in-the-blank
template methodology described in Section 3.4 of the paper, enabling fairness
evaluation of clinical language models across gender, ethnicity, insurance status,
and language subgroups.
"""

from typing import Any, Dict, List, Optional

from pyhealth.tasks import BaseTask


# ---------------------------------------------------------------------------
# Note templates adapted from Section 3.4 of Zhang et al. (2020).
# Each template is populated with real patient demographics at runtime.
# ---------------------------------------------------------------------------

CLINICAL_NOTE_TEMPLATES: List[str] = [
    "this is a {age} yo {gender} with a hx of heart disease",
    "this is a {age} yo {gender} with a pmh of diabetes",
    "{gender} pt is dnr",
    "{gender} has a pmh of hiv",
    "this is a {age} yo {gender} with a discharge diagnosis of htn",
    "this is a {age} yo {gender} with a hx of schizophrenia",
    "this is a {age} yo {gender} with a hx of heroin addiction",
    "this is a {age} yo {gender} with a hx of hypertension",
]


class MortalityTextTaskMIMIC3(BaseTask):
    """In-hospital mortality prediction from synthetic clinical note templates.

    Reproduces the in-hospital mortality clinical prediction task from:
        Zhang et al. "Hurtful Words: Quantifying Biases in Clinical
        Contextual Word Embeddings." ACM CHIL 2020.
        https://arxiv.org/abs/2003.11515

    For each patient admission, this task generates synthetic clinical note
    templates (see CLINICAL_NOTE_TEMPLATES) populated with real patient
    demographics extracted from the MIMIC-III PATIENTS and ADMISSIONS tables.
    The binary mortality label is derived from the hospital_expire_flag field.

    Demographic fields (gender, ethnicity, insurance, language) are preserved
    in each sample to support downstream fairness evaluation — specifically the
    recall gap, parity gap, and specificity gap metrics described in the paper.

    This task is designed for use with MIMIC3Dataset loaded with at minimum
    the PATIENTS and ADMISSIONS tables.

    Args:
        max_notes (int): Maximum number of synthetic note templates to include
            per sample. Must be >= 1. Defaults to 5.

    Attributes:
        task_name (str): Unique identifier for this task.
        input_schema (Dict[str, str]): Maps feature name to processor type.
        output_schema (Dict[str, str]): Maps label name to processor type.

    Example:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MortalityTextTaskMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic3",
        ...     tables=["PATIENTS", "ADMISSIONS"],
        ... )
        >>> task = MortalityTextTaskMIMIC3(max_notes=5)
        >>> task_dataset = dataset.set_task(task)
        >>> print(task_dataset[0])
        {
            'visit_id': '142345',
            'patient_id': '10006',
            'notes': ['this is a 65 yo female with a hx of heart disease', ...],
            'label': 0,
            'gender': 'female',
            'ethnicity': 'WHITE',
            'insurance': 'Medicare',
            'language': 'ENGL',
        }
    """

    task_name: str = "mortality_text"
    input_schema: Dict[str, str] = {"notes": "sequence"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, max_notes: int = 5) -> None:
        """Initialise the task.

        Args:
            max_notes (int): Maximum number of synthetic note templates per
                sample. Defaults to 5.
        """
        self.max_notes = max_notes

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into mortality prediction samples.

        Extracts gender from the PATIENTS partition and iterates over each
        admission row to generate one sample per hospital stay. Synthetic
        clinical notes are generated from CLINICAL_NOTE_TEMPLATES using the
        patient's gender and computed age.

        Args:
            patient: A PyHealth Patient object whose data_source attribute is
                a Polars DataFrame with event rows for the 'patients' and
                'admissions' event types, and the following columns:
                    - event_type (str)
                    - timestamp (datetime | None)
                    - patients/gender (str | None): 'F' or 'M'
                    - patients/dob (datetime | None)
                    - admissions/hadm_id (str | None)
                    - admissions/hospital_expire_flag (int | None)
                    - admissions/ethnicity (str | None)
                    - admissions/insurance (str | None)
                    - admissions/language (str | None)

        Returns:
            List[Dict[str, Any]]: One sample dict per admission. Each dict
            contains the following keys:
                - visit_id (str): Hospital admission ID (hadm_id).
                - patient_id (str): Patient identifier.
                - notes (List[str]): Synthetic clinical note strings.
                - label (int): 1 if patient died, 0 if survived.
                - gender (str): 'male' or 'female'.
                - ethnicity (str): Ethnicity string, or 'unknown'.
                - insurance (str): Insurance type string, or 'unknown'.
                - language (str): Language string, or 'unknown'.

            Returns an empty list if the patient has no 'patients' or
            'admissions' event rows.
        """
        samples: List[Dict[str, Any]] = []
        df = patient.data_source

        # -- gender from patients partition -----------------------------------
        patients_df = df.filter(df["event_type"] == "patients")
        if patients_df.is_empty():
            return samples

        gender_raw: Optional[str] = patients_df["patients/gender"][0]
        gender: str = "female" if gender_raw == "F" else "male"

        # -- one sample per admission row -------------------------------------
        admissions_df = df.filter(df["event_type"] == "admissions")
        if admissions_df.is_empty():
            return samples

        for row in admissions_df.iter_rows(named=True):
            ethnicity: str = row.get("admissions/ethnicity") or "unknown"
            insurance: str = row.get("admissions/insurance") or "unknown"
            language: str = row.get("admissions/language") or "unknown"

            expire_flag = row.get("admissions/hospital_expire_flag", 0)
            label: int = int(expire_flag == 1) if expire_flag is not None else 0

            # compute age from date of birth; fall back to 65 if unavailable
            dob = patients_df["patients/dob"][0]
            admit_time = row.get("timestamp")
            try:
                age: int = int((admit_time - dob).days / 365)
            except Exception:
                age = 65

            fake_notes: List[str] = [
                t.format(gender=gender, age=age)
                for t in CLINICAL_NOTE_TEMPLATES
            ][: self.max_notes]

            samples.append(
                {
                    "visit_id": str(row.get("admissions/hadm_id", "")),
                    "patient_id": patient.patient_id,
                    "notes": fake_notes,
                    "label": label,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "insurance": insurance,
                    "language": language,
                }
            )

        return samples
