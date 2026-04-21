"""
Tests for MortalityTextTask.

Tests use fully synthetic data — no real MIMIC data required.
All tests complete in milliseconds.

Reference paper:
    Zhang et al. "Hurtful Words: Quantifying Biases in Clinical
    Contextual Word Embeddings." ACM CHIL 2020.
    https://arxiv.org/abs/2003.11515
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock

import polars as pl
from pyhealth.tasks import BaseTask


# ---------------------------------------------------------------------------
# Task definition (copy of pyhealth/tasks/mortality_text_task.py)
# ---------------------------------------------------------------------------

TEMPLATES = [
    "this is a {age} yo {gender} with a hx of heart disease",
    "this is a {age} yo {gender} with a pmh of diabetes",
    "{gender} pt is dnr",
    "{gender} has a pmh of hiv",
    "this is a {age} yo {gender} with a discharge diagnosis of htn",
    "this is a {age} yo {gender} with a hx of schizophrenia",
    "this is a {age} yo {gender} with a hx of heroin addiction",
    "this is a {age} yo {gender} with a hx of hypertension",
]


class MortalityTextTask(BaseTask):
    """In-hospital mortality prediction using synthetic demographic note templates.

    Reproduces the clinical prediction task from:
        Zhang et al. "Hurtful Words: Quantifying Biases in Clinical
        Contextual Word Embeddings." ACM CHIL 2020.

    For each patient admission, generates synthetic clinical note templates
    populated with real patient demographics (gender, age) and assigns a
    binary mortality label from the ADMISSIONS table.

    Args:
        max_notes (int): Maximum number of note templates per sample.
            Defaults to 5.

    Input schema:
        notes (sequence): List of synthetic clinical note strings.

    Output schema:
        label (binary): 1 if patient died during admission, 0 otherwise.

    Example:
        >>> task = MortalityTextTask(max_notes=5)
        >>> dataset = MIMIC3Dataset(root="...", tables=["PATIENTS", "ADMISSIONS"])
        >>> task_dataset = dataset.set_task(task)
    """

    task_name: str = "mortality_text"
    input_schema: dict = {"notes": "sequence"}
    output_schema: dict = {"label": "binary"}

    def __init__(self, max_notes: int = 5) -> None:
        self.max_notes = max_notes

    def __call__(self, patient: object) -> list:
        """Process a single patient into mortality prediction samples.

        Args:
            patient: PyHealth Patient object with a data_source Polars
                DataFrame containing event rows for 'patients' and
                'admissions' event types.

        Returns:
            List of sample dicts, one per admission, each containing:
                - visit_id (str): Hospital admission ID.
                - patient_id (str): Patient identifier.
                - notes (list[str]): Synthetic clinical note templates.
                - label (int): 1 = died, 0 = survived.
                - gender (str): 'male' or 'female'.
                - ethnicity (str): Patient ethnicity string.
                - insurance (str): Insurance type string.
                - language (str): Language string.
        """
        samples = []
        df = patient.data_source

        # -- gender from patients partition --
        patients_df = df.filter(df["event_type"] == "patients")
        if patients_df.is_empty():
            return samples

        gender_raw = patients_df["patients/gender"][0]
        gender = "female" if gender_raw == "F" else "male"

        # -- one sample per admission --
        admissions_df = df.filter(df["event_type"] == "admissions")
        if admissions_df.is_empty():
            return samples

        for row in admissions_df.iter_rows(named=True):
            ethnicity: str = row.get("admissions/ethnicity") or "unknown"
            insurance: str = row.get("admissions/insurance") or "unknown"
            language: str = row.get("admissions/language") or "unknown"

            expire_flag = row.get("admissions/hospital_expire_flag", 0)
            label: int = int(expire_flag == 1) if expire_flag is not None else 0

            # compute age; fall back to 65 if timestamps are unavailable
            dob = patients_df["patients/dob"][0]
            admit_time = row.get("timestamp")
            try:
                age: int = int((admit_time - dob).days / 365)
            except Exception:
                age = 65

            fake_notes = [
                t.format(gender=gender, age=age) for t in TEMPLATES
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_patient(
    patient_id: str = "test_001",
    gender: str = "F",
    dob: datetime = datetime(1960, 1, 1),
    admit_time: datetime = datetime(2020, 1, 1),
    expire_flag: int = 0,
    ethnicity: str = "WHITE",
    insurance: str = "Medicare",
    language: str = "ENGL",
    n_admissions: int = 1,
) -> MagicMock:
    """Build a minimal synthetic patient object backed by a Polars DataFrame.

    Args:
        patient_id: Unique patient identifier string.
        gender: Raw gender code, 'F' or 'M'.
        dob: Date of birth as datetime.
        admit_time: Admission timestamp as datetime.
        expire_flag: 1 if patient died, 0 otherwise.
        ethnicity: Ethnicity label string.
        insurance: Insurance type string.
        language: Language code string.
        n_admissions: Number of admission rows to generate.

    Returns:
        MagicMock patient object with patient_id and data_source attributes.
    """
    patient_rows = {
        "patient_id": [patient_id],
        "event_type": ["patients"],
        "timestamp": [None],
        "patients/gender": [gender],
        "patients/dob": [dob],
        "admissions/hadm_id": [None],
        "admissions/hospital_expire_flag": [None],
        "admissions/ethnicity": [None],
        "admissions/insurance": [None],
        "admissions/language": [None],
    }

    admission_rows = {
        "patient_id": [patient_id] * n_admissions,
        "event_type": ["admissions"] * n_admissions,
        "timestamp": [admit_time] * n_admissions,
        "patients/gender": [None] * n_admissions,
        "patients/dob": [None] * n_admissions,
        "admissions/hadm_id": [f"10000{i}" for i in range(n_admissions)],
        "admissions/hospital_expire_flag": [expire_flag] * n_admissions,
        "admissions/ethnicity": [ethnicity] * n_admissions,
        "admissions/insurance": [insurance] * n_admissions,
        "admissions/language": [language] * n_admissions,
    }

    df = pl.concat([pl.DataFrame(patient_rows), pl.DataFrame(admission_rows)])
    patient = MagicMock()
    patient.patient_id = patient_id
    patient.data_source = df
    return patient


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestMortalityTextTaskSampleProcessing(unittest.TestCase):
    """Tests for basic sample processing."""

    def setUp(self) -> None:
        self.task = MortalityTextTask(max_notes=5)

    def test_returns_list(self) -> None:
        """__call__ should always return a list."""
        patient = make_synthetic_patient()
        result = self.task(patient)
        self.assertIsInstance(result, list)

    def test_single_admission_one_sample(self) -> None:
        """One admission row should produce exactly one sample."""
        patient = make_synthetic_patient(n_admissions=1)
        result = self.task(patient)
        self.assertEqual(len(result), 1)

    def test_multiple_admissions_multiple_samples(self) -> None:
        """Two admission rows should produce two samples."""
        patient = make_synthetic_patient(n_admissions=2)
        result = self.task(patient)
        self.assertEqual(len(result), 2)

    def test_sample_has_required_keys(self) -> None:
        """Every sample must contain all required keys."""
        patient = make_synthetic_patient()
        result = self.task(patient)
        required = {
            "visit_id", "patient_id", "notes",
            "label", "gender", "ethnicity", "insurance", "language",
        }
        self.assertTrue(required.issubset(result[0].keys()))

    def test_patient_id_preserved(self) -> None:
        """Sample patient_id must match the input patient."""
        patient = make_synthetic_patient(patient_id="p_42")
        result = self.task(patient)
        self.assertEqual(result[0]["patient_id"], "p_42")


class TestMortalityTextTaskLabelGeneration(unittest.TestCase):
    """Tests for mortality label generation."""

    def setUp(self) -> None:
        self.task = MortalityTextTask(max_notes=5)

    def test_label_zero_for_survivor(self) -> None:
        """hospital_expire_flag=0 should produce label 0."""
        patient = make_synthetic_patient(expire_flag=0)
        result = self.task(patient)
        self.assertEqual(result[0]["label"], 0)

    def test_label_one_for_death(self) -> None:
        """hospital_expire_flag=1 should produce label 1."""
        patient = make_synthetic_patient(expire_flag=1)
        result = self.task(patient)
        self.assertEqual(result[0]["label"], 1)

    def test_label_is_integer(self) -> None:
        """Label must be a plain Python int."""
        patient = make_synthetic_patient(expire_flag=1)
        result = self.task(patient)
        self.assertIsInstance(result[0]["label"], int)

    def test_none_expire_flag_defaults_to_zero(self) -> None:
        """A None expire flag should safely default to label 0."""
        patient = make_synthetic_patient(expire_flag=0)
        # manually null out the flag in the dataframe
        df = patient.data_source
        df = df.with_columns(
            pl.when(pl.col("event_type") == "admissions")
            .then(None)
            .otherwise(pl.col("admissions/hospital_expire_flag"))
            .alias("admissions/hospital_expire_flag")
        )
        patient.data_source = df
        result = self.task(patient)
        self.assertEqual(result[0]["label"], 0)


class TestMortalityTextTaskFeatureExtraction(unittest.TestCase):
    """Tests for synthetic note feature extraction."""

    def setUp(self) -> None:
        self.task = MortalityTextTask(max_notes=5)

    def test_notes_is_list(self) -> None:
        """Notes field should be a list."""
        patient = make_synthetic_patient()
        result = self.task(patient)
        self.assertIsInstance(result[0]["notes"], list)

    def test_notes_are_strings(self) -> None:
        """Every note should be a plain string."""
        patient = make_synthetic_patient()
        result = self.task(patient)
        for note in result[0]["notes"]:
            self.assertIsInstance(note, str)

    def test_max_notes_respected(self) -> None:
        """Notes list length must not exceed max_notes."""
        task = MortalityTextTask(max_notes=3)
        patient = make_synthetic_patient()
        result = task(patient)
        self.assertLessEqual(len(result[0]["notes"]), 3)

    def test_max_notes_one(self) -> None:
        """max_notes=1 should return exactly one note."""
        task = MortalityTextTask(max_notes=1)
        patient = make_synthetic_patient()
        result = task(patient)
        self.assertEqual(len(result[0]["notes"]), 1)

    def test_notes_contain_gender_female(self) -> None:
        """Notes should reference 'female' for F-coded patients."""
        patient = make_synthetic_patient(gender="F")
        result = self.task(patient)
        combined = " ".join(result[0]["notes"])
        self.assertIn("female", combined)

    def test_notes_contain_gender_male(self) -> None:
        """Notes should reference 'male' for M-coded patients."""
        patient = make_synthetic_patient(gender="M")
        result = self.task(patient)
        combined = " ".join(result[0]["notes"])
        self.assertIn("male", combined)

    def test_notes_contain_age(self) -> None:
        """Notes should embed the computed patient age."""
        dob = datetime(1960, 1, 1)
        admit = datetime(2020, 1, 1)
        expected_age = str(int((admit - dob).days / 365))
        patient = make_synthetic_patient(dob=dob, admit_time=admit)
        result = self.task(patient)
        combined = " ".join(result[0]["notes"])
        self.assertIn(expected_age, combined)


class TestMortalityTextTaskDemographics(unittest.TestCase):
    """Tests that demographic fields are preserved in samples."""

    def setUp(self) -> None:
        self.task = MortalityTextTask(max_notes=5)

    def test_gender_female_mapping(self) -> None:
        """'F' should map to 'female' in the sample."""
        patient = make_synthetic_patient(gender="F")
        result = self.task(patient)
        self.assertEqual(result[0]["gender"], "female")

    def test_gender_male_mapping(self) -> None:
        """'M' should map to 'male' in the sample."""
        patient = make_synthetic_patient(gender="M")
        result = self.task(patient)
        self.assertEqual(result[0]["gender"], "male")

    def test_ethnicity_preserved(self) -> None:
        """Ethnicity string should pass through unchanged."""
        patient = make_synthetic_patient(ethnicity="BLACK/AFRICAN AMERICAN")
        result = self.task(patient)
        self.assertEqual(result[0]["ethnicity"], "BLACK/AFRICAN AMERICAN")

    def test_insurance_preserved(self) -> None:
        """Insurance type should pass through unchanged."""
        patient = make_synthetic_patient(insurance="Medicaid")
        result = self.task(patient)
        self.assertEqual(result[0]["insurance"], "Medicaid")

    def test_language_preserved(self) -> None:
        """Language code should pass through unchanged."""
        patient = make_synthetic_patient(language="SPAN")
        result = self.task(patient)
        self.assertEqual(result[0]["language"], "SPAN")


class TestMortalityTextTaskEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self) -> None:
        self.task = MortalityTextTask(max_notes=5)

    def test_no_patients_row_returns_empty(self) -> None:
        """If no 'patients' event rows exist, return empty list."""
        df = pl.DataFrame({
            "patient_id": ["p_edge"],
            "event_type": ["admissions"],
            "timestamp": [datetime(2020, 1, 1)],
            "patients/gender": [None],
            "patients/dob": [None],
            "admissions/hadm_id": ["999"],
            "admissions/hospital_expire_flag": [0],
            "admissions/ethnicity": ["WHITE"],
            "admissions/insurance": ["Medicare"],
            "admissions/language": ["ENGL"],
        })
        patient = MagicMock()
        patient.patient_id = "p_edge"
        patient.data_source = df
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_no_admissions_row_returns_empty(self) -> None:
        """If no 'admissions' event rows exist, return empty list."""
        df = pl.DataFrame({
            "patient_id": ["p_edge2"],
            "event_type": ["patients"],
            "timestamp": [None],
            "patients/gender": ["F"],
            "patients/dob": [datetime(1960, 1, 1)],
            "admissions/hadm_id": [None],
            "admissions/hospital_expire_flag": [None],
            "admissions/ethnicity": [None],
            "admissions/insurance": [None],
            "admissions/language": [None],
        })
        patient = MagicMock()
        patient.patient_id = "p_edge2"
        patient.data_source = df
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_missing_ethnicity_defaults_to_unknown(self) -> None:
        """None ethnicity should default to 'unknown'."""
        patient = make_synthetic_patient(ethnicity=None)
        result = self.task(patient)
        self.assertEqual(result[0]["ethnicity"], "unknown")

    def test_missing_insurance_defaults_to_unknown(self) -> None:
        """None insurance should default to 'unknown'."""
        patient = make_synthetic_patient(insurance=None)
        result = self.task(patient)
        self.assertEqual(result[0]["insurance"], "unknown")

    def test_bad_dob_falls_back_to_age_65(self) -> None:
        """If age cannot be computed, notes should use fallback age 65."""
        patient = make_synthetic_patient()
        # null out dob so age computation fails
        df = patient.data_source.with_columns(
            pl.when(pl.col("event_type") == "patients")
            .then(None)
            .otherwise(pl.col("patients/dob"))
            .alias("patients/dob")
        )
        patient.data_source = df
        result = self.task(patient)
        combined = " ".join(result[0]["notes"])
        self.assertIn("65", combined)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
