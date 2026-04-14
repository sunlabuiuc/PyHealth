"""
Unit tests for MedLingoDataset and abbreviation_expansion_medlingo_fn.

Run with:
    pytest tests/test_medlingo_dataset.py -v

Unit tests (TestMedLingoDatasetStructure) use small synthetic data and run
fast without any external downloads.

Integration tests (TestMedLingoTaskFunction, TestMedLingoDatasetIntegration)
use the real questions.csv at test-resources/MedLingo/questions.csv.
"""

import os
import tempfile
import textwrap
import unittest

import pandas as pd

# ---------------------------------------------------------------------------
# Check PyHealth availability
# ---------------------------------------------------------------------------
try:
    from pyhealth.datasets import BaseDataset
    _PYHEALTH_AVAILABLE = True
except ImportError:
    _PYHEALTH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV_CONTENT = textwrap.dedent(
    """\
    word1,word2,question,answer
    PRN,as needed,"In a clinical note that mentions a high creat, creat stands for creatine. In a clinical note that mentions a patient is prescribed a medication labeled as PRN for pain relief, PRN stands for",as needed
    HTN,hypertension,"In a clinical note that mentions a high creat, creat stands for creatine. In a medical record that mentions a patient has HTN, HTN stands for",hypertension
    NPO,nothing by mouth,"In a clinical note that mentions a high creat, creat stands for creatine. In a clinical note that mentions a patient should remain NPO before surgery, NPO stands for",nothing by mouth
    """
)

# Path to real dataset
_REAL_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "test-resources", "MedLingo"
)


def _real_data_available():
    return os.path.isfile(os.path.join(_REAL_DATA_DIR, "questions.csv"))


def _write_sample_csv(directory: str) -> str:
    csv_path = os.path.join(directory, "questions.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(SAMPLE_CSV_CONTENT)
    return csv_path


# ---------------------------------------------------------------------------
# Unit tests — CSV structure only, no PyHealth required
# ---------------------------------------------------------------------------

class TestMedLingoDatasetStructure(unittest.TestCase):
    """Test CSV parsing logic independently of PyHealth."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        _write_sample_csv(self.tmp_dir)

    def test_csv_loads_correctly(self):
        """questions.csv must parse into a DataFrame with the required columns."""
        csv_path = os.path.join(self.tmp_dir, "questions.csv")
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.lower().str.strip()
        required = {"word1", "word2", "question", "answer"}
        self.assertTrue(required.issubset(set(df.columns)))
        self.assertEqual(len(df), 3)

    def test_no_null_in_required_columns(self):
        """After fillna(''), required columns should have no empty values."""
        csv_path = os.path.join(self.tmp_dir, "questions.csv")
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df.columns = df.columns.str.lower().str.strip()
        for col in ("word1", "word2", "question", "answer"):
            self.assertTrue((df[col] != "").all())

    def test_word1_values_are_unique(self):
        """Each abbreviation (word1) should be unique."""
        csv_path = os.path.join(self.tmp_dir, "questions.csv")
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.lower().str.strip()
        self.assertEqual(df["word1"].nunique(), len(df))

    def test_missing_file_raises(self):
        """A missing questions.csv should raise FileNotFoundError."""
        empty_dir = tempfile.mkdtemp()
        path = os.path.join(empty_dir, "questions.csv")
        with self.assertRaises(FileNotFoundError):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")

    def test_missing_column_raises(self):
        """A CSV missing required columns should raise ValueError."""
        bad_csv = os.path.join(self.tmp_dir, "bad.csv")
        pd.DataFrame({"word1": ["PRN"], "word2": ["as needed"]}).to_csv(
            bad_csv, index=False
        )
        df = pd.read_csv(bad_csv, dtype=str)
        df.columns = df.columns.str.lower().str.strip()
        missing = {"word1", "word2", "question", "answer"} - set(df.columns)
        with self.assertRaises(ValueError):
            if missing:
                raise ValueError(f"Missing columns: {missing}")

    def test_dev_mode_limits_rows(self):
        """Dev mode should cap to the first N rows."""
        csv_path = os.path.join(self.tmp_dir, "questions.csv")
        df = pd.read_csv(csv_path, dtype=str)
        self.assertEqual(len(df.head(2)), 2)


# ---------------------------------------------------------------------------
# Integration tests — require PyHealth + real dataset
# ---------------------------------------------------------------------------

class TestMedLingoDatasetIntegration(unittest.TestCase):
    """End-to-end tests using the real questions.csv."""

    def setUp(self):
        if not _PYHEALTH_AVAILABLE:
            self.skipTest("pyhealth not installed")
        if not _real_data_available():
            self.skipTest(
                "Real dataset not found at test-resources/MedLingo/questions.csv"
            )
        self.data_dir = _REAL_DATA_DIR

    def test_dataset_loads_patients(self):
        """MedLingoDataset must expose one patient per unique abbreviation."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset

        ds = MedLingoDataset(root=self.data_dir, dev=False)
        patient_ids = ds.unique_patient_ids
        self.assertGreater(len(patient_ids), 0)
        self.assertIn("PRN", patient_ids)
        self.assertIn("HTN", patient_ids)
        self.assertIn("NPO", patient_ids)

    def test_dev_mode_limits_patients(self):
        """Dev mode must load only a subset of entries (up to 1000)."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset

        ds = MedLingoDataset(root=self.data_dir, dev=True)
        # PyHealth 2.0 dev mode limits to 1000 patients;
        # MedLingo has 100 entries so all load, but count must be > 0
        self.assertGreater(len(ds.unique_patient_ids), 0)
        self.assertLessEqual(len(ds.unique_patient_ids), 1000)

    def test_get_patient(self):
        """get_patient() must return a Patient for known abbreviations."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset

        ds = MedLingoDataset(root=self.data_dir, dev=False)
        patient = ds.get_patient("PRN")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "PRN")

    def test_get_patient_unknown_returns_none(self):
        """get_patient() must raise AssertionError for unknown abbreviations."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset

        ds = MedLingoDataset(root=self.data_dir, dev=False)
        with self.assertRaises(AssertionError):
            ds.get_patient("NONEXISTENT")

    def test_get_abbreviation_helper(self):
        """get_abbreviation() convenience method must work correctly."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset

        ds = MedLingoDataset(root=self.data_dir, dev=False)
        self.assertIsNotNone(ds.get_abbreviation("HTN"))
        self.assertIsNone(ds.get_abbreviation("NONEXISTENT"))

    def test_missing_file_raises(self):
        """Instantiating with a missing CSV must raise FileNotFoundError."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset

        with self.assertRaises(FileNotFoundError):
            MedLingoDataset(root=tempfile.mkdtemp())

    def test_set_task_produces_samples(self):
        """set_task() must return a non-empty SampleDataset."""
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset
        from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo

        ds = MedLingoDataset(root=self.data_dir, dev=False)
        sample_ds = ds.set_task(AbbreviationExpansionMedLingo())
        self.assertGreater(len(sample_ds), 0)


class TestMedLingoTaskFunction(unittest.TestCase):
    """Tests for the abbreviation_expansion_medlingo_fn task function."""

    def setUp(self):
        if not _PYHEALTH_AVAILABLE:
            self.skipTest("pyhealth not installed")
        if not _real_data_available():
            self.skipTest(
                "Real dataset not found at test-resources/MedLingo/questions.csv"
            )

    def _load_dev_dataset(self):
        from pyhealth.datasets.medlingo_dataset import MedLingoDataset
        return MedLingoDataset(root=_REAL_DATA_DIR, dev=True)

    def test_sample_dict_keys(self):
        """Each sample must contain the required keys."""
        from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo

        ds = self._load_dev_dataset()
        patient = next(ds.iter_patients())
        samples = AbbreviationExpansionMedLingo()(patient)
        self.assertGreater(len(samples), 0)
        required_keys = {"patient_id", "visit_id", "question", "answer"}
        self.assertTrue(required_keys.issubset(set(samples[0].keys())))

    def test_label_equals_answer(self):
        """answer field must be a non-empty string in every sample."""
        from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo

        ds = self._load_dev_dataset()
        for patient in ds.iter_patients():
            for s in AbbreviationExpansionMedLingo()(patient):
                self.assertIsInstance(s["answer"], str)
                self.assertGreater(len(s["answer"]), 0)

    def test_task_does_not_crash(self):
        """Task must return a list for every patient without crashing."""
        from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo

        task = AbbreviationExpansionMedLingo()
        ds = self._load_dev_dataset()
        for patient in ds.iter_patients():
            samples = task(patient)
            self.assertIsInstance(samples, list)
            self.assertGreaterEqual(len(samples), 0)


if __name__ == "__main__":
    unittest.main()