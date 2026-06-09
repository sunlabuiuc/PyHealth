"""Tests for the MortalityPredictionSynthea task.

Author: Justin Xu

Paper: Raphael Poulain, Mehak Gupta, and Rahmatollah Beheshti.
    "CEHR-GAN-BERT: Incorporating Temporal Information from Structured EHR
    Data to Improve Prediction Tasks." MLHC 2022, Section A.2.
    https://proceedings.mlr.press/v182/poulain22a.html

Description: Validates cohort construction, label assignment, and schema
    for the Synthea post-discharge mortality prediction task.
"""

import unittest
from pathlib import Path

from pyhealth.datasets import SyntheaDataset
from pyhealth.tasks import MortalityPredictionSynthea


class TestSyntheaMortalityTask(unittest.TestCase):
    """Test MortalityPredictionSynthea with demo data."""

    @classmethod
    def setUpClass(cls):
        """Load dataset and apply task once for all tests."""
        test_dir = Path(__file__).parent.parent.parent
        demo_path = str(
            test_dir / "test-resources" / "core" / "syntheademo"
        )
        cls.dataset = SyntheaDataset(
            root=demo_path,
            tables=["conditions", "medications", "procedures"],
        )
        cls.task = MortalityPredictionSynthea(prediction_window_days=365)
        cls.sample_dataset = cls.dataset.set_task(cls.task)

    # ------------------------------------------------------------------
    # Basic sanity
    # ------------------------------------------------------------------

    def test_set_task_returns_samples(self):
        """set_task should produce a non-empty sample dataset."""
        self.assertIsNotNone(self.sample_dataset)
        self.assertGreater(len(self.sample_dataset), 0)

    def test_sample_schema(self):
        """Each sample should contain the expected keys."""
        required_keys = {
            "visit_id",
            "patient_id",
            "conditions",
            "procedures",
            "medications",
            "mortality",
        }
        for i in range(len(self.sample_dataset)):
            sample = self.sample_dataset[i]
            for key in required_keys:
                self.assertIn(key, sample, f"Missing key: {key}")

    def test_mortality_is_binary(self):
        """Mortality label should be 0 or 1."""
        for i in range(len(self.sample_dataset)):
            sample = self.sample_dataset[i]
            label = sample["mortality"]
            # May be a tensor; coerce to int
            label_int = int(label.item()) if hasattr(label, "item") else int(label)
            self.assertIn(label_int, [0, 1])

    # ------------------------------------------------------------------
    # Patient-specific labels (365-day window)
    # ------------------------------------------------------------------

    def _find_sample_for_patient(self, patient_id):
        """Return the sample whose patient_id matches, or None."""
        for i in range(len(self.sample_dataset)):
            s = self.sample_dataset[i]
            pid = s["patient_id"]
            if hasattr(pid, "item"):
                pid = pid.item()
            if str(pid) == patient_id:
                return s
        return None

    def test_patient1_mortality_positive(self):
        """p001 dies ~100 days post-discharge -> mortality=1."""
        sample = self._find_sample_for_patient("p001")
        self.assertIsNotNone(sample, "p001 should produce a sample")
        label = sample["mortality"]
        label_int = int(label.item()) if hasattr(label, "item") else int(label)
        self.assertEqual(label_int, 1)

    def test_patient2_mortality_negative(self):
        """p002 is alive -> mortality=0."""
        sample = self._find_sample_for_patient("p002")
        self.assertIsNotNone(sample, "p002 should produce a sample")
        label = sample["mortality"]
        label_int = int(label.item()) if hasattr(label, "item") else int(label)
        self.assertEqual(label_int, 0)

    def test_patient3_mortality_negative_365(self):
        """p003 dies ~431 days post-discharge -> mortality=0 for 365-day."""
        sample = self._find_sample_for_patient("p003")
        self.assertIsNotNone(sample, "p003 should produce a sample")
        label = sample["mortality"]
        label_int = int(label.item()) if hasattr(label, "item") else int(label)
        self.assertEqual(label_int, 0)

    def test_patient4_excluded(self):
        """p004 dies during encounter -> excluded from cohort."""
        sample = self._find_sample_for_patient("p004")
        self.assertIsNone(sample, "p004 should be excluded (died during encounter)")

    def test_patient5_mortality_negative(self):
        """p005 is alive, multiple encounters -> mortality=0."""
        sample = self._find_sample_for_patient("p005")
        self.assertIsNotNone(sample, "p005 should produce a sample")
        label = sample["mortality"]
        label_int = int(label.item()) if hasattr(label, "item") else int(label)
        self.assertEqual(label_int, 0)

    # ------------------------------------------------------------------
    # Prediction window variation
    # ------------------------------------------------------------------

    def test_patient3_mortality_positive_730(self):
        """p003 dies ~431 days post-discharge -> mortality=1 for 730-day."""
        task_730 = MortalityPredictionSynthea(prediction_window_days=730)
        sample_dataset_730 = self.dataset.set_task(task_730)
        sample = None
        for i in range(len(sample_dataset_730)):
            s = sample_dataset_730[i]
            pid = s["patient_id"]
            if hasattr(pid, "item"):
                pid = pid.item()
            if str(pid) == "p003":
                sample = s
                break
        self.assertIsNotNone(sample, "p003 should produce a sample with 730-day window")
        label = sample["mortality"]
        label_int = int(label.item()) if hasattr(label, "item") else int(label)
        self.assertEqual(label_int, 1)

    # ------------------------------------------------------------------
    # Cohort size
    # ------------------------------------------------------------------

    def test_excluded_count(self):
        """With 5 patients and 1 excluded, should have 4 samples."""
        self.assertEqual(len(self.sample_dataset), 4)


if __name__ == "__main__":
    unittest.main()
