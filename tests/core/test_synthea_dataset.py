"""Tests for the Synthea dataset loader.

Author: Justin Xu

Paper: Raphael Poulain, Mehak Gupta, and Rahmatollah Beheshti.
    "CEHR-GAN-BERT: Incorporating Temporal Information from Structured EHR
    Data to Improve Prediction Tasks." MLHC 2022.
    https://proceedings.mlr.press/v182/poulain22a.html

Description: Verifies that SyntheaDataset correctly loads the demo CSV
    files, exposes patient records, and reports basic statistics.
"""

import unittest
from pathlib import Path

from pyhealth.datasets import SyntheaDataset


class TestSyntheaDataset(unittest.TestCase):
    """Test SyntheaDataset with demo data from test-resources."""

    def setUp(self):
        """Load the demo dataset once per test."""
        test_dir = Path(__file__).parent.parent.parent
        self.demo_path = str(
            test_dir / "test-resources" / "core" / "syntheademo"
        )
        self.dataset = SyntheaDataset(
            root=self.demo_path,
            tables=["conditions", "medications", "procedures"],
        )

    def test_patient_count(self):
        """Dataset should contain exactly 5 patients."""
        patient_ids = self.dataset.unique_patient_ids
        self.assertEqual(len(patient_ids), 5)

    def test_stats(self):
        """stats() should run without error."""
        self.dataset.stats()

    def test_get_patient(self):
        """get_patient should return a non-None patient object."""
        patient_ids = self.dataset.unique_patient_ids
        patient = self.dataset.get_patient(patient_ids[0])
        self.assertIsNotNone(patient)

    def test_patient_has_encounter_events(self):
        """Each patient should have at least one encounter event."""
        for pid in self.dataset.unique_patient_ids:
            patient = self.dataset.get_patient(pid)
            encounters = patient.get_events(event_type="encounters")
            self.assertGreater(
                len(encounters), 0,
                f"Patient {pid} should have encounters",
            )

    def test_patient_has_condition_events(self):
        """Each patient should have at least one condition event."""
        for pid in self.dataset.unique_patient_ids:
            patient = self.dataset.get_patient(pid)
            conditions = patient.get_events(event_type="conditions")
            self.assertGreater(
                len(conditions), 0,
                f"Patient {pid} should have conditions",
            )


if __name__ == "__main__":
    unittest.main()
