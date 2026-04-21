"""
Contributor: Adrianne Sun, Ruoyi Xie
NetID: ajsun2, ruoyix2
Paper Title: Reproducible Survival Prediction with SEER Cancer Data
Paper Link: https://proceedings.mlr.press/v85/hegselmann18a/hegselmann18a.pdf
Description: Test suite for the SEER Survival Prediction task.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from pyhealth.tasks.seer_survival_prediction import SEERSurvivalPrediction

class TestSEERSurvivalPrediction(unittest.TestCase):
    
    def setUp(self):
        """Initializes a fresh task and a valid mocked patient for each test."""
        self.task = SEERSurvivalPrediction()

        # Create synthetic  event 
        self.mock_event = MagicMock()
        self.mock_event.attr_dict = {
            "label": 1,
            "age": 55,
            "year_dx": 2005,
            "race_White": 1,
            "stage_Localized": 1,
        }

        # Attach it to a synthetic  patient
        self.valid_patient = MagicMock()
        self.valid_patient.patient_id = "p1"
        self.valid_patient.get_events.return_value = [self.mock_event]


    def test_seer_task_generates_samples(self) -> None:
        """Test that the SEER task generates valid samples."""
        samples = self.task(self.valid_patient)

        self.assertEqual(len(samples), 1)

        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["visit_id"], "p1_seer")
        self.assertIn("features", sample)
        self.assertIn("label", sample)


    def test_seer_task_feature_extraction(self) -> None:
        """Test that task extracts features with the correct dimension."""
        samples = self.task(self.valid_patient)
        features = samples[0]["features"]

        # label is excluded, leaving exactly 4 feature columns in our mock
        self.assertEqual(features.shape[0], 4)
        self.assertIsInstance(features, np.ndarray)


    def test_seer_task_label_generation(self) -> None:
        """Test that labels are preserved as binary outputs."""
        samples = self.task(self.valid_patient)
        self.assertEqual(samples[0]["label"], 1)


    def test_seer_task_feature_names_saved(self) -> None:
        """Test that feature names are saved consistently."""
        self.task(self.valid_patient)

        self.assertIsNotNone(self.task.feature_names)
        self.assertIn("age", self.task.feature_names)
        self.assertIn("year_dx", self.task.feature_names)
        self.assertNotIn("label", self.task.feature_names)


    def test_seer_task_invalid_label_raises(self) -> None:
        """Test that a non-binary label raises a ValueError."""
        # Corrupt the label in memory
        self.mock_event.attr_dict["label"] = 2

        with self.assertRaisesRegex(ValueError, "Label must be binary 0/1"):
            self.task(self.valid_patient)


    def test_seer_task_non_numeric_feature_raises(self) -> None:
        """Test that a non-numeric feature raises a ValueError."""
        # Corrupt a feature in memory
        self.mock_event.attr_dict["age"] = "bad_data"

        with self.assertRaisesRegex(ValueError, "Feature column"):
            self.task(self.valid_patient)


    def test_seer_task_empty_visit_handling(self) -> None:
        """Test that empty visits/no events are handled gracefully (Edge Case)."""
        # Create a patient with no events
        empty_patient = MagicMock()
        empty_patient.patient_id = "p_empty"
        empty_patient.get_events.return_value = []

        samples = self.task(empty_patient)
        self.assertEqual(len(samples), 0)


if __name__ == '__main__':
    unittest.main()