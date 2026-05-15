"""
Contributor: Adrianne Sun, Ruoyi Xie
NetID: ajsun2, ruoyix2
Paper Title: Reproducible Survival Prediction with SEER Cancer Data
Paper Link: https://proceedings.mlr.press/v85/hegselmann18a/hegselmann18a.pdf
Description: Test suite for the SEER dataset using synthetic data only.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import SEERDataset

np.random.seed(42)
torch.manual_seed(42)


class TestSEERDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a tiny synthetic SEER dataset once for all tests."""
        cls.test_dir = tempfile.mkdtemp()
        processed_dir = Path(cls.test_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # 3 synthetic patients, with patient seer_1 having 2 visits/events
        df = pd.DataFrame({
            "patient_id": ["seer_0", "seer_1", "seer_1", "seer_2"],
            "visit_id": ["v_0", "v_1", "v_2", "v_3"],
            "event_time": ["2005-01-01", "2006-01-01", "2006-06-01", "2007-01-01"],
            "age": [55, 63, 63, 47],
            "year_dx": [2005, 2006, 2006, 2007],
            "label": [1, 0, 0, 1],
        })

        cls.raw_df = df.copy()

        # Read required SEER columns from config and fill missing ones with zeros
        base_dir = Path(__file__).resolve().parent.parent
        yaml_path = base_dir / "pyhealth" / "datasets" / "configs" / "seer.yaml"

        with open(yaml_path, "r", encoding="utf-8") as f:
            required_cols = [
                line.strip().strip('- "')
                for line in f
                if line.strip().startswith("-")
            ]

        df = df.reindex(columns=df.columns.union(required_cols), fill_value=0)
        cls.df = df

        df.to_csv(processed_dir / "seer_pyhealth.csv", index=False)

        cls.dataset = SEERDataset(root=cls.test_dir, tables=["seer"], dev=True)

    def test_dataset_loads_expected_number_of_patients(self):
        """Test data loading with a tiny synthetic dataset."""
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_patient_parsing(self):
        """Test that a patient can be parsed correctly."""
        patient = self.dataset.get_patient("seer_1")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "seer_1")

    def test_event_parsing(self):
        """Test that visits/events are attached to the correct patient."""
        patient = self.dataset.get_patient("seer_1")
        events = patient.get_events()
        self.assertEqual(len(events), 2)

    def test_data_integrity(self):
        """Test that key synthetic fields are preserved correctly."""
        self.assertIn("patient_id", self.df.columns)
        self.assertIn("visit_id", self.df.columns)
        self.assertIn("event_time", self.df.columns)
        self.assertIn("age", self.df.columns)
        self.assertIn("year_dx", self.df.columns)
        self.assertIn("label", self.df.columns)

        self.assertEqual(set(self.raw_df["patient_id"]), {"seer_0", "seer_1", "seer_2"})
        self.assertTrue((self.raw_df["age"] > 0).all())
        self.assertTrue((self.raw_df["year_dx"] >= 2005).all())

    @classmethod
    def tearDownClass(cls):
        """Remove temporary files after tests complete."""
        shutil.rmtree(cls.test_dir)


if __name__ == "__main__":
    unittest.main()