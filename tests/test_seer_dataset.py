"""
Contributor: Adrianne Sun, Ruoyi Xie
NetID: ajsun2, ruoyix2
Paper Title: Reproducible Survival Prediction with SEER Cancer Data
Paper Link: https://proceedings.mlr.press/v85/hegselmann18a/hegselmann18a.pdf
Description: Test suite for the SEER dataset.
"""
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from pyhealth.datasets import SEERDataset

np.random.seed(42)
torch.manual_seed(42)

class TestSEERDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary directory 
        self.test_dir = tempfile.mkdtemp()
        processed_dir = Path(self.test_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Generate CSV with minimal synthetic data
        df = pd.DataFrame({
            "patient_id": ["seer_0", "seer_1", "seer_2"],
            "visit_id": ["v_0", "v_1", "v_2"],
            "event_time": ["2005-01-01", "2006-01-01", "2007-01-01"],
            "age": [55, 63, 47],
            "year_dx": [2005, 2006, 2007],
            "label": [1, 0, 1]
        })

        base_dir = Path(__file__).resolve().parent.parent
        yaml_path = base_dir / "pyhealth" / "datasets" / "configs" / "seer.yaml"

        with open(yaml_path, "r", encoding="utf-8") as f:
            required_cols = [
                line.strip().strip('- "') 
                for line in f if line.strip().startswith("-")
            ]
        df = df.reindex(columns=df.columns.union(required_cols), fill_value=0)

        df.to_csv(processed_dir / "seer_pyhealth.csv", index=False)

        self.dataset = SEERDataset(root=self.test_dir, tables=["seer"], dev=True)

    def test_dataset_functionality(self):
        # Test that data loads correctly
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

        # Test that parsing produces expected format
        patient = self.dataset.get_patient("seer_1")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "seer_1")
        self.assertTrue(len(patient.get_events()) > 0)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()