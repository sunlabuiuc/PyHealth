import unittest
import numpy as np
import os
import sys
from pathlib import Path
import torch
import pandas as pd
from unittest.mock import patch
import yaml
import polars as pl

# Add root directory to path
current = Path(__file__).parent
repo_root = current.parent.parent.parent
sys.path.append(str(repo_root))

from pyhealth.datasets.dreamt_e4 import DREAMTE4Dataset
from pyhealth.tasks.dreamt_sleeping_stage_classification import DREAMTE4SleepingStageClassification


class TestDREAMTE4Dataset(unittest.TestCase):
    """Test suite for DREAMTE4Dataset validation."""
    
    # Dataset configuration
    # ROOT = "pyhealth/unittests/test_datasets/dreamt_e4_test_data"  # Test data subset
    EXPECTED_SAMPLE_COUNT = 800
    EXPECTED_INPUT_SCHEMA = {"features": "NumericToTensor"}
    EXPECTED_OUTPUT_SCHEMA = {"label": "binary"}

    
    @classmethod
    def setUpClass(cls):
        """Initialize dataset once for all tests."""
        # 1) Load YAML to get exact attribute list
        cfg_path = Path(__file__).parent.parent.parent / "datasets" / "configs" / "dreamt_e4.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        attributes = cfg["tables"]["dreamt_features"]["attributes"]

        # 2) Generate dummy pandas DataFrame
        def generate_dummy_df(n_sids=80, recs_per_sid=10, seed=42):
            np.random.seed(seed)
            rows = []
            for pid in range(1, n_sids + 1):
                sid = f"S{pid:03d}"
                bmi = float(np.clip(np.random.normal(25,5), 15, 50))
                obesity = 1.0 if bmi >= 30 else 0.0
                circ_vals = np.random.rand(3)
                for rec in range(recs_per_sid):
                    row = {}
                    for col in attributes:
                        if col == "sid":
                            row[col] = sid
                        elif col == "Sleep_Stage":
                            row[col] = float(np.random.randint(0,1))
                        elif col in {"Central_Apnea","Obstructive_Apnea","Multiple_Events","Hypopnea","artifact"}:
                            row[col] = float(np.random.binomial(1,0.1))
                        elif col == "AHI_Severity":
                            row[col] = float(np.random.uniform(0,10))
                        elif col == "Obesity":
                            row[col] = obesity
                        elif col == "BMI":
                            row[col] = bmi
                        elif col in {"circadian_decay","circadian_linear","circadian_cosine"}:
                            idx = {"circadian_decay":0,"circadian_linear":1,"circadian_cosine":2}[col]
                            row[col] = float(circ_vals[idx])
                        elif col == "timestamp_start":
                            # ISO string so Polars infers Utf8
                            row[col] = f"2025-01-19T00:00:{rec:02d}"
                        else:
                            # domain/rolling/gaussian features
                            row[col] = float(np.random.rand())
                    rows.append(row)
            return pd.DataFrame(rows, columns=attributes)

        cls.dummy_pd = generate_dummy_df()
        cls.dummy_pl_lazy = pl.from_pandas(cls.dummy_pd).lazy()
        cls.dummy_pl_eager = pl.from_pandas(cls.dummy_pd)

        # 3) Patch file I/O calls
        cls.patch_exists = patch.object(os.path, "exists", return_value=True)
        cls.patch_pd_csv = patch("pandas.read_csv", return_value=cls.dummy_pd)
        cls.patch_pl_read = patch("polars.read_csv",  lambda *a,**k: cls.dummy_pl_eager)
        cls.patch_pl_scan = patch("polars.scan_csv",  lambda *a,**k: cls.dummy_pl_lazy)
        cls.patch_exists.start()
        cls.patch_pd_csv.start()
        cls.patch_pl_read.start()
        cls.patch_pl_scan.start()

        # 4) Instantiate dataset and generate samples
        cls.dataset = DREAMTE4Dataset(root="does_not_matter")
        cls.task = DREAMTE4SleepingStageClassification()
        cls.samples = cls.dataset.set_task(cls.task)


    def test_sample_count(self):
        """Verify the dataset contains expected number of samples."""
        self.assertEqual(len(self.samples), self.EXPECTED_SAMPLE_COUNT)

    def test_schemas(self):
        """Verify input and output schemas match expectations."""
        self.assertEqual(self.samples.input_schema, self.EXPECTED_INPUT_SCHEMA)
        self.assertEqual(self.samples.output_schema, self.EXPECTED_OUTPUT_SCHEMA)

    def test_sample_structure(self):
        """Validate structure of individual samples."""
        sample = self.samples[0]
        
        # Check required keys exist
        self.assertIn("patient_id", sample)
        self.assertIn("features", sample)
        self.assertIn("label", sample)
        
        # Validate feature array
        self.assertIsInstance(sample["features"], torch.Tensor)
        self.assertEqual(len(sample["features"].shape), 2)  # Should be 2D
        self.assertEqual(sample["features"].shape, (1, 358)) # 358 features in yaml
        
        # Validate label
        self.assertIsInstance(sample["label"], torch.Tensor)
        self.assertEqual(len(sample["label"].shape), 1) # binary label

    def test_feature_consistency(self):
        """Check all samples have consistent feature dimensions."""
        first_sample_shape = self.samples[0]["features"].shape[1]  # Get feature dim
        
        for sample in self.samples[1:100]:  # Check first 100 samples
            self.assertEqual(
                sample["features"].shape[1], 
                first_sample_shape,
                msg="Inconsistent feature dimensions across samples"
            )


    def test_patient_consistency(self):
        """Check patient IDs match expected format."""
        patient_ids = {sample["patient_id"] for sample in self.samples}
        
        # Verify patient IDs are strings and match expected pattern
        for pid in patient_ids:
            self.assertIsInstance(pid, str)

    

if __name__ == "__main__":
    unittest.main(verbosity=2)