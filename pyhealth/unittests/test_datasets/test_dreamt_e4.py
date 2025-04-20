import unittest
import numpy as np
import os
import sys
from pathlib import Path
import torch

# Add root directory to path
current = Path(__file__).parent
repo_root = current.parent.parent.parent
sys.path.append(str(repo_root))

from pyhealth.datasets.dreamt_e4 import DREAMTE4Dataset
from pyhealth.tasks.dreamt_sleeping_stage_classification import DREAMTE4SleepingStageClassification


class TestDREAMTE4Dataset(unittest.TestCase):
    """Test suite for DREAMTE4Dataset validation."""
    
    # Dataset configuration
    ROOT = "pyhealth/unittests/test_datasets/dreamt_e4_test_data"  # Test data subset
    EXPECTED_SAMPLE_COUNT = 800
    EXPECTED_INPUT_SCHEMA = {"features": "NumericToTensor"}
    EXPECTED_OUTPUT_SCHEMA = {"label": "binary"}


    @classmethod
    def setUpClass(cls):
        """Initialize dataset once for all tests."""
        cls.dataset = DREAMTE4Dataset(root=cls.ROOT)
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