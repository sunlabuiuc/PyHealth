import unittest
import tempfile
import shutil
import io
import sys
from pathlib import Path
import pandas as pd
import torch

from pyhealth.datasets import Support2Dataset
from pyhealth.tasks import SurvivalPreprocessSupport2


class TestSupport2Dataset(unittest.TestCase):
    """Test Support2Dataset with synthetic test data."""

    def setUp(self):
        """Set up test data files and directory structure with synthetic data."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        support2_data = {
            'sno': ['1', '2', '3'],
            'age': [62.85, 60.34, 52.75],
            'death': [0, 1, 1],
            'sex': ['male', 'female', 'female'],
            'hospdead': [0, 1, 0],
            'slos': [5, 4, 17],
            'd.time': [2029, 4, 47],
            'dzgroup': ['Lung Cancer', 'Cirrhosis', 'Cirrhosis'],
            'dzclass': ['Cancer', 'COPD/CHF/Cirrhosis', 'COPD/CHF/Cirrhosis'],
            'num.co': [0, 2, 2],
            'edu': [11, 12, 12],
            'income': ['$11-$25k', '$11-$25k', 'under $11k'],
            'scoma': [0, 44, 0],
            'charges': [9715.0, 34496.0, 41094.0],
            'race': ['other', 'white', 'white'],
            'sps': [33.9, 52.7, 20.5],
            'aps': [20, 74, 45],
            'surv2m': [0.26, 0.001, 0.79],
            'surv6m': [0.037, 0.0, 0.66],
            'hday': [1, 3, 4],
            'diabetes': [0, 0, 0],
            'dementia': [0, 0, 0],
            'ca': ['metastatic', 'no', 'no'],
            'meanbp': [97.0, 43.0, 70.0],
            'wblc': [6.0, 17.1, 8.5],
            'hrt': [69.0, 112.0, 88.0],
            'resp': [22, 34, 28],
            'temp': [36.0, 34.6, 37.4],
            'pafi': [388.0, 98.0, 231.7],
            'alb': [1.8, None, None],
            'bili': [0.2, None, 2.2],
            'crea': [1.2, 5.5, 2.0],
            'sod': [141, 132, 134],
            'ph': [7.46, 7.25, 7.46],
            'glucose': [None, None, None],
            'bun': [None, None, None],
            'urine': [None, None, None],
            'adlp': [7, 1, 1],
            'adls': [7, None, 0],
            'sfdm2': [None, '<2 mo. follow-up', '<2 mo. follow-up'],
            'adlsc': [7.0, 1.0, 0.0],
            'prg2m': [0.5, 0.0, 0.75],
            'prg6m': [0.25, 0.0, 0.5],
            'dnr': ['no dnr', None, 'no dnr'],
            'dnrday': [5, None, 17],
            'totcst': [None, None, None],
            'totmcst': [None, None, None],
            'avtisst': [7.0, 29.0, 13.0]
        }

        support2_df = pd.DataFrame(support2_data)
        support2_df.to_csv(self.root / "support2.csv", index=False)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test Support2Dataset initialization."""
        print("Dataset Tests:")
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        self.assertIsInstance(dataset, Support2Dataset)
        self.assertEqual(dataset.dataset_name, "support2")
        print("Test passed: dataset_initialization\n")

    def test_load_data(self):
        """Test that data loads correctly."""
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        self.assertIsNotNone(dataset.global_event_df)
        print("Test passed: load_data\n")

    def test_patient_count(self):
        """Test that the dataset contains the expected number of patients."""
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        unique_patients = dataset.unique_patient_ids
        self.assertEqual(len(unique_patients), 3)
        print("Test passed: patient_count\n")

    def test_get_patient(self):
        """Test retrieving a single patient by ID."""
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        patient = dataset.get_patient("1")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "1")
        print("Test passed: get_patient\n")

    def test_stats(self):
        """Test that stats method executes without errors."""
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        # Suppress stats output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        dataset.stats()
        sys.stdout = old_stdout
        print("Test passed: stats\n")

    def test_survival_preprocess_2m(self):
        """Test SurvivalPreprocessSupport2 task with 2-month horizon."""
        print("\nTask Tests:")
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        
        task = SurvivalPreprocessSupport2(time_horizon="2m")
        self.assertEqual(task.task_name, "SurvivalPreprocessSupport2_2m")
        self.assertEqual(task.survival_field, "surv2m")
        
        # Test input and output schemas
        self.assertIn("demographics", task.input_schema)
        self.assertIn("disease_codes", task.input_schema)
        self.assertIn("vitals", task.input_schema)
        self.assertIn("labs", task.input_schema)
        self.assertIn("scores", task.input_schema)
        self.assertIn("comorbidities", task.input_schema)
        self.assertIn("survival_probability", task.output_schema)
        self.assertEqual(task.output_schema["survival_probability"], "regression")
        
        sample_dataset = dataset.set_task(task)
        self.assertIsNotNone(sample_dataset)
        self.assertTrue(hasattr(sample_dataset, "samples"))
        self.assertEqual(len(sample_dataset.samples), 3)
        
        # Check first sample structure
        sample = sample_dataset.samples[0]
        required_keys = [
            "patient_id",
            "demographics",
            "disease_codes",
            "vitals",
            "labs",
            "scores",
            "comorbidities",
            "survival_probability",
        ]
        for key in required_keys:
            self.assertIn(key, sample, f"Sample should contain key: {key}")
        
        # Verify survival probabilities are in valid range [0, 1]
        for s in sample_dataset.samples:
            survival_prob = s["survival_probability"]
            self.assertIsInstance(survival_prob, torch.Tensor)
            prob_value = survival_prob.item()
            self.assertGreaterEqual(prob_value, 0.0)
            self.assertLessEqual(prob_value, 1.0)
        
        # Check that features are tensors after processing
        self.assertIsInstance(sample["demographics"], torch.Tensor)
        self.assertIsInstance(sample["disease_codes"], torch.Tensor)
        self.assertIsInstance(sample["vitals"], torch.Tensor)
        self.assertIsInstance(sample["labs"], torch.Tensor)
        self.assertIsInstance(sample["scores"], torch.Tensor)
        self.assertIsInstance(sample["comorbidities"], torch.Tensor)
        
        # Check that tensors are non-empty
        self.assertGreater(len(sample["demographics"]), 0)
        self.assertGreater(len(sample["disease_codes"]), 0)
        print("Test passed: survival_preprocess_2m\n")

    def test_survival_preprocess_6m(self):
        """Test SurvivalPreprocessSupport2 task with 6-month horizon."""
        dataset = Support2Dataset(root=str(self.root), tables=["support2"])
        
        task = SurvivalPreprocessSupport2(time_horizon="6m")
        self.assertEqual(task.task_name, "SurvivalPreprocessSupport2_6m")
        self.assertEqual(task.survival_field, "surv6m")
        
        sample_dataset = dataset.set_task(task)
        self.assertIsNotNone(sample_dataset)
        self.assertEqual(len(sample_dataset.samples), 3)
        
        # Verify all samples have valid survival probabilities
        for s in sample_dataset.samples:
            survival_prob = s["survival_probability"]
            self.assertIsInstance(survival_prob, torch.Tensor)
            prob_value = survival_prob.item()
            self.assertGreaterEqual(prob_value, 0.0)
            self.assertLessEqual(prob_value, 1.0)
        print("Test passed: survival_preprocess_6m\n")

    def test_survival_preprocess_invalid_horizon(self):
        """Test that invalid time_horizon raises ValueError."""
        with self.assertRaises(ValueError):
            SurvivalPreprocessSupport2(time_horizon="12m")
        
        with self.assertRaises(ValueError):
            SurvivalPreprocessSupport2(time_horizon="invalid")
        print("Test passed: survival_preprocess_invalid_horizon\n")


if __name__ == "__main__":
    unittest.main()
