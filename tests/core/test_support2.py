import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from pyhealth.datasets import Support2Dataset


class TestSupport2Dataset(unittest.TestCase):
    """Test Support2Dataset with synthetic test data."""

    def setUp(self):
        """Set up test data files and directory structure"""
        # Use test-resources for actual test data
        test_dir = Path(__file__).parent.parent.parent
        self.test_data_path = test_dir / "test-resources" / "core" / "support2"
        
        # Check if test data exists, if not create synthetic data
        if not self.test_data_path.exists() or not (self.test_data_path / "support2.csv").exists():
            # Fallback: create temporary directory with synthetic data
            self.temp_dir = tempfile.mkdtemp()
            self.root = Path(self.temp_dir)
            self.use_temp = True
        else:
            self.root = self.test_data_path
            self.use_temp = False

        # Create minimal synthetic support2.csv with 3 patients (only if using temp)
        if self.use_temp:
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
        """Clean up temporary files"""
        if hasattr(self, 'use_temp') and self.use_temp:
            shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test Support2Dataset initialization"""
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"]
        )
        self.assertIsInstance(dataset, Support2Dataset)
        self.assertEqual(dataset.dataset_name, "support2")

    def test_load_data(self):
        """Test that data loads correctly"""
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"]
        )
        self.assertIsNotNone(dataset.global_event_df)

    def test_patient_count(self):
        """Test that we have the expected number of patients"""
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"]
        )
        unique_patients = dataset.unique_patient_ids
        # Should have patients (3 for synthetic, 9105 for real dataset)
        self.assertGreater(len(unique_patients), 0)

    def test_get_patient(self):
        """Test getting a single patient"""
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"]
        )
        patient = dataset.get_patient("1")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "1")

    def test_stats(self):
        """Test stats method"""
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"]
        )
        # Should not raise an error
        dataset.stats()


if __name__ == "__main__":
    unittest.main()

