import unittest
import os
from pathlib import Path

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.dka import DKAPredictionMIMIC4


class TestMIMIC4DKAPrediction(unittest.TestCase):
    """Test MIMIC-4 DKA prediction task with demo data from local test resources."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        self._setup_dataset_path()
        self._load_dataset()

    def _setup_dataset_path(self):
        """Get path to local MIMIC-IV demo dataset in test resources."""
        # Get the path to the test-resources/core/mimic4demo directory
        test_dir = Path(__file__).parent.parent
        self.demo_dataset_path = str(
            test_dir / "test-resources" / "core" / "mimic4demo"
        )

        print(f"\n{'='*60}")
        print(f"Setting up MIMIC-IV demo dataset for DKA prediction")
        print(f"Dataset path: {self.demo_dataset_path}")

        # Verify the dataset exists
        if not os.path.exists(self.demo_dataset_path):
            raise unittest.SkipTest(
                f"MIMIC-IV demo dataset not found at {self.demo_dataset_path}"
            )

        # List files in the hosp directory
        hosp_path = os.path.join(self.demo_dataset_path, "hosp")
        if os.path.exists(hosp_path):
            files = os.listdir(hosp_path)
            print(f"Found {len(files)} files in hosp directory:")
            for f in sorted(files):
                file_path = os.path.join(hosp_path, f)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {f} ({size:.1f} KB)")
        print(f"{'='*60}\n")

    def _load_dataset(self):
        """Load the dataset for testing."""
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"]
        print(f"Loading MIMIC4Dataset with tables: {tables}")
        self.dataset = MIMIC4Dataset(root=self.demo_dataset_path, tables=tables)
        print(f"✓ Dataset loaded successfully")
        print(f"  Total patients: {len(self.dataset.patients)}")
        print()

    def test_dataset_stats(self):
        """Test that the dataset loads correctly and stats() works."""
        print(f"\n{'='*60}")
        print("TEST: test_dataset_stats()")
        print(f"{'='*60}")
        try:
            print("Calling dataset.stats()...")
            self.dataset.stats()
            print("✓ dataset.stats() executed successfully")
        except Exception as e:
            print(f"✗ dataset.stats() failed with error: {e}")
            self.fail(f"dataset.stats() failed: {e}")

    def test_dka_prediction_task_initialization(self):
        """Test DKAPredictionMIMIC4 task initialization."""
        print(f"\n{'='*60}")
        print("TEST: test_dka_prediction_task_initialization()")
        print(f"{'='*60}")

        # Test default initialization
        print("Testing default initialization...")
        task = DKAPredictionMIMIC4()
        self.assertEqual(task.task_name, "DKAPredictionMIMIC4")
        self.assertEqual(task.dka_window_days, 90)
        self.assertEqual(task.padding, 0)
        self.assertIn("diagnoses", task.input_schema)
        self.assertIn("labs", task.input_schema)
        self.assertIn("label", task.output_schema)
        print(f"✓ Default task initialized: {task.task_name}")
        print(f"  DKA window: {task.dka_window_days} days")
        print(f"  Input schema: {list(task.input_schema.keys())}")
        print(f"  Output schema: {list(task.output_schema.keys())}")

        # Test custom initialization
        print("\nTesting custom initialization...")
        custom_task = DKAPredictionMIMIC4(dka_window_days=30, padding=5)
        self.assertEqual(custom_task.dka_window_days, 30)
        self.assertEqual(custom_task.padding, 5)
        print(f"✓ Custom task initialized with window={custom_task.dka_window_days}, padding={custom_task.padding}")

    def test_dka_prediction_class_variables(self):
        """Test that class variables are properly defined."""
        print(f"\n{'='*60}")
        print("TEST: test_dka_prediction_class_variables()")
        print(f"{'='*60}")

        # Test T1DM codes
        self.assertEqual(DKAPredictionMIMIC4.T1DM_ICD10_PREFIX, "E10")
        self.assertIn("25001", DKAPredictionMIMIC4.T1DM_ICD9_CODES)
        print(f"✓ T1DM ICD-10 prefix: {DKAPredictionMIMIC4.T1DM_ICD10_PREFIX}")
        print(f"✓ T1DM ICD-9 codes: {len(DKAPredictionMIMIC4.T1DM_ICD9_CODES)} codes")

        # Test DKA codes
        self.assertEqual(DKAPredictionMIMIC4.DKA_ICD10_PREFIX, "E101")
        self.assertIn("25011", DKAPredictionMIMIC4.DKA_ICD9_CODES)
        print(f"✓ DKA ICD-10 prefix: {DKAPredictionMIMIC4.DKA_ICD10_PREFIX}")
        print(f"✓ DKA ICD-9 codes: {len(DKAPredictionMIMIC4.DKA_ICD9_CODES)} codes")

        # Test lab categories
        self.assertEqual(len(DKAPredictionMIMIC4.LAB_CATEGORY_ORDER), 6)
        expected_categories = ["glucose", "bicarbonate", "anion_gap", "potassium", "sodium", "chloride"]
        self.assertEqual(DKAPredictionMIMIC4.LAB_CATEGORY_ORDER, expected_categories)
        print(f"✓ Lab categories: {DKAPredictionMIMIC4.LAB_CATEGORY_ORDER}")
        print(f"✓ Total lab item IDs: {len(DKAPredictionMIMIC4.ALL_LAB_ITEMIDS)}")

    def test_dka_prediction_set_task(self):
        """Test DKAPredictionMIMIC4 task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_dka_prediction_set_task()")
        print(f"{'='*60}")

        print("Initializing DKAPredictionMIMIC4 task...")
        task = DKAPredictionMIMIC4()

        # Test using set_task method
        try:
            print("\nCalling dataset.set_task()...")
            sample_dataset = self.dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            self.assertTrue(
                hasattr(sample_dataset, "samples"), "Sample dataset should have samples"
            )
            print(f"✓ set_task() completed")

            # Check sample count
            num_samples = len(sample_dataset.samples)
            print(f"✓ Generated {num_samples} DKA prediction samples")

            if num_samples > 0:
                sample = sample_dataset.samples[0]
                required_keys = ["patient_id", "record_id", "diagnoses", "labs", "label"]

                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")

                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")

                # Verify diagnoses format (tuple of times and sequences)
                diagnoses = sample["diagnoses"]
                self.assertIsInstance(diagnoses, tuple, "diagnoses should be a tuple")
                self.assertEqual(len(diagnoses), 2, "diagnoses tuple should have 2 elements")
                print(f"  - diagnoses: {len(diagnoses[1])} admission(s)")

                # Verify labs format (tuple of times and sequences)
                labs = sample["labs"]
                self.assertIsInstance(labs, tuple, "labs should be a tuple")
                self.assertEqual(len(labs), 2, "labs tuple should have 2 elements")
                print(f"  - labs: {len(labs[1])} lab vector(s)")

                # Verify label is binary
                self.assertIn(sample["label"], [0, 1], "Label should be binary (0 or 1)")
                print(f"  - label: {sample['label']}")

                # Count label distribution
                label_counts = {0: 0, 1: 0}
                for s in sample_dataset.samples:
                    label_counts[s["label"]] += 1

                print(f"\nLabel distribution:")
                print(f"  - No DKA (0): {label_counts[0]} ({label_counts[0]/num_samples*100:.1f}%)")
                print(f"  - Has DKA (1): {label_counts[1]} ({label_counts[1]/num_samples*100:.1f}%)")

            print(f"\n✓ test_dka_prediction_set_task() passed successfully")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed to use set_task with DKAPredictionMIMIC4: {e}")

    def test_dka_prediction_helper_methods(self):
        """Test helper methods of DKAPredictionMIMIC4."""
        print(f"\n{'='*60}")
        print("TEST: test_dka_prediction_helper_methods()")
        print(f"{'='*60}")

        task = DKAPredictionMIMIC4()

        # Test _normalize_icd
        print("Testing _normalize_icd()...")
        self.assertEqual(task._normalize_icd("E10.10"), "E1010")
        self.assertEqual(task._normalize_icd("e10.10"), "E1010")
        self.assertEqual(task._normalize_icd(None), "")
        self.assertEqual(task._normalize_icd("  25001  "), "25001")
        print("✓ _normalize_icd() works correctly")

        # Test _is_t1dm_code
        print("\nTesting _is_t1dm_code()...")
        self.assertTrue(task._is_t1dm_code("E10.10", 10))
        self.assertTrue(task._is_t1dm_code("E1010", "10"))
        self.assertTrue(task._is_t1dm_code("25001", 9))
        self.assertTrue(task._is_t1dm_code("25001", "9"))
        self.assertFalse(task._is_t1dm_code("E11.0", 10))  # Type 2
        self.assertFalse(task._is_t1dm_code("25000", 9))  # Type 2
        self.assertFalse(task._is_t1dm_code(None, 10))
        print("✓ _is_t1dm_code() works correctly")

        # Test _is_dka_code
        print("\nTesting _is_dka_code()...")
        self.assertTrue(task._is_dka_code("E10.10", 10))
        self.assertTrue(task._is_dka_code("E1011", "10"))
        self.assertTrue(task._is_dka_code("25011", 9))
        self.assertTrue(task._is_dka_code("25013", "9"))
        self.assertFalse(task._is_dka_code("E10.65", 10))  # Not DKA
        self.assertFalse(task._is_dka_code(None, 10))
        print("✓ _is_dka_code() works correctly")

        # Test _deduplicate_preserve_order
        print("\nTesting _deduplicate_preserve_order()...")
        self.assertEqual(
            task._deduplicate_preserve_order(["A", "B", "A", "C", "B"]),
            ["A", "B", "C"]
        )
        self.assertEqual(
            task._deduplicate_preserve_order([]),
            []
        )
        print("✓ _deduplicate_preserve_order() works correctly")

        print(f"\n✓ test_dka_prediction_helper_methods() passed successfully")


if __name__ == "__main__":
    unittest.main()

