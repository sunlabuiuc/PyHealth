import unittest
import os
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.mortality_prediction import (
    MortalityPredictionMIMIC3,
    MultimodalMortalityPredictionMIMIC3,
)


class TestMIMIC3MortalityPrediction(unittest.TestCase):
    """Test MIMIC-3 mortality prediction tasks with demo data from local test resources."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        self._setup_dataset_path()
        self._load_dataset()

    def _setup_dataset_path(self):
        """Get path to local MIMIC-III demo dataset in test resources."""
        # Get the path to the test-resources/core/mimic3demo directory
        test_dir = Path(__file__).parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "mimic3demo")
        
        print(f"\n{'='*60}")
        print(f"Setting up MIMIC-III demo dataset for mortality prediction")
        print(f"Dataset path: {self.demo_dataset_path}")
        
        # Verify the dataset exists
        if not os.path.exists(self.demo_dataset_path):
            raise unittest.SkipTest(
                f"MIMIC-III demo dataset not found at {self.demo_dataset_path}"
            )
        
        # List files in the dataset directory
        files = os.listdir(self.demo_dataset_path)
        print(f"Found {len(files)} files in dataset directory:")
        for f in sorted(files):
            file_path = os.path.join(self.demo_dataset_path, f)
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")
        print(f"{'='*60}\n")

    def _load_dataset(self):
        """Load the dataset for testing."""
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions"]
        print(f"Loading MIMIC3Dataset with tables: {tables}")
        self.dataset = MIMIC3Dataset(root=self.demo_dataset_path, tables=tables)
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

    def test_mortality_prediction_mimic3_set_task(self):
        """Test MortalityPredictionMIMIC3 task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_mortality_prediction_mimic3_set_task()")
        print(f"{'='*60}")
        
        print("Initializing MortalityPredictionMIMIC3 task...")
        task = MortalityPredictionMIMIC3()

        # Test that task is properly initialized
        print(f"✓ Task initialized: {task.task_name}")
        self.assertEqual(task.task_name, "MortalityPredictionMIMIC3")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("mortality", task.output_schema)
        print(f"  Input schema: {list(task.input_schema.keys())}")
        print(f"  Output schema: {list(task.output_schema.keys())}")

        # Test using set_task method
        try:
            print("\nCalling dataset.set_task()...")
            sample_dataset = self.dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            self.assertTrue(
                hasattr(sample_dataset, "samples"), "Sample dataset should have samples"
            )
            print(f"✓ set_task() completed")

            # Verify we got some samples
            num_samples = len(sample_dataset.samples)
            self.assertGreater(num_samples, 0, "Should generate at least one sample")
            print(f"✓ Generated {num_samples} mortality prediction samples")

            # Test sample structure
            if num_samples > 0:
                sample = sample_dataset.samples[0]
                required_keys = [
                    "hadm_id",
                    "patient_id",
                    "conditions",
                    "procedures",
                    "drugs",
                    "mortality",
                ]
                
                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")
                
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")
                    if key in ["conditions", "procedures", "drugs"]:
                        print(f"  - {key}: {len(sample[key])} items")
                    else:
                        print(f"  - {key}: {sample[key]}")

                # Verify mortality label is binary (0 or 1)
                self.assertIn(
                    sample["mortality"], [0, 1], "Mortality label should be 0 or 1"
                )
                
                # Count mortality distribution
                mortality_counts = {0: 0, 1: 0}
                for s in sample_dataset.samples:
                    mortality_counts[s["mortality"]] += 1
                print(f"\nMortality label distribution:")
                print(f"  Survived (0): {mortality_counts[0]} ({mortality_counts[0]/num_samples*100:.1f}%)")
                print(f"  Died (1): {mortality_counts[1]} ({mortality_counts[1]/num_samples*100:.1f}%)")
                
                print(f"\n✓ test_mortality_prediction_mimic3_set_task() passed successfully")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed to use set_task with MortalityPredictionMIMIC3: {e}")

    @unittest.skip("Skipping multimodal test - noteevents not included in test resources")
    def test_multimodal_mortality_prediction_mimic3_set_task(self):
        """Test MultimodalMortalityPredictionMIMIC3 task with set_task() method."""
        task = MultimodalMortalityPredictionMIMIC3()

        # Test that task is properly initialized
        self.assertEqual(task.task_name, "MultimodalMortalityPredictionMIMIC3")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("clinical_notes", task.input_schema)
        self.assertIn("mortality", task.output_schema)

        # Test using set_task method
        try:
            sample_dataset = self.dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            self.assertTrue(
                hasattr(sample_dataset, "samples"), "Sample dataset should have samples"
            )

            # Verify we got some samples
            self.assertGreater(
                len(sample_dataset.samples), 0, "Should generate at least one sample"
            )

            # Test sample structure
            if len(sample_dataset.samples) > 0:
                sample = sample_dataset.samples[0]
                required_keys = [
                    "hadm_id",
                    "patient_id",
                    "conditions",
                    "procedures",
                    "drugs",
                    "clinical_notes",
                    "mortality",
                ]
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")

                # Verify data types
                self.assertIsInstance(
                    sample["clinical_notes"], str, "clinical_notes should be a string"
                )
                self.assertIn(
                    sample["mortality"], [0, 1], "Mortality label should be 0 or 1"
                )

                print(f"Generated {len(sample_dataset.samples)} multimodal samples")
                print(f"Clinical notes length: {len(sample['clinical_notes'])}")

        except Exception as e:
            self.fail(
                f"Failed to use set_task with MultimodalMortalityPredictionMIMIC3: {e}"
            )


if __name__ == "__main__":
    unittest.main()
