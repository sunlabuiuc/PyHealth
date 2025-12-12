import unittest
import os
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.length_of_stay_prediction import LengthOfStayPredictionMIMIC3


class TestMIMIC3LengthOfStayPrediction(unittest.TestCase):
    """Test MIMIC-3 length of stay prediction task with demo data from local test resources."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        self._setup_dataset_path()
        self._load_dataset()

    def _setup_dataset_path(self):
        """Get path to local MIMIC-III demo dataset in test resources."""
        # Get the path to the test-resources/core/mimic3demo directory
        test_dir = Path(__file__).parent.parent
        self.demo_dataset_path = str(
            test_dir / "test-resources" / "core" / "mimic3demo"
        )

        print(f"\n{'='*60}")
        print(f"Setting up MIMIC-III demo dataset for length of stay prediction")
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
        tables = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"]
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

    def test_length_of_stay_prediction_mimic3_set_task(self):
        """Test LengthOfStayPredictionMIMIC3 task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_length_of_stay_prediction_mimic3_set_task()")
        print(f"{'='*60}")

        print("Initializing LengthOfStayPredictionMIMIC3 task...")
        task = LengthOfStayPredictionMIMIC3()

        # Test that task is properly initialized
        print(f"✓ Task initialized: {task.task_name}")
        self.assertEqual(task.task_name, "LengthOfStayPredictionMIMIC3")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("los", task.output_schema)
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
            print(f"✓ Generated {num_samples} length of stay prediction samples")

            # Test sample structure
            if num_samples > 0:
                sample = sample_dataset.samples[0]
                required_keys = [
                    "visit_id",
                    "patient_id",
                    "conditions",
                    "procedures",
                    "drugs",
                    "los",
                ]

                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")

                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")
                    if key in ["conditions", "procedures", "drugs"]:
                        print(f"  - {key}: {len(sample[key])} items")
                    else:
                        print(f"  - {key}: {sample[key]}")

                # Verify los label is in valid range (0-9 for 10 categories)
                self.assertIn(
                    sample["los"],
                    list(range(10)),
                    "Length of stay category should be 0-9",
                )

                # Count LOS distribution
                los_counts = {i: 0 for i in range(10)}
                for s in sample_dataset.samples:
                    los_counts[s["los"]] += 1

                print(f"\nLength of stay category distribution:")
                category_labels = [
                    "< 1 day",
                    "1 day",
                    "2 days",
                    "3 days",
                    "4 days",
                    "5 days",
                    "6 days",
                    "7 days",
                    "1-2 weeks",
                    "> 2 weeks",
                ]
                for i in range(10):
                    if los_counts[i] > 0:
                        pct = los_counts[i] / num_samples * 100
                        print(
                            f"  Category {i} ({category_labels[i]}): {los_counts[i]} ({pct:.1f}%)"
                        )

                print(
                    f"\n✓ test_length_of_stay_prediction_mimic3_set_task() passed successfully"
                )

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback

            traceback.print_exc()
            self.fail(f"Failed to use set_task with LengthOfStayPredictionMIMIC3: {e}")


if __name__ == "__main__":
    unittest.main()
