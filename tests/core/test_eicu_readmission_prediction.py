import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import ReadmissionPredictionEICU


class TesteICUReadmissionPrediction(unittest.TestCase):
    """Test eICU readmission prediction task with demo data."""

    @classmethod
    def setUpClass(cls):
        """Set up demo dataset path for each test."""
        test_dir = Path(__file__).parent.parent.parent
        demo_dataset_path = str(test_dir / "test-resources" / "core" / "eicudemo")

        print(f"\n{'='*60}")
        print(f"Setting up eICU demo dataset for readmission prediction")
        print(f"Dataset path: {demo_dataset_path}")
        print(f"{'='*60}\n")

        # Load dataset with required tables
        tables = ["diagnosis", "medication", "physicalexam"]
        print(f"Loading eICUDataset with tables: {tables}")
        cls.dataset = eICUDataset(
            root=demo_dataset_path,
            tables=tables,
            cache_dir=tempfile.TemporaryDirectory().name,
        )
        print("✓ Dataset loaded successfully")

    def test_readmission_prediction_eicu_set_task(self):
        """Test ReadmissionPredictionEICU task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_readmission_prediction_eicu_set_task()")
        print(f"{'='*60}")

        # Initialize task
        print("\nInitializing ReadmissionPredictionEICU task...")
        task = ReadmissionPredictionEICU()

        # Verify task schema
        self.assertEqual(task.task_name, "ReadmissionPredictionEICU")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("readmission", task.output_schema)
        print(f"✓ Task initialized: {task.task_name}")
        print(f"  Input schema: {list(task.input_schema.keys())}")
        print(f"  Output schema: {list(task.output_schema.keys())}")
        print(f"  Exclude minors: {task.exclude_minors}")

        # Apply task
        try:
            print("\nCalling dataset.set_task()...")
            sample_dataset = self.dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            print(f"✓ set_task() completed")

            # Check samples
            num_samples = len(sample_dataset)
            print(f"✓ Generated {num_samples} readmission prediction samples")

            if num_samples > 0:
                sample = sample_dataset[0]
                required_keys = ["visit_id", "patient_id", "conditions", "procedures", "drugs", "readmission"]

                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")

                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")

                # Verify readmission is binary
                readmission = sample["readmission"]
                self.assertIn(int(readmission.item()) if hasattr(readmission, 'item') else int(readmission), [0, 1])

                # Count readmission distribution
                readmission_counts = {0: 0, 1: 0}
                for s in sample_dataset:
                    label = int(s["readmission"].item()) if hasattr(s["readmission"], 'item') else int(s["readmission"])
                    readmission_counts[label] += 1

                print(f"\nReadmission distribution:")
                print(f"  No readmission (0): {readmission_counts[0]}")
                print(f"  Readmission (1): {readmission_counts[1]}")

                print(f"\n✓ test_readmission_prediction_eicu_set_task() passed")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed: {e}")

    def test_admissions_of_minors_are_excluded(self):
        """Test the ReadmissionPredictionEICU task exclude_minors param."""
        print(f"\n{'='*60}")
        print("TEST: test_admissions_of_minors_are_excluded()")
        print(f"{'='*60}")

        patients = [ s["patient_id"] for s in self.dataset.set_task(ReadmissionPredictionEICU()) ]
        self.assertNotIn("035-10434", patients)

        patients = [ s["patient_id"] for s in self.dataset.set_task(ReadmissionPredictionEICU(exclude_minors=False)) ]
        self.assertIn("035-10434", patients)


if __name__ == "__main__":
    unittest.main()
