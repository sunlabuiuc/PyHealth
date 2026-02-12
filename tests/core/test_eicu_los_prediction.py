import unittest
import os
from pathlib import Path

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import LengthOfStayPredictioneICU


class TesteICULengthOfStayPrediction(unittest.TestCase):
    """Test eICU length of stay prediction task with demo data."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "eicudemo")

        print(f"\n{'='*60}")
        print(f"Setting up eICU demo dataset for length of stay prediction")
        print(f"Dataset path: {self.demo_dataset_path}")
        print(f"{'='*60}\n")

    def test_length_of_stay_prediction_eicu_set_task(self):
        """Test LengthOfStayPredictioneICU task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_length_of_stay_prediction_eicu_set_task()")
        print(f"{'='*60}")

        # Load dataset with required tables
        tables = ["diagnosis", "medication", "physicalexam"]
        print(f"Loading eICUDataset with tables: {tables}")
        dataset = eICUDataset(root=self.demo_dataset_path, tables=tables)
        print("✓ Dataset loaded successfully")

        # Initialize task
        print("\nInitializing LengthOfStayPredictioneICU task...")
        task = LengthOfStayPredictioneICU()
        
        # Verify task schema
        self.assertEqual(task.task_name, "LengthOfStayPredictioneICU")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("los", task.output_schema)
        print(f"✓ Task initialized: {task.task_name}")
        print(f"  Input schema: {list(task.input_schema.keys())}")
        print(f"  Output schema: {list(task.output_schema.keys())}")

        # Apply task
        try:
            print("\nCalling dataset.set_task()...")
            sample_dataset = dataset.set_task(task)
            self.assertIsNotNone(sample_dataset, "set_task should return a dataset")
            print(f"✓ set_task() completed")

            # Check samples
            num_samples = len(sample_dataset)
            print(f"✓ Generated {num_samples} length of stay prediction samples")

            if num_samples > 0:
                sample = sample_dataset[0]
                required_keys = ["visit_id", "patient_id", "conditions", "procedures", "drugs", "los"]
                
                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")
                
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")
                
                # Verify LOS is in valid range (0-9)
                los = sample["los"]
                los_val = int(los.item()) if hasattr(los, 'item') else int(los)
                self.assertIn(los_val, list(range(10)), "LOS category should be 0-9")
                
                # Count LOS distribution
                los_counts = {i: 0 for i in range(10)}
                for s in sample_dataset:
                    label = int(s["los"].item()) if hasattr(s["los"], 'item') else int(s["los"])
                    los_counts[label] += 1
                
                category_labels = [
                    "< 1 day", "1 day", "2 days", "3 days", "4 days",
                    "5 days", "6 days", "7 days", "1-2 weeks", "> 2 weeks"
                ]
                
                print(f"\nLength of stay category distribution:")
                for i in range(10):
                    if los_counts[i] > 0:
                        pct = los_counts[i] / num_samples * 100
                        print(f"  Category {i} ({category_labels[i]}): {los_counts[i]} ({pct:.1f}%)")
                
                print(f"\n✓ test_length_of_stay_prediction_eicu_set_task() passed")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed: {e}")


if __name__ == "__main__":
    unittest.main()





