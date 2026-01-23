import unittest
import os
from pathlib import Path

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import MortalityPredictionEICU, MortalityPredictionEICU2


class TesteICUMortalityPrediction(unittest.TestCase):
    """Test eICU mortality prediction tasks with demo data."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "eicudemo")

        print(f"\n{'='*60}")
        print(f"Setting up eICU demo dataset for mortality prediction")
        print(f"Dataset path: {self.demo_dataset_path}")
        print(f"{'='*60}\n")

    def test_mortality_prediction_eicu_set_task(self):
        """Test MortalityPredictionEICU task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_mortality_prediction_eicu_set_task()")
        print(f"{'='*60}")

        # Load dataset with required tables
        tables = ["diagnosis", "medication", "physicalexam"]
        print(f"Loading eICUDataset with tables: {tables}")
        dataset = eICUDataset(root=self.demo_dataset_path, tables=tables)
        print("✓ Dataset loaded successfully")

        # Initialize task
        print("\nInitializing MortalityPredictionEICU task...")
        task = MortalityPredictionEICU()
        
        # Verify task schema
        self.assertEqual(task.task_name, "MortalityPredictionEICU")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs", task.input_schema)
        self.assertIn("mortality", task.output_schema)
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
            print(f"✓ Generated {num_samples} mortality prediction samples")

            if num_samples > 0:
                sample = sample_dataset[0]
                required_keys = ["visit_id", "patient_id", "conditions", "procedures", "drugs", "mortality"]
                
                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")
                
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")
                
                # Verify mortality is binary
                mortality = sample["mortality"]
                self.assertIn(int(mortality.item()) if hasattr(mortality, 'item') else int(mortality), [0, 1])
                
                print(f"✓ test_mortality_prediction_eicu_set_task() passed")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed: {e}")

    def test_mortality_prediction_eicu2_set_task(self):
        """Test MortalityPredictionEICU2 task with alternative coding."""
        print(f"\n{'='*60}")
        print("TEST: test_mortality_prediction_eicu2_set_task()")
        print(f"{'='*60}")

        # Load dataset with alternative tables
        tables = ["diagnosis", "admissiondx", "treatment"]
        print(f"Loading eICUDataset with tables: {tables}")
        dataset = eICUDataset(root=self.demo_dataset_path, tables=tables)
        print("✓ Dataset loaded successfully")

        # Initialize task
        print("\nInitializing MortalityPredictionEICU2 task...")
        task = MortalityPredictionEICU2()
        
        self.assertEqual(task.task_name, "MortalityPredictionEICU2")
        print(f"✓ Task initialized: {task.task_name}")

        # Apply task
        try:
            print("\nCalling dataset.set_task()...")
            sample_dataset = dataset.set_task(task)
            self.assertIsNotNone(sample_dataset)
            
            num_samples = len(sample_dataset)
            print(f"✓ Generated {num_samples} samples")
            print(f"✓ test_mortality_prediction_eicu2_set_task() passed")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed: {e}")


if __name__ == "__main__":
    unittest.main()



