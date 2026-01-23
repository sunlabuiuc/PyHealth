import unittest
import os
from pathlib import Path

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import DrugRecommendationEICU


class TesteICUDrugRecommendation(unittest.TestCase):
    """Test eICU drug recommendation task with demo data."""

    def setUp(self):
        """Set up demo dataset path for each test."""
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(test_dir / "test-resources" / "core" / "eicudemo")

        print(f"\n{'='*60}")
        print(f"Setting up eICU demo dataset for drug recommendation")
        print(f"Dataset path: {self.demo_dataset_path}")
        print(f"{'='*60}\n")

    def test_drug_recommendation_eicu_set_task(self):
        """Test DrugRecommendationEICU task with set_task() method."""
        print(f"\n{'='*60}")
        print("TEST: test_drug_recommendation_eicu_set_task()")
        print(f"{'='*60}")

        # Load dataset with required tables
        tables = ["diagnosis", "medication", "physicalexam"]
        print(f"Loading eICUDataset with tables: {tables}")
        dataset = eICUDataset(root=self.demo_dataset_path, tables=tables)
        print("✓ Dataset loaded successfully")

        # Initialize task
        print("\nInitializing DrugRecommendationEICU task...")
        task = DrugRecommendationEICU()
        
        # Verify task schema
        self.assertEqual(task.task_name, "DrugRecommendationEICU")
        self.assertIn("conditions", task.input_schema)
        self.assertIn("procedures", task.input_schema)
        self.assertIn("drugs_hist", task.input_schema)
        self.assertIn("drugs", task.output_schema)
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
            print(f"✓ Generated {num_samples} drug recommendation samples")

            if num_samples > 0:
                sample = sample_dataset[0]
                required_keys = ["visit_id", "patient_id", "conditions", "procedures", "drugs_hist", "drugs"]
                
                print(f"\nFirst sample structure:")
                print(f"  Sample keys: {list(sample.keys())}")
                
                for key in required_keys:
                    self.assertIn(key, sample, f"Sample should contain key: {key}")
                
                # Verify nested structure for conditions, procedures, drugs_hist
                print(f"\nVerifying nested sequence structure:")
                conditions = sample["conditions"]
                procedures = sample["procedures"]
                drugs_hist = sample["drugs_hist"]
                
                # These should be nested lists (list of lists per visit)
                print(f"  conditions shape: {len(conditions)} visits")
                print(f"  procedures shape: {len(procedures)} visits")
                print(f"  drugs_hist shape: {len(drugs_hist)} visits")
                
                # Target drugs should be a flat list
                drugs = sample["drugs"]
                print(f"  drugs (target): {len(drugs)} items")
                
                print(f"\n✓ test_drug_recommendation_eicu_set_task() passed")

        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Failed: {e}")


if __name__ == "__main__":
    unittest.main()



