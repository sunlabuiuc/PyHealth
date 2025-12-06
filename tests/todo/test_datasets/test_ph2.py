import unittest
import os
import torch
import shutil
from pyhealth.datasets import PH2Dataset
from pyhealth.tasks import PH2MelanomaClassification

# To run: 
# export PH2_ROOT="/path/to/PH2Dataset"
# python tests/todo/test_datasets/test_ph2.py 

class TestPH2Dataset(unittest.TestCase):
    # Get the root path from environment variable
    ROOT = os.environ.get("PH2_ROOT")

    def setUp(self):
        """Set up the test environment."""
        if self.ROOT is None or not os.path.exists(self.ROOT):
            self.skipTest("PH2_ROOT environment variable not set or path not found. Skipping.")

        csv_path = os.path.join(self.ROOT, "ph2_metadata_pyhealth.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)

        self.dataset = PH2Dataset(root=self.ROOT)

    def test_initialization(self):
        """Test 1: Does the dataset load the correct number of patients?"""
        self.assertEqual(len(self.dataset.unique_patient_ids), 200)
        
        # Verify the CSV was recreated
        metadata_path = os.path.join(self.ROOT, "ph2_metadata_pyhealth.csv")
        self.assertTrue(os.path.exists(metadata_path))

    def test_task_generation(self):
        """Test 2: Does set_task generate valid samples?"""
        sample_dataset = self.dataset.set_task(task=PH2MelanomaClassification())
        
        # Check dataset size
        self.assertEqual(len(sample_dataset), 200)
        
        # Check the first sample
        first_sample = sample_dataset[0]
        
        self.assertIn("image", first_sample)
        self.assertIn("label", first_sample)
        self.assertTrue(torch.is_tensor(first_sample["image"]))
        self.assertEqual(first_sample["image"].shape, (3, 224, 224))

        # Check label validity
        # Note: labels are now integer class IDs: 0, 1, 2 
        # where 0 corresponds to atypical_nevus, 1 to common_nevus
        # and 2 to melanoma.
        valid_label_ids = [0, 1, 2]
        self.assertTrue(torch.is_tensor(first_sample["label"]))
        self.assertIn(first_sample["label"].item(), valid_label_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)



