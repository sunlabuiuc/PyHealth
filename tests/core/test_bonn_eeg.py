import unittest
import os
import shutil
import numpy as np
from pyhealth.datasets import BonnEEGDataset
from pyhealth.tasks.bonn_eeg_tasks import BonnEEGSeizureDetection

class TestBonnEEGDataset(unittest.TestCase):
    def setUp(self):
        self.root = "test_bonn_mem_load"
        if os.path.exists(self.root): shutil.rmtree(self.root)
        os.makedirs(self.root)

        # Create dummy data structure: Z (Healthy) and S (Seizure)
        # Creating 2 files per class = 4 total samples expected
        for cls in ["Z", "S"]:
            os.makedirs(os.path.join(self.root, cls), exist_ok=True)
            for i in range(2):
                data = np.random.randn(4097) # Random noise
                np.savetxt(os.path.join(self.root, cls, f"{cls}{i}.txt"), data)

    def tearDown(self):
        if os.path.exists(self.root): shutil.rmtree(self.root)

    def test_pipeline(self):
        # 1. Initialize Dataset (Generates index.csv automatically)
        dataset = BonnEEGDataset(root=self.root, dev=False)
        
        # 2. Run Task (Streams data from disk)
        task = BonnEEGSeizureDetection()
        samples = dataset.set_task(task)

        # 3. Validation
        # Check total count (2 Z + 2 S = 4)
        self.assertEqual(len(samples), 4)
        
        # Check signal integrity
        self.assertIn("signal", samples[0])
        self.assertEqual(samples[0]["signal"].shape, (1, 4097))
        
        # Check Label Logic
        # We assume sample order might vary, but we must have both classes
        labels = [s["label"] for s in samples]
        self.assertIn(1, labels) # Should capture Seizure (S)
        self.assertIn(0, labels) # Should capture Healthy (Z)

if __name__ == "__main__":
    unittest.main()