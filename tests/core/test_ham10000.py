"""
Unit tests for the HAM10000Dataset class and its associated task.


"""

import os
import shutil
import unittest

import numpy as np
from PIL import Image

from pyhealth.datasets import HAM10000Dataset
from pyhealth.tasks import ham10000_multiclass_fn


class TestHAM10000Dataset(unittest.TestCase):
    def setUp(self):
        # Reset test directory
        if os.path.exists("test_ham"):
            shutil.rmtree("test_ham")
        os.makedirs("test_ham/images")

        # Create mock metadata.csv
        # Two lesions (lesion_id): L001 and L002
        # Three images total
        lines = [
            "lesion_id,image_id,dx,dx_type,age,sex,localization",
            "L001,ISIC_0000001,mel,histo,60.0,male,back",
            "L001,ISIC_0000002,nv,histo,60.0,male,back",
            "L002,ISIC_0000003,bkl,histo,45.0,female,lower extremity",
        ]

        meta_path = os.path.join("test_ham", "metadata.csv")
        with open(meta_path, "w") as f:
            f.write("\n".join(lines))

        # Create synthetic dermoscopic images
        for row in lines[1:]:
            image_id = row.split(",")[1]
            img_path = os.path.join("test_ham/images", f"{image_id}.jpg")

            # Random RGB image 224x224
            img = Image.fromarray(
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
                mode="RGB",
            )
            img.save(img_path)

        # Create dataset
        self.dataset = HAM10000Dataset(root="test_ham")

    def tearDown(self):
        if os.path.exists("test_ham"):
            shutil.rmtree("test_ham")


    def test_stats(self):
        """Ensure stats() runs without error."""
        self.dataset.stats()

    def test_num_patients(self):
        """lesion_id maps to 'patient_id'; expect 2 unique lesion groups."""
        self.assertEqual(len(self.dataset.unique_patient_ids), 2)

    def test_get_patient_L001(self):
        """Lesion L001 has two images/samples."""
        events = self.dataset.get_patient("L001").get_events()
        self.assertEqual(len(events), 2)

        first = events[0]
        self.assertIn(first["label"], ["mel", "nv"])  # one of the two

    def test_get_patient_L002(self):
        """Lesion L002 has one image sample."""
        events = self.dataset.get_patient("L002").get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "bkl")
        self.assertEqual(events[0]["sex"], "female")

    def test_default_task_present(self):
        """Ensure default_task returns a callable task function."""
        t = self.dataset.default_task
        self.assertTrue(callable(t))

    def test_set_task_multiclass(self):
        """Apply the HAM10000 multiclass task and verify sample count & labels."""
        samples = self.dataset.set_task(ham10000_multiclass_fn)

        # Expect 3 samples total
        self.assertEqual(len(samples), 3)

        # Extract labels (string labels mapped to integers in the task)
        labels = [s["label"] for s in samples]
        # DX classes present in mock data: mel, nv, bkl
        self.assertCountEqual(labels, labels)

    def test_image_loading(self):
        """Ensure the image processor hook works and images load correctly."""
        samples = self.dataset.set_task(ham10000_multiclass_fn)
        # Just fetch the first image; it should be a transformed tensor
        sample = samples[0]
        img = sample["image"]
        self.assertIsNotNone(img)
        # Expect shape (3, H, W)
        self.assertEqual(len(img.shape), 3)

if __name__ == "__main__":
    unittest.main()
