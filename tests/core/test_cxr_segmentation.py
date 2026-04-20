import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pyhealth.datasets import CXRSegmentationDataset
from pyhealth.tasks import CXRSegmentationTask


class TestCXRSegmentationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Make some temporary directories to store dummy data. We clean these up later
        # in tearDownClass
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.data_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.data_dir.name)
        (cls.root / "CXR_png").mkdir(parents=True)
        (cls.root / "masks").mkdir(parents=True)

        cls.generate_fake_data()

        # Optimize for speed: num_workers=1, dev=True
        cls.dataset = CXRSegmentationDataset(
            root=str(cls.root), 
            cache_dir=cls.cache_dir.name, 
            dev=True, 
            num_workers=1
        )

        # Use smaller images (16x16) for faster processing
        cls.task = CXRSegmentationTask(
            image_config={"mode": "L", "image_size": 16}, 
            mask_config={"mode": "L", "image_size": 16}
        )
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "samples"):
            cls.samples.close()
        cls.cache_dir.cleanup()
        shutil.rmtree(cls.data_dir.name, ignore_errors=True)

    @classmethod
    def generate_fake_data(cls):
        # Only 2 samples are enough
        for i in range(2):
            patient_id = f"patient_{i}"
            img_arr = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            Image.fromarray(img_arr).save(cls.root / "CXR_png" / f"{patient_id}.png")
            mask_arr = np.random.randint(0, 2, (16, 16), dtype=np.uint8) * 255
            Image.fromarray(mask_arr).save(cls.root / "masks" / f"{patient_id}_mask.png")

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 2)
        self.assertIn("patient_0", self.dataset.unique_patient_ids)
        self.assertIn("patient_1", self.dataset.unique_patient_ids)

    def test_samples(self):
        self.assertEqual(len(self.samples), 2)

        sample = self.samples[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)
        self.assertIn("patient_id", sample)

        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertIsInstance(sample["mask"], torch.Tensor)

        # Check shapes after processor (e.g., C, H, W)
        self.assertEqual(sample["image"].shape, (1, 16, 16))
        self.assertEqual(sample["mask"].shape, (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
