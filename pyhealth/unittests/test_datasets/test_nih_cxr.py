import os
import sys
import shutil
import tempfile
import unittest
from PIL import Image

# Allow imports from repo root
current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)

from pyhealth.datasets.nih_cxr import NIHChestXrayDataset


class TestNIHCxrDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for the demo dataset
        cls.temp_dir = tempfile.mkdtemp(prefix="nih_demo_")

        # Create one images_001/images folder and drop in a dummy PNG
        images_folder = os.path.join(cls.temp_dir, "images_001", "images")
        os.makedirs(images_folder, exist_ok=True)
        cls.img_name = "00000001_000.png"
        dummy_path = os.path.join(images_folder, cls.img_name)
        Image.new("RGB", (10, 10)).save(dummy_path)

        # Write split text files listing that image
        with open(os.path.join(cls.temp_dir, "train_val_list.txt"), "w") as f:
            f.write(cls.img_name + "\n")
        with open(os.path.join(cls.temp_dir, "test_list.txt"), "w") as f:
            f.write(cls.img_name + "\n")

        # Instantiate both splits (download=False since data is already on disk)
        cls.train_ds = NIHChestXrayDataset(
            root=cls.temp_dir, split="training", download=False
        )
        cls.test_ds = NIHChestXrayDataset(
            root=cls.temp_dir, split="test", download=False
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up
        shutil.rmtree(cls.temp_dir)

    def test_length(self):
        # Both splits should contain exactly one sample
        self.assertEqual(len(self.train_ds), 1)
        self.assertEqual(len(self.test_ds), 1)

    def test_sample_loading(self):
        # Ensure sample is a PIL Image of size 10×10
        img = self.train_ds[0]
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (10, 10))

    def test_index_error(self):
        # Out-of-bounds should raise IndexError
        with self.assertRaises(IndexError):
            _ = self.train_ds[1]

    def test_stat_output(self):
        stats = self.train_ds.stat()
        # Matches the lines printed by stat()
        self.assertIn("Dataset: NIH Chest X-ray", stats)
        self.assertIn("Split:   training", stats)
        self.assertIn("Samples: 1", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
