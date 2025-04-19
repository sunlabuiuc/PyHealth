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
    """
    Unittest suite for NIHChestXrayDataset.

    Verifies that a minimal NIH CXR demo dataset can be processed
    correctly, including indexing, loading, error handling, and stats.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a minimal demo dataset on disk.

        This creates a temporary directory with:
          - images_001/images/00000001_000.png (dummy image)
          - train_val_list.txt and test_list.txt listing that image
          - Data_Entry_2017.csv with a single 'No Finding' entry
        Then instantiates both training and test dataset splits.
        """
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

        # Create a minimal Data_Entry_2017.csv with our dummy image → "No Finding"
        csv_path = os.path.join(cls.temp_dir, "Data_Entry_2017.csv")
        with open(csv_path, "w") as f:
            f.write("Image Index,Finding Labels\n")
            f.write(f"{cls.img_name},No Finding\n")

        # Instantiate both splits (download=False since data is already on disk)
        cls.train_ds = NIHChestXrayDataset(
            root=cls.temp_dir, split="training", download=False
        )
        cls.test_ds = NIHChestXrayDataset(
            root=cls.temp_dir, split="test", download=False
        )
        

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the temporary demo dataset directory.
        Deletes the entire temporary directory tree.
        """
        shutil.rmtree(cls.temp_dir)

    def test_length(self):
        """
        Test that both training and test splits contain exactly one sample.

        Verifies:
          - len(dataset) == 1 for both splits.
        """
        self.assertEqual(len(self.train_ds), 1)
        self.assertEqual(len(self.test_ds), 1)

    def test_sample_loading(self):
        """
        Test that samples load correctly as PIL images.

        Verifies:
          - type of ds[0] is PIL.Image.Image
          - image size matches (10, 10) dummy image.
        """
        img = self.train_ds[0]
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (10, 10))

    def test_index_error(self):
        """
        Test that accessing out-of-range index raises IndexError.

        Verifies:
          - ds[1] for a single-sample dataset raises IndexError.
        """
        with self.assertRaises(IndexError):
            _ = self.train_ds[1]

    def test_stat_output(self):
        """
        Test that the stat() output includes correct dataset info.

        Verifies:
          - stat string contains dataset name, split, and sample count.
        """
        stats = self.train_ds.stat()
        self.assertIn("Dataset: NIH Chest X-ray", stats)
        self.assertIn("Split:   training", stats)
        self.assertIn("Samples: 1", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
