import os
import sys
import unittest
from PIL import Image

# Adjust the repository path so that imports work in your testing environment.
current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)

from pyhealth.datasets.nih_cxr import NIHChestXrayDataset

# This test suite verifies the NIH Chest X‑ray demo dataset is loaded correctly
# and produces the expected sample and statistics.
class TestNIHCxrDataset(unittest.TestCase):
    DATASET_NAME = "nih-demo"
    # For demonstration, we use a locally available directory.
    # This path should contain a demo NIH Chest X-ray dataset extracted in the
    # expected folder structure (i.e. a "database" folder with "train" and "test"
    # subdirectories).
    ROOT = "/tmp/nih_chestxray_demo"
    SPLIT = "training"

    # Create the dataset instance.
    # Set download to False if you expect the dataset to be already present at ROOT.
    dataset = NIHChestXrayDataset(
        root=ROOT,
        split=SPLIT,
        download=False,
        transform=None,
    )

    def setUp(self):
        # This method can be used for additional initialization if needed.
        pass

    def test_sample(self):
        """Test that a sample image is correctly loaded from the dataset.

        Verifies:
          - The dataset has at least one sample.
          - The first sample is a valid PIL Image object with nonzero dimensions.
        """
        # Ensure the dataset mapping is not empty.
        self.assertGreater(
            len(self.dataset.patients),
            0,
            "Dataset should have at least 1 sample.",
        )
        # Retrieve the first sample.
        sample = self.dataset[0]
        self.assertIsInstance(
            sample,
            Image.Image,
            "Sample should be a PIL Image object.",
        )
        # Check that the loaded image dimensions are valid.
        width, height = sample.size
        self.assertGreater(width, 0, "Image width should be greater than 0.")
        self.assertGreater(height, 0, "Image height should be greater than 0.")

    def test_statistics(self):
        """Test that the dataset's statistics are reported correctly.

        Verifies:
          - The statistics string includes the dataset name.
          - The reported number of samples matches the length of the dataset.
        """
        stats_str = self.dataset.stat()
        self.assertIn(
            self.dataset.dataset_name,
            stats_str,
            "Dataset statistics should contain the dataset name.",
        )
        expected_num_samples = len(self.dataset.patients)
        self.assertIn(
            f"Number of samples: {expected_num_samples}",
            stats_str,
            "Dataset statistics should report the correct number of samples.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
