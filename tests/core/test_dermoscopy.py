"""
test_dermoscopy.py

Unit tests for the PyHealth DermoscopyDataset integration.
This test suite uses a mocked subset of the HAM10000 dataset.

Dataset Citation:
Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection 
of multi-source dermatoscopic images of common pigmented skin lesions. 
Sci. Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

Author:
    Mumme, Raymond Paul rmumme2@illinois.edu
"""

import os
import shutil
# Stop Polars from spinning up 16 threads for 3 tiny images
os.environ["POLARS_MAX_THREADS"] = "1"

import unittest
import warnings
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor

class TestDermoscopyDataset(unittest.TestCase):
    def setUp(self):
        """Initializes the dataset pointer and suppresses known PyHealth warnings."""
        # ignores not being able to close the PyHealth cached .ld files
        warnings.simplefilter("ignore", ResourceWarning)
        # Ensure this directory contains ONLY 2-5 tiny images!
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_dir = os.path.join(current_dir, "../../test-resources/core/dermoscopy")

        local_cache = os.path.join(test_data_dir, ".test_cache")

        self.dataset = DermoscopyDataset(root=test_data_dir, dev=False, cache_dir=local_cache)

    def tearDown(self):
        """
        Cleans up the localized test cache to prevent artifact bloat in the repo.
        Wiped after every test to prove functionality.
        """
        if os.path.exists(self.local_cache):
            shutil.rmtree(self.local_cache, ignore_errors=True)

    def test_load_data(self):
        """Verifies the base dataset successfully parses the metadata CSV into a Polars DataFrame."""
        # Checking the PyHealth 2.0 DataFrame instead of the old dictionary
        self.assertIsNotNone(self.dataset.global_event_df)

    def test_task_and_processor(self):
        """Verifies the dataset correctly applies tasks, modes, and image processors."""
        processor = DermoscopyImageProcessor(mode="high_whole")

        # Architecture natively supports a single string input or a list of strings
        task = DermoscopyMelanomaClassification(source_datasets="ham10000")

        # Capture the newly generated SampleDataset
        task_dataset = self.dataset.set_task(
            task=task,
            input_processors={"image": processor},
            num_workers=1
        )

        # Check the samples
        sample = task_dataset[0]
        self.assertEqual(sample['image'].shape, (3, 224, 224))

if __name__ == "__main__":
    unittest.main()