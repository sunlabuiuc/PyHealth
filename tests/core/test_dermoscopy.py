# test_dermoscopy.py
import os
import unittest
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor

class TestDermoscopyDataset(unittest.TestCase):
    def setUp(self):
        # Ensure this directory contains ONLY 2-5 tiny synthetic images!
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_dir = os.path.join(current_dir, "../../test-resources/dermoscopy")
        self.dataset = DermoscopyDataset(root=test_data_dir, dataset_name="isic2018", dev=False)

    def test_load_data(self):
        self.assertGreater(len(self.dataset.patients), 0)

    def test_task_and_processor(self):
        processor = DermoscopyImageProcessor(mode="high_whole")
        self.dataset.set_task(
            task=DermoscopyMelanomaClassification, # Fixed from task_fn
            input_processors={"image": processor}
        )
        sample = self.dataset.samples[0]
        self.assertEqual(sample['image'].shape, (3, 224, 224))