"""
Unit tests for the MRIDataset, AlzheimerDiseaseClassification classes.

Author:
    Soheil Golara (sgolara@illinois.edu), Karan Desai (kdesai2@illinois.edu)
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from pyhealth.datasets import MRIDataset
from pyhealth.tasks import MRIBinaryClassification

class TestMRIDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parent.parent.parent / "test-resources" / "core" / "mri"
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.dataset = MRIDataset(cls.root, cache_dir=cls.cache_dir.name, download=True, partial=True)

        cls.samples_alzheimer = cls.dataset.set_task(MRIBinaryClassification(disease="alzheimer"))

    @classmethod
    def tearDownClass(cls):
        cls.samples_alzheimer.close()

        Path(cls.dataset.root / "mri-metadata-pyhealth.csv").unlink()

    @classmethod
    def generate_fake_images(cls):
        with open(Path(cls.root / "oasis_longitudinal.csv"), 'r') as f:
            lines = f.readlines()

        for line in lines[1:]: # Skip header row
            name = line.split(',')[0]
            mri = Image.fromarray(np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8))
            mri.save(Path(cls.root / "oasis_longitudinal_nifti" / name))

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 436)

    def test_get_patient_1(self):
        events = self.dataset.get_patient("OAS1_0001_MR1").get_events()

        self.assertEqual(len(events), 1)

        self.assertEqual(events[0]["gender"], "F")
        self.assertEqual(events[0]["dominant_hand"], "R")
        self.assertEqual(events[0]["age"], "74")
        self.assertEqual(events[0]["clinical_dementia_rating"], "0.0")
        self.assertEqual(events[0]["alzheimer"], "0")

    def test_get_patient_2(self):
        events = self.dataset.get_patient("OAS1_0002_MR1").get_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gender"], "F")
        self.assertEqual(events[0]["age"], "55")
        self.assertEqual(events[0]["clinical_dementia_rating"], "0.0")
        self.assertEqual(events[0]["alzheimer"], "0")

    def test_get_patient_3(self):
        events = self.dataset.get_patient("OAS1_0003_MR1").get_events()

        self.assertEqual(len(events), 1)

        self.assertEqual(events[0]["gender"], "F")
        self.assertEqual(events[0]["age"], "73")
        self.assertEqual(events[0]["clinical_dementia_rating"], "0.5")
        self.assertEqual(events[0]["alzheimer"], "1")

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, MRIBinaryClassification)

    def test_task_given_invalid_disease(self):
        with self.assertRaises(ValueError):
            _ = MRIBinaryClassification(disease="arthritis")


if __name__ == "__main__":
    unittest.main()
