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
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_get_patient_1(self):
        events = self.dataset.get_patient('1').get_events()

        self.assertEqual(len(events), 3)

        self.assertEqual(events[0]['visit_id'], '0')
        self.assertEqual(events[0]['patient_age'], '57')
        self.assertEqual(events[0]['mild_demented'], '1')
        self.assertEqual(events[0]['effusion'], '0')
        self.assertEqual(events[0]['very_mild_demented'], '0')
        self.assertEqual(events[0]['non_demented'], '0')

    def test_get_patient_2(self):
        events = self.dataset.get_patient('2').get_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['visit_id'], '0')
        self.assertEqual(events[0]['patient_age'], '80')
        self.assertEqual(events[0]['mild_demented'], '0')
        self.assertEqual(events[0]['very_mild_demented'], '0')
        self.assertEqual(events[0]['non_demented'], '1')

    def test_get_patient_3(self):
        events = self.dataset.get_patient('3').get_events()

        self.assertEqual(len(events), 6)

        self.assertEqual(events[0]['mild_demented'], '0')
        self.assertEqual(events[0]['very_mild_demented'], '0')
        self.assertEqual(events[0]['non_demented'], '1')

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, MRIBinaryClassification)

    def test_task_given_invalid_disease(self):
        with self.assertRaises(ValueError):
            _ = MRIBinaryClassification(disease="arthritis")

    def test_task_classify_cardiomegaly(self):
        self.assertEqual(len(self.samples_alzheimer), 10)
        self.assertEqual(sum(sample["mild_demented"] for sample in self.samples_alzheimer), 3)


if __name__ == "__main__":
    unittest.main()
