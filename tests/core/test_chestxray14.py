"""
Unit tests for the ChestXray14Dataset, ChestXray14BinaryClassification, and ChestXray14MultilabelClassification classes.

Author:
    Eric Schrock (ejs9@illinois.edu)
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from pyhealth.datasets import ChestXray14Dataset
from pyhealth.tasks import ChestXray14BinaryClassification
from pyhealth.tasks import ChestXray14MultilabelClassification

class TestChestXray14Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parent.parent.parent / "test-resources" / "core" / "chestxray14"
        cls.generate_fake_images()
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.dataset = ChestXray14Dataset(cls.root, cache_dir=cls.cache_dir.name)

        cls.samples_cardiomegaly = cls.dataset.set_task(ChestXray14BinaryClassification(disease="cardiomegaly"))
        cls.samples_hernia = cls.dataset.set_task(ChestXray14BinaryClassification(disease="hernia"))
        cls.samples_multilabel = cls.dataset.set_task()

    @classmethod
    def tearDownClass(cls):
        cls.samples_cardiomegaly.close()
        cls.samples_hernia.close()
        cls.samples_multilabel.close()

        Path(cls.dataset.root / "chestxray14-metadata-pyhealth.csv").unlink()
        cls.delete_fake_images()

    @classmethod
    def generate_fake_images(cls):
        with open(Path(cls.root / "Data_Entry_2017_v2020.csv"), 'r') as f:
            lines = f.readlines()

        for line in lines[1:]: # Skip header row
            name = line.split(',')[0]
            img = Image.fromarray(np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8))
            img.save(Path(cls.root / "images" / name))

    @classmethod
    def delete_fake_images(cls):
        for png in Path(cls.root / "images").glob("*.png"):
            png.unlink()

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_get_patient_1(self):
        events = self.dataset.get_patient('1').get_events()

        self.assertEqual(len(events), 3)

        self.assertEqual(events[0]['visit_id'], '0')
        self.assertEqual(events[0]['patient_age'], '57')
        self.assertEqual(events[0]['cardiomegaly'], '1')
        self.assertEqual(events[0]['effusion'], '0')
        self.assertEqual(events[0]['emphysema'], '0')

        self.assertEqual(events[1]['visit_id'], '1')
        self.assertEqual(events[1]['patient_age'], '58')
        self.assertEqual(events[1]['cardiomegaly'], '1')
        self.assertEqual(events[1]['effusion'], '0')
        self.assertEqual(events[1]['emphysema'], '1')

        self.assertEqual(events[2]['visit_id'], '2')
        self.assertEqual(events[2]['patient_age'], '58')
        self.assertEqual(events[2]['cardiomegaly'], '1')
        self.assertEqual(events[2]['effusion'], '1')
        self.assertEqual(events[2]['emphysema'], '0')

    def test_get_patient_2(self):
        events = self.dataset.get_patient('2').get_events()

        self.assertEqual(len(events), 1)

        self.assertEqual(events[0]['visit_id'], '0')
        self.assertEqual(events[0]['patient_age'], '80')
        self.assertEqual(events[0]['patient_sex'], 'M')
        self.assertEqual(events[0]['view_position'], 'PA')
        self.assertEqual(events[0]['original_image_width'], '2500')
        self.assertEqual(events[0]['original_image_height'], '2048')
        self.assertEqual(events[0]['original_image_pixel_spacing_x'], '0.171')
        self.assertEqual(events[0]['original_image_pixel_spacing_y'], '0.171')
        self.assertEqual(events[0]['atelectasis'], '0')
        self.assertEqual(events[0]['cardiomegaly'], '0')
        self.assertEqual(events[0]['consolidation'], '0')
        self.assertEqual(events[0]['edema'], '0')
        self.assertEqual(events[0]['effusion'], '0')
        self.assertEqual(events[0]['emphysema'], '0')
        self.assertEqual(events[0]['fibrosis'], '0')
        self.assertEqual(events[0]['hernia'], '0')
        self.assertEqual(events[0]['infiltration'], '0')
        self.assertEqual(events[0]['mass'], '0')
        self.assertEqual(events[0]['nodule'], '0')
        self.assertEqual(events[0]['pleural_thickening'], '0')
        self.assertEqual(events[0]['pneumonia'], '0')
        self.assertEqual(events[0]['pneumothorax'], '0')

    def test_get_patient_3(self):
        events = self.dataset.get_patient('3').get_events()

        self.assertEqual(len(events), 6)

        self.assertEqual(events[0]['patient_sex'], 'F')

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ChestXray14MultilabelClassification)

    def test_task_given_invalid_disease(self):
        with self.assertRaises(ValueError):
            _ = ChestXray14BinaryClassification(disease="toothache")

    def test_task_classify_cardiomegaly(self):
        self.assertEqual(len(self.samples_cardiomegaly), 10)
        self.assertEqual(sum(sample["label"] for sample in self.samples_cardiomegaly), 3)

    def test_task_classify_hernia(self):
        self.assertEqual(len(self.samples_hernia), 10)
        self.assertEqual(sum(sample["label"] for sample in self.samples_hernia), 6)

    def test_task_classify_all(self):
        self.assertEqual(len(self.samples_multilabel), 10)

        actual_labels = [sample["labels"].tolist() for sample in self.samples_multilabel]

        expected_labels = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]

        self.assertCountEqual(actual_labels, expected_labels)

if __name__ == "__main__":
    unittest.main()
