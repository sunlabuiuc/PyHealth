"""Tests for ISIC2020MelanomaClassification using synthetic images."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import ISIC2020Dataset, PatientSplit
from pyhealth.tasks import ISIC2020MelanomaClassification
from tests.core.test_isic2020_dataset import make_isic2020_root


class TestISIC2020MelanomaClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_isic2020_root(cls.tmp / "isic2020")
        cls.dataset = ISIC2020Dataset(
            root=str(cls.tmp / "isic2020"), cache_dir=str(cls.tmp / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_schema(self):
        task = ISIC2020MelanomaClassification(image_size=16)
        self.assertEqual(task.output_schema, {"malignant": "binary"})
        self.assertEqual(task.input_schema["image"][1]["image_size"], 16)
        self.assertEqual(
            ISIC2020MelanomaClassification.input_schema["image"][1]["image_size"],
            224,
        )

    def test_samples(self):
        samples = self.dataset.set_task(ISIC2020MelanomaClassification(image_size=16))
        self.assertEqual(len(samples), 6)
        labels = sorted(int(samples[i]["malignant"].item()) for i in range(6))
        self.assertEqual(labels, [0, 0, 0, 1, 1, 1])
        self.assertEqual(tuple(samples[0]["image"].shape), (3, 16, 16))
        self.assertEqual(samples[0]["image"].dtype, torch.float32)
        samples.close()

    def test_patient_split_has_no_overlap(self):
        parts = self.dataset.set_task(
            ISIC2020MelanomaClassification(image_size=16),
            split=PatientSplit((0.34, 0.33, 0.33), seed=0),
        )
        patient_sets = [
            {part[i]["patient_id"] for i in range(len(part))} for part in parts
        ]
        self.assertEqual(sum(len(part) for part in parts), 6)
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertFalse(patient_sets[a] & patient_sets[b])
        for part in parts:
            part.close()

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ISIC2020MelanomaClassification)


if __name__ == "__main__":
    unittest.main()
