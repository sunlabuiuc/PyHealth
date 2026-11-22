"""Tests for HAM10000Classification using synthetic images."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import HAM10000Dataset, PatientSplit
from pyhealth.tasks import HAM10000Classification
from tests.core.test_ham10000_dataset import make_ham10000_root


def _labels(samples):
    proc = samples.output_processors["diagnosis"]
    inverse = {v: k for k, v in proc.label_vocab.items()}
    return sorted(inverse[int(samples[i]["diagnosis"])] for i in range(len(samples)))


class TestHAM10000Classification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_ham10000_root(cls.tmp / "isic2018")
        cls.dataset = HAM10000Dataset(
            root=str(cls.tmp / "isic2018"),
            splits=("train", "val", "test"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_ham7_samples(self):
        samples = self.dataset.set_task(
            HAM10000Classification(image_size=16, split="train")
        )
        self.assertEqual(samples.output_processors["diagnosis"].size(), 7)
        self.assertEqual(
            _labels(samples),
            ["AKIEC", "BCC", "BKL", "DF", "MEL", "MEL", "NV", "VASC"],
        )
        self.assertEqual(tuple(samples[0]["image"].shape), (3, 16, 16))
        samples.close()

    def test_climb5_merges_into_other(self):
        samples = self.dataset.set_task(
            HAM10000Classification(label_scheme="climb5", image_size=16, split="train")
        )
        self.assertEqual(samples.output_processors["diagnosis"].size(), 5)
        self.assertEqual(
            _labels(samples),
            ["AKIEC", "BCC", "MEL", "MEL", "NV", "OTHER", "OTHER", "OTHER"],
        )
        samples.close()

    def test_split_filter(self):
        samples = self.dataset.set_task(
            HAM10000Classification(image_size=16, split="val")
        )
        self.assertEqual(_labels(samples), ["MEL", "NV"])
        samples.close()

    def test_patient_split_keeps_lesions_together(self):
        parts = self.dataset.set_task(
            HAM10000Classification(image_size=16, split="train"),
            split=PatientSplit((0.5, 0.25, 0.25), seed=0),
        )
        owners = {}
        for index, part in enumerate(parts):
            for i in range(len(part)):
                owners.setdefault(part[i]["patient_id"], set()).add(index)
        self.assertEqual(len(owners["HAM_1"]), 1)
        self.assertTrue(all(len(v) == 1 for v in owners.values()))
        for part in parts:
            part.close()

    def test_bad_label_scheme_raises(self):
        with self.assertRaises(ValueError):
            HAM10000Classification(label_scheme="isic8")


if __name__ == "__main__":
    unittest.main()
