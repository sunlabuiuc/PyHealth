"""Tests for ISIC2019Classification using synthetic images."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import ISIC2019Dataset, PatientSplit
from pyhealth.tasks import ISIC2019Classification
from tests.core.test_isic2019_dataset import make_isic2019_root


def _labels(samples):
    proc = samples.output_processors["diagnosis"]
    inverse = {v: k for k, v in proc.label_vocab.items()}
    return sorted(inverse[int(samples[i]["diagnosis"])] for i in range(len(samples)))


class TestISIC2019Classification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_isic2019_root(cls.tmp / "isic2019")
        cls.dataset = ISIC2019Dataset(
            root=str(cls.tmp / "isic2019"),
            splits=("train", "test"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_isic8_training_samples(self):
        samples = self.dataset.set_task(
            ISIC2019Classification(image_size=16, split="train")
        )
        self.assertEqual(samples.output_processors["diagnosis"].size(), 8)
        self.assertEqual(
            _labels(samples),
            ["AK", "BCC", "BKL", "DF", "MEL", "MEL", "NV", "SCC", "VASC"],
        )
        self.assertEqual(tuple(samples[0]["image"].shape), (3, 16, 16))
        samples.close()

    def test_climb5(self):
        samples = self.dataset.set_task(
            ISIC2019Classification(label_scheme="climb5", image_size=16, split="train")
        )
        self.assertEqual(samples.output_processors["diagnosis"].size(), 5)
        self.assertEqual(
            _labels(samples),
            ["AKIEC", "AKIEC", "BCC", "MEL", "MEL", "NV", "OTHER", "OTHER", "OTHER"],
        )
        samples.close()

    def test_climb5_test_split(self):
        samples = self.dataset.set_task(
            ISIC2019Classification(label_scheme="climb5", image_size=16, split="test")
        )
        # UNK (ISIC_0060001) is skipped; SCC maps to AKIEC.
        self.assertEqual(_labels(samples), ["AKIEC", "MEL", "NV"])
        self.assertTrue(all(samples[i]["split"] == "test" for i in range(len(samples))))
        samples.close()

    def test_bad_split_raises(self):
        with self.assertRaises(ValueError):
            ISIC2019Classification(split="val")

    def test_exclude_ham10000_and_skip_unk(self):
        samples = self.dataset.set_task(
            ISIC2019Classification(image_size=16, exclude_ham10000=True)
        )
        ids = {samples[i]["image_id"] for i in range(len(samples))}
        # HAM_ training lesions, ISIC 2018 holdout test images and UNK are gone
        self.assertEqual(len(ids), 8)
        for image in (
            "ISIC_0000003",
            "ISIC_0000004",
            "ISIC_0034400",
            "ISIC_0036064",
            "ISIC_0060001",
        ):
            self.assertNotIn(image, ids)
        self.assertIn("ISIC_0060000", ids)
        samples.close()

    def test_patient_split_keeps_lesions_together(self):
        parts = self.dataset.set_task(
            ISIC2019Classification(image_size=16, split="train"),
            split=PatientSplit((0.5, 0.25, 0.25), seed=0),
        )
        owners = {}
        for index, part in enumerate(parts):
            for i in range(len(part)):
                owners.setdefault(part[i]["patient_id"], set()).add(index)
        self.assertTrue(all(len(v) == 1 for v in owners.values()))
        self.assertEqual(sum(len(p) for p in parts), 9)
        for part in parts:
            part.close()

    def test_bad_label_scheme_raises(self):
        with self.assertRaises(ValueError):
            ISIC2019Classification(label_scheme="ham7")


if __name__ == "__main__":
    unittest.main()
