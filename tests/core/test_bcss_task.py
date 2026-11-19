"""Tests for BCSSTileClassification (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import BCSSDataset
from pyhealth.tasks import BCSSTileClassification
from tests.core.test_bcss_dataset import write_fixture


class TestBCSSTileClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        write_fixture(cls.tmp / "bcss")
        cls.dataset = BCSSDataset(
            root=str(cls.tmp / "bcss"), tile_size=4, cache_dir=str(cls.tmp / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_bad_classes(self):
        with self.assertRaises(ValueError):
            BCSSTileClassification(classes=["tumor", "tumor"])

    def test_call_and_class_subset(self):
        patient = self.dataset.get_patient("TCGA-A2-BBBB")
        labels = sorted(s["label"] for s in BCSSTileClassification()(patient))
        self.assertEqual(
            labels,
            [
                "lymphocytic_infiltrate",
                "lymphocytic_infiltrate",
                "necrosis_or_debris",
                "tumor",
            ],
        )
        subset = BCSSTileClassification(classes=["tumor"])(patient)
        self.assertEqual([s["label"] for s in subset], ["tumor"])

    def test_official_split_set_task(self):
        train, test = self.dataset.set_task(
            BCSSTileClassification(image_size=4), split=self.dataset.official_split()
        )
        try:
            self.assertEqual(len(train), 6)
            self.assertEqual(len(test), 3)
            self.assertEqual(tuple(train[0]["image"].shape), (3, 4, 4))
            self.assertEqual(train.output_processors["label"].size(), 4)
            self.assertEqual(
                {test[i]["patient_id"] for i in range(len(test))}, {"TCGA-E2-CCCC"}
            )
        finally:
            train.close()
            test.close()


if __name__ == "__main__":
    unittest.main()
