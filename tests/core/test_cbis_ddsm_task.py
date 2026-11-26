"""Tests for the CBIS-DDSM image-level classification tasks."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import CBISDDSMDataset, get_dataloader
from pyhealth.tasks import (
    CBISDDSMBIRADSClassification,
    CBISDDSMPathologyClassification,
)
from tests.core.cbis_ddsm_test_utils import build_cbis_root, pydicom


def _labels(samples, key):
    return {
        sample["record_id"]: int(sample[key].argmax()) if sample[key].numel() > 1
        else int(sample[key].item())
        for sample in samples
    }


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCBISDDSMTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root = cls.temp_dir / "cbis"
        root.mkdir()
        build_cbis_root(root)
        cls.dataset = CBISDDSMDataset(
            root=str(root), cache_dir=str(cls.temp_dir / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_pathology(self):
        samples = self.dataset.set_task(
            CBISDDSMPathologyClassification(), num_workers=1
        )
        self.assertEqual(
            _labels(samples, "malignant"),
            {
                "P_00001_LEFT_CC_train": 1,  # one malignant lesion is enough
                "P_00002_RIGHT_MLO_train": 0,  # BENIGN_WITHOUT_CALLBACK
                "P_00003_LEFT_MLO_test": 1,
                "P_00004_LEFT_CC_train": 0,
                "P_00004_LEFT_CC_test": 1,
            },
        )
        image = samples.input_processors["image"].materialize(samples[0]["image"])
        self.assertEqual(tuple(image.shape), (1, 512, 512))
        batch = next(iter(get_dataloader(samples, batch_size=2)))
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 512, 512))
        self.assertIsInstance(batch["malignant"], torch.Tensor)

    def test_birads_drops_zero_and_takes_max(self):
        samples = self.dataset.set_task(CBISDDSMBIRADSClassification(), num_workers=1)
        self.assertEqual(samples.output_processors["birads"].size(), 5)
        # class index = BI-RADS - 1
        self.assertEqual(
            _labels(samples, "birads"),
            {
                "P_00001_LEFT_CC_train": 3,  # max(4, 3)
                "P_00002_RIGHT_MLO_train": 1,  # BI-RADS 0 lesion dropped
                "P_00003_LEFT_MLO_test": 4,
                "P_00004_LEFT_CC_train": 3,
                "P_00004_LEFT_CC_test": 3,
            },
        )

    def test_split_filter(self):
        samples = self.dataset.set_task(
            CBISDDSMBIRADSClassification(split="test"), num_workers=1
        )
        self.assertEqual({sample["split"] for sample in samples}, {"test"})
        self.assertEqual(len(samples), 2)
        with self.assertRaises(ValueError):
            CBISDDSMPathologyClassification(split="val")


if __name__ == "__main__":
    unittest.main()
