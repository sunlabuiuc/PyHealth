"""Tests for CBISDDSMAbnormalityClassification on a synthetic CBIS-DDSM root."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import CBISDDSMDataset, get_dataloader
from pyhealth.tasks import CBISDDSMAbnormalityClassification
from tests.core.cbis_ddsm_test_utils import ROWS, build_cbis_root, pydicom

# P_00001's left CC image also gets a calcification row in train, so it has
# both types and is skipped.
_ROWS_WITH_BOTH = ROWS + [
    ("calc", "train", "P_00001", "LEFT", "CC", 3, 3, "BENIGN", "1.1", True),
]


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCBISDDSMAbnormalityClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root = cls.temp_dir / "cbis"
        root.mkdir()
        build_cbis_root(root, rows=_ROWS_WITH_BOTH)
        cls.dataset = CBISDDSMDataset(
            root=str(root), cache_dir=str(cls.temp_dir / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_labels_and_mixed_image_skipped(self):
        samples = self.dataset.set_task(
            CBISDDSMAbnormalityClassification(), num_workers=1
        )
        labels = {s["record_id"]: int(s["mass"].item()) for s in samples}
        self.assertEqual(
            labels,
            {
                "P_00002_RIGHT_MLO_train": 0,
                "P_00003_LEFT_MLO_test": 1,
                "P_00004_LEFT_CC_train": 0,
                "P_00004_LEFT_CC_test": 1,
            },
        )
        batch = next(iter(get_dataloader(samples, batch_size=2)))
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 512, 512))

    def test_split_filter(self):
        task = CBISDDSMAbnormalityClassification(split="train")
        samples = task(self.dataset.get_patient("P_00004"))
        self.assertEqual([(s["split"], s["mass"]) for s in samples], [("train", 0)])
        with self.assertRaises(ValueError):
            CBISDDSMAbnormalityClassification(split="val")


if __name__ == "__main__":
    unittest.main()
