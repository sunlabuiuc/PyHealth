"""Tests for LNDbTextureClassification on a synthetic LNDb tree."""

import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from pyhealth.datasets import LNDbDataset
from pyhealth.tasks import LNDbTextureClassification
from pyhealth.tasks.lndb_texture import lndb_texture_class
from tests.core.test_lndb_dataset import build_lndb_root


class TestLNDbTextureClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "lndb"
        build_lndb_root(root, with_scans=(1, 3))
        gt_path = root / "trainset_csv" / "trainNodules_gt.csv"
        gt = pd.read_csv(gt_path)
        # LNDb-0001_1 (>= 3 mm) becomes part-solid; add a GGO nodule to CT 3.
        gt.loc[(gt.LNDbID == 1) & (gt.FindingID == 1), "Text"] = 3.0
        extra = gt[(gt.LNDbID == 3) & (gt.FindingID == 1)].copy()
        extra["FindingID"], extra["Text"], extra["x"] = 3, 1.5, -2.0
        pd.concat([gt, extra]).to_csv(gt_path, index=False)
        cls.dataset = LNDbDataset(root=str(root), cache_dir=str(cls.tmp / "cache"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_only_rated_nodules_ge_3mm_with_official_classes(self):
        task = LNDbTextureClassification(patch_size=8, patch_mode="2.5d")
        samples = self.dataset.set_task(task, num_workers=1)
        try:
            # Non-nodules and the < 3 mm nodule (rated 3.0) are left out.
            self.assertEqual(len(samples), 3)
            self.assertEqual(samples.output_processors["texture_class"].size(), 3)
            self.assertEqual(tuple(samples[0]["patch"].shape), (3, 8, 8))
            by_record = {s["record_id"]: int(s["texture_class"]) for s in samples}
            self.assertEqual(
                by_record,
                {"LNDb-0001_1": 1, "LNDb-0003_1": 2, "LNDb-0003_3": 0},
            )
            self.assertEqual(Counter(by_record.values()), Counter({0: 1, 1: 1, 2: 1}))
        finally:
            samples.close()

    def test_thresholds_match_official_script(self):
        self.assertEqual(lndb_texture_class(7 / 3 - 1e-6), 0)
        self.assertEqual(lndb_texture_class(7 / 3), 1)
        self.assertEqual(lndb_texture_class(11 / 3), 1)
        self.assertEqual(lndb_texture_class(11 / 3 + 1e-6), 2)

    def test_task_name_and_invalid_mode(self):
        self.assertEqual(
            self.dataset.default_task.task_name, "LNDbFindingClassification"
        )
        with self.assertRaises(ValueError):
            LNDbTextureClassification(patch_mode="2d")


if __name__ == "__main__":
    unittest.main()
