"""Tests for LNDbFindingClassification on a synthetic LNDb tree."""

import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from pyhealth.datasets import LNDbDataset, get_dataloader
from pyhealth.tasks import LNDbFindingClassification
from tests.core.test_lndb_dataset import build_lndb_root


class TestLNDbFindingClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "lndb"
        build_lndb_root(root, with_scans=(1, 3))
        cls.dataset = LNDbDataset(root=str(root), cache_dir=str(cls.tmp / "cache"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_three_class_3d_patches(self):
        samples = self.dataset.set_task(
            LNDbFindingClassification(patch_size=8), num_workers=1
        )
        try:
            # 4 findings have a CT on disk; LNDb-0002 has none.
            self.assertEqual(len(samples), 4)
            self.assertEqual(
                samples.output_processors["finding_class"].size(), 3
            )
            patch = samples[0]["patch"]
            self.assertEqual(tuple(patch.shape), (1, 8, 8, 8))
            self.assertGreaterEqual(float(patch.min()), 0.0)
            self.assertLessEqual(float(patch.max()), 1.0)
            labels = Counter(int(s["finding_class"]) for s in samples)
            self.assertEqual(labels, Counter({2: 2, 1: 1, 0: 1}))
            batch = next(iter(get_dataloader(samples, batch_size=4)))
            self.assertEqual(tuple(batch["patch"].shape), (4, 1, 8, 8, 8))
        finally:
            samples.close()

    def test_binary_25d_patches(self):
        task = LNDbFindingClassification(
            patch_size=6, patch_mode="2.5d", binary=True
        )
        samples = self.dataset.set_task(task, num_workers=1)
        try:
            self.assertEqual(tuple(samples[0]["patch"].shape), (3, 6, 6))
            labels = sorted(int(s["nodule_ge_3mm"]) for s in samples)
            self.assertEqual(labels, [0, 0, 1, 1])
        finally:
            samples.close()

    def test_binary_needs_both_labels(self):
        # With min_agreement=2 only nodules >= 3 mm remain, so the binary
        # label processor sees one label and fails with a clear error.
        task = LNDbFindingClassification(
            patch_size=4, binary=True, min_agreement=2
        )
        with self.assertRaisesRegex(ValueError, "2 unique labels"):
            self.dataset.set_task(task, num_workers=1)

    def test_min_agreement_filters(self):
        task = LNDbFindingClassification(patch_size=4, min_agreement=2)
        samples = self.dataset.set_task(task, num_workers=1)
        try:
            self.assertEqual(len(samples), 2)
        finally:
            samples.close()

    def test_crop_pads_outside_volume_with_air(self):
        task = LNDbFindingClassification(patch_size=4, hu_window=(-1000, 0))
        volume = np.zeros((4, 4, 4), dtype=np.int16)
        patch = task.crop(volume, (100, 100, 100))
        self.assertTrue(np.all(patch == 0.0))
        patch = task.crop(volume, (2, 2, 2))
        self.assertTrue(np.all(patch == 1.0))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            LNDbFindingClassification(patch_mode="2d")
        with self.assertRaises(ValueError):
            LNDbFindingClassification(hu_window=(10, -10))


if __name__ == "__main__":
    unittest.main()
