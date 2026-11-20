"""Tests for the KiTS23 malignancy tasks on a synthetic KiTS23 tree."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyhealth.datasets import KiTS23Dataset, get_dataloader
from pyhealth.tasks import (
    KiTS23MalignancyClassification,
    KiTS23MalignancySliceClassification,
)
from tests.core.test_kits23_dataset import build_kits23_root, nib


@unittest.skipUnless(nib is not None, "nibabel is not installed")
class TestKiTS23Tasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        dataset_dir = build_kits23_root(cls.tmp / "kits23")
        cls.dataset = KiTS23Dataset(root=str(dataset_dir), cache_dir=str(cls.tmp / "cache"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_volume_task(self):
        task = KiTS23MalignancyClassification(target_shape=(4, 8, 8))
        samples = self.dataset.set_task(task, num_workers=1)
        try:
            # case_00002 has no imaging and case_00003 no label.
            self.assertEqual(len(samples), 2)
            self.assertEqual(
                sorted(int(s["malignant"].item()) for s in samples), [0, 1]
            )
            batch = next(iter(get_dataloader(samples, batch_size=2)))
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 4, 8, 8))
            self.assertGreaterEqual(float(batch["image"].min()), 0.0)
            self.assertLessEqual(float(batch["image"].max()), 1.0)
        finally:
            samples.close()

    def test_slice_task(self):
        task = KiTS23MalignancySliceClassification(image_size=12)
        samples = self.dataset.set_task(task, num_workers=1)
        try:
            self.assertEqual(len(samples), 2)
            self.assertEqual(tuple(samples[0]["image"].shape), (3, 12, 12))
        finally:
            samples.close()

    def test_slice_task_skips_case_without_segmentation(self):
        root = self.tmp / "no_seg"
        dataset_dir = build_kits23_root(root)
        (dataset_dir / "case_00001" / "segmentation.nii.gz").unlink()
        dataset = KiTS23Dataset(
            root=str(dataset_dir), cache_dir=str(self.tmp / "cache_no_seg")
        )
        slice_task = KiTS23MalignancySliceClassification(image_size=12)
        volume_task = KiTS23MalignancyClassification(target_shape=(4, 8, 8))
        # Called per patient: a set_task over one label would fail in the
        # binary label processor, which is not what is tested here.
        no_seg = dataset.get_patient("case_00001")
        self.assertEqual(slice_task(no_seg), [])
        # It still has imaging, so only the 2.5D task drops it.
        self.assertEqual(len(volume_task(no_seg)), 1)
        self.assertEqual(len(slice_task(dataset.get_patient("case_00000"))), 1)

    def test_slice_task_skips_shape_mismatch(self):
        dataset_dir = build_kits23_root(self.tmp / "mismatch")
        seg = np.zeros((8, 8, 8), dtype=np.uint8)
        seg[2:6, 2:6, 3] = 2
        nib.save(
            nib.Nifti1Image(seg, np.eye(4)),
            dataset_dir / "case_00001" / "segmentation.nii.gz",
        )
        dataset = KiTS23Dataset(
            root=str(dataset_dir), cache_dir=str(self.tmp / "cache_mismatch")
        )
        task = KiTS23MalignancySliceClassification(image_size=12)
        with self.assertLogs("pyhealth.tasks.kits23_classification", "WARNING") as logs:
            self.assertEqual(task(dataset.get_patient("case_00001")), [])
        self.assertIn("case_00001", logs.output[0])
        self.assertEqual(len(task(dataset.get_patient("case_00000"))), 1)

    def test_best_slice_prefers_tumour_then_kidney(self):
        seg = np.zeros((6, 4, 4), dtype=int)
        seg[1, :, :] = 1
        self.assertEqual(KiTS23MalignancySliceClassification.best_slice(seg), 1)
        seg[4, :1, :1] = 2
        self.assertEqual(KiTS23MalignancySliceClassification.best_slice(seg), 4)
        self.assertEqual(
            KiTS23MalignancySliceClassification.best_slice(np.zeros((6, 2, 2))), 3
        )

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            KiTS23MalignancyClassification(target_shape=(8, 8))
        with self.assertRaises(ValueError):
            KiTS23MalignancySliceClassification(hu_window=(1, 0))


if __name__ == "__main__":
    unittest.main()
