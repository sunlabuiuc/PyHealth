"""Tests for KiTS23Segmentation on a synthetic KiTS23 tree."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import KiTS23Dataset, get_dataloader
from pyhealth.tasks import KiTS23Segmentation
from tests.core.test_kits23_dataset import build_kits23_root, nib


@unittest.skipUnless(nib is not None, "nibabel is not installed")
class TestKiTS23Segmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        dataset_dir = build_kits23_root(cls.tmp / "kits23")
        # case_00001 keeps its CT but loses its mask.
        (dataset_dir / "case_00001" / "segmentation.nii.gz").unlink()
        cls.dataset = KiTS23Dataset(
            root=str(dataset_dir), cache_dir=str(cls.tmp / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_samples_need_ct_and_mask_not_label(self):
        samples = self.dataset.set_task(
            KiTS23Segmentation(target_shape=(4, 8, 8)), num_workers=1
        )
        try:
            # case_00000 (malignant) and case_00003 (no malignant label);
            # case_00001 has no mask and case_00002 no CT.
            self.assertEqual(
                sorted(s["patient_id"] for s in samples),
                ["case_00000", "case_00003"],
            )
            self.assertEqual(samples.output_processors["mask"].num_classes, 4)
            batch = next(iter(get_dataloader(samples, batch_size=2)))
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 4, 8, 8))
            self.assertEqual(tuple(batch["mask"].shape), (2, 1, 4, 8, 8))
            self.assertGreaterEqual(float(batch["image"].min()), 0.0)
            self.assertLessEqual(float(batch["image"].max()), 1.0)
            self.assertTrue(set(batch["mask"].unique().tolist()) <= {0, 1, 2})
        finally:
            samples.close()

    def test_unet3d_one_step(self):
        from pyhealth.models import UNet3D

        samples = self.dataset.set_task(
            KiTS23Segmentation(target_shape=(4, 8, 8)), num_workers=1
        )
        try:
            model = UNet3D(samples, base_channels=2, depth=2)
            self.assertEqual(model.num_classes, 4)
            batch = next(iter(get_dataloader(samples, batch_size=2)))
            out = model(**batch)
            out["loss"].backward()
            self.assertEqual(tuple(out["y_prob"].shape), (2, 4, 4, 8, 8))
        finally:
            samples.close()

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            KiTS23Segmentation(target_shape=(8, 8))
        with self.assertRaises(ValueError):
            KiTS23Segmentation(target_spacing=(1.0, 1.0))
        with self.assertRaises(ValueError):
            KiTS23Segmentation(hu_window=(1, 0))


if __name__ == "__main__":
    unittest.main()
