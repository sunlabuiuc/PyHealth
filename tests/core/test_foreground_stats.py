"""Tests for foreground intensity statistics."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.datasets import compute_foreground_intensity_stats, create_sample_dataset


class TestForegroundStats(unittest.TestCase):
    def test_collects_only_foreground_voxels(self):
        stats = compute_foreground_intensity_stats(
            [
                {
                    "image": np.array([[-1000.0, 10.0], [20.0, 30.0]]),
                    "mask": np.array([[0, 1], [1, 1]]),
                }
            ]
        )
        self.assertAlmostEqual(stats["mean"], 20.0, places=5)
        self.assertAlmostEqual(stats["percentile_00_5"], 10.1, places=5)
        self.assertAlmostEqual(stats["percentile_99_5"], 29.9, places=5)

    def test_modality_filter(self):
        samples = [
            {"modality": "CT", "image": np.array([0.0, 10.0]), "mask": np.array([0, 1])},
            {"modality": "MR", "image": np.array([0.0, 100.0]), "mask": np.array([0, 1])},
        ]
        stats = compute_foreground_intensity_stats(samples, modality="CT")
        self.assertAlmostEqual(stats["mean"], 10.0, places=5)

    def test_max_voxels_keeps_reservoir_bounded(self):
        from pyhealth.datasets.imaging.stats import _update_reservoir

        reservoir = np.empty(0, dtype=np.float32)
        reservoir, seen = _update_reservoir(
            reservoir,
            np.arange(100.0, dtype=np.float32),
            0,
            7,
            np.random.default_rng(0),
        )
        self.assertEqual(reservoir.size, 7)
        self.assertEqual(seen, 100)
        stats = compute_foreground_intensity_stats(
            [{"image": np.arange(100.0), "mask": np.ones(100, dtype=int)}],
            max_voxels=7,
        )
        self.assertTrue(np.isfinite(stats["mean"]))

    def test_no_foreground_raises(self):
        with self.assertRaises(ValueError):
            compute_foreground_intensity_stats(
                [{"image": np.array([1.0, 2.0]), "mask": np.array([0, 0])}]
            )

    @unittest.skipUnless(nib is not None, "nibabel is not installed")
    def test_lazy_dataset_uses_raw_values(self):
        temp_dir = tempfile.mkdtemp()
        try:
            image_path = Path(temp_dir) / "image.nii.gz"
            mask_path = Path(temp_dir) / "mask.nii.gz"
            image = np.array([[[0.0, 100.0], [200.0, 300.0]]], dtype=np.float32)
            mask = np.array([[[0, 1], [1, 1]]], dtype=np.int16)
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
            dataset = create_sample_dataset(
                [{"patient_id": "p0", "image": str(image_path), "mask": str(mask_path)}],
                {"image": ("nifti", {"lazy": True})},
                {"mask": ("nifti", {"lazy": True, "is_mask": True})},
            )
            stats = compute_foreground_intensity_stats(dataset)
            self.assertAlmostEqual(stats["mean"], 200.0, places=4)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
