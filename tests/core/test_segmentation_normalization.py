"""Tests for nnU-Net-style intensity normalisation."""

import unittest

import numpy as np

from pyhealth.processors.image_io.transforms import normalize_intensity


class TestSegmentationNormalization(unittest.TestCase):
    def test_ct_uses_supplied_stats(self):
        data = np.array([-1000.0, 0.0, 100.0], dtype=np.float32)
        stats = {
            "percentile_00_5": -500.0,
            "percentile_99_5": 500.0,
            "mean": 0.0,
            "std": 100.0,
        }
        result = normalize_intensity(data, "ct", intensity_stats=stats)
        np.testing.assert_allclose(result, [-5.0, 0.0, 1.0], atol=1e-6)

    def test_ct_without_stats_does_not_crash(self):
        data = np.array([-1000.0, 0.0, 50.0, 100.0, 200.0], dtype=np.float32)
        result = normalize_intensity(data, "ct")
        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_mri_normalises_nonzero_voxels_only(self):
        data = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = normalize_intensity(data, "mri")
        np.testing.assert_allclose(result, [0.0, -1.0, 1.0], atol=1e-6)

    def test_mri_suppresses_low_level_background_noise(self):
        data = np.concatenate(
            [
                np.zeros(50, dtype=np.float32),
                np.ones(10, dtype=np.float32),
                np.array([100.0, 200.0, 300.0], dtype=np.float32),
            ]
        )
        result = normalize_intensity(data, "mri", mri_background_threshold=0.05)
        np.testing.assert_allclose(result[:60], 0.0)
        self.assertTrue(np.any(result[60:] != 0.0))

    def test_mri_can_normalise_all_voxels(self):
        data = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = normalize_intensity(data, "mri", use_nonzero_mask=False)
        self.assertAlmostEqual(float(result.mean()), 0.0, places=6)

    def test_auto_must_be_resolved_first(self):
        with self.assertRaises(ValueError):
            normalize_intensity(np.array([1.0, 2.0]), "auto")


if __name__ == "__main__":
    unittest.main()
