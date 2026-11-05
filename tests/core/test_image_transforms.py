"""Tests for the array-level medical image transforms."""

import unittest

import numpy as np

from pyhealth.processors.image_io.transforms import (
    _normalize_ct,
    _normalize_mri,
    apply_window,
    canonicalize_volume,
    normalize_intensity,
    resample_volume,
)


class TestCanonicalizeVolume(unittest.TestCase):
    def test_matches_nibabel_canonical(self):
        import nibabel as nib

        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.array(
            [
                [0.0, -2.0, 0.0, 10.0],
                [3.0, 0.0, 0.0, -5.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected = nib.as_closest_canonical(nib.Nifti1Image(data, affine))

        canonical, canonical_affine = canonicalize_volume(data, affine)

        np.testing.assert_array_equal(canonical, np.asanyarray(expected.dataobj))
        np.testing.assert_allclose(canonical_affine, expected.affine)


class TestResampleVolume(unittest.TestCase):
    def test_target_shape(self):
        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

        resampled, resampled_affine = resample_volume(
            data, affine, target_shape=(2, 3, 4), order=0
        )

        self.assertEqual(resampled.shape, (2, 3, 4))
        self.assertEqual(resampled_affine.shape, (4, 4))

    def test_target_spacing(self):
        data = np.zeros((4, 5, 6), dtype=np.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

        resampled, _ = resample_volume(data, affine, target_spacing=(1, 1, 1))

        self.assertEqual(resampled.shape, (8, 15, 24))

    def test_without_target_returns_input(self):
        data = np.ones((2, 3, 4), dtype=np.float32)
        affine = np.eye(4)

        resampled, resampled_affine = resample_volume(data, affine)

        self.assertEqual(resampled.shape, data.shape)
        np.testing.assert_allclose(resampled_affine, affine)

    def test_invalid_targets_raise(self):
        data = np.zeros((2, 3, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "three positive"):
            resample_volume(data, np.eye(4), target_shape=(0, 2, 3))
        with self.assertRaisesRegex(ValueError, "three positive"):
            resample_volume(data, np.eye(4), target_spacing=(1, 2))


class TestApplyWindow(unittest.TestCase):
    def test_scales_to_unit_range(self):
        result = apply_window(np.array([0.0, 50.0, 100.0]), 50.0, 100.0)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_zero_width_raises(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            apply_window(np.array([0.0, 1.0]), 0.5, 0.0)


class TestNormalizeIntensity(unittest.TestCase):
    def test_none_returns_float32_copy(self):
        result = normalize_intensity(np.array([1, 2, 3], dtype=np.int64))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_minmax_and_constant_input(self):
        np.testing.assert_allclose(
            normalize_intensity(np.array([0.0, 5.0, 10.0]), "minmax"),
            [0.0, 0.5, 1.0],
        )
        np.testing.assert_allclose(
            normalize_intensity(np.array([3.0, 3.0]), "minmax"), [0.0, 0.0]
        )

    def test_zscore_and_constant_input(self):
        np.testing.assert_allclose(
            normalize_intensity(np.array([0.0, 1.0, 2.0]), "zscore"),
            [-1.2247449, 0.0, 1.2247449],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            normalize_intensity(np.array([3.0, 3.0]), "zscore"), [0.0, 0.0]
        )

    def test_percentile_and_hu(self):
        np.testing.assert_allclose(
            normalize_intensity(
                np.array([0.0, 5.0, 10.0]),
                "percentile",
                percentile_range=(0, 100),
            ),
            [0.0, 0.5, 1.0],
        )
        np.testing.assert_allclose(
            normalize_intensity(np.array([-2000.0, 0.0, 2000.0]), "hu"),
            [0.0, 0.5, 1.0],
        )

    def test_clip_before_normalisation(self):
        result = normalize_intensity(
            np.array([-10.0, 0.0, 10.0]), "minmax", clip=(0.0, 10.0)
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0])

    def test_auto_and_unknown_modes_raise(self):
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            normalize_intensity(np.array([0.0, 1.0]), "auto")
        with self.assertRaisesRegex(ValueError, "Unknown intensity mode"):
            normalize_intensity(np.array([0.0, 1.0]), "missing")


class TestCtAndMriNormalisation(unittest.TestCase):
    def test_ct_fallback_handles_constant_volume(self):
        result = _normalize_ct(np.full((2, 2), 5.0), None, eps=1e-8)
        np.testing.assert_allclose(result, np.zeros((2, 2)))

    def test_mri_nonzero_mask_keeps_background_zero(self):
        result = normalize_intensity(
            np.array([0.0, 1.0, 2.0]), "mri", use_nonzero_mask=True
        )
        self.assertEqual(float(result[0]), 0.0)
        np.testing.assert_allclose(result[1:], [-1.0, 1.0])

    def test_mri_all_zero_returns_copy(self):
        data = np.zeros((2, 2), dtype=np.float32)
        result = normalize_intensity(data, "mri", use_nonzero_mask=True)
        np.testing.assert_allclose(result, data)
        self.assertIsNot(result, data)

    def test_mri_background_threshold_zeroes_noise(self):
        result = normalize_intensity(
            np.array([0.0, 1.0, 2.0]),
            "mri",
            mri_background_threshold=0.5,
        )
        self.assertEqual(float(result[0]), 0.0)
        self.assertGreater(float(result[-1]), float(result[1]))

    def test_mri_without_nonzero_mask_z_scores_everything(self):
        result = normalize_intensity(
            np.array([0.0, 1.0, 2.0]), "mri", use_nonzero_mask=False
        )
        np.testing.assert_allclose(result, [-1.2247449, 0.0, 1.2247449], rtol=1e-6)

    def test_mri_background_threshold_without_foreground(self):
        data = np.zeros((2, 2), dtype=np.float32)
        result = _normalize_mri(
            data, use_nonzero_mask=True, eps=1e-8, background_threshold=0.5
        )
        np.testing.assert_allclose(result, data)


if __name__ == "__main__":
    unittest.main()
