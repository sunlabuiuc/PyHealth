"""Tests for the array-level medical image transforms."""

import unittest

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.processors.image_io.transforms import (
    _normalize_ct,
    _normalize_mri,
    apply_window,
    canonicalize_volume,
    normalize_intensity,
    resample_volume,
)


class TestImageTransforms(unittest.TestCase):
    @unittest.skipUnless(nib is not None, "nibabel is required")
    def test_canonicalize(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.array(
            [[0, -2, 0, 10], [3, 0, 0, -5], [0, 0, 4, 2], [0, 0, 0, 1]], float
        )
        expected = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
        out, out_affine = canonicalize_volume(data, affine)
        np.testing.assert_array_equal(out, np.asanyarray(expected.dataobj))
        np.testing.assert_allclose(out_affine, expected.affine)

    @unittest.skipUnless(nib is not None, "nibabel is required")
    def test_resample(self):
        data = np.zeros((4, 5, 6), dtype=np.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        self.assertEqual(
            resample_volume(data, affine, target_shape=(2, 3, 4), order=0)[0].shape,
            (2, 3, 4),
        )
        self.assertEqual(
            resample_volume(data, affine, target_spacing=(1, 1, 1))[0].shape,
            (8, 15, 24),
        )
        same, same_affine = resample_volume(data, affine)
        self.assertEqual(same.shape, data.shape)
        np.testing.assert_allclose(same_affine, affine)
        for kwargs in ({"target_shape": (0, 2, 3)}, {"target_spacing": (1, 2)}):
            with self.assertRaisesRegex(ValueError, "three positive"):
                resample_volume(data, affine, **kwargs)

    def test_window(self):
        np.testing.assert_allclose(
            apply_window(np.array([0.0, 50.0, 100.0]), 50, 100), [0, 0.5, 1]
        )
        with self.assertRaisesRegex(ValueError, "non-zero"):
            apply_window(np.array([0.0, 1.0]), 0.5, 0)

    def test_modes(self):
        data = np.array([0.0, 5.0, 10.0])
        np.testing.assert_allclose(normalize_intensity(data), data)
        np.testing.assert_allclose(
            normalize_intensity(np.array([-10.0, 0, 10.0]), clip=(0, 10)),
            [0, 0, 10],
        )
        np.testing.assert_allclose(normalize_intensity(data, "minmax"), [0, 0.5, 1])
        np.testing.assert_allclose(
            normalize_intensity(np.array([-2000.0, 0, 2000.0]), "hu"), [0, 0.5, 1]
        )
        np.testing.assert_allclose(
            normalize_intensity(data, "percentile", percentile_range=(0, 100)),
            [0, 0.5, 1],
        )
        np.testing.assert_allclose(
            normalize_intensity(np.array([-10.0, 0, 10.0]), "minmax", clip=(0, 10)),
            [0, 0, 1],
        )
        np.testing.assert_allclose(
            normalize_intensity(np.array([0.0, 1, 2]), "zscore"),
            [-1.2247449, 0, 1.2247449],
            rtol=1e-6,
        )
        for mode in ("minmax", "zscore"):
            np.testing.assert_allclose(
                normalize_intensity(np.array([3.0, 3.0]), mode), [0, 0]
            )
        np.testing.assert_allclose(
            _normalize_ct(np.full((2, 2), 5.0), None, eps=1e-8), np.zeros((2, 2))
        )
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            normalize_intensity(data, "auto")
        with self.assertRaisesRegex(ValueError, "Unknown intensity mode"):
            normalize_intensity(data, "missing")

    def test_mri(self):
        result = normalize_intensity(np.array([0.0, 1, 2]), "mri")
        self.assertEqual(float(result[0]), 0)
        np.testing.assert_allclose(result[1:], [-1, 1])
        np.testing.assert_allclose(
            normalize_intensity(np.array([0.0, 1, 2]), "mri", use_nonzero_mask=False),
            [-1.2247449, 0, 1.2247449],
            rtol=1e-6,
        )
        zeros = np.zeros((2, 2), dtype=np.float32)
        result = normalize_intensity(zeros, "mri")
        np.testing.assert_allclose(result, zeros)
        self.assertIsNot(result, zeros)
        np.testing.assert_allclose(
            _normalize_mri(zeros, True, 1e-8, background_threshold=0.5), zeros
        )
        out = normalize_intensity(
            np.array([0.0, 1, 2]), "mri", mri_background_threshold=0.5
        )
        self.assertEqual(float(out[0]), 0)
        self.assertGreater(float(out[-1]), float(out[1]))


if __name__ == "__main__":
    unittest.main()
