"""Tests for the shared per-recording IBI sleep-staging helpers.

All waveforms and beat times are synthetic; no real dataset is required.
"""

import unittest

import numpy as np

from pyhealth.tasks.sleep_staging_ibi import (
    SleepStagingIBIDREAMT,
    SleepStagingIBIMESA,
    SleepStagingIBISHHS,
    _beat_annotations_to_ibi,
    _detect_peaks,
    _heart_rate_to_ibi,
    _ibi_to_epoch_grid,
    _peaks_to_ibi,
    _to_three_class,
    _truncate_epochs,
)


class TestIBIHelpers(unittest.TestCase):
    def test_peaks_to_ibi_sample_and_hold(self):
        ibi = _peaks_to_ibi([0, 100, 250], 100, 2.0, length=300)
        self.assertEqual(ibi.shape, (300,))
        np.testing.assert_allclose(ibi[:100], 1.0)
        np.testing.assert_allclose(ibi[100:250], 1.5)
        np.testing.assert_allclose(ibi[250:], 0.0)

    def test_peaks_to_ibi_long_interval_and_few_peaks(self):
        ibi = _peaks_to_ibi([0, 300], 100, 2.0, length=400)
        np.testing.assert_allclose(ibi, 0.0)
        np.testing.assert_allclose(_peaks_to_ibi([10], 100, 2.0, 20), 0.0)

    def test_beat_annotations_delegate_to_peak_expansion(self):
        left = _beat_annotations_to_ibi([0, 50, 110], 50, 1.5, 120)
        right = _peaks_to_ibi([0, 50, 110], 50, 1.5, 120)
        np.testing.assert_array_equal(left, right)

    def test_heart_rate_to_ibi(self):
        ibi = _heart_rate_to_ibi([0.0, 1.0, 2.0], [60.0, 60.0, 120.0], 2.0)
        self.assertEqual(ibi.shape, (3,))
        np.testing.assert_allclose(ibi[:2], [1.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(ibi[2]), 0.5)

    def test_ibi_to_epoch_grid_stride(self):
        ibi = np.arange(4 * 100, dtype=np.float64)
        grid = _ibi_to_epoch_grid(ibi, 30.0, 100, 4, downsample_factor=1)
        self.assertEqual(grid.shape, (4, 100))
        self.assertEqual(grid.dtype, np.float32)
        np.testing.assert_allclose(grid[0], ibi[:100])

    def test_ibi_to_epoch_grid_resamples_when_stride_mismatches(self):
        ibi = np.ones(4 * 120)
        grid = _ibi_to_epoch_grid(ibi, 30.0, 100, 4, downsample_factor=7)
        self.assertEqual(grid.shape, (4, 100))
        np.testing.assert_allclose(grid, 1.0, atol=0.1)

    def test_stage_maps(self):
        self.assertEqual(_to_three_class("W"), 0)
        self.assertEqual(_to_three_class("N1"), 1)
        self.assertEqual(_to_three_class("N4"), 1)
        self.assertEqual(_to_three_class("R"), 2)
        self.assertEqual(_to_three_class("Missing"), -1)
        self.assertEqual(_to_three_class("P"), -1)
        self.assertEqual(_to_three_class("S4", scheme="rk"), 1)
        labels = _to_three_class([0, 1, 2, 3, 4, -1])
        self.assertEqual(labels.tolist(), [0, 1, 1, 1, 2, -1])
        self.assertEqual(labels.dtype, np.int64)

    def test_truncate_epochs(self):
        signal, label = _truncate_epochs(np.ones((5, 750)), np.arange(5), 3)
        self.assertEqual(signal.shape, (3, 750))
        self.assertEqual(label.tolist(), [0, 1, 2])

    def test_detect_peaks_scipy_fallback(self):
        waveform = np.zeros(300)
        waveform[50] = waveform[150] = waveform[250] = 10.0
        peaks = _detect_peaks(waveform, 100, "ecg", backend="scipy")
        self.assertEqual(peaks.tolist(), [50, 150, 250])

    def test_invalid_constructor_arguments(self):
        with self.assertRaises(ValueError):
            SleepStagingIBIDREAMT(epoch_seconds=0)
        with self.assertRaises(ValueError):
            SleepStagingIBIDREAMT(points_per_epoch=0)
        with self.assertRaises(ValueError):
            SleepStagingIBIDREAMT(max_epochs=0)
        with self.assertRaises(ValueError):
            SleepStagingIBIDREAMT(peak_backend="nope")
        with self.assertRaises(ValueError):
            SleepStagingIBIDREAMT(signal_source="nope")
        with self.assertRaises(ValueError):
            SleepStagingIBISHHS(source_fs=0)
        with self.assertRaises(ValueError):
            SleepStagingIBIMESA(downsample_factor=0)


if __name__ == "__main__":
    unittest.main()
