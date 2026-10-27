"""Regression tests for HistogramBinning's bin-count setting.

Author: Shenqi Zhou
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from pyhealth.calib.calibration import HistogramBinning


class _CalibrationModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.device = torch.device("cpu")


class TestHistogramBinning(unittest.TestCase):
    def _calibrate(self, mode, n_samples, nbins=None, one_hot=False):
        positive_prob = np.linspace(0.1, 0.9, n_samples)
        labels = (positive_prob > 0.5).astype(int)
        if mode == "binary":
            y_prob = positive_prob[:, None]
            y_true = labels[:, None]
        else:
            y_prob = np.column_stack([1 - positive_prob, positive_prob])
            y_true = labels
            if mode == "multilabel" or one_hot:
                y_true = np.column_stack([1 - labels, labels])

        calibrator = HistogramBinning(_CalibrationModel(mode))
        # Supply model outputs directly; fitting still uses real HB_binary objects.
        with patch(
            "pyhealth.calib.calibration.hb.prepare_numpy_dataset",
            return_value={"y_prob": y_prob, "y_true": y_true},
        ):
            if nbins is None:
                calibrator.calibrate(cal_dataset=[])
            else:
                calibrator.calibrate(cal_dataset=[], nbins=nbins)
        return calibrator

    def test_two_bins_with_fewer_than_fifteen_samples(self):
        for mode, one_hot in [
            ("binary", False),
            ("multiclass", False),
            ("multiclass", True),
            ("multilabel", False),
        ]:
            with self.subTest(mode=mode, one_hot=one_hot):
                calibrator = self._calibrate(
                    mode, n_samples=8, nbins=2, one_hot=one_hot
                )
                for fitted in calibrator.calib:
                    self.assertEqual(fitted.mean_pred_values.shape, (2,))
                    self.assertEqual(fitted.num_calibration_examples_in_bin.sum(), 8)
                    np.testing.assert_allclose(
                        fitted.predict_proba(np.array([0.15, 0.85])),
                        [0.0, 1.0],
                        atol=1e-8,
                    )

    def test_custom_bin_count_above_default(self):
        for mode in ("binary", "multiclass", "multilabel"):
            with self.subTest(mode=mode):
                calibrator = self._calibrate(mode, n_samples=24, nbins=20)
                for fitted in calibrator.calib:
                    self.assertEqual(fitted.mean_pred_values.shape, (20,))
                    self.assertEqual(fitted.num_calibration_examples_in_bin.sum(), 24)

    def test_default_is_fifteen_bins_in_all_modes(self):
        for mode in ("binary", "multiclass", "multilabel"):
            with self.subTest(mode=mode):
                calibrator = self._calibrate(mode, n_samples=18)
                for fitted in calibrator.calib:
                    self.assertEqual(fitted.mean_pred_values.shape, (15,))
                    self.assertEqual(fitted.num_calibration_examples_in_bin.sum(), 18)


if __name__ == "__main__":
    unittest.main()
