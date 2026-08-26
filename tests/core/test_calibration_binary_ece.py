import unittest
from unittest.mock import patch

import numpy as np

from pyhealth.metrics import binary_metrics_fn
from pyhealth.metrics.calibration import ece_confidence_binary


class TestBinaryECE(unittest.TestCase):
    def test_binary_metrics_fn_ece_does_not_crash(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
        for metric in ("ECE", "ECE_adapt"):
            out = binary_metrics_fn(y_true, y_prob, metrics=[metric])
            self.assertIn(metric, out)
            self.assertTrue(np.isfinite(out[metric]))
            self.assertGreaterEqual(out[metric], 0.0)
            self.assertLessEqual(out[metric], 1.0)

    def test_two_dim_inputs_use_positive_class(self):
        prob = np.array([[0.2, 0.8], [0.7, 0.3]])
        label = np.array([[0, 1], [1, 0]])

        with patch(
            "pyhealth.metrics.calibration._ECE_confidence",
            return_value=(None, 0.0),
        ) as ece:
            ece_confidence_binary(prob, label)

        frame = ece.call_args.args[0]
        np.testing.assert_array_equal(frame["conf"].to_numpy(), prob[:, 1])
        np.testing.assert_array_equal(frame["acc"].to_numpy(), label[:, 1])
