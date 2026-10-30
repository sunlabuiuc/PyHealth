"""Tests for EnbPI (Xu and Xie 2021).

The deterministic tests use hand-computed values from the paper's
Algorithm 1 with constant fake models and fixed bootstrap samples.

Fixture: three models predict the constants 1, 2 and 4, trained on

    b0: [0, 0, 1, 1]  -> out of bag for points {2, 3}
    b1: [2, 2, 2, 2]  -> out of bag for points {0, 1, 3}
    b2: [0, 1, 2, 2]  -> out of bag for point  {3}

so the OOB predictions are 2, 2, 1 and aggregate(1, 2, 4) for point 3, which
is 7/3 under the mean and 2 under the median. With targets [0, 3, 1, 5] the
mean residuals are [2, 1, 0, 8/3].
"""

import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import EnbPI
from pyhealth.calib.predictionset.timeseries import conformal_quantile
from pyhealth.calib.predictionset.timeseries.enbpi import _oob_aggregate
from pyhealth.datasets import get_dataloader
from pyhealth.metrics.prediction_interval import miscoverage_pi
from pyhealth.models import MLP
from tests.core.timeseries_cp_test_utils import (
    ConstantModel,
    make_ar_stream,
    make_stream,
)

# Model outputs and forward() results are float32, so compare at float32 precision.
RTOL = 1e-6

BAGS = np.array([[0, 0, 1, 1], [2, 2, 2, 2], [0, 1, 2, 2]])
Y_TRAIN = [0.0, 3.0, 1.0, 5.0]


def _fitted_enbpi(bags=BAGS, models=None, **kwargs):
    """EnbPI whose training installs fixed fake models and bags.

    ``enbpi.train_calls`` counts how often calibrate trained.
    """
    enbpi = EnbPI(model_fn=lambda: None, **kwargs)
    if models is None:
        models = [ConstantModel(1.0), ConstantModel(2.0), ConstantModel(4.0)]
    enbpi.train_calls = 0

    def install(train_dataset):
        enbpi.models = torch.nn.ModuleList(models)
        enbpi.bootstrap_indices = np.asarray(bags)
        enbpi.train_calls += 1

    enbpi._train_models = install
    return enbpi


class TestOutOfBag(unittest.TestCase):
    def test_mean_residuals(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [2.0, 1.0, 0.0, 8 / 3], rtol=RTOL
        )

    def test_median_residuals(self):
        enbpi = _fitted_enbpi(alpha=0.5, aggregation="median")
        enbpi.calibrate(make_stream(Y_TRAIN))
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [2.0, 1.0, 0.0, 3.0], rtol=RTOL
        )

    def test_point_in_every_bag_is_dropped(self):
        bags = np.array([[0, 0, 3, 1], [2, 2, 3, 3], [0, 1, 2, 3]])
        enbpi = _fitted_enbpi(bags=bags, alpha=0.5)
        with self.assertLogs(
            "pyhealth.calib.predictionset.timeseries.enbpi", "WARNING"
        ):
            enbpi.calibrate(make_stream(Y_TRAIN))
        # Point 3 has no OOB model; points 0, 1 use b1 (2) and point 2 uses b0 (1).
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [2.0, 1.0, 0.0], rtol=RTOL
        )
        self.assertEqual(enbpi.residuals.maxlen, 4)

    def test_no_out_of_bag_model_raises(self):
        # Every bag contains every point, so no OOB prediction exists.
        bags = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2]])
        enbpi = _fitted_enbpi(bags=bags, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "no out-of-bag"):
            enbpi.calibrate(make_stream(Y_TRAIN))
        self.assertFalse(enbpi.calibrated)

    def test_no_out_of_bag_model_from_training_raises(self):
        # The reported case: 2 samples, 1 model, and random_state=1 draws
        # the bag [0, 1], which contains both points.
        enbpi = EnbPI(
            model_fn=lambda: ConstantModel(1.0),
            alpha=0.5,
            n_bootstrap=1,
            random_state=1,
            train_kwargs={"epochs": 0},
        )
        with self.assertRaisesRegex(ValueError, "no out-of-bag"):
            enbpi.calibrate(make_stream([0.0, 1.0]))
        self.assertFalse(enbpi.calibrated)


class TestInterval(unittest.TestCase):
    def setUp(self):
        self.test = make_stream([1.0, 9.0, 2.0])
        self.batch = next(iter(get_dataloader(self.test, 3, shuffle=False)))

    def test_aggregate_center_and_width(self):
        # OOB ensembles predict [2, 2, 1, 7/3] everywhere, mean 11/6.
        # Window [2, 1, 0, 8/3], alpha 0.5: rank ceil(0.5 * 5) = 3 -> 2.
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        out = enbpi(**self.batch)
        np.testing.assert_allclose(
            out["y_prob"].numpy().ravel(), [11 / 6] * 3, rtol=RTOL
        )
        np.testing.assert_allclose(
            out["y_lower"].numpy().ravel(), [11 / 6 - 2] * 3, rtol=RTOL
        )
        np.testing.assert_allclose(
            out["y_upper"].numpy().ravel(), [11 / 6 + 2] * 3, rtol=RTOL
        )

    def test_quantile_center(self):
        # Algorithm 1 line 12: np.quantile([2, 2, 1, 7/3], 0.5) = 2.
        enbpi = _fitted_enbpi(alpha=0.5, center="quantile")
        enbpi.calibrate(make_stream(Y_TRAIN))
        np.testing.assert_allclose(
            enbpi(**self.batch)["y_prob"].numpy().ravel(), [2.0] * 3, rtol=RTOL
        )

    def test_window_slides_after_each_feedback_batch(self):
        # Center 11/6; test residuals |1 - 11/6| = 5/6, |9 - 11/6| = 43/6.
        enbpi = _fitted_enbpi(alpha=0.5, feedback_size=2)
        enbpi.calibrate(make_stream(Y_TRAIN))
        out = enbpi.predict_sequential(self.test)
        # The third residual is |2 - 11/6| = 1/6; the window keeps T = 4.
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [8 / 3, 5 / 6, 43 / 6, 1 / 6], rtol=RTOL
        )
        # The first two points share the calibration window, quantile 2.
        # The third uses [0, 8/3, 5/6, 43/6]: rank 3 of 4 -> 8/3.
        np.testing.assert_allclose(
            out["y_upper"] - out["y_pred"], [2.0, 2.0, 8 / 3], rtol=RTOL
        )

    def test_forward_does_not_change_window(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        enbpi(**self.batch)
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [2.0, 1.0, 0.0, 8 / 3], rtol=RTOL
        )

    def test_forward_returns_columns_and_loss(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        out = enbpi(**self.batch)
        for key in ("y_prob", "y_lower", "y_upper", "y_true"):
            self.assertEqual(out[key].shape, (3, 1))
        self.assertIn("loss", out)

    def test_update_slides_window_in_order(self):
        # Residuals |1 - 11/6| = 5/6 then |9 - 11/6| = 43/6 evict 2 and 1.
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        enbpi.update(make_stream([1.0, 9.0]))
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [0.0, 8 / 3, 5 / 6, 43 / 6], rtol=RTOL
        )

    def test_feedback_size_one_updates_every_step(self):
        # Widths: 2 from [2, 1, 0, 8/3]; then [1, 0, 8/3, 5/6] -> 1;
        # then [0, 8/3, 5/6, 43/6] -> 8/3.
        enbpi = _fitted_enbpi(alpha=0.5, feedback_size=1)
        enbpi.calibrate(make_stream(Y_TRAIN))
        out = enbpi.predict_sequential(self.test)
        np.testing.assert_allclose(
            out["y_upper"] - out["y_pred"], [2.0, 1.0, 8 / 3], rtol=RTOL
        )


class TestTimeOrder(unittest.TestCase):
    def test_dataset_order_is_time_order(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        out = enbpi.predict_sequential(make_stream([3.0, 1.0, 2.0]))
        np.testing.assert_array_equal(out["y_true"], [3.0, 1.0, 2.0])


class TestScaling(unittest.TestCase):
    """The memory- and time-saving paths match the direct computation."""

    def test_oob_aggregate_matches_masked_reference(self):
        rng = np.random.default_rng(0)
        oob = rng.random((6, 40)) < 0.4
        oob[0, ~oob.any(axis=0)] = True  # every column needs an OOB model
        preds = rng.normal(size=(6, 5))
        masked = np.where(oob.T[:, :, None], preds[None, :, :], np.nan)
        for aggregation, reference in (("mean", np.nanmean), ("median", np.nanmedian)):
            with self.subTest(aggregation=aggregation):
                # A tiny chunk cap forces the median through several chunks.
                np.testing.assert_allclose(
                    _oob_aggregate(oob, preds, aggregation, max_chunk_elements=50),
                    reference(masked, axis=1),
                )

    def test_sorted_window_matches_conformal_quantile(self):
        enbpi = _fitted_enbpi(alpha=0.2)
        enbpi.calibrate(make_stream(Y_TRAIN))
        rng = np.random.default_rng(0)
        # Integer residuals give ties, which removal from the sorted copy
        # must handle.
        for residual in rng.integers(0, 5, size=30).astype(float):
            enbpi._push_residuals(np.zeros(1), np.array([residual]))
            window = np.asarray(enbpi.residuals)
            self.assertEqual(enbpi.sorted_residuals, sorted(window.tolist()))
            self.assertEqual(enbpi._width(), conformal_quantile(window, 0.2))


class TestInterface(unittest.TestCase):
    def test_forward_before_calibrate_raises(self):
        batch = next(iter(get_dataloader(make_stream(Y_TRAIN), 4, shuffle=False)))
        with self.assertRaises(RuntimeError):
            _fitted_enbpi(alpha=0.5)(**batch)

    def test_non_regression_model_fn_raises_before_training(self):
        enbpi = EnbPI(model_fn=lambda: ConstantModel(1.0, mode="binary"), alpha=0.1)
        with self.assertRaises(NotImplementedError):
            enbpi.calibrate(make_stream(Y_TRAIN))

    def test_recalibrate_retrains(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        enbpi.update(make_stream([1.0, 9.0]))
        enbpi.calibrate(make_stream(Y_TRAIN))
        self.assertEqual(enbpi.train_calls, 2)
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [2.0, 1.0, 0.0, 8 / 3], rtol=RTOL
        )

    def test_reset_restores_initial_window(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        enbpi.update(make_stream([1.0, 9.0]))
        enbpi.reset()
        self.assertEqual(enbpi.train_calls, 1)
        np.testing.assert_allclose(
            np.asarray(enbpi.residuals), [2.0, 1.0, 0.0, 8 / 3], rtol=RTOL
        )
        self.assertEqual(enbpi.sorted_residuals, sorted(enbpi.residuals))
        self.assertEqual(enbpi.residuals.maxlen, 4)

    def test_reset_reruns_give_same_intervals(self):
        enbpi = _fitted_enbpi(alpha=0.5)
        enbpi.calibrate(make_stream(Y_TRAIN))
        test = make_stream([1.0, 9.0, 2.0])
        first = enbpi.predict_sequential(test)
        enbpi.reset()
        second = enbpi.predict_sequential(test)
        np.testing.assert_array_equal(first["y_upper"], second["y_upper"])

    def test_reset_before_calibrate_raises(self):
        with self.assertRaises(RuntimeError):
            _fitted_enbpi(alpha=0.5).reset()

    def test_invalid_arguments_raise(self):
        for kwargs in (
            {"alpha": 0.0},
            {"alpha": 0.1, "aggregation": "trimmed"},
            {"alpha": 0.1, "center": "mode"},
            {"alpha": 0.1, "n_bootstrap": 0},
            {"alpha": 0.1, "feedback_size": 0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                EnbPI(model_fn=lambda: None, **kwargs)


class TestEndToEnd(unittest.TestCase):
    def test_coverage_on_ar_series(self):
        torch.manual_seed(42)
        stream = make_ar_stream(n=300, seed=42)
        train, test = stream.subset(range(100)), stream.subset(range(100, 300))
        enbpi = EnbPI(
            model_fn=lambda: MLP(
                dataset=stream,
                feature_keys=["x"],
                label_key="y",
                mode="regression",
                hidden_dim=4,
            ),
            alpha=0.1,
            n_bootstrap=5,
            train_kwargs={"epochs": 5},
            device="cpu",
            random_state=42,
        )
        enbpi.calibrate(train)
        out = enbpi.predict_sequential(test)
        self.assertTrue(np.all(out["y_lower"] <= out["y_pred"]))
        self.assertTrue(np.all(out["y_pred"] <= out["y_upper"]))
        miscoverage = miscoverage_pi(out["y_lower"], out["y_upper"], out["y_true"])
        self.assertGreater(miscoverage, 0.02)
        self.assertLess(miscoverage, 0.2)


if __name__ == "__main__":
    unittest.main()
