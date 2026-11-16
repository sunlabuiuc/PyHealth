"""Prediction-set miscoverage for multiclass and multilabel targets."""

import unittest
import warnings

import numpy as np

from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.metrics.prediction_set import miscoverage_overall_ps


class TestPredictionSetMiscoverage(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([[1, 1, 0], [0, 0, 1]])
        self.y_predset = np.array([[1, 0, 1], [0, 0, 1]])

    def test_fractional_miscoverage_averages_samples_equally(self):
        # Per-sample miss fractions are 1/2 and 0: mean 1/4, not the pooled
        # missed-label fraction (1/3) or the fraction of incomplete sets (1/2).
        self.assertEqual(miscoverage_overall_ps(self.y_predset, self.y_true), 0.25)

    def test_empty_and_full_prediction_sets(self):
        self.assertEqual(
            miscoverage_overall_ps(np.zeros_like(self.y_true), self.y_true), 1.0
        )
        self.assertEqual(
            miscoverage_overall_ps(np.ones_like(self.y_true), self.y_true), 0.0
        )

    def test_false_positives_do_not_change_label_miscoverage(self):
        without_false_positive = np.array([[1, 0, 0], [0, 0, 1]])
        self.assertEqual(
            miscoverage_overall_ps(self.y_predset, self.y_true),
            miscoverage_overall_ps(without_false_positive, self.y_true),
        )

    def test_boolean_integer_and_float_binary_masks(self):
        for dtype in (bool, np.int64, np.float64):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    miscoverage_overall_ps(
                        self.y_predset.astype(dtype), self.y_true.astype(dtype)
                    ),
                    0.25,
                )

    def test_empty_targets_are_excluded(self):
        y_true = np.vstack([self.y_true, [0, 0, 0], [0, 0, 0]])
        y_predset = np.vstack([self.y_predset, [1, 1, 1], [0, 0, 0]])
        self.assertEqual(miscoverage_overall_ps(y_predset, y_true), 0.25)

    def test_all_empty_targets_and_empty_batches_return_nan_without_warnings(self):
        for shape in ((2, 3), (0, 3), (2, 0)):
            with self.subTest(shape=shape), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                value = miscoverage_overall_ps(np.ones(shape), np.zeros(shape))
                self.assertTrue(np.isnan(value))

    def test_invalid_multilabel_shapes_are_rejected(self):
        for predictions, truth in (
            (np.ones((2, 1)), self.y_true),  # Must not silently broadcast.
            (np.ones((1, 3)), self.y_true),
            (np.ones((2, 3, 1)), np.ones((2, 3, 1))),
        ):
            with self.subTest(predictions=predictions.shape, truth=truth.shape):
                with self.assertRaisesRegex(ValueError, "matching.*shapes"):
                    miscoverage_overall_ps(predictions, truth)

    def test_multiclass_behavior_is_preserved(self):
        labels = np.array([0, 2])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])
        self.assertEqual(miscoverage_overall_ps(predictions, labels), 0.5)
        self.assertEqual(
            miscoverage_overall_ps(np.zeros_like(predictions), labels), 1.0
        )
        self.assertEqual(miscoverage_overall_ps(np.ones_like(predictions), labels), 0.0)

    def test_one_hot_targets_match_multiclass_targets(self):
        labels = np.array([0, 2])
        predictions = np.array([[1, 0, 0], [0, 1, 0]])
        one_hot = np.eye(3, dtype=int)[labels]
        self.assertEqual(
            miscoverage_overall_ps(predictions, labels),
            miscoverage_overall_ps(predictions, one_hot),
        )

    def test_multiclass_shape_mismatch_is_rejected(self):
        for predictions in (np.ones((1, 3)), np.ones(2)):
            with self.subTest(shape=predictions.shape):
                with self.assertRaisesRegex(ValueError, "one prediction-set row"):
                    miscoverage_overall_ps(predictions, np.array([0, 2]))

    def test_unsupported_target_dimensions_are_rejected(self):
        for truth in (np.array(0), np.zeros((2, 3, 1))):
            with self.subTest(shape=truth.shape):
                with self.assertRaises(ValueError):
                    miscoverage_overall_ps(np.ones((2, 3)), truth)

    def test_nonbinary_masks_are_rejected(self):
        for value in (-1, 0.5, 2, np.nan, np.inf):
            invalid = self.y_true.astype(float)
            invalid[0, 0] = value

            for predictions, truth in (
                (invalid, self.y_true),
                (self.y_predset, invalid),
            ):
                with self.subTest(value=value, truth=truth):
                    with self.assertRaisesRegex(ValueError, "binary membership masks"):
                        miscoverage_overall_ps(predictions, truth)

    def test_scorer_uses_prediction_sets_not_thresholded_probabilities(self):
        probabilities = np.zeros_like(self.y_true, dtype=float)
        snapshots = [
            array.copy() for array in (self.y_true, probabilities, self.y_predset)
        ]
        result = multilabel_metrics_fn(
            self.y_true,
            probabilities,
            metrics=["miscoverage_overall_ps", "recall_samples"],
            threshold=0.0,  # Ordinary recall sees every label as predicted.
            y_predset=self.y_predset,
        )
        self.assertEqual(
            result, {"miscoverage_overall_ps": 0.25, "recall_samples": 1.0}
        )

        for actual, before in zip(
            (self.y_true, probabilities, self.y_predset), snapshots
        ):
            np.testing.assert_array_equal(actual, before)

    def test_scorer_preserves_undefined_miscoverage(self):
        y_true = np.zeros((2, 3))
        result = multilabel_metrics_fn(
            y_true,
            np.zeros_like(y_true),
            metrics=["miscoverage_overall_ps"],
            y_predset=np.ones_like(y_true),
        )
        self.assertTrue(np.isnan(result["miscoverage_overall_ps"]))


if __name__ == "__main__":
    unittest.main()
