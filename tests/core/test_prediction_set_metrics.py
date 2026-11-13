"""Prediction-set denominators, multilabel scoring and weighted cost/value."""

import unittest
import warnings

import numpy as np

from pyhealth.metrics import multiclass_metrics_fn, multilabel_metrics_fn
from pyhealth.metrics import prediction_set as pset


class TestUndefinedPredictionSetMetrics(unittest.TestCase):
    def test_unobserved_classes_are_nan_and_excluded_from_class_averages(self):
        predictions = np.array([[1, 0, 0], [1, 0, 0]])
        labels = np.array([0, 1])
        for truth in (labels, np.eye(3, dtype=int)[labels]):
            with self.subTest(shape=truth.shape):
                np.testing.assert_allclose(
                    pset.miscoverage_ps(predictions, truth),
                    [0.0, 1.0, np.nan],
                    equal_nan=True,
                )
                np.testing.assert_allclose(
                    pset.error_ps(predictions, truth),
                    [0.0, 1.0, np.nan],
                    equal_nan=True,
                )
                # Keep the genuine zero; omit only the undefined class.
                self.assertEqual(pset.miscoverage_mean_ps(predictions, truth), 0.5)
                self.assertEqual(pset.error_mean_ps(predictions, truth), 0.5)

    def test_absent_classes_do_not_make_class_averages_look_better(self):
        predictions = np.array([[0, 1, 0], [0, 1, 0]])
        truth = np.array([0, 0])
        result = multiclass_metrics_fn(
            truth,
            np.zeros_like(predictions, dtype=float),
            y_predset=predictions,
            metrics=["miscoverage_mean_ps", "error_mean_ps"],
        )
        self.assertEqual(result, {"miscoverage_mean_ps": 1.0, "error_mean_ps": 1.0})

    def test_singleton_error_uses_only_classes_observed_among_singletons(self):
        truth = np.array([0, 1])
        predictions = np.array([[1, 0, 0], [1, 1, 0]])
        np.testing.assert_allclose(
            pset.error_ps(predictions, truth),
            [0.0, np.nan, np.nan],
            equal_nan=True,
        )
        self.assertEqual(pset.error_mean_ps(predictions, truth), 0.0)
        self.assertEqual(pset.error_overall_ps(predictions, truth), 0.0)

    def test_no_singletons_means_undefined_error_not_zero(self):
        truth = np.array([0, 1])
        predictions = np.array([[1, 1, 0], [1, 1, 0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = multiclass_metrics_fn(
                truth,
                np.zeros_like(predictions, dtype=float),
                y_predset=predictions,
                metrics=["error_ps", "error_mean_ps", "error_overall_ps"],
            )
        for value in result.values():
            self.assertTrue(np.all(np.isnan(value)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for function in (pset.error_ps, pset.error_mean_ps, pset.error_overall_ps):
                with self.subTest(function=function.__name__):
                    self.assertTrue(np.all(np.isnan(function(predictions, truth))))

    def test_empty_multiclass_batch_has_no_defined_prediction_set_averages(self):
        predictions = np.zeros((0, 3))
        truth = np.array([], dtype=int)
        metrics = [
            "set_size",
            "rejection_rate",
            "miscoverage_ps",
            "miscoverage_mean_ps",
            "miscoverage_overall_ps",
            "error_ps",
            "error_mean_ps",
            "error_overall_ps",
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = multiclass_metrics_fn(
                truth,
                predictions,
                metrics=metrics,
                y_predset=predictions,
            )
        self.assertEqual(set(result), set(metrics))
        for value in result.values():
            self.assertTrue(np.all(np.isnan(value)))

    def test_empty_multilabel_batch_returns_nan_without_warnings(self):
        predictions = np.zeros((0, 3))
        metrics = [
            "set_size",
            "tp",
            "fp",
            "tp_value",
            "fp_cost",
            "miscoverage_ps",
            "miscoverage_mean_ps",
            "miscoverage_overall_ps",
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = multilabel_metrics_fn(
                predictions, predictions, metrics=metrics, y_predset=predictions
            )
        self.assertEqual(set(result), set(metrics))
        for value in result.values():
            self.assertTrue(np.all(np.isnan(value)))

    def test_empty_true_label_sets_are_undefined_only_for_recall_denominators(self):
        truth = np.zeros((2, 3))
        predictions = np.array([[1, 0, 0], [1, 1, 0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = multilabel_metrics_fn(
                truth,
                np.zeros_like(truth),
                y_predset=predictions,
                metrics=[
                    "set_size",
                    "tp",
                    "fp",
                    "miscoverage_ps",
                    "miscoverage_mean_ps",
                    "miscoverage_overall_ps",
                ],
            )
        self.assertEqual(result["set_size"], 1.5)
        self.assertEqual(result["tp"], 0.0)
        self.assertEqual(result["fp"], 1.5)
        for name in ("miscoverage_ps", "miscoverage_mean_ps", "miscoverage_overall_ps"):
            self.assertTrue(np.all(np.isnan(result[name])))


class TestEmptyPredictionSetDenominators(unittest.TestCase):
    def test_empty_multiclass_helpers_return_nan_without_warnings(self):
        predictions = np.zeros((0, 3))
        truth = np.array([], dtype=int)
        functions = (
            pset.miscoverage_ps,
            pset.miscoverage_mean_ps,
            pset.miscoverage_overall_ps,
            pset.error_ps,
            pset.error_mean_ps,
            pset.error_overall_ps,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for function in functions:
                with self.subTest(function=function.__name__):
                    self.assertTrue(np.all(np.isnan(function(predictions, truth))))

    def test_overall_multilabel_excludes_empty_targets(self):
        truth = np.array([[1, 1, 0], [0, 0, 0]])
        predictions = np.array([[1, 0, 0], [1, 1, 1]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertEqual(pset.miscoverage_overall_ps(predictions, truth), 0.5)
            self.assertTrue(
                np.isnan(pset.miscoverage_overall_ps(predictions, np.zeros_like(truth)))
            )


class TestMultilabelPredictionSetMetrics(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[1, 1, 0], [0, 0, 1]])
        self.predictions = np.array([[1, 0, 1], [0, 0, 1]])

    def test_existing_size_and_class_missrate_helpers_are_exposed(self):
        result = multilabel_metrics_fn(
            self.truth,
            np.zeros_like(self.truth, dtype=float),
            y_predset=self.predictions,
            metrics=[
                "set_size",
                "miscoverage_ps",
                "miscoverage_mean_ps",
                "miscoverage_overall_ps",
            ],
        )
        self.assertEqual(result["set_size"], 1.5)
        np.testing.assert_array_equal(result["miscoverage_ps"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(result["miscoverage_mean_ps"], 1 / 3)
        self.assertEqual(result["miscoverage_overall_ps"], 0.25)

    def test_multilabel_class_average_excludes_unobserved_classes(self):
        truth = np.array([[1, 0, 0]])
        result = multilabel_metrics_fn(
            truth,
            np.zeros_like(truth, dtype=float),
            y_predset=np.array([[0, 1, 0]]),
            metrics=["miscoverage_ps", "miscoverage_mean_ps"],
        )
        np.testing.assert_allclose(
            result["miscoverage_ps"], [1.0, np.nan, np.nan], equal_nan=True
        )
        self.assertEqual(result["miscoverage_mean_ps"], 1.0)


class TestWeightedPredictionSetMetrics(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[1, 0, 1], [0, 1, 0]])
        self.predictions = np.array([[1, 1, 1], [1, 1, 0]])

    def test_class_weights_preserve_original_units_and_unweighted_counts(self):
        result = multilabel_metrics_fn(
            self.truth,
            np.zeros_like(self.truth, dtype=float),
            metrics=["tp", "fp", "fp_cost", "tp_value", "set_size"],
            y_predset=self.predictions,
            cost_weights=[2, 3, 5],
            value_weights=[2, 3, 5],
        )
        self.assertEqual(
            result,
            {"tp": 1.5, "fp": 1.0, "fp_cost": 2.5, "tp_value": 5.0, "set_size": 2.5},
        )

    def test_default_weights_match_tp_and_fp(self):
        result = multilabel_metrics_fn(
            self.truth,
            np.zeros_like(self.truth, dtype=float),
            metrics=["tp", "fp", "fp_cost", "tp_value"],
            y_predset=self.predictions,
        )
        self.assertEqual(result["fp_cost"], result["fp"])
        self.assertEqual(result["tp_value"], result["tp"])

    def test_scalar_zero_and_vector_weights(self):
        self.assertEqual(pset.fp_cost(self.predictions, self.truth, 2), 2.0)
        self.assertEqual(pset.tp_value(self.predictions, self.truth, 4), 6.0)
        self.assertEqual(pset.fp_cost(self.predictions, self.truth, 0), 0.0)
        self.assertEqual(pset.tp_value(self.predictions, self.truth, [0, 0, 0]), 0.0)

    def test_independent_cost_and_value_weights(self):
        result = multilabel_metrics_fn(
            self.truth,
            np.zeros_like(self.truth, dtype=float),
            y_predset=self.predictions,
            metrics=["fp_cost", "tp_value"],
            cost_weights=[2, 3, 5],
            value_weights=[1, 4, 7],
        )
        self.assertEqual(result, {"fp_cost": 2.5, "tp_value": 6.0})

    def test_weighted_metrics_accept_boolean_masks(self):
        truth, predictions = self.truth.astype(bool), self.predictions.astype(bool)
        self.assertEqual(pset.fp_cost(predictions, truth, [2, 3, 5]), 2.5)
        self.assertEqual(pset.tp_value(predictions, truth, [2, 3, 5]), 5.0)

    def test_empty_true_label_sets_still_contribute_false_positive_cost(self):
        truth = np.array([[0, 0, 0], [0, 1, 0]])
        self.assertEqual(pset.fp_cost(self.predictions, truth, [2, 3, 5]), 6.0)
        self.assertEqual(pset.tp_value(self.predictions, truth, [2, 3, 5]), 1.5)

    def test_weights_reject_broadcasting_negative_and_nonfinite_values(self):
        for weights in (
            [1, 2],
            [[1, 2, 3]],
            -1,
            [1, -1, 1],
            np.nan,
            np.inf,
            [1, np.inf, 1],
        ):
            for function in (pset.fp_cost, pset.tp_value):
                with self.subTest(weights=weights, function=function.__name__):
                    with self.assertRaisesRegex(
                        ValueError, "finite non-negative scalar or class vector"
                    ):
                        function(self.predictions, self.truth, weights)

    def test_weighted_metrics_reject_nonbinary_or_misaligned_masks(self):
        for function in (pset.fp_cost, pset.tp_value):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(self.predictions[:, :1], self.truth)
                with self.assertRaises(ValueError):
                    function(self.predictions * 0.5, self.truth)
                with self.assertRaises(ValueError):
                    function(self.predictions, self.truth * 0.5)

    def test_metric_order_and_weights_do_not_mutate_inputs(self):
        probabilities = np.zeros_like(self.truth, dtype=float)
        cost, value = np.array([2, 3, 5]), np.array([1, 4, 7])
        arrays = (self.truth, self.predictions, probabilities, cost, value)
        before = [array.copy() for array in arrays]
        names = ["tp", "fp", "fp_cost", "tp_value"]
        first = multilabel_metrics_fn(
            self.truth,
            probabilities,
            metrics=names,
            y_predset=self.predictions,
            cost_weights=cost,
            value_weights=value,
        )
        second = multilabel_metrics_fn(
            self.truth,
            probabilities,
            metrics=list(reversed(names)),
            y_predset=self.predictions,
            cost_weights=cost,
            value_weights=value,
        )
        self.assertEqual(first, second)
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)


class TestCostViolationRate(unittest.TestCase):
    def test_equal_mean_costs_can_have_different_violation_rates(self):
        truth = np.zeros((2, 2), dtype=int)
        variable = np.array([[0, 0], [1, 1]])
        constant = np.array([[1, 0], [0, 1]])
        self.assertEqual(pset.fp_cost(variable, truth), pset.fp_cost(constant, truth))
        self.assertEqual(pset.fp_cost_violation_rate(variable, truth, 1), 0.5)
        self.assertEqual(pset.fp_cost_violation_rate(constant, truth, 1), 0.0)

    def test_weighted_costs_use_original_units_and_strict_boundary(self):
        truth = np.zeros((2, 2), dtype=int)
        predictions = np.array([[1, 0], [0, 1]])
        self.assertEqual(
            pset.fp_cost_violation_rate(predictions, truth, 2, [2, 3]), 0.5
        )
        self.assertEqual(
            pset.fp_cost_violation_rate(predictions, truth, 20, [20, 30]), 0.5
        )
        self.assertEqual(
            pset.fp_cost_violation_rate(predictions, truth, 3, [2, 3]), 0.0
        )
        self.assertEqual(pset.fp_cost_violation_rate(predictions, truth, 1, 2), 1.0)

    def test_empty_true_sets_count_as_samples_and_zero_costs_do_not_violate_zero(self):
        truth = np.zeros((2, 2), dtype=bool)
        predictions = np.array([[False, False], [True, False]])
        self.assertEqual(pset.fp_cost_violation_rate(predictions, truth, 0), 0.5)
        self.assertEqual(pset.fp_cost_violation_rate(predictions, truth, 0, 0), 0.0)

    def test_empty_batch_returns_nan_without_warnings(self):
        empty = np.zeros((0, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertTrue(np.isnan(pset.fp_cost_violation_rate(empty, empty, 1)))
            result = multilabel_metrics_fn(
                empty,
                empty,
                metrics=["fp_cost_violation_rate"],
                y_predset=empty,
                target_cost=1,
            )
        self.assertTrue(np.isnan(result["fp_cost_violation_rate"]))

    def test_invalid_limits_are_rejected(self):
        masks = np.zeros((2, 2))
        for limit in (
            None,
            -1,
            np.nan,
            np.inf,
            -np.inf,
            True,
            np.bool_(False),
            "1",
            [1],
            np.array([1]),
        ):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "target_cost"),
            ):
                pset.fp_cost_violation_rate(masks, masks, limit)

    def test_weights_and_masks_share_existing_cost_validation(self):
        masks = np.ones((2, 2))
        for weights in ([1], [[1, 2]], -1, [1, np.nan]):
            with (
                self.subTest(weights=weights),
                self.assertRaisesRegex(ValueError, "weights"),
            ):
                pset.fp_cost_violation_rate(masks, masks, 1, weights)

        for prediction, truth in (
            (masks[:, :1], masks),
            (masks * 0.5, masks),
            (masks, masks * 0.5),
        ):
            with (
                self.subTest(prediction=prediction, truth=truth),
                self.assertRaises(ValueError),
            ):
                pset.fp_cost_violation_rate(prediction, truth, 1)

    def test_scorer_requires_limit_only_when_metric_and_prediction_sets_are_present(
        self,
    ):
        truth = np.array([[1, 0], [0, 1]])
        probabilities = np.zeros_like(truth, dtype=float)
        with self.assertRaisesRegex(ValueError, "target_cost"):
            multilabel_metrics_fn(
                truth,
                probabilities,
                metrics=["fp_cost_violation_rate"],
                y_predset=truth,
            )

        self.assertEqual(
            multilabel_metrics_fn(
                truth, probabilities, metrics=["fp_cost_violation_rate"]
            ),
            {},
        )
        self.assertEqual(
            multilabel_metrics_fn(
                truth,
                probabilities,
                metrics=["fp_cost"],
                y_predset=truth,
                target_cost=None,
            ),
            {"fp_cost": 0.0},
        )

    def test_scorer_uses_prediction_masks_and_does_not_mutate_inputs(self):
        truth = np.array([[1, 0, 1], [0, 1, 0]])
        predictions = np.array([[1, 1, 1], [1, 1, 0]])
        probabilities = np.zeros_like(truth, dtype=float)
        weights = np.array([2.0, 3.0, 5.0])
        arrays = (truth, predictions, probabilities, weights)
        before = [values.copy() for values in arrays]
        result = multilabel_metrics_fn(
            truth,
            probabilities,
            metrics=["fp_cost", "fp_cost_violation_rate", "fp"],
            y_predset=predictions,
            cost_weights=weights,
            target_cost=2.0,
        )
        self.assertEqual(
            result, {"fp_cost": 2.5, "fp_cost_violation_rate": 0.5, "fp": 1.0}
        )
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)


class TestExistingScalarTypes(unittest.TestCase):
    def test_defined_set_metrics_preserve_numpy_scalar_types(self):
        labels = np.array([0, 1, 2])
        for dtype in (bool, np.int64, np.float32, np.float64):
            predictions = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype)
            selected = predictions[np.arange(3), labels]
            expected = {
                "size": predictions.sum(axis=1).mean(),
                "rejection_rate": np.mean(predictions.sum(axis=1) != 1),
                "miscoverage_overall_ps": 1 - selected.mean(),
                "error_overall_ps": 1 - selected.mean(),
                "miscoverage_mean_ps": pset.miscoverage_ps(predictions, labels).mean(),
                "error_mean_ps": pset.error_ps(predictions, labels).mean(),
            }
            for name, reference in expected.items():
                with self.subTest(dtype=dtype, name=name):
                    function = getattr(pset, name)
                    value = (
                        function(predictions)
                        if name in {"size", "rejection_rate"}
                        else function(predictions, labels)
                    )
                    self.assertIs(type(value), type(reference))
                    self.assertEqual(value.item(), reference.item())

    def test_empty_averages_preserve_numpy_scalar_types_without_warnings(self):
        for dtype in (bool, np.int64, np.float32, np.float64):
            empty = np.zeros((0, 3), dtype=dtype)
            expected_dtype = np.float32 if dtype == np.float32 else np.float64
            with self.subTest(dtype=dtype), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                size = pset.size(empty)
                self.assertIs(type(size), expected_dtype)
                self.assertTrue(np.isnan(size.item()))
                rejection = pset.rejection_rate(empty)
                self.assertIs(type(rejection), np.float64)
                self.assertTrue(np.isnan(rejection.item()))
                counts = multilabel_metrics_fn(
                    empty, empty.astype(float), metrics=["tp", "fp"], y_predset=empty
                )
                for value in counts.values():
                    self.assertIs(type(value), expected_dtype)
                    self.assertTrue(np.isnan(value.item()))

    def test_unweighted_multilabel_counts_preserve_scalar_types(self):
        for dtype in (bool, np.int64, np.float32, np.float64):
            truth = np.array([[1, 0, 1], [0, 1, 0]], dtype=dtype)
            predictions = np.array([[1, 1, 1], [1, 1, 0]], dtype=dtype)
            expected = {
                "tp": (truth * predictions).sum(axis=1).mean(),
                "fp": ((1 - truth) * predictions).sum(axis=1).mean(),
            }
            result = multilabel_metrics_fn(
                truth,
                np.zeros_like(truth, dtype=float),
                metrics=["tp", "fp"],
                y_predset=predictions,
                cost_weights=2.0,
                value_weights=3.0,
            )
            for name, reference in expected.items():
                with self.subTest(dtype=dtype, name=name):
                    self.assertIs(type(result[name]), type(reference))
                    self.assertEqual(result[name].item(), reference.item())


class TestReviewRegressions(unittest.TestCase):
    def test_decimal_ties_do_not_violate_the_cost_limit(self):
        truth = np.zeros((1, 2), dtype=int)
        predictions = np.ones_like(truth)
        for scale in (1e-15, 1.0, 1e15):
            with self.subTest(scale=scale):
                weights = np.array([0.1, 0.2]) * scale
                self.assertEqual(
                    pset.fp_cost_violation_rate(
                        predictions, truth, 0.3 * scale, weights
                    ),
                    0.0,
                )
                # A genuine excess, beyond rounding tolerance, still violates.
                self.assertEqual(
                    pset.fp_cost_violation_rate(
                        predictions, truth, (0.3 - 1e-10) * scale, weights
                    ),
                    1.0,
                )

        self.assertEqual(
            pset.fp_cost_violation_rate(predictions, truth, 0, [1e-30, 0]), 1.0
        )

    def test_float32_limits_treat_rounding_ties_as_equal(self):
        import torch

        truth = np.zeros((2, 2), dtype=int)
        predictions = np.eye(2, dtype=int)
        for limit in (
            np.float32(0.7),
            np.array(0.7, dtype=np.float32),
            torch.tensor(0.7, dtype=torch.float32),
            torch.tensor(0.7, dtype=torch.float32, requires_grad=True),
        ):
            with self.subTest(limit=limit):
                result = multilabel_metrics_fn(
                    truth,
                    truth.astype(float),
                    metrics=["fp_cost_violation_rate"],
                    y_predset=predictions,
                    target_cost=limit,
                    cost_weights=0.7,
                )
                self.assertEqual(result["fp_cost_violation_rate"], 0.0)
                self.assertEqual(
                    pset.fp_cost_violation_rate(
                        predictions, truth, limit, [0.7, 0.70001]
                    ),
                    0.5,
                )

    def test_bfloat16_scalar_limits_remain_supported(self):
        import torch

        truth = np.zeros((1, 1), dtype=int)
        predictions = np.ones_like(truth)
        self.assertEqual(
            pset.fp_cost_violation_rate(
                predictions, truth, torch.tensor(0.7, dtype=torch.bfloat16), 0.7
            ),
            0.0,
        )

    def test_float32_weights_treat_rounding_ties_as_equal(self):
        truth = np.zeros((2, 2), dtype=int)
        predictions = np.eye(2, dtype=int)
        for weights in (
            np.float32(0.1),
            np.array([0.1, 0.1], dtype=np.float32),
            [np.float32(0.1), np.float32(0.1)],
        ):
            with self.subTest(weights=weights):
                self.assertEqual(
                    pset.fp_cost_violation_rate(predictions, truth, 0.1, weights),
                    0.0,
                )
                self.assertEqual(
                    pset.fp_cost_violation_rate(predictions, truth, 0.09999, weights),
                    1.0,
                )
                self.assertEqual(
                    pset.fp_cost_violation_rate(predictions, truth, 0, weights),
                    1.0,
                )

    def test_multilabel_scorer_accepts_lists_for_each_array_input(self):
        from itertools import product

        truth = np.array([[1, 1, 0], [0, 0, 1]])
        probabilities = np.array([[0.8, 0.1, 0.7], [0.1, 0.2, 0.9]])
        predictions = np.array([[1, 0, 1], [0, 0, 1]])
        metrics = [
            "accuracy", "tp", "fp", "tp_value", "fp_cost",
            "fp_cost_violation_rate", "set_size", "miscoverage_ps",
            "miscoverage_mean_ps", "miscoverage_overall_ps",
        ]
        expected = multilabel_metrics_fn(
            truth, probabilities, metrics=metrics, y_predset=predictions, target_cost=1
        )
        for use_lists in product((False, True), repeat=3):
            inputs = [
                array.tolist() if as_list else array.copy()
                for array, as_list in zip((truth, probabilities, predictions), use_lists)
            ]
            with self.subTest(use_lists=use_lists):
                result = multilabel_metrics_fn(
                    inputs[0], inputs[1], metrics=metrics,
                    y_predset=inputs[2], target_cost=1,
                )
                self.assertEqual(result.keys(), expected.keys())
                for name in metrics:
                    np.testing.assert_allclose(result[name], expected[name])
                for actual, original in zip(inputs, (truth, probabilities, predictions)):
                    np.testing.assert_array_equal(actual, original)

        for metric in metrics[1:]:
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                multilabel_metrics_fn(
                    truth.tolist(), probabilities.tolist(), metrics=[metric],
                    y_predset=[[1, 0], [0, 1]], target_cost=1,
                )

    def test_numpy_and_torch_scalar_limits(self):
        import torch

        truth = np.zeros((1, 2), dtype=int)
        predictions = np.ones_like(truth)
        for limit in (
            np.array(3.0),
            np.float32(3.0),
            torch.tensor(3.0),
            torch.tensor(3.0, requires_grad=True),
        ):
            with self.subTest(limit=limit):
                result = multilabel_metrics_fn(
                    truth,
                    truth.astype(float),
                    metrics=["fp_cost_violation_rate"],
                    y_predset=predictions,
                    target_cost=limit,
                    cost_weights=[1, 2],
                )
                self.assertEqual(result, {"fp_cost_violation_rate": 0.0})

        for limit in (np.array(True), np.array(-1.0), torch.tensor([3.0])):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "target_cost"),
            ):
                pset.fp_cost_violation_rate(predictions, truth, limit)

    def test_all_multilabel_set_metrics_reject_the_same_bad_inputs(self):
        masks = np.ones((2, 3), dtype=int)
        invalid_pairs = [(masks[:, :2], masks), (masks, masks[:, :2])]
        invalid_pairs += [(masks[0], masks), (masks, masks[0])]
        for value in (-1, 0.5, 2, np.nan, np.inf):
            invalid = masks.astype(float)
            invalid[0, 0] = value
            invalid_pairs += [(invalid, masks), (masks, invalid)]

        metrics = [
            "tp",
            "fp",
            "tp_value",
            "fp_cost",
            "fp_cost_violation_rate",
            "set_size",
            "miscoverage_ps",
            "miscoverage_mean_ps",
            "miscoverage_overall_ps",
        ]
        for prediction, truth in invalid_pairs:
            for metric in metrics:
                with self.subTest(metric=metric, truth=truth, prediction=prediction):
                    with self.assertRaises(ValueError):
                        multilabel_metrics_fn(
                            truth,
                            np.zeros_like(truth, dtype=float),
                            metrics=[metric],
                            y_predset=prediction,
                            target_cost=1,
                        )

            for function in (
                pset.miscoverage_ps,
                pset.miscoverage_mean_ps,
                pset.error_ps,
                pset.error_mean_ps,
            ):
                with self.subTest(function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(prediction, truth)

    def test_multiclass_labels_and_prediction_masks_are_validated(self):
        masks = np.ones((2, 3), dtype=int)
        functions = (
            pset.miscoverage_ps,
            pset.miscoverage_mean_ps,
            pset.miscoverage_overall_ps,
            pset.error_ps,
            pset.error_mean_ps,
            pset.error_overall_ps,
        )
        for function in functions:
            for truth in (
                np.array([-1, 1]), np.array([0, 3]),
                np.array([0.0, 1.0]), np.array([0.0, 1.5]),
            ):
                with self.subTest(function=function.__name__, truth=truth):
                    with self.assertRaises(ValueError):
                        function(masks, truth)

            with self.assertRaisesRegex(ValueError, "binary membership masks"):
                function(masks * 0.5, np.array([0, 1]))

        for function in (pset.size, pset.rejection_rate):
            for prediction in (masks[0], masks * 0.5):
                with self.subTest(function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(prediction)

    def test_defined_overall_miscoverage_returns_numpy_scalars(self):
        labels = np.array([0, 1])
        for dtype in (bool, np.int64, np.float32, np.float64):
            masks = np.array([[1, 0, 0], [1, 0, 0]], dtype=dtype)
            for truth in (labels, np.eye(3, dtype=dtype)[labels]):
                with self.subTest(dtype=dtype, shape=truth.shape):
                    value = pset.miscoverage_overall_ps(masks, truth)
                    self.assertIsInstance(value, np.floating)
                    self.assertEqual(value.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
