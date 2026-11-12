"""Hand-calculated IPCW and train-only preprocessing tests.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import unittest

import numpy as np

from pyhealth.metrics.competing_risks import (
    CensoringDistribution,
    MetricSupportError,
    competing_risk_brier,
    competing_risk_concordance,
)
from pyhealth.tasks import CompetingRisksPreprocessor


class TestCompetingRisks(unittest.TestCase):
    def test_impute_then_scale_and_restore(self) -> None:
        train = [
            {"features": [1.0, np.nan], "time": 2.0, "event": 0},
            {"features": [3.0, np.nan], "time": 4.0, "event": 1},
            {"features": [np.nan, np.nan], "time": 3.0, "event": 2},
        ]
        processor = CompetingRisksPreprocessor().fit(train)
        self.assertEqual(processor.state["all_missing_indices"], [1])
        np.testing.assert_allclose(processor.state["scale"], [np.sqrt(2 / 3), 1.0])
        original_state = dict(processor.state)
        restored = CompetingRisksPreprocessor.from_state(processor.state)
        result = restored.transform([{"features": [100.0, np.nan], "time": 8.0}])[0]
        self.assertAlmostEqual(result["features"][0], 98 / np.sqrt(2 / 3))
        self.assertEqual(result["missing_mask"], [0.0, 1.0])
        self.assertEqual(result["time"], 2.0)
        self.assertEqual(processor.state, original_state)
        self.assertIsNot(processor.state, restored.state)
        with self.assertRaises(ValueError):
            restored.transform([{"features": [1.0], "time": 1.0}])
        with self.assertRaises(ValueError):
            restored.transform([{"features": [1.0, 2.0], "time": -1.0}])
        with self.assertRaises(ValueError):
            CompetingRisksPreprocessor().fit([])
        with self.assertRaises(RuntimeError):
            CompetingRisksPreprocessor().transform(train)

    def test_km_ties_and_left_limit(self) -> None:
        censoring = CensoringDistribution(
            np.array([1.0, 1.0, 2.0, 3.0]),
            np.array([0, 1, 0, 2]),
        )
        np.testing.assert_allclose(
            censoring.probability(np.array([0.0, 1.0, 2.0])),
            [1.0, 2 / 3, 1 / 3],
        )
        np.testing.assert_allclose(
            censoring.probability(np.array([1.0, 2.0]), before=True),
            [1.0, 2 / 3],
        )
        with self.assertRaises(ValueError):
            censoring.probability(-1.0)

    def test_reverse_km_independent_discrete_ties(self) -> None:
        times = np.array([1.0, 1.0, 1.0, 2.0])
        events = np.array([1, 1, 0, 1])
        censoring = CensoringDistribution(times, events)
        np.testing.assert_allclose(censoring.probability([1.0, 2.0]), [0.5, 0.5])
        self.assertAlmostEqual(
            competing_risk_brier(times, events, np.ones(4), 1.0, 1, censoring),
            0.5,
        )
        np.testing.assert_allclose(
            CensoringDistribution(np.array([1.0, 1.0]), np.array([1, 2])).survival,
            [1.0],
        )

    def test_horizon_support_is_required_without_weighted_observations(self) -> None:
        exhausted = CensoringDistribution(np.array([1.0]), np.array([0]))
        for function in (competing_risk_brier, competing_risk_concordance):
            with self.assertRaisesRegex(MetricSupportError, "training follow-up"):
                function(
                    np.array([0.5]),
                    np.array([0]),
                    np.array([0.0]),
                    2.0,
                    1,
                    exhausted,
                )
            with self.assertRaisesRegex(MetricSupportError, "support exhausted"):
                function(
                    np.array([0.5]),
                    np.array([0]),
                    np.array([0.0]),
                    1.0,
                    1,
                    exhausted,
                )
        with self.assertRaisesRegex(MetricSupportError, "training follow-up"):
            competing_risk_concordance(
                np.array([0.2, 0.5]),
                np.array([1, 2]),
                np.array([0.9, 0.1]),
                2.0,
                1,
                CensoringDistribution(np.array([1.0]), np.array([1])),
            )

    def test_brier_counts_competing_negative(self) -> None:
        censoring = CensoringDistribution(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0, 1, 2, 1]),
        )
        score = competing_risk_brier(
            np.array([2.0, 2.0, 4.0]),
            np.array([1, 2, 0]),
            np.array([0.8, 0.4, 0.2]),
            3.0,
            1,
            censoring,
        )
        self.assertAlmostEqual(score, ((0.2**2 + 0.4**2 + 0.2**2) / 0.75) / 3)

    def test_concordance_prior_competing_and_ties(self) -> None:
        times, events = np.array([1.0, 2.0, 3.0]), np.array([2, 1, 0])
        censoring = CensoringDistribution(times, events)
        score, count = competing_risk_concordance(
            times,
            events,
            np.array([0.1, 0.9, 0.1]),
            2.0,
            1,
            censoring,
        )
        self.assertEqual((score, count), (1.0, 2))
        self.assertEqual(
            competing_risk_concordance(
                times,
                events,
                np.ones(3),
                2.0,
                1,
                censoring,
            ),
            (0.5, 2),
        )

    def test_independent_metric_support(self) -> None:
        times, events = np.array([1.0, 2.0]), np.array([1, 2])
        censoring = CensoringDistribution(times, events)
        self.assertEqual(
            competing_risk_brier(
                times,
                events,
                np.array([1.0, 0.0]),
                1.0,
                1,
                censoring,
            ),
            0.0,
        )
        with self.assertRaises(MetricSupportError):
            competing_risk_concordance(times, events, np.ones(2), 0.0, 1, censoring)
        exhausted = CensoringDistribution(np.array([1.0]), np.array([0]))
        with self.assertRaises(MetricSupportError):
            competing_risk_brier(
                np.array([2.0]), np.array([0]), np.array([0.0]), 1.0, 1, exhausted
            )

    def test_concordance_uses_distinct_competing_pair_weights(self) -> None:
        censoring = CensoringDistribution(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1, 0, 2, 1]),
        )
        score, pairs = competing_risk_concordance(
            np.array([1.0, 3.0, 4.0]),
            np.array([2, 1, 0]),
            np.array([0.1, 0.8, 0.9]),
            3.0,
            1,
            censoring,
        )
        # Correct early-competing pair: weight 1/(2/3); incorrect later pair:
        # weight 1/(2/3)**2. Their weighted agreement is 1.5/(1.5+2.25).
        self.assertAlmostEqual(score, 0.4)
        self.assertEqual(pairs, 2)

    def test_concordance_censoring_ties_use_right_limit_for_later_pairs(self) -> None:
        censoring = CensoringDistribution(
            np.array([1.0, 1.0, 2.0, 3.0]), np.array([1, 0, 1, 1])
        )
        score, pairs = competing_risk_concordance(
            np.array([1.0, 1.0, 2.0, 3.0]),
            np.array([1, 2, 1, 0]),
            np.array([0.8, 0.9, 0.9, 0.1]),
            3.0,
            1,
            censoring,
        )
        # Later-pair weights are 1.5, 1.5, 2.25; early-competing weights
        # are 1, 1.5. Concordant mass including the prediction tie is 4.5.
        self.assertAlmostEqual(score, 4.5 / 7.75)
        self.assertEqual(pairs, 5)

    def test_invalid_metrics(self) -> None:
        times, events = np.array([1.0, 2.0]), np.array([1, 2])
        censoring = CensoringDistribution(times, events)
        for function in (competing_risk_brier, competing_risk_concordance):
            for horizon, risk in ((-1.0, 1), (np.nan, 1), (1.0, 0), (1.0, 1.5)):
                with self.assertRaises(ValueError):
                    function(
                        times, events, np.array([0.1, 0.2]), horizon, risk, censoring
                    )
        with self.assertRaises(ValueError):
            CensoringDistribution(times, np.array([1.0, 0.5]))
