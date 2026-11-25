"""Tests for the repeated-run significance helpers.

Synthetic score lists only; every test is millisecond-fast. Welch results are
cross-checked against scipy where scipy exposes them.
"""

import unittest

import numpy as np
from scipy import stats

from pyhealth.metrics import benjamini_hochberg, welch_relative_improvement_test


class TestBenjaminiHochberg(unittest.TestCase):
    def test_known_vector(self):
        p_values = [0.001, 0.008, 0.039, 0.041, 0.206]
        q_values, rejected = benjamini_hochberg(p_values, alpha=0.05)
        np.testing.assert_allclose(q_values, [0.005, 0.02, 0.05125, 0.05125, 0.206])
        self.assertEqual(rejected.tolist(), [True, True, False, False, False])

    def test_smallest_p_alone_can_be_rejected(self):
        # p1 <= alpha / m is enough, even when nothing else is
        q_values, rejected = benjamini_hochberg([0.01, 0.5, 0.6, 0.7], alpha=0.05)
        self.assertAlmostEqual(q_values[0], 0.04)
        self.assertEqual(rejected.tolist(), [True, False, False, False])

    def test_preserves_input_order(self):
        ordered = [0.041, 0.001, 0.206, 0.039, 0.008]
        q_ordered, rejected_ordered = benjamini_hochberg(ordered, alpha=0.05)
        q_sorted, rejected_sorted = benjamini_hochberg(sorted(ordered), alpha=0.05)
        np.testing.assert_allclose(q_ordered, q_sorted[[3, 0, 4, 2, 1]])
        self.assertEqual(rejected_ordered.tolist(), rejected_sorted[[3, 0, 4, 2, 1]].tolist())

    def test_matches_scipy_reference(self):
        if not hasattr(stats, "false_discovery_control"):
            self.skipTest("scipy too old for false_discovery_control")
        rng = np.random.default_rng(0)
        p_values = rng.random(12) ** 2
        expected = stats.false_discovery_control(p_values, method="bh")
        q_values, _ = benjamini_hochberg(p_values, alpha=0.05)
        np.testing.assert_allclose(q_values, expected, atol=1e-12)

    def test_alpha_and_edge_cases(self):
        p_values = [0.001, 0.008, 0.039, 0.041, 0.206]
        _, loose = benjamini_hochberg(p_values, alpha=0.5)
        self.assertEqual(loose.tolist(), [True] * 5)

        q_values, rejected = benjamini_hochberg([], alpha=0.05)
        self.assertEqual(q_values.size, 0)
        self.assertEqual(rejected.size, 0)

        q_values, rejected = benjamini_hochberg([0.0, 1.0], alpha=0.05)
        np.testing.assert_allclose(q_values, [0.0, 1.0])
        self.assertEqual(rejected.tolist(), [True, False])

    def test_rejects_out_of_range_p_values(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.0, 1.5])


class TestWelchRelativeImprovementTest(unittest.TestCase):
    def test_row_matches_hand_and_scipy_computation(self):
        scores = {"all/roc_auc_macro": {18: [0.90, 0.91, 0.92], 34: [0.93, 0.94, 0.95]}}
        (row,) = welch_relative_improvement_test(scores, baseline=18)

        self.assertTrue(row["tested"])
        self.assertEqual(row["reason"], "")
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["baseline_n"], 3)
        self.assertAlmostEqual(row["mean"], 0.94)
        self.assertAlmostEqual(row["std"], 0.01)
        self.assertAlmostEqual(row["baseline_mean"], 0.91)
        self.assertAlmostEqual(row["baseline_std"], 0.01)
        self.assertAlmostEqual(row["improvement"], 0.03)
        self.assertAlmostEqual(row["relative_improvement"], 100.0 * 0.03 / 0.91)

        reference = stats.ttest_ind([0.93, 0.94, 0.95], [0.90, 0.91, 0.92], equal_var=False)
        self.assertAlmostEqual(row["t_statistic"], float(reference.statistic), places=10)
        self.assertAlmostEqual(row["p_value"], float(reference.pvalue), places=10)
        df = getattr(reference, "df", None)
        if df is not None:
            self.assertAlmostEqual(row["degrees_of_freedom"], float(df), places=10)
        else:  # pragma: no cover - older scipy
            self.assertAlmostEqual(row["degrees_of_freedom"], 4.0, places=10)

        # single test in the family, so q == p
        self.assertAlmostEqual(row["q_value"], row["p_value"])
        self.assertTrue(row["significant"])

    def test_welch_does_not_assume_equal_variance(self):
        arm = [4.0, 5.0, 6.0, 7.0]
        base = [1.5, 5.5]
        pooled = stats.ttest_ind(arm, base, equal_var=True)
        reference = stats.ttest_ind(arm, base, equal_var=False)
        (row,) = welch_relative_improvement_test({"f1": {"a": arm, "b": base}}, baseline="b")

        self.assertTrue(row["tested"])
        self.assertAlmostEqual(row["t_statistic"], float(reference.statistic), places=10)
        self.assertAlmostEqual(row["p_value"], float(reference.pvalue), places=10)
        self.assertLess(row["degrees_of_freedom"], 3.0)
        self.assertNotAlmostEqual(row["t_statistic"], float(pooled.statistic), places=6)
        self.assertNotAlmostEqual(row["p_value"], float(pooled.pvalue), places=6)

    def test_decreased_improvement_is_excluded_from_testing(self):
        scores = {
            "rhythm/roc_auc_macro": {
                18: [0.92, 0.91, 0.93],
                50: [0.90, 0.91, 0.92],  # worse than baseline
            }
        }
        (row,) = welch_relative_improvement_test(scores, baseline=18)
        self.assertFalse(row["tested"])
        self.assertFalse(row["significant"])
        self.assertIsNone(row["p_value"])
        self.assertIsNone(row["q_value"])
        self.assertEqual(row["reason"], "decreased relative improvement")
        self.assertLess(row["relative_improvement"], 0.0)
        # means are still reported, so the sweep table stays complete
        self.assertAlmostEqual(row["mean"], float(np.mean([0.90, 0.91, 0.92])))

    def test_equal_means_are_not_an_improvement(self):
        (row,) = welch_relative_improvement_test(
            {"x": {0: [0.5, 0.6], 1: [0.5, 0.6]}}, baseline=0
        )
        self.assertFalse(row["tested"])
        self.assertEqual(row["reason"], "decreased relative improvement")
        self.assertAlmostEqual(row["relative_improvement"], 0.0)

    def test_zero_variance_in_both_settings(self):
        scores = {
            "equal": {18: [0.9, 0.9], 34: [0.95, 0.95]},
            "same": {18: [0.9, 0.9], 50: [0.9, 0.9]},
        }
        rows = {row["setting"]: row for row in welch_relative_improvement_test(scores, baseline=18)}
        self.assertEqual(rows["equal"]["reason"], "zero variance in both settings")
        self.assertAlmostEqual(rows["equal"]["p_value"], 0.0)
        self.assertTrue(rows["equal"]["significant"])
        self.assertEqual(rows["same"]["reason"], "decreased relative improvement")
        self.assertIsNone(rows["same"]["p_value"])

    def test_short_inputs_are_reported_not_tested(self):
        # the paper's single run per split gives one score per arm
        (row,) = welch_relative_improvement_test({"f1": {18: [0.8], 34: [0.9]}}, baseline=18)
        self.assertFalse(row["tested"])
        self.assertEqual(row["reason"], "fewer than 2 split scores")
        self.assertEqual(row["n"], 1)
        self.assertAlmostEqual(row["mean"], 0.9)
        self.assertIsNone(row["std"])

    def test_missing_baseline_arm(self):
        (row,) = welch_relative_improvement_test({"f1": {34: [0.9, 0.91]}}, baseline=18)
        self.assertFalse(row["tested"])
        self.assertIn("baseline arm 18 missing", row["reason"])
        self.assertIsNone(row["baseline_mean"])

    def test_baseline_mean_of_zero_leaves_relative_improvement_undefined(self):
        (row,) = welch_relative_improvement_test(
            {"metric": {"base": [0.0, 0.0], "arm": [0.5, 0.6]}}, baseline="base"
        )
        self.assertFalse(row["tested"])
        self.assertEqual(row["reason"], "baseline mean is zero, relative improvement undefined")
        self.assertAlmostEqual(row["improvement"], 0.55)
        self.assertIsNone(row["relative_improvement"])

    def test_bh_spans_every_setting(self):
        # 5 split scores per arm, as in the paper's five trial splits
        base = [0.900, 0.910, 0.920, 0.910, 0.920]
        strong = [0.9245, 0.925, 0.9245, 0.925, 0.9245]
        weak = [0.9165, 0.917, 0.916, 0.9165, 0.917]
        scores = {"a/roc_auc": {18: base, 50: strong}, "b/roc_auc": {18: base, 50: weak}}
        rows = welch_relative_improvement_test(scores, baseline=18)

        self.assertEqual([row["setting"] for row in rows], ["a/roc_auc", "b/roc_auc"])
        self.assertTrue(all(row["tested"] for row in rows))

        raw_p = [row["p_value"] for row in rows]
        expected_q, expected_reject = benjamini_hochberg(raw_p, alpha=0.05)
        np.testing.assert_allclose([row["q_value"] for row in rows], expected_q)
        self.assertEqual([row["significant"] for row in rows], expected_reject.tolist())

        # one test alone would pass at 0.05, but in a 2-test family it does not
        self.assertLess(raw_p[0], 0.05)
        self.assertAlmostEqual(rows[0]["q_value"], 2.0 * raw_p[0])
        self.assertGreater(rows[0]["q_value"], 0.05)
        self.assertFalse(rows[0]["significant"])
        # the largest p-value of the family is left uncorrected
        self.assertAlmostEqual(rows[1]["q_value"], raw_p[1])

    def test_alpha_is_applied_to_the_whole_family(self):
        base = [0.900, 0.910, 0.920, 0.910, 0.920]
        strong = [0.9245, 0.925, 0.9245, 0.925, 0.9245]
        weak = [0.9165, 0.917, 0.916, 0.9165, 0.917]
        scores = {"a/roc_auc": {18: base, 50: strong}, "b/roc_auc": {18: base, 50: weak}}
        strict = welch_relative_improvement_test(scores, baseline=18, alpha=0.05)
        loose = welch_relative_improvement_test(scores, baseline=18, alpha=0.50)
        self.assertFalse(any(row["significant"] for row in strict))
        self.assertTrue(all(row["significant"] for row in loose))

    def test_empty_input(self):
        self.assertEqual(welch_relative_improvement_test({}, baseline=18), [])

    def test_rows_exclude_the_baseline_arm(self):
        rows = welch_relative_improvement_test(
            {"s": {18: [0.9, 0.91], 34: [0.92, 0.93], 50: [0.94, 0.95]}}, baseline=18
        )
        self.assertEqual([row["arm"] for row in rows], [34, 50])
        self.assertTrue(all(row["baseline"] == 18 for row in rows))


if __name__ == "__main__":
    unittest.main()
