"""Tests for pyhealth.metrics.fairness_utils.capacity_fairness_metrics.

``_reference_evaluate`` is the bootstrap part of ``evaluate`` in
``mimic/analysis groups.ipynb`` of Jeanselme's ClinicalPresenceFairness at
commit fcb580b (https://github.com/Jeanselme/ClinicalPresenceFairness; MIT
License, Copyright (c) 2022 Vincent Jeanselme). It is verbatim apart from line
breaks, a loop in place of the repeated ``result`` entries, and dropping the
Brier score, FPR at 90% TPR and TPR at 10% FPR, which draw no random numbers.
"""

import inspect
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pyhealth.metrics.fairness_utils import capacity_fairness_metrics

METRICS = ("auc", "prioritization_rate", "fnr", "fpr")

# Our metric -> the reference's name for it
REFERENCE_NAMES = {
    "auc": "AUC ROC",
    "prioritization_rate": "Prioritized",
    "fnr": "Wrongly not prioritized (FNR)",
    "fpr": "Wrongly prioritized (FPR)",
}

# Hand example: k = 3, so the threshold is 0.7 and samples 0-2 are prioritised
HAND_Y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
HAND_P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
HAND_S = np.array([0, 0, 1, 0, 1, 0, 0, 1, 1, 1])
# Group -> (auc, prioritization_rate, fnr, fpr)
HAND_EXPECTED = {
    "protected": (4 / 6, 1 / 5, 2 / 3, 0),
    "unprotected": (3 / 6, 2 / 5, 1 / 2, 1 / 3),
    "difference": (1 / 6, -1 / 5, 1 / 6, -1 / 3),
    "population": (14 / 25, 3 / 10, 3 / 5, 1 / 5),
}


def _reference_evaluate(y_true, groups, groups_unique, y_pred, iterations=100, p=0.5):
    rocs, screened, screened_fpr, screened_fnr = (
        {b: [] for b in groups_unique},
        {b: [] for b in groups_unique},
        {b: [] for b in groups_unique},
        {b: [] for b in groups_unique},
    )

    # Threshold screening
    threshold_top = pd.Series(y_pred).nlargest(int(len(y_pred) * p), keep="all").min()

    for group in groups_unique:
        if group == "Overall":
            y_pred_group = y_pred
            y_true_group = y_true
        else:
            y_pred_group = y_pred[groups == group]
            y_true_group = y_true[groups == group]

        for i in range(iterations):
            bootstrap = np.random.choice(
                np.arange(len(y_pred_group)), size=len(y_pred_group), replace=True
            )
            y_pred_iteration = y_pred_group[bootstrap]
            y_true_iteration = y_true_group[bootstrap]

            rocs[group].append(roc_auc_score(y_true_iteration, y_pred_iteration))

            # Percentage screened-out in bottom 30 %
            selected = y_pred_iteration >= threshold_top
            screened[group].append(np.mean(selected))
            screened_fnr[group].append(
                (y_true_iteration[~selected]).sum() / y_true_iteration.sum()
            )
            screened_fpr[group].append(
                (1 - y_true_iteration[selected]).sum() / (1 - y_true_iteration).sum()
            )

    result = {}
    named = [
        ("AUC ROC", rocs),
        ("Prioritized", screened),
        ("Wrongly prioritized (FPR)", screened_fpr),
        ("Wrongly not prioritized (FNR)", screened_fnr),
    ]
    if len(groups_unique) == 2:
        difference = "Difference {} - {}".format(  # noqa: UP032
            groups_unique[0], groups_unique[1]
        )
        for name, values in named:
            diff = np.array(values[groups_unique[0]]) - np.array(
                values[groups_unique[1]]
            )
            result[(difference, name, "Mean")] = np.mean(diff)
            result[(difference, name, "Std")] = np.std(diff)
    for group in groups_unique:
        for name, values in named:
            result[(group, name, "Mean")] = np.mean(values[group])
            result[(group, name, "Std")] = np.std(values[group])
    return result


def synthetic(n=3000, prevalence=0.08, protected_share=0.12, seed=1):
    """Labels, scores with ties (rounded to 3 decimals) and group flags."""
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < prevalence).astype(int)
    p = np.clip(0.3 * y + rng.normal(0.2, 0.15, size=n), 0, 1).round(3)
    s = (rng.uniform(size=n) < protected_share).astype(int)
    return y, p, s


def balanced(n):
    """Synthetic data whose groups keep both classes in every resample."""
    return synthetic(n, prevalence=0.3, protected_share=0.4)


class _NoResample:
    """Stands in for RandomState: every resample is the group itself."""

    def __init__(self, seed=None):
        pass

    def choice(self, a, size, replace):
        return np.arange(a)


def no_resample():
    return mock.patch("numpy.random.RandomState", _NoResample)


class TestAgainstReference(unittest.TestCase):
    """Golden comparison with the reference, which seeds before each call.

    Both draw the same index stream for any number of resamples, so 10 keep the
    test fast; ``test_defaults_are_the_papers`` covers the 100 used in the paper.
    """

    N_BOOTSTRAP = 10

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def reference(self, y, p, s, capacity):
        np.random.seed(42)
        if s is None:
            groups, groups_unique = None, ["Overall"]
        else:
            groups = np.where(s == 1, "Marginalised", "Rest")
            groups_unique = ["Marginalised", "Rest"]
        return _reference_evaluate(
            y, groups, groups_unique, p, iterations=self.N_BOOTSTRAP, p=capacity
        )

    def test_matches_reference(self):
        y, p, s = synthetic()
        # Our group -> the reference's group
        cases = [
            (
                s,
                {
                    "protected": "Marginalised",
                    "unprotected": "Rest",
                    "difference": "Difference Marginalised - Rest",
                },
            ),
            (None, {"population": "Overall"}),
        ]
        for capacity in (0.05, 0.3, 0.5):
            for sensitive, groups in cases:
                with self.subTest(capacity=capacity, groups=list(groups)):
                    ours = capacity_fairness_metrics(
                        y, p, sensitive, capacity, n_bootstrap=self.N_BOOTSTRAP
                    )
                    ref = self.reference(y, p, sensitive, capacity)
                    for group, ref_group in groups.items():
                        for m in METRICS:
                            name = REFERENCE_NAMES[m]
                            self.assertAlmostEqual(
                                ours[f"{m}_{group}"],
                                ref[(ref_group, name, "Mean")],
                                places=12,
                            )
                            self.assertAlmostEqual(
                                ours[f"{m}_{group}_std"],
                                ref[(ref_group, name, "Std")],
                                places=12,
                            )

    def test_threshold_matches_reference(self):
        y, p, s = synthetic()
        for capacity in (0.05, 0.3, 0.5, 1.0):
            with self.subTest(capacity=capacity):
                k = int(len(p) * capacity)
                expected = pd.Series(p).nlargest(k, keep="all").min()
                result = capacity_fairness_metrics(y, p, s, capacity, n_bootstrap=1)
                self.assertEqual(result["threshold"], expected)

    def test_defaults_are_the_papers(self):
        params = inspect.signature(capacity_fairness_metrics).parameters
        defaults = {
            name: params[name].default for name in ("capacity", "n_bootstrap", "seed")
        }
        self.assertEqual(defaults, {"capacity": 0.3, "n_bootstrap": 100, "seed": 42})


class TestHandComputed(unittest.TestCase):
    """Exact values on 10 samples, with resampling switched off."""

    def test_ten_samples(self):
        with no_resample():
            grouped = capacity_fairness_metrics(HAND_Y, HAND_P, HAND_S, n_bootstrap=2)
            population = capacity_fairness_metrics(HAND_Y, HAND_P, n_bootstrap=2)
        for result in (grouped, population):
            self.assertEqual(result["threshold"], 0.7)
            for key, value in result.items():
                if key.endswith("_std"):
                    self.assertAlmostEqual(value, 0.0, places=12, msg=key)
        for group, expected in HAND_EXPECTED.items():
            result = population if group == "population" else grouped
            for m, value in zip(METRICS, expected):
                with self.subTest(group=group, metric=m):
                    self.assertAlmostEqual(result[f"{m}_{group}"], value, places=12)

    def test_k_truncates(self):
        # 10 * 0.29 = 2.9 -> k = 2, as int() does in the reference
        for capacity, threshold in [(0.1, 0.9), (0.25, 0.8), (0.29, 0.8), (0.39, 0.7)]:
            with self.subTest(capacity=capacity), no_resample():
                result = capacity_fairness_metrics(
                    HAND_Y, HAND_P, capacity=capacity, n_bootstrap=1
                )
                self.assertEqual(result["threshold"], threshold)

    def test_ties_at_threshold_all_prioritised(self):
        y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1])
        with no_resample():
            result = capacity_fairness_metrics(y, p, capacity=0.3, n_bootstrap=1)
        self.assertEqual(result["threshold"], 0.5)
        # k = 3, but all four samples tied at 0.5 count
        self.assertAlmostEqual(result["prioritization_rate_population"], 0.6)
        self.assertAlmostEqual(result["fnr_population"], 1 / 4)
        self.assertAlmostEqual(result["fpr_population"], 3 / 6)

    def test_full_capacity_prioritises_everyone(self):
        with no_resample():
            result = capacity_fairness_metrics(
                HAND_Y, HAND_P, HAND_S, capacity=1.0, n_bootstrap=1
            )
        self.assertEqual(result["threshold"], 0.05)
        for group in ("protected", "unprotected"):
            self.assertEqual(result[f"prioritization_rate_{group}"], 1.0)
            self.assertEqual(result[f"fnr_{group}"], 0.0)
            self.assertEqual(result[f"fpr_{group}"], 1.0)


class TestConventions(unittest.TestCase):
    def test_difference_is_protected_minus_unprotected(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.2, size=2000)
        s = rng.binomial(1, 0.3, size=2000)
        p = 1 / (1 + np.exp(-(2 * y - s + rng.normal(size=2000))))
        result = capacity_fairness_metrics(y, p, s, n_bootstrap=20)
        # The protected group scores lower, so it is prioritised less
        self.assertLess(result["prioritization_rate_difference"], 0)
        self.assertGreater(result["fnr_difference"], 0)
        for m in METRICS:
            self.assertAlmostEqual(
                result[f"{m}_difference"],
                result[f"{m}_protected"] - result[f"{m}_unprotected"],
                places=12,
            )

    def test_one_marks_protected(self):
        with no_resample():
            result = capacity_fairness_metrics(HAND_Y, HAND_P, HAND_S, n_bootstrap=1)
            flipped = capacity_fairness_metrics(
                HAND_Y, HAND_P, 1 - HAND_S, n_bootstrap=1
            )
        for m in METRICS:
            self.assertEqual(flipped[f"{m}_protected"], result[f"{m}_unprotected"])
            self.assertEqual(flipped[f"{m}_unprotected"], result[f"{m}_protected"])
            self.assertAlmostEqual(
                flipped[f"{m}_difference"], -result[f"{m}_difference"], places=12
            )

    def test_keys_and_types(self):
        y, p, s = balanced(500)
        for sensitive, groups in [
            (s, ("protected", "unprotected", "difference")),
            (None, ("population",)),
        ]:
            result = capacity_fairness_metrics(y, p, sensitive, n_bootstrap=5)
            expected = {"threshold"} | {
                f"{m}_{g}{suffix}"
                for m in METRICS
                for g in groups
                for suffix in ("", "_std")
            }
            self.assertEqual(set(result), expected)
            for key, value in result.items():
                self.assertIs(type(value), float, key)
                self.assertTrue(np.isfinite(value), key)

    def test_input_forms(self):
        y, p, s = balanced(500)
        expected = capacity_fairness_metrics(y, p, s, n_bootstrap=5)
        for name, args in [
            ("bool", (y.astype(bool), p, s.astype(bool))),
            ("float", (y.astype(float), p, s.astype(float))),
            ("column", (y.reshape(-1, 1), p.reshape(-1, 1), s.reshape(-1, 1))),
            ("list", (y.tolist(), p.tolist(), s.tolist())),
        ]:
            with self.subTest(form=name):
                result = capacity_fairness_metrics(*args, n_bootstrap=5)
                self.assertEqual(result, expected)


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.y, self.p, self.s = balanced(1000)

    def run_metrics(self, seed):
        return capacity_fairness_metrics(
            self.y, self.p, self.s, n_bootstrap=10, seed=seed
        )

    def test_same_seed_same_result(self):
        self.assertEqual(self.run_metrics(0), self.run_metrics(0))

    def test_seed_changes_only_the_resamples(self):
        a, b = self.run_metrics(0), self.run_metrics(1)
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertNotEqual(a["auc_protected"], b["auc_protected"])

    def test_seed_none_draws_fresh_resamples(self):
        self.assertNotEqual(self.run_metrics(None), self.run_metrics(None))

    def test_global_rng_untouched(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        expected = np.random.random_sample()
        np.random.seed(0)
        self.run_metrics(42)
        self.assertEqual(np.random.random_sample(), expected)


class TestErrors(unittest.TestCase):
    def test_invalid_inputs(self):
        y, p, s = balanced(500)
        one_positive = np.r_[1, np.zeros(99, dtype=int)]
        cases = [
            ("length", (y, p[:-1]), {}, "same length"),
            ("y not binary", (np.where(y == 1, 2, 0), p), {}, "y_true must contain"),
            ("y NaN", (np.where(y == 1, np.nan, y), p), {}, "y_true must contain"),
            ("p NaN", (y, np.where(y == 1, np.nan, p)), {}, "NaN or infinite"),
            ("p inf", (y, np.where(y == 1, np.inf, p)), {}, "NaN or infinite"),
            ("capacity 0", (y, p), {"capacity": 0}, r"in \(0, 1\]"),
            ("capacity > 1", (y, p), {"capacity": 1.5}, r"in \(0, 1\]"),
            ("capacity NaN", (y, p), {"capacity": np.nan}, r"in \(0, 1\]"),
            ("k = 0", (y, p), {"capacity": 0.001}, "no sample can be prioritised"),
            ("n_bootstrap 0", (y, p), {"n_bootstrap": 0}, "n_bootstrap"),
            ("s length", (y, p, s[:-1]), {}, "same length"),
            ("s not 0/1", (y, p, 2 * s), {}, "sensitive_attributes must contain"),
            ("no protected", (y, p, 0 * s), {}, "The protected group has no"),
            ("no unprotected", (y, p, 0 * s + 1), {}, "The unprotected group has no"),
            ("protected one class", (y, p, y), {}, "The protected group needs both"),
            ("population one class", (0 * y, p), {}, "The population group needs both"),
            (
                "resample loses a class",
                (one_positive, np.linspace(0, 1, 100)),
                {},
                "resample of the population group has only one class",
            ),
        ]
        for name, args, kwargs, message in cases:
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                capacity_fairness_metrics(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
