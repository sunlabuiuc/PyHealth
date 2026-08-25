"""Tests for pyhealth.metrics.fairness_utils.group: disparate_impact and
statistical_parity_difference, focused on the empty-subgroup NaN bug.
"""

import unittest

import numpy as np

from pyhealth.metrics.fairness_utils import (
    disparate_impact,
    statistical_parity_difference,
)


class TestFairnessEmptySubgroup(unittest.TestCase):
    """Regression tests: an empty subgroup must raise ValueError, not
    silently return NaN.

    Previously, an empty group made the favorable-outcome rate a numpy
    0/0 NaN. The guard in disparate_impact checked `rate == 0`, which
    NaN never satisfies, so the ValueError never fired and NaN was
    returned silently. statistical_parity_difference had no guard at
    all. Both cases are fixed by validating group non-emptiness before
    computing the rate.
    """

    def setUp(self):
        # sensitive_attributes: 1 = protected, 0 = unprotected.
        self.no_unprotected = np.array([1, 1, 1])
        self.no_protected = np.array([0, 0, 0])
        self.y_pred = np.array([1, 0, 1])

    def test_disparate_impact_raises_when_unprotected_empty(self):
        with self.assertRaises(ValueError):
            disparate_impact(self.no_unprotected, self.y_pred)

    def test_disparate_impact_raises_when_protected_empty(self):
        """The original bug's guard only ever checked the unprotected
        group; the protected group being empty was never checked at
        all and also produced a silent NaN."""
        with self.assertRaises(ValueError):
            disparate_impact(self.no_protected, self.y_pred)

    def test_disparate_impact_empty_group_raises_even_with_allow_zero_division(self):
        """An empty group is a different failure mode than a non-empty
        group with a genuinely-zero rate: allow_zero_division must not
        paper over a group we have zero information about."""
        with self.assertRaises(ValueError):
            disparate_impact(
                self.no_unprotected, self.y_pred, allow_zero_division=True
            )

    def test_statistical_parity_difference_raises_when_unprotected_empty(self):
        with self.assertRaises(ValueError):
            statistical_parity_difference(self.no_unprotected, self.y_pred)

    def test_statistical_parity_difference_raises_when_protected_empty(self):
        with self.assertRaises(ValueError):
            statistical_parity_difference(self.no_protected, self.y_pred)

    def test_no_nan_ever_returned(self):
        """Direct regression test for the core symptom: an empty group
        must never let a NaN escape as a return value. Catches the
        exception outside the assertion so a hypothetical future
        regression that silently returns NaN (instead of raising) would
        actually be caught by the isnan check, rather than the test
        vacuously passing because nothing after a raise executes."""
        for sa in (self.no_unprotected, self.no_protected):
            try:
                result = disparate_impact(sa, self.y_pred)
            except ValueError:
                result = None
            if result is not None:
                self.assertFalse(np.isnan(result), "NaN silently returned instead of raising")

            try:
                result = statistical_parity_difference(sa, self.y_pred)
            except ValueError:
                result = None
            if result is not None:
                self.assertFalse(np.isnan(result), "NaN silently returned instead of raising")


class TestFairnessNormalOperation(unittest.TestCase):
    """Sanity checks that the fix doesn't change behavior for non-empty
    groups (the common, legitimate case)."""

    def setUp(self):
        # unprotected (sa=0): preds [1, 0] -> 1/2 favorable
        # protected (sa=1):   preds [1, 1, 0] -> 2/3 favorable
        self.sensitive_attributes = np.array([0, 0, 1, 1, 1])
        self.y_pred = np.array([1, 0, 1, 1, 0])

    def test_disparate_impact_normal(self):
        result = disparate_impact(self.sensitive_attributes, self.y_pred)
        self.assertAlmostEqual(result, (2 / 3) / (1 / 2))

    def test_statistical_parity_difference_normal(self):
        result = statistical_parity_difference(
            self.sensitive_attributes, self.y_pred
        )
        self.assertAlmostEqual(result, (2 / 3) - (1 / 2))

    def test_disparate_impact_zero_rate_non_empty_group_still_raises_by_default(self):
        """A non-empty group with a genuinely-zero favorable rate is NOT
        the same failure mode as an empty group, but should still raise
        by default (existing, unchanged behavior) unless the caller
        opts in via allow_zero_division."""
        sa = np.array([0, 0, 1, 1])
        yp = np.array([0, 0, 1, 1])  # unprotected group: 0% favorable, but non-empty
        with self.assertRaises(ValueError):
            disparate_impact(sa, yp)

    def test_disparate_impact_allow_zero_division_for_non_empty_zero_rate_group(self):
        """allow_zero_division should still work for its intended case:
        a non-empty group whose rate happens to be exactly 0."""
        sa = np.array([0, 0, 1, 1])
        yp = np.array([0, 0, 1, 1])
        result = disparate_impact(sa, yp, allow_zero_division=True, epsilon=1e-8)
        self.assertAlmostEqual(result, 1.0 / 1e-8)


if __name__ == "__main__":
    unittest.main()
