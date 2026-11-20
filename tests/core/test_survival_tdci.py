"""Tests for the time-dependent concordance index and its bootstrap helpers."""

import math
import unittest

import numpy as np

from pyhealth.metrics import (
    mean_tdci_bootstrap_ci,
    mean_tdci_paired_bootstrap_diff,
    outcomes_from_labels,
    time_dependent_concordance_index,
)


def _authors_tdci(rows, times, horizons):
    """Reference: the MAS authors' c_index applied per (t, delta_t) cell.

    ``rows`` are ``(patient, landmark_time, risk, event_time, event)``. Pairs
    need ``T_i < T_j`` and ``T_i <= t + delta_t`` with an event for ``i``;
    risks must not tie (the authors count ties as 0, we count one half).
    """
    out = np.full((len(times), len(horizons)), np.nan)
    for k, t in enumerate(times):
        latest = {}
        for patient, landmark, risk, event_time, event in rows:
            newer = patient not in latest or landmark >= latest[patient][0]
            if event_time > t and landmark <= t and newer:
                latest[patient] = (landmark, risk, event_time, event)
        if not latest:
            continue
        values = np.array([v[1:] for v in latest.values()])
        risk, time, event = values[:, 0], values[:, 1], values[:, 2]
        for h, delta in enumerate(horizons):
            later = time[:, None] < time[None, :]
            case = ((time <= t + delta) & (event == 1))[:, None]
            higher = risk[:, None] > risk[None, :]
            pairs = (later & case).sum()
            out[k, h] = np.nan if pairs == 0 else (later & case & higher).sum() / pairs
    return out


class TestTimeDependentConcordance(unittest.TestCase):
    # Patient a dies at 2.0, b dies at 6.0, c is censored at 8.0.
    # Visits at 0 and 1; c's visit at 1 has a misleadingly high risk.
    patient = ("a", "a", "b", "b", "c", "c")
    landmark = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    event_time = (2.0, 2.0, 6.0, 6.0, 8.0, 8.0)
    event = (1, 1, 1, 1, 0, 0)
    risk = (3.0, 3.0, 2.0, 2.0, 1.0, 9.0)

    def _tdci(self, **kwargs):
        return time_dependent_concordance_index(
            self.event_time,
            self.event,
            self.risk,
            self.landmark,
            self.patient,
            **kwargs,
        )

    def test_hand_computed_cells(self):
        out = self._tdci(prediction_times=[0.5, 1.5], horizons=[2.0, 6.0])
        # t=0.5 uses day-0 risks (3, 2, 1). Within 2.5 y only a dies: pairs
        # (a,b), (a,c), both concordant. Within 6.5 y b also dies: (b,c) too.
        # t=1.5 uses visit-1 risks (3, 2, 9): (a,c) and (b,c) are discordant.
        np.testing.assert_allclose(out["tdci"], [[1.0, 1.0], [0.5, 1 / 3]])
        np.testing.assert_array_equal(out["n_at_risk"], [3, 3])
        np.testing.assert_array_equal(out["n_events"], [[1, 2], [1, 2]])
        self.assertAlmostEqual(out["mean_tdci"], np.mean([1.0, 1.0, 0.5, 1 / 3]))

    def test_uses_no_visit_after_the_prediction_time(self):
        # At t=0.5 only day-0 visits count, so c's risk of 9 at year 1 is unseen.
        before = self._tdci(prediction_times=[0.5], horizons=[2.0])["tdci"]
        self.assertEqual(before[0, 0], 1.0)

    def test_patients_with_event_by_t_are_excluded(self):
        out = self._tdci(prediction_times=[3.0], horizons=[4.0])
        self.assertEqual(out["n_at_risk"][0], 2)  # a died at 2.0
        # b dies at 6 with risk 2; c outlives b with risk 9: discordant.
        self.assertEqual(out["tdci"][0, 0], 0.0)

    def test_cells_without_cases_are_nan_and_skipped_in_mean(self):
        out = self._tdci(prediction_times=[0.5, 7.0], horizons=[1.0])
        self.assertTrue(math.isnan(out["tdci"][1, 0]))  # only c left, censored
        self.assertTrue(math.isnan(out["tdci"][0, 0]))  # no death in (0.5, 1.5]
        self.assertTrue(math.isnan(out["mean_tdci"]))

    def test_matches_authors_definition(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            rows = []
            for p in range(int(rng.integers(5, 25))):
                event_time = float(rng.uniform(0.3, 12))
                event = int(rng.random() < 0.5)
                visits = np.concatenate([[0.0], rng.uniform(0, event_time, 3)])
                for landmark in np.sort(visits):
                    rows.append((f"p{p}", landmark, rng.normal(), event_time, event))
            patient, landmark, risk, event_time, event = map(np.array, zip(*rows))
            ours = time_dependent_concordance_index(
                event_time, event, risk, landmark, patient
            )["tdci"]
            reference = _authors_tdci(rows, (0.5, 1, 3, 5), (1, 3, 5, 7))
            np.testing.assert_allclose(ours, reference, equal_nan=True)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            time_dependent_concordance_index([1.0], [1], [0.5, 0.2], [0.0], ["a"])
        with self.assertRaises(ValueError):
            time_dependent_concordance_index([1.0], [2], [0.5], [0.0], ["a"])
        with self.assertRaises(ValueError):
            time_dependent_concordance_index([1.0], [1], [np.nan], [0.0], ["a"])
        with self.assertRaises(ValueError):
            time_dependent_concordance_index([], [], [], [], [])


class TestOutcomesFromLabels(unittest.TestCase):
    def test_counting_process_labels(self):
        y_true = [[0.0, 1.0, 0], [1.0, 2.5, 1], [0.0, 4.0, 0]]
        event_time, event = outcomes_from_labels(y_true, [0, 1, 0], ["a", "a", "b"])
        np.testing.assert_array_equal(event_time, [2.5, 2.5, 4.0])
        np.testing.assert_array_equal(event, [1, 1, 0])

    def test_landmark_labels(self):
        y_true = [[2.5, 1], [1.5, 1], [4.0, 0]]
        event_time, event = outcomes_from_labels(y_true, [0, 1, 0], ["a", "a", "b"])
        np.testing.assert_allclose(event_time, [2.5, 2.5, 4.0])
        np.testing.assert_array_equal(event, [1, 1, 0])

    def test_rejects_inconsistent_or_malformed_labels(self):
        with self.assertRaises(ValueError):
            outcomes_from_labels([[2.0, 1], [2.0, 1]], [0, 1], ["a", "a"])
        with self.assertRaises(ValueError):
            outcomes_from_labels([[1.0]], [0], ["a"])


class TestTdciBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        rows = []
        for p in range(30):
            event_time = float(rng.uniform(1, 12))
            for landmark in (0.0, 0.5, 2.0):
                if landmark < event_time:
                    rows.append((p, landmark, event_time, int(p % 2 == 0)))
        self.patient, self.landmark, self.event_time, self.event = map(
            np.array, zip(*rows)
        )
        self.risk = -self.event_time + rng.normal(0, 2, len(rows))

    def test_ci_brackets_the_estimate_and_is_reproducible(self):
        args = (self.event_time, self.event, self.risk, self.landmark, self.patient)
        ci = mean_tdci_bootstrap_ci(*args, n_boot=50, seed=0)
        estimate = time_dependent_concordance_index(*args)["mean_tdci"]
        self.assertAlmostEqual(ci["estimate"], estimate)
        self.assertLessEqual(ci["lower"], ci["estimate"])
        self.assertLessEqual(ci["estimate"], ci["upper"])
        self.assertEqual(ci, mean_tdci_bootstrap_ci(*args, n_boot=50, seed=0))

    def test_paired_difference_of_identical_models_is_zero(self):
        diff = mean_tdci_paired_bootstrap_diff(
            self.event_time,
            self.event,
            self.risk,
            self.risk,
            self.landmark,
            self.patient,
            n_boot=30,
        )
        self.assertEqual((diff["lower"], diff["upper"]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
