"""Tests for counting-process sample construction in liver_survival."""

import itertools
import math
import unittest

import numpy as np
import pandas as pd

from pyhealth.tasks.liver_survival import (
    DAYS_PER_YEAR,
    build_binary_horizon_samples,
    build_counting_process_samples,
    build_landmark_samples,
    counting_process_samples_from_frame,
    event_indicator,
    forward_fill,
)

NAN = float("nan")


def _fixture() -> pd.DataFrame:
    """Five synthetic patients with outcomes that can be checked by hand.

    1: died on day 913 after visits on days 0, 182, 365, 730.
    2: censored on day 1461 after yearly visits.
    3: transplanted on day 548.
    4: one visit, censored on day 292.
    5: died on day 2191; AST missing at the day-365 visit.
    """
    rows = [
        # id, day, futime, status, bili, ast
        (1, 0, 913, 2, 1.0, 100.0),
        (1, 182, 913, 2, 1.5, 110.0),
        (1, 365, 913, 2, 2.0, 120.0),
        (1, 730, 913, 2, 3.0, 130.0),
        (2, 0, 1461, 0, 0.5, 50.0),
        (2, 365, 1461, 0, 0.6, 55.0),
        (2, 730, 1461, 0, 0.7, 60.0),
        (2, 1095, 1461, 0, 0.8, 65.0),
        (3, 0, 548, 1, 4.0, 200.0),
        (3, 182, 548, 1, 5.0, 210.0),
        (3, 365, 548, 1, 6.0, 220.0),
        (4, 0, 292, 0, 1.2, 80.0),
        (5, 0, 2191, 2, 0.9, 90.0),
        (5, 365, 2191, 2, 1.1, NAN),
        (5, 730, 2191, 2, 1.3, 95.0),
    ]
    return pd.DataFrame(rows, columns=["id", "day", "futime", "status", "bili", "ast"])


def _years(day: float) -> float:
    return day / DAYS_PER_YEAR


class TestCountingProcessSamples(unittest.TestCase):
    """Builds [start, stop, event] rows from the five-patient fixture."""

    @classmethod
    def setUpClass(cls):
        cls.samples = counting_process_samples_from_frame(_fixture(), ["bili", "ast"])
        cls.by_patient = {}
        for sample in cls.samples:
            cls.by_patient.setdefault(sample["patient_id"], []).append(sample)

    def test_one_sample_per_visit(self):
        self.assertEqual(len(self.samples), 15)
        self.assertEqual(
            {pid: len(rows) for pid, rows in self.by_patient.items()},
            {"1": 4, "2": 4, "3": 3, "4": 1, "5": 3},
        )

    def test_sample_keys_and_types(self):
        sample = self.samples[0]
        self.assertEqual(
            set(sample),
            {
                "patient_id",
                "visit_id",
                "landmark_time",
                "features",
                "survival_interval",
            },
        )
        self.assertIsInstance(sample["patient_id"], str)
        self.assertEqual(sample["visit_id"], "1_0")
        self.assertEqual(len(sample["features"]), 2)
        self.assertEqual(len(sample["survival_interval"]), 3)

    def test_intervals_of_patient_who_died(self):
        intervals = [row["survival_interval"] for row in self.by_patient["1"]]
        expected = [
            [_years(0), _years(182), 0],
            [_years(182), _years(365), 0],
            [_years(365), _years(730), 0],
            [_years(730), _years(913), 1],
        ]
        np.testing.assert_allclose(np.array(intervals), np.array(expected))
        self.assertAlmostEqual(intervals[-1][1], 2.4997, places=4)

    def test_intervals_are_contiguous_and_ordered(self):
        for rows in self.by_patient.values():
            intervals = [row["survival_interval"] for row in rows]
            self.assertEqual(intervals[0][0], 0.0)
            for interval in intervals:
                self.assertLess(interval[0], interval[1])
            for previous, current in itertools.pairwise(intervals):
                self.assertEqual(previous[1], current[0])

    def test_event_only_on_last_interval(self):
        for rows in self.by_patient.values():
            events = [row["survival_interval"][2] for row in rows]
            self.assertEqual(sum(events[:-1]), 0)
        self.assertEqual(self.by_patient["1"][-1]["survival_interval"][2], 1)
        self.assertEqual(self.by_patient["2"][-1]["survival_interval"][2], 0)

    def test_transplant_is_censored_by_default(self):
        last = self.by_patient["3"][-1]["survival_interval"]
        self.assertEqual(last[2], 0)
        self.assertAlmostEqual(last[1], _years(548))

    def test_transplant_counts_with_composite_event(self):
        samples = counting_process_samples_from_frame(
            _fixture(), ["bili"], event="death_or_transplant"
        )
        patient3 = [s for s in samples if s["patient_id"] == "3"]
        self.assertEqual(patient3[-1]["survival_interval"][2], 1)

    def test_single_visit_patient_is_kept(self):
        (row,) = self.by_patient["4"]
        self.assertEqual(row["survival_interval"], [0.0, _years(292), 0])
        self.assertAlmostEqual(row["survival_interval"][1], 0.7995, places=4)

    def test_interval_uses_labs_from_its_starting_visit(self):
        bili = [row["features"][0] for row in self.by_patient["1"]]
        self.assertEqual(bili, [1.0, 1.5, 2.0, 3.0])
        landmarks = [row["landmark_time"] for row in self.by_patient["1"]]
        starts = [row["survival_interval"][0] for row in self.by_patient["1"]]
        self.assertEqual(landmarks, starts)

    def test_missing_lab_is_forward_filled(self):
        ast = [row["features"][1] for row in self.by_patient["5"]]
        self.assertEqual(ast, [90.0, 90.0, 95.0])

    def test_later_visits_do_not_change_earlier_samples(self):
        changed = _fixture()
        changed.loc[(changed["id"] == 1) & (changed["day"] == 730), "bili"] = 99.0
        samples = counting_process_samples_from_frame(changed, ["bili", "ast"])
        patient1 = [s for s in samples if s["patient_id"] == "1"]
        self.assertEqual(
            [s["features"] for s in patient1[:3]],
            [s["features"] for s in self.by_patient["1"][:3]],
        )


class TestLandmarkSamples(unittest.TestCase):
    """Builds [duration, event] samples counted from each visit."""

    def test_durations_from_each_visit(self):
        rows = build_landmark_samples(
            "1", [0, 182, 365, 730], [[1.0], [NAN], [2.0], [3.0]], futime=913, status=2
        )
        np.testing.assert_allclose(
            [row["survival"][0] for row in rows],
            np.array([913, 731, 548, 183]) / DAYS_PER_YEAR,
        )
        self.assertEqual([row["survival"][1] for row in rows], [1, 1, 1, 1])
        self.assertEqual([row["features"][0] for row in rows], [1.0, 1.0, 2.0, 3.0])
        self.assertEqual(rows[1]["landmark_time"], 182 / DAYS_PER_YEAR)

    def test_censored_and_validated_like_counting_process(self):
        rows = build_landmark_samples("2", [0], [[1.0]], futime=100, status=1)
        self.assertEqual(rows[0]["survival"][1], 0)
        with self.assertRaises(ValueError):
            build_landmark_samples("2", [0, 100], [[1.0], [2.0]], futime=100, status=0)


class TestBinaryHorizonSamples(unittest.TestCase):
    """Labels an event within the horizon and drops unknown labels."""

    def test_labels_and_drop_rule(self):
        rows = build_binary_horizon_samples(
            "1", [0, 182, 365, 730], [[1.0]] * 4, futime=913, status=2, horizon=1.0
        )
        # Years left: 2.50, 2.00, 1.50, 0.50 -> only the last is within 1 year.
        self.assertEqual([r["label"] for r in rows], [0, 0, 0, 1])
        censored = build_binary_horizon_samples(
            "2", [0, 365], [[1.0]] * 2, futime=500, status=0, horizon=1.0
        )
        self.assertEqual([r["visit_id"] for r in censored], ["2_0"])

    def test_rejects_non_positive_horizon(self):
        with self.assertRaises(ValueError):
            build_binary_horizon_samples("1", [0], [[1.0]], 100, 2, horizon=0)

    def test_ffill_switch(self):
        rows = build_counting_process_samples(
            "1", [0, 100], [[1.0], [NAN]], futime=300, status=0, ffill=False
        )
        self.assertTrue(math.isnan(rows[1]["features"][0]))


class TestBuildCountingProcessValidation(unittest.TestCase):
    """Rejects visit data that cannot form valid intervals."""

    def _build(self, days, futime=1000.0, status=0, n_features=1):
        features = [[1.0] * n_features for _ in days]
        return build_counting_process_samples("p", days, features, futime, status)

    def test_rejects_empty_visits(self):
        with self.assertRaises(ValueError):
            self._build([])

    def test_rejects_unsorted_or_duplicate_days(self):
        with self.assertRaises(ValueError):
            self._build([0, 300, 200])
        with self.assertRaises(ValueError):
            self._build([0, 300, 300])

    def test_rejects_non_finite_day(self):
        with self.assertRaises(ValueError):
            self._build([0, NAN, 200])

    def test_rejects_non_integer_status(self):
        frame = _fixture()
        frame["status"] = frame["status"].astype(float)
        frame.loc[frame["id"] == 1, "status"] = 2.6
        with self.assertRaises(ValueError):
            counting_process_samples_from_frame(frame, ["bili"])

    def test_rejects_negative_day(self):
        with self.assertRaises(ValueError):
            self._build([-1, 300])

    def test_rejects_visit_at_or_after_futime(self):
        with self.assertRaises(ValueError):
            self._build([0, 1000], futime=1000.0)

    def test_rejects_feature_row_mismatch(self):
        with self.assertRaises(ValueError):
            build_counting_process_samples("p", [0, 100], [[1.0]], 500.0, 0)

    def test_rejects_missing_column(self):
        with self.assertRaises(ValueError):
            counting_process_samples_from_frame(_fixture(), ["bili", "albumin"])

    def test_rejects_inconsistent_outcome_within_patient(self):
        frame = _fixture()
        frame.loc[1, "futime"] = 999
        with self.assertRaises(ValueError):
            counting_process_samples_from_frame(frame, ["bili"])


class TestHelpers(unittest.TestCase):
    """Event mapping and forward-fill helpers."""

    def test_event_indicator(self):
        self.assertEqual(event_indicator(2), 1)
        self.assertEqual(event_indicator(1), 0)
        self.assertEqual(event_indicator(0), 0)
        self.assertEqual(event_indicator(1, "death_or_transplant"), 1)
        with self.assertRaises(ValueError):
            event_indicator(3)
        with self.assertRaises(ValueError):
            event_indicator(2, "graft_failure")

    def test_forward_fill_keeps_unobserved_as_nan(self):
        filled = forward_fill([[NAN, 1.0], [2.0, NAN], [NAN, NAN]])
        self.assertTrue(math.isnan(filled[0, 0]))
        np.testing.assert_array_equal(filled[1:], [[2.0, 1.0], [2.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
