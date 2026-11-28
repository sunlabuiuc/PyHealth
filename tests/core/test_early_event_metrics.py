"""Synthetic tests. Run: python -m unittest -v test_early_event_metrics.py"""
import unittest
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "mortality_prediction"
sys.path.insert(0, str(EXAMPLE_DIR))

from early_event_metrics import (
    prepare_records, evaluate_threshold, evaluate_validation_selected_test,
    select_validation_threshold,
)


def row(stay, time, score, label=0, event=False, last_t=9):
    return {"stay": stay, "period_length": time, "last_t": last_t,
            "ever_event": int(event), "y_true": label, "score": score}


class EventEvaluationTests(unittest.TestCase):
    def test_counts_recall_precision_and_false_alarms(self):
        records = [row("a", 2, .9, 1, True), row("a", 4, .8, 1, True),
                   row("b", 1, .9), row("c", 3, .1, 0, True)]
        result = evaluate_threshold(prepare_records(records), .5)
        self.assertEqual(result["total_alarms"], 3)
        self.assertEqual(result["true_alarms"], 2)
        self.assertEqual(result["false_alarms"], 1)
        self.assertEqual(result["detected_events"], 1)
        self.assertAlmostEqual(result["event_recall"], .5)
        self.assertAlmostEqual(result["alarm_precision"], 2/3)
        self.assertAlmostEqual(result["false_alarms_per_100_stays"], 100/3)
        self.assertAlmostEqual(result["median_lead_steps"], 8)

    def test_silence_uses_times_not_row_positions(self):
        data = prepare_records([row("a", 1, .8), row("a", 3, .8)])
        self.assertEqual(evaluate_threshold(data, .5, silence_steps=1)["total_alarms"], 2)
        data = prepare_records([row("a", 1, .8), row("a", 2, .8)])
        self.assertEqual(evaluate_threshold(data, .5, silence_steps=1)["total_alarms"], 1)

    def test_no_alarms_is_not_perfect_precision(self):
        result = evaluate_threshold(prepare_records([row("a", 1, .1, event=True)]), .9)
        self.assertIsNone(result["alarm_precision"])
        self.assertEqual(result["event_recall"], 0.0)

    def test_no_events_has_undefined_recall(self):
        result = evaluate_threshold(prepare_records([row("a", 1, .9)]), .5)
        self.assertIsNone(result["event_recall"])

    def test_validation_threshold_is_frozen_on_test(self):
        validation = [row("a", 1, .9, 1, True), row("b", 1, .4)]
        test = [row("c", 1, .4, 1, True), row("d", 1, .9)]
        out = evaluate_validation_selected_test(validation, test, [.4, .9], .75)
        self.assertEqual(out["validation"]["threshold"], .9)
        self.assertEqual(out["test"]["threshold"], .9)
        self.assertEqual(out["test"]["event_recall"], 0)
        self.assertEqual(out["test"]["alarm_precision"], 0)

    def test_threshold_without_alarm_is_not_eligible(self):
        data = prepare_records([row("a", 1, .2, 1, True)])
        self.assertIsNone(select_validation_threshold(data, [1.0], .7))

    def test_reject_duplicate_times(self):
        with self.assertRaises(ValueError):
            prepare_records([row("a", 1, .2), row("a", 1, .3)])

    def test_reject_inconsistent_event_status(self):
        with self.assertRaises(ValueError):
            prepare_records([row("a", 1, .2, event=True), row("a", 2, .3, event=False)])


if __name__ == "__main__":
    unittest.main()
