"""Tests of the event-level alarm metrics, on tiny hand-built stays."""

import math
import unittest

import numpy as np

from pyhealth.metrics.event_alarm import (
    alarm_curve,
    alarm_event_auprc,
    alarm_policy,
    event_alarm_metrics,
)

HORIZON = 3


def _event_stay(scores):
    """A stay of 6 hourly steps whose event falls 6 h after the first step.

    ``time_to_event`` is 6, 5, ..., 1, so the last 3 steps are in the window.
    """
    return {
        "scores": scores,
        "times": list(range(6)),
        "time_to_event": [6, 5, 4, 3, 2, 1],
        "event": [1] * 6,
    }


def _quiet_stay(scores):
    """A stay of 6 hourly steps without an event (censored)."""
    return {
        "scores": scores,
        "times": list(range(6)),
        "time_to_event": [5, 4, 3, 2, 1, 0],
        "event": [0] * 6,
    }


def _stack(*stays):
    return {k: [s[k] for s in stays] for k in stays[0]}


def _metrics(stays, **kwargs):
    kwargs.setdefault("silence", 1.0)
    return event_alarm_metrics(horizon=HORIZON, **_stack(*stays), **kwargs)


class TestAlarmPolicy(unittest.TestCase):
    def test_threshold_without_silence(self):
        alarms = alarm_policy([0.2, 0.6, 0.5, 0.4], [0, 1, 2, 3], 0.5)
        self.assertEqual(alarms.tolist(), [False, True, True, False])

    def test_silence_suppresses_following_alarms(self):
        alarms = alarm_policy([0.9] * 5, [0, 1, 2, 3, 4], 0.5, silence=2)
        self.assertEqual(alarms.tolist(), [True, False, True, False, True])

    def test_silence_counts_time_not_steps(self):
        alarms = alarm_policy([0.9] * 3, [0.0, 0.5, 3.0], 0.5, silence=1)
        self.assertEqual(alarms.tolist(), [True, False, True])

    def test_silenced_step_does_not_restart_the_silence(self):
        # the alarm at 0 silences 1; step 2 is 2 h after the last alarm
        alarms = alarm_policy([0.9] * 3, [0, 1, 2], 0.5, silence=2)
        self.assertEqual(alarms.tolist(), [True, False, True])

    def test_no_scores_above_threshold(self):
        self.assertFalse(alarm_policy([0.1, 0.2], [0, 1], 0.5).any())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            alarm_policy([0.1, 0.2], [0], 0.5)
        with self.assertRaises(ValueError):
            alarm_policy([0.1], [0], 0.5, silence=-1)


class TestEventMetrics(unittest.TestCase):
    def test_perfect_early_alarm(self):
        # alarm only in the window, first at 3 h before the event
        stays = [
            _event_stay([0, 0, 0, 0.9, 0, 0]),
            _quiet_stay([0] * 6),
        ]
        m = _metrics(stays, thresholds=[0.5])
        self.assertEqual(m["event_recall@80"], 1.0)
        self.assertEqual(m["first_alarm_distance@80"], 3.0)
        self.assertEqual(m["n_events"], 1.0)

    def test_alarm_before_the_window_is_false(self):
        # the only alarm is 5 h before the event: outside the 3 h horizon
        stays = [_event_stay([0, 0.9, 0, 0, 0, 0])]
        curve = alarm_curve(
            **_stack(*stays), horizon=HORIZON, silence=1, thresholds=[0.5]
        )
        self.assertEqual(curve["alarm_precision"].tolist(), [0.0])
        self.assertEqual(curve["event_recall"].tolist(), [0.0])
        self.assertTrue(math.isnan(curve["first_alarm_distance"][0]))

    def test_alarms_on_stays_without_event_lower_precision(self):
        stays = [
            _event_stay([0, 0, 0, 0.9, 0, 0]),
            _quiet_stay([0.9, 0, 0, 0, 0, 0]),
            _quiet_stay([0, 0.9, 0, 0, 0, 0]),
            _quiet_stay([0, 0, 0.9, 0, 0, 0]),
        ]
        curve = alarm_curve(
            **_stack(*stays), horizon=HORIZON, silence=1, thresholds=[0.5]
        )
        self.assertEqual(curve["alarm_precision"][0], 0.25)  # 1 of 4 alarms
        self.assertEqual(curve["event_recall"][0], 1.0)

    def test_several_true_alarms_for_one_event_count_once_for_recall(self):
        # 3 alarms in the window (3, 2 and 1 h before): precision 1, recall 1
        stays = [_event_stay([0, 0, 0, 0.9, 0.9, 0.9])]
        curve = alarm_curve(
            **_stack(*stays), horizon=HORIZON, silence=1, thresholds=[0.5]
        )
        self.assertEqual(curve["n_alarms"][0], 3)
        self.assertEqual(curve["alarm_precision"][0], 1.0)
        self.assertEqual(curve["event_recall"][0], 1.0)
        self.assertEqual(curve["first_alarm_distance"][0], 3.0)

    def test_silencing_changes_the_alarms_and_the_distance(self):
        # an early alarm (5 h before) silences the one at 3 h before
        stays = [_event_stay([0, 0.9, 0.9, 0.9, 0.9, 0])]
        short = alarm_curve(
            **_stack(*stays), horizon=HORIZON, silence=1, thresholds=[0.5]
        )
        long = alarm_curve(
            **_stack(*stays), horizon=HORIZON, silence=3, thresholds=[0.5]
        )
        self.assertEqual(short["first_alarm_distance"][0], 3.0)
        self.assertEqual(short["n_alarms"][0], 4)
        # alarms at 1 and 4: the second is at 2 h before the event
        self.assertEqual(long["n_alarms"][0], 2)
        self.assertEqual(long["event_recall"][0], 1.0)
        self.assertEqual(long["first_alarm_distance"][0], 2.0)

    def test_recall_counts_events_not_alarms(self):
        stays = [
            _event_stay([0, 0, 0, 0.9, 0, 0]),  # caught
            _event_stay([0, 0, 0, 0, 0, 0]),  # missed
        ]
        curve = alarm_curve(
            **_stack(*stays), horizon=HORIZON, silence=1, thresholds=[0.5]
        )
        self.assertEqual(curve["event_recall"][0], 0.5)

    def test_recall_at_precision_picks_the_best_qualifying_threshold(self):
        # event stay scores high in the window; quiet stay scores lower
        stays = [
            _event_stay([0.1, 0.1, 0.1, 0.9, 0.8, 0.7]),
            _quiet_stay([0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
        ]
        m = _metrics(stays, thresholds=[0.95, 0.65, 0.5], precisions=(0.8, 0.2))
        # tau 0.65: alarms at steps 3, 4, 5 (all true): precision 1, recall 1
        self.assertEqual(m["event_recall@80"], 1.0)
        self.assertEqual(m["first_alarm_distance@80"], 3.0)
        # tau 0.5 adds six false alarms: precision 3/9 = 0.33, still >= 0.2
        self.assertEqual(m["event_recall@20"], 1.0)

    def test_precision_not_reached_gives_zero_recall(self):
        stays = [
            _event_stay([0, 0, 0, 0, 0, 0.9]),
            _quiet_stay([0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
        ]
        m = _metrics(stays, thresholds=[0.5], silence=1, precisions=(0.8,))
        self.assertEqual(m["event_recall@80"], 0.0)
        self.assertTrue(math.isnan(m["first_alarm_distance@80"]))

    def test_perfect_ranking_gives_auprc_one(self):
        stays = [
            _event_stay([0, 0, 0, 0.9, 0.9, 0.9]),
            _quiet_stay([0.1] * 6),
        ]
        m = _metrics(stays, n_thresholds=20)
        self.assertAlmostEqual(m["alarm_event_auprc"], 1.0)

    def test_auprc_of_a_flat_scorer_is_the_base_precision(self):
        # all scores equal: any threshold alarms (after silencing) everywhere
        stays = [_event_stay([0.5] * 6), _quiet_stay([0.5] * 6)]
        curve = alarm_curve(**_stack(*stays), horizon=HORIZON, silence=1)
        # 12 alarms, 3 in the window, one event caught
        area = alarm_event_auprc(curve["alarm_precision"], curve["event_recall"])
        self.assertAlmostEqual(area, 3 / 12)

    def test_no_events(self):
        m = _metrics([_quiet_stay([0.9] * 6)], thresholds=[0.5])
        self.assertEqual(m["n_events"], 0.0)
        self.assertEqual(m["event_recall@60"], 0.0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            event_alarm_metrics([[0.1]], [[0]], [[1]], [[1]], horizon=0)
        with self.assertRaises(ValueError):
            event_alarm_metrics([[0.1, 0.2]], [[0]], [[1]], [[1]], horizon=1)
        with self.assertRaises(ValueError):
            event_alarm_metrics([[0.1]], [[0]], [[1]], [], horizon=1)

    def test_window_matches_the_binary_label_of_the_task(self):
        # in the window iff event and 0 < time_to_event <= horizon
        stay = _event_stay([0.9] * 6)
        curve = alarm_curve(
            **_stack(stay), horizon=HORIZON, silence=0, thresholds=[0.5]
        )
        self.assertEqual(curve["n_alarms"][0], 6)
        self.assertAlmostEqual(curve["alarm_precision"][0], 3 / 6)

    def test_inputs_may_be_arrays(self):
        stay = _event_stay(np.array([0, 0, 0, 0.9, 0, 0]))
        stay = {k: np.asarray(v) for k, v in stay.items()}
        m = _metrics([stay], thresholds=[0.5])
        self.assertEqual(m["event_recall@60"], 1.0)


if __name__ == "__main__":
    unittest.main()
