"""MortalityPredictionOMOP with ``min_followup``: the last visit is a sample.

With ``min_followup`` the horizon is ``visit_end + min_followup``, so a visit
needs no next visit to get a label. A death after the patient's last visit
must therefore reach the samples, and a last visit without enough observed
follow-up must still be dropped.
"""

import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from pyhealth.tasks import MortalityPredictionOMOP


class _Patient:
    """Minimal stand-in for ``Patient.get_events``."""

    def __init__(self, events, patient_id="p1"):
        self.patient_id = patient_id
        self.events = events

    def get_events(self, event_type=None, start=None, end=None, filters=None):
        events = list(self.events.get(event_type, []))
        for attr, _, value in filters or []:
            events = [e for e in events if getattr(e, attr) == value]
        return events


def _event(timestamp=None, **attributes):
    return SimpleNamespace(timestamp=timestamp, **attributes)


def _patient(visits, period, death=None):
    """A patient whose visits each have one condition, procedure and drug.

    Args:
        visits: ``(visit_id, start, end_string)`` tuples in time order.
        period: ``(start_date, end_date)`` of the one observation period.
        death: Death datetime, or None.
    """
    events = {
        "visit_occurrence": [],
        "condition_occurrence": [],
        "procedure_occurrence": [],
        "drug_exposure": [],
        "death": [] if death is None else [_event(death)],
        "observation_period": [
            _event(
                observation_period_start_date=period[0],
                observation_period_end_date=period[1],
            )
        ],
    }
    for visit_id, start, end in visits:
        events["visit_occurrence"].append(
            _event(start, visit_occurrence_id=visit_id, visit_end_datetime=end)
        )
        events["condition_occurrence"].append(
            _event(visit_occurrence_id=visit_id, condition_concept_id=201826)
        )
        events["procedure_occurrence"].append(
            _event(visit_occurrence_id=visit_id, procedure_concept_id=4163872)
        )
        events["drug_exposure"].append(
            _event(visit_occurrence_id=visit_id, drug_concept_id=1503297)
        )
    return _Patient(events)


FIRST = ("v1", datetime(2150, 1, 1, 8, 0, 0), "2150-01-03 12:00:00")
LAST = ("v2", datetime(2150, 6, 1, 8, 0, 0), "2150-06-05 12:00:00")


class TestMortalityMinFollowupLastVisit(unittest.TestCase):
    def setUp(self):
        self.task = MortalityPredictionOMOP(min_followup=timedelta(days=30))

    def test_death_after_last_visit_is_a_positive_sample(self):
        patient = _patient(
            [FIRST, LAST],
            ("2150-01-01", "2150-12-31"),
            death=datetime(2150, 6, 15, 0, 0, 0),
        )
        labels = {s["visit_id"]: s["mortality"] for s in self.task(patient)}
        self.assertEqual(labels, {"v1": 0, "v2": 1})

    def test_single_visit_patient_gets_a_sample(self):
        patient = _patient(
            [LAST],
            ("2150-06-01", "2150-12-31"),
            death=datetime(2150, 6, 15, 0, 0, 0),
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["mortality"], 1)

    def test_last_visit_with_full_followup_is_negative(self):
        patient = _patient([FIRST, LAST], ("2150-01-01", "2150-12-31"))
        labels = {s["visit_id"]: s["mortality"] for s in self.task(patient)}
        self.assertEqual(labels, {"v1": 0, "v2": 0})

    def test_last_visit_without_followup_is_dropped(self):
        # Observation ends on the last discharge day: no evidence either way.
        patient = _patient([FIRST, LAST], ("2150-01-01", "2150-06-05"))
        labels = {s["visit_id"]: s["mortality"] for s in self.task(patient)}
        self.assertEqual(labels, {"v1": 0})

    def test_zero_min_followup_keeps_the_last_visit(self):
        task = MortalityPredictionOMOP(min_followup=timedelta(0))
        patient = _patient([FIRST, LAST], ("2150-01-01", "2150-06-05"))
        self.assertEqual([s["visit_id"] for s in task(patient)], ["v1", "v2"])

    def test_next_visit_horizon_still_skips_the_last_visit(self):
        patient = _patient(
            [FIRST, LAST],
            ("2150-01-01", "2150-12-31"),
            death=datetime(2150, 6, 15, 0, 0, 0),
        )
        for task in (
            MortalityPredictionOMOP(),
            MortalityPredictionOMOP(check_observation_period=True),
        ):
            self.assertEqual([s["visit_id"] for s in task(patient)], ["v1"])


if __name__ == "__main__":
    unittest.main()
