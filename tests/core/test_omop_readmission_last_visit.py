"""ReadmissionPredictionOMOP with ``check_observation_period``: the last visit.

With the check on, a non-readmitted visit is kept when its observation period
reaches ``discharge + window``. That holds for the patient's last visit too,
so it must become a negative sample, and a last visit observed for less than
the window must still be dropped. Without the check the last visit stays out.
"""

import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from pyhealth.tasks import ReadmissionPredictionOMOP


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


def _patient(visits, period):
    """An adult whose visits each have one condition, procedure and drug.

    Args:
        visits: ``(visit_id, start, end_string)`` tuples in time order.
        period: ``(start_date, end_date)`` of the one observation period.
    """
    events = {
        "person": [
            _event(year_of_birth="2100", month_of_birth="1", day_of_birth="1")
        ],
        "visit_occurrence": [],
        "condition_occurrence": [],
        "procedure_occurrence": [],
        "drug_exposure": [],
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
READMIT = ("v2", datetime(2150, 1, 10, 8, 0, 0), "2150-01-12 12:00:00")
LAST = ("v3", datetime(2150, 6, 1, 8, 0, 0), "2150-06-05 12:00:00")


class TestReadmissionObservationPeriodLastVisit(unittest.TestCase):
    def setUp(self):
        self.task = ReadmissionPredictionOMOP(
            window=timedelta(days=15), check_observation_period=True
        )

    def test_observed_last_visit_is_a_negative_sample(self):
        patient = _patient([FIRST, READMIT, LAST], ("2150-01-01", "2150-12-31"))
        labels = {s["visit_id"]: s["readmission"] for s in self.task(patient)}
        self.assertEqual(labels, {"v1": 1, "v2": 0, "v3": 0})

    def test_single_visit_patient_gets_a_sample(self):
        patient = _patient([LAST], ("2150-06-01", "2150-12-31"))
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["readmission"], 0)

    def test_last_visit_without_full_window_is_dropped(self):
        # Observation ends 10 days after the last discharge, before the window.
        patient = _patient([FIRST, READMIT, LAST], ("2150-01-01", "2150-06-15"))
        labels = {s["visit_id"]: s["readmission"] for s in self.task(patient)}
        self.assertEqual(labels, {"v1": 1, "v2": 0})

    def test_without_the_check_the_last_visit_is_skipped(self):
        task = ReadmissionPredictionOMOP(window=timedelta(days=15))
        patient = _patient([FIRST, READMIT, LAST], ("2150-01-01", "2150-12-31"))
        self.assertEqual([s["visit_id"] for s in task(patient)], ["v1", "v2"])
        self.assertEqual(task(_patient([LAST], ("2150-06-01", "2150-12-31"))), [])


if __name__ == "__main__":
    unittest.main()
