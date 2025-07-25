import unittest
from datetime import datetime

import polars as pl
import polars.testing as pt

from pyhealth.data import Event, Patient


class TestEvent(unittest.TestCase):
    def setUp(self):
        self.event = Event(
            event_type="DIAGNOSES_ICD",
            timestamp=datetime(2012, 1, 1, 0, 0),
            attr_dict={"code": "428.0", "vocabulary": "ICD9CM"},
        )

    def test_type(self):
        self.assertIsInstance(self.event.event_type, str)
        self.assertIsInstance(self.event.timestamp, datetime)

    def test_attr(self):
        self.assertEqual(self.event.event_type, "DIAGNOSES_ICD")
        self.assertEqual(self.event.timestamp, datetime(2012, 1, 1, 0, 0))
        attr_dict = self.event.attr_dict
        self.assertEqual(attr_dict["code"], "428.0")
        self.assertEqual(attr_dict["vocabulary"], "ICD9CM")


class TestPatient(unittest.TestCase):
    def setUp(self):
        self.event1 = Event(
            event_type="diagnosis",
            timestamp=datetime(2023, 5, 17),
            attr_dict={"a": 0, "b": 0},
        )
        self.event2 = Event(
            event_type="diagnosis",
            timestamp=datetime(2023, 5, 18),
            attr_dict={"a": 1, "b": 1},
        )
        self.event3 = Event(event_type="prescription", timestamp=datetime(2023, 5, 19))
        self.event4 = Event(event_type="lab_test", timestamp=None)
        self.patient = Patient.from_events(
            patient_id="12345",
            events=[self.event1, self.event2, self.event3, self.event4],
        )

    def test_get_events(self):
        events = self.patient.get_events()
        expected = [self.event4, self.event1, self.event2, self.event3]
        self.assertListEqual(events, expected)

    def test_get_events_diagnosis(self):
        events = self.patient.get_events(event_type="diagnosis")
        expected = [self.event1, self.event2]
        self.assertListEqual(events, expected)

    def test_get_events_between_times(self):
        events = self.patient.get_events(
            start=datetime(2023, 5, 18), end=datetime(2023, 5, 19)
        )
        expected = [self.event2, self.event3]
        self.assertListEqual(events, expected)

    def test_get_events_one_filter(self):
        events = self.patient.get_events("diagnosis", filters=[("a", ">=", 0)])
        expected = [self.event1, self.event2]
        self.assertListEqual(events, expected)

    def test_get_events_one_filter_with_time(self):
        events = self.patient.get_events(
            "diagnosis", start=datetime(2023, 5, 18), filters=[("a", ">=", 0)]
        )
        expected = [self.event2]
        self.assertListEqual(events, expected)

    def test_get_events_multi_filters(self):
        events = self.patient.get_events(
            "diagnosis", filters=[("a", ">=", 0), ("b", "==", 0)]
        )
        expected = [self.event1]
        self.assertListEqual(events, expected)

    def test_get_events_as_df(self):
        events = self.patient.get_events(return_df=True)
        expected = pl.DataFrame(
            e.to_dict() for e in [self.event4, self.event1, self.event2, self.event3]
        )
        pt.assert_frame_equal(events, expected, check_column_order=False)


if __name__ == "__main__":
    unittest.main()
