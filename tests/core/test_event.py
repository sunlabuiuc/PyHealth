"""Tests for pyhealth.data.Event: missing timestamps and copy/pickle support."""

import copy
import pickle
import unittest
from datetime import datetime

import polars as pl

from pyhealth.data import Event, Patient
from pyhealth.datasets import create_sample_dataset


def _patient() -> Patient:
    # A timeless demographics row (like MIMIC's `patients` table) and a timed row.
    df = pl.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "event_type": ["patients", "admissions"],
            "timestamp": [None, datetime(2164, 10, 23, 21, 9)],
            "patients/gender": ["F", None],
            "admissions/hadm_id": [None, "142345"],
        },
        schema_overrides={"timestamp": pl.Datetime("ms")},
    )
    return Patient(patient_id="p1", data_source=df)


class TestEventTimestamp(unittest.TestCase):
    def test_missing_timestamp_is_none(self):
        self.assertIsNone(Event("note").timestamp)
        self.assertIsNone(Event("note", timestamp=None).timestamp)

    def test_explicit_timestamp_is_kept(self):
        ts = datetime(2020, 1, 2, 3, 4)
        self.assertEqual(Event("note", timestamp=ts).timestamp, ts)

    def test_from_dict_keeps_null_timestamp(self):
        event = Event.from_dict(
            {"event_type": "patients", "timestamp": None, "patients/gender": "F"}
        )
        self.assertIsNone(event.timestamp)
        self.assertEqual(event.gender, "F")

    def test_get_events_matches_dataframe_and_is_stable(self):
        patient = _patient()
        df_ts = patient.get_events("patients", return_df=True)["timestamp"].to_list()
        first = patient.get_events("patients")[0].timestamp
        second = patient.get_events("patients")[0].timestamp
        self.assertEqual(df_ts, [None])
        self.assertIsNone(first)
        self.assertIsNone(second)
        admission = patient.get_events("admissions")[0]
        self.assertEqual(admission.timestamp, datetime(2164, 10, 23, 21, 9))


class TestEventCopyAndPickle(unittest.TestCase):
    def setUp(self):
        self.event = Event(
            "admissions", timestamp=datetime(2164, 10, 23), hadm_id="142345"
        )

    def _assert_same(self, other: Event):
        self.assertIsNot(other, self.event)
        self.assertEqual(other, self.event)
        self.assertEqual(other.hadm_id, "142345")
        self.assertEqual(other.timestamp, datetime(2164, 10, 23))

    def test_pickle_round_trip(self):
        self._assert_same(pickle.loads(pickle.dumps(self.event)))

    def test_copy_and_deepcopy(self):
        self._assert_same(copy.copy(self.event))
        clone = copy.deepcopy(self.event)
        self._assert_same(clone)
        self.assertIsNot(clone.attr_dict, self.event.attr_dict)

    def test_attribute_access(self):
        self.assertEqual(self.event["hadm_id"], "142345")
        self.assertIn("hadm_id", self.event)
        with self.assertRaises(AttributeError):
            _ = self.event.not_a_field
        self.assertFalse(hasattr(self.event, "not_a_field"))

    def test_event_inside_a_sample(self):
        samples = [
            {"patient_id": f"p{i}", "codes": ["a"], "admission": self.event, "label": i % 2}
            for i in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence", "admission": "raw"},
            output_schema={"label": "binary"},
            dataset_name="test_event",
        )
        self._assert_same(dataset[0]["admission"])


if __name__ == "__main__":
    unittest.main()
