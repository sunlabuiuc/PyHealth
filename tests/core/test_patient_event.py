from datetime import datetime, timedelta
import pandas as pd

from tests.base import BaseTestCase
from pyhealth.data.data import Event, Patient


class PatientEventTestCase(BaseTestCase):
    def setUp(self):
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        self.event_rows = [
            {
                "patient_id": "p1",
                "event_type": "med",
                "timestamp": base_time + timedelta(days=1),
                "med/dose": 10,
            },
            {
                "patient_id": "p1",
                "event_type": "diag",
                "timestamp": base_time + timedelta(days=2),
                "diag/code": "A",
            },
            {
                "patient_id": "p1",
                "event_type": "med",
                "timestamp": base_time + timedelta(days=4),
                "med/dose": 20,
            },
            {
                "patient_id": "p1",
                "event_type": "diag",
                "timestamp": base_time + timedelta(days=3),
                "diag/code": "B",
                "diag/severity": 2,
            },
            {
                "patient_id": "p1",
                "event_type": "lab",
                "timestamp": pd.NaT,
                "lab/value": 99,
            },
        ]
        unsorted_df = pd.DataFrame(self.event_rows)
        self.patient = Patient(patient_id="p1", data_source=unsorted_df)
        self.base_time = base_time

    def test_event_accessors_and_from_dict(self):
        ts = datetime(2024, 5, 1, 8, 0, 0)
        event = Event(event_type="diag", timestamp=ts, code="X", value=1)
        self.assertEqual(event["timestamp"], ts)
        self.assertEqual(event["event_type"], "diag")
        self.assertEqual(event["code"], "X")
        self.assertIn("value", event)
        self.assertEqual(event.value, 1)

        raw = {"event_type": "diag", "timestamp": ts, "diag/code": "Y", "diag/score": 5}
        reconstructed = Event.from_dict(raw)
        self.assertEqual(reconstructed.event_type, "diag")
        self.assertEqual(reconstructed.attr_dict, {"code": "Y", "score": 5})

    def test_patient_sorting_and_partitions(self):
        # timestamps should be sorted with NaT at the end
        timestamps = list(self.patient.data_source["timestamp"])
        sorted_without_nat = sorted([ts for ts in timestamps if pd.notna(ts)])
        self.assertEqual(timestamps[:-1], sorted_without_nat)
        self.assertTrue(pd.isna(timestamps[-1]))

        diag_partition = self.patient.event_type_partitions[("diag",)]
        self.assertListEqual(list(diag_partition["diag/code"]), ["A", "B"])

        med_partition = self.patient.event_type_partitions[("med",)]
        self.assertListEqual(list(med_partition["med/dose"]), [10, 20])

    def test_get_events_by_type_and_time(self):
        diag_df = self.patient.get_events(event_type="diag", return_df=True)
        self.assertEqual(len(diag_df), 2)
        self.assertListEqual(list(diag_df["diag/code"]), ["A", "B"])

        # Time filtering should include bounds and drop NaT
        start = self.base_time + timedelta(days=2)
        end = self.base_time + timedelta(days=4)
        ranged = self.patient.get_events(start=start, end=end, return_df=True)
        self.assertListEqual(list(ranged["event_type"]), ["diag", "diag", "med"])
        self.assertTrue(ranged["timestamp"].between(start, end, inclusive="both").all())
        self.assertFalse(ranged["timestamp"].isna().any())

    def test_attribute_filters(self):
        filtered_events = self.patient.get_events(
            event_type="diag", filters=[("code", "==", "B")]
        )
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0].attr_dict, {"code": "B", "severity": 2})

        filtered_df = self.patient.get_events(
            event_type="diag", filters=[("code", "!=", "B")], return_df=True
        )
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df.iloc[0]["diag/code"], "A")

    def test_filters_require_event_type(self):
        with self.assertRaises(AssertionError):
            self.patient.get_events(filters=[("code", "==", "A")])
