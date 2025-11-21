import unittest
from datetime import datetime

import dask.dataframe as dd
import pandas as pd

from pyhealth.data import Patient


class TestPatientGetEvents(unittest.TestCase):
    def setUp(self):
        timestamps = [
            datetime(2021, 1, 1),
            datetime(2021, 1, 5),
            datetime(2021, 2, 1),
        ]
        pdf = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p1"],
                "event_type": ["labs", "labs", "visit"],
                "timestamp": timestamps,
                "labs/result": [1.0, 2.0, None],
                "labs/unit": ["mg/dL", "mg/dL", None],
                "visit/location": [None, None, "icu"],
            }
        )
        self.ddf = dd.from_pandas(pdf, npartitions=1)
        self.patient = Patient(patient_id="p1", data_source=self.ddf)

    def test_returns_event_objects_by_default(self):
        events = self.patient.get_events()
        self.assertEqual(len(events), 3)
        self.assertEqual(
            sorted([e.event_type for e in events]), ["labs", "labs", "visit"]
        )
        self.assertEqual(events[0].attr_dict["result"], 1.0)

    def test_return_df_flag(self):
        labs_df = self.patient.get_events(event_type="labs", return_df=True)
        labs_pdf = labs_df.compute()
        self.assertEqual(len(labs_pdf), 2)
        self.assertTrue((labs_pdf["event_type"] == "labs").all())

    def test_time_range_filter(self):
        start = datetime(2021, 1, 2)
        end = datetime(2021, 1, 31)
        events = self.patient.get_events(start=start, end=end)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].timestamp, datetime(2021, 1, 5))

    def test_event_type_and_attribute_filters(self):
        filters = [("result", ">=", 2)]
        events = self.patient.get_events(event_type="labs", filters=filters)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attr_dict["result"], 2.0)

    def test_filters_require_event_type(self):
        with self.assertRaises(AssertionError):
            self.patient.get_events(filters=[("result", "==", 1)])

    def test_missing_column_in_filters_raises(self):
        with self.assertRaises(KeyError):
            self.patient.get_events(
                event_type="labs", filters=[("does_not_exist", "==", 1)]
            )


if __name__ == "__main__":
    unittest.main()
