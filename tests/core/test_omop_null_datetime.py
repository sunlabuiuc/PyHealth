"""OMOPDataset with empty *_datetime columns.

In the OMOP CDM, event dates (e.g. visit_start_date) are required and the
matching datetimes (e.g. visit_start_datetime) are optional. These tests copy
the OMOP test resources, blank the datetime columns, and check that event
timestamps fall back to the date and that the OMOP tasks still run.
"""

import csv
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import polars as pl

from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import LengthOfStayPredictionOMOP, ReadmissionPredictionOMOP

SOURCE = Path(__file__).parent.parent.parent / "test-resources" / "omop"

# This visit keeps its datetimes, to check that present values are not replaced.
KEPT_VISIT_ID = "6788074353249374815"
KEPT_PERSON_ID = "-3210373572193940939"


def _blank_datetimes(path: Path) -> None:
    """Empties every *_datetime column in a CSV, except birth_datetime."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row.get("visit_occurrence_id") == KEPT_VISIT_ID and path.stem == (
            "visit_occurrence"
        ):
            continue
        for col in fields:
            if col.endswith("_datetime") and col != "birth_datetime":
                row[col] = ""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestOMOPNullDatetime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.cache_dir = tempfile.mkdtemp()
        for path in SOURCE.glob("*.csv"):
            shutil.copy(path, cls.root)
            _blank_datetimes(Path(cls.root) / path.name)

        cls.dataset = OMOPDataset(
            root=cls.root,
            tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
            cache_dir=cls.cache_dir,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_no_null_event_timestamps(self):
        nulls = (
            self.dataset.global_event_df.filter(pl.col("event_type") != "person")
            .filter(pl.col("timestamp").is_null())
            .collect()
        )
        self.assertEqual(nulls.height, 0)

    def test_timestamp_falls_back_to_date(self):
        patient = self.dataset.get_patient(KEPT_PERSON_ID)
        conditions = patient.get_events(event_type="condition_occurrence")
        self.assertGreater(len(conditions), 0)
        for event in conditions:
            date = datetime.strptime(event.condition_start_date, "%Y-%m-%d")
            self.assertEqual(event.timestamp, date)
            self.assertEqual(
                event.condition_start_datetime,
                f"{event.condition_start_date} 00:00:00",
            )

    def test_present_datetime_is_kept(self):
        patient = self.dataset.get_patient(KEPT_PERSON_ID)
        visit = next(
            v
            for v in patient.get_events(event_type="visit_occurrence")
            if v.visit_occurrence_id == KEPT_VISIT_ID
        )
        self.assertEqual(visit.timestamp, datetime(2146, 12, 3, 15, 0, 0))
        self.assertEqual(visit.visit_end_datetime, "2146-12-03 15:00:00")

    def test_readmission_task_runs(self):
        samples = self.dataset.set_task(ReadmissionPredictionOMOP())
        try:
            self.assertGreater(len(samples), 0)
        finally:
            samples.close()

    def test_length_of_stay_task_runs(self):
        samples = self.dataset.set_task(LengthOfStayPredictionOMOP())
        try:
            self.assertGreater(len(samples), 0)
        finally:
            samples.close()


if __name__ == "__main__":
    unittest.main()
