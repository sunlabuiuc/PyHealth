"""Events that share a timestamp must come out in source-file order.

The cached event table is built by a parallel sort on ``patient_id`` only, so
the order of a patient's rows differed between cache builds, and ``Patient``
sorted by timestamp alone with an unstable sort. Events with equal timestamps
(all diagnoses of a MIMIC admission, same-day prescriptions, admissions that
start at the same time) therefore came out in a different order on different
builds, which changed task outputs (for example which admission counts as the
last one) and lost the ``seq_num`` priority order of diagnoses.
"""

import random
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import BaseDataset
from pyhealth.datasets.base_dataset import EVENT_FORMAT, SOURCE_ROW

CONFIG = """version: "1.0"
tables:
  admissions:
    file_path: "admissions.csv"
    patient_id: "patient_id"
    timestamp: "admittime"
    attributes:
      - "hadm_id"
  diagnoses:
    file_path: "diagnoses.csv"
    patient_id: "patient_id"
    timestamp: "time"
    attributes:
      - "hadm_id"
      - "code"
"""

# Codes in priority order, deliberately not sorted, and all at one timestamp.
CODES = ["428.0", "250.00", "401.9", "038.9", "486", "584.9", "276.1", "285.9"]


def _write_data(root: Path) -> None:
    (root / "config.yaml").write_text(CONFIG)
    (root / "admissions.csv").write_text(
        "patient_id,admittime,hadm_id\n"
        # Two admissions of p1 start at the same time.
        "p1,2020-01-01 12:00:00,h2\n"
        "p1,2020-01-01 12:00:00,h1\n"
        "p1,2020-03-01 12:00:00,h3\n"
    )
    rows = ["patient_id,time,hadm_id,code"]
    rows += [f"p1,2020-01-05 00:00:00,h1,{code}" for code in CODES]
    (root / "diagnoses.csv").write_text("\n".join(rows) + "\n")


def _events(n: int) -> pl.DataFrame:
    t = datetime(2020, 1, 1)  # noqa: DTZ001 - event timestamps are naive
    return pl.DataFrame(
        {
            "patient_id": ["p1"] * n,
            "event_type": ["diagnoses"] * n,
            "timestamp": [t] * n,
            SOURCE_ROW: list(range(n)),
            "diagnoses/code": [f"c{i:03d}" for i in range(n)],
        }
    )


class TestPatientTieOrder(unittest.TestCase):
    def test_tied_events_follow_source_row_whatever_the_input_order(self):
        events = _events(200)
        expected = events["diagnoses/code"].to_list()
        for seed in range(5):
            rows = list(range(events.height))
            random.Random(seed).shuffle(rows)
            patient = Patient("p1", events[rows])
            got = [e.code for e in patient.get_events(event_type="diagnoses")]
            self.assertEqual(got, expected)

    def test_caches_without_source_row_keep_their_row_order(self):
        events = _events(50).drop(SOURCE_ROW).reverse()
        patient = Patient("p1", events)
        got = [e.code for e in patient.get_events(event_type="diagnoses")]
        self.assertEqual(got, events["diagnoses/code"].to_list())

    def test_source_row_is_not_an_event_attribute(self):
        patient = Patient("p1", _events(3))
        event = patient.get_events(event_type="diagnoses")[0]
        self.assertNotIn(SOURCE_ROW, event.attr_dict)


class TestDatasetTieOrder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "data"
        self.root.mkdir()
        _write_data(self.root)
        self.cache = Path(tmp.name) / "cache"

    def _dataset(self) -> BaseDataset:
        return BaseDataset(
            root=str(self.root),
            tables=["admissions", "diagnoses"],
            dataset_name="TieOrderDataset",
            config_path=str(self.config_path),
            cache_dir=str(self.cache),
        )

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    def test_source_order_survives_the_cache_build(self):
        patient = self._dataset().get_patient("p1")
        admissions = [e.hadm_id for e in patient.get_events(event_type="admissions")]
        self.assertEqual(admissions, ["h2", "h1", "h3"])
        codes = [e.code for e in patient.get_events(event_type="diagnoses")]
        self.assertEqual(codes, CODES)

    def test_event_table_has_source_row(self):
        columns = self._dataset().global_event_df.collect_schema().names()
        self.assertIn(SOURCE_ROW, columns)

    def test_cache_key_includes_event_format(self):
        # Caches written before SOURCE_ROW existed must not be reused.
        current = self._dataset().cache_dir
        self.assertEqual(self._dataset().cache_dir, current)
        with mock.patch(
            "pyhealth.datasets.base_dataset.EVENT_FORMAT", EVENT_FORMAT - 1
        ):
            self.assertNotEqual(self._dataset().cache_dir, current)


if __name__ == "__main__":
    unittest.main()
