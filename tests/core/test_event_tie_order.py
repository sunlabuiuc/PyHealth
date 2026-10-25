"""Events that share a timestamp must come out in source-file order.

The cached event table is built by a parallel sort on ``patient_id`` only, so
the order of a patient's rows differed between cache builds, and ``Patient``
sorted by timestamp alone with an unstable sort. Events with equal timestamps
(all diagnoses of a MIMIC admission, same-day prescriptions, admissions that
start at the same time) therefore came out in a different order on different
builds, which changed task outputs (for example which admission counts as the
last one) and lost the ``seq_num`` priority order of diagnoses.
"""

import json
import random
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC4FHIR, BaseDataset
from pyhealth.datasets.base_dataset import (
    EVENT_FORMAT,
    SOURCE_ROW,
    _event_parquet_options,
    _source_row_as_int32,
)

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

# Diagnoses joined to a code lookup with an outer join.
JOIN_CONFIG = """version: "1.0"
tables:
  diagnoses:
    file_path: "diagnoses.csv"
    patient_id: "patient_id"
    timestamp: "time"
    attributes:
      - "code"
      - "label"
    join:
      - file_path: "codes.csv"
        "on": "code"
        how: "outer"
        columns:
          - "label"
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

    def test_tied_events_of_different_tables_are_grouped_by_table(self):
        t = datetime(2020, 1, 1)  # noqa: DTZ001 - event timestamps are naive
        events = pl.DataFrame(
            {
                "patient_id": ["p1"] * 4,
                "event_type": ["prescriptions", "diagnoses"] * 2,
                "timestamp": [t] * 4,
                SOURCE_ROW: [1, 1, 0, 0],
                "diagnoses/code": [None, "d1", None, "d0"],
                "prescriptions/code": ["p1", None, "p0", None],
            }
        )
        patient = Patient("p1", events)
        got = [(e.event_type, e.code) for e in patient.get_events()]
        self.assertEqual(
            got,
            [
                ("diagnoses", "d0"),
                ("diagnoses", "d1"),
                ("prescriptions", "p0"),
                ("prescriptions", "p1"),
            ],
        )

    def test_caches_without_source_row_keep_their_row_order(self):
        events = _events(50).drop(SOURCE_ROW).reverse()
        patient = Patient("p1", events)
        got = [e.code for e in patient.get_events(event_type="diagnoses")]
        self.assertEqual(got, events["diagnoses/code"].to_list())

    def test_source_row_is_not_an_event_attribute(self):
        patient = Patient("p1", _events(3))
        event = patient.get_events(event_type="diagnoses")[0]
        self.assertNotIn(SOURCE_ROW, event.attr_dict)
        self.assertNotIn(SOURCE_ROW, event)
        self.assertIn("timestamp", event)
        self.assertEqual(event["timestamp"], event.timestamp)
        self.assertEqual(event["event_type"], "diagnoses")

    def test_a_time_window_keeps_tied_events_in_source_order(self):
        tied = _events(20)
        t = tied["timestamp"][0]
        others = pl.DataFrame(
            {
                "patient_id": ["p1", "p1"],
                "event_type": ["diagnoses", "diagnoses"],
                "timestamp": [t - timedelta(days=1), t + timedelta(days=1)],
                SOURCE_ROW: [100, 101],
                "diagnoses/code": ["before", "after"],
            }
        )
        events = pl.concat([others, tied.reverse()])
        patient = Patient("p1", events)
        got = [e.code for e in patient.get_events(event_type="diagnoses", start=t, end=t)]
        self.assertEqual(got, tied["diagnoses/code"].to_list())


class _DropsSourceRow(BaseDataset):
    """A preprocess hook that rebuilds the frame without the source-row column."""

    def preprocess_diagnoses(self, df):
        return df.select("patient_id", "time", "hadm_id", "code")


class _ReordersRows(BaseDataset):
    """A preprocess hook that reorders rows but keeps the source-row column."""

    def preprocess_diagnoses(self, df):
        return df.sort("code")


class TestDatasetTieOrder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "data"
        self.root.mkdir()
        _write_data(self.root)
        self.cache = Path(tmp.name) / "cache"

    def _dataset(self, cls=BaseDataset) -> BaseDataset:
        return cls(
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

    def test_event_table_stores_source_row_as_delta_encoded_int32(self):
        dataset = self._dataset()
        self.assertEqual(dataset.global_event_df.collect_schema()[SOURCE_ROW], pl.Int32)
        files = sorted((dataset.cache_dir / "global_event_df.parquet").glob("*.parquet"))
        meta = pq.ParquetFile(files[0]).metadata
        column = meta.row_group(0).column(meta.schema.names.index(SOURCE_ROW))
        self.assertIn("DELTA_BINARY_PACKED", column.encodings)

    def test_hook_that_drops_source_row_gets_rows_numbered_again(self):
        patient = self._dataset(_DropsSourceRow).get_patient("p1")
        codes = [e.code for e in patient.get_events(event_type="diagnoses")]
        self.assertEqual(codes, CODES)
        rows = (
            self._dataset(_DropsSourceRow).global_event_df
            .filter(pl.col("event_type") == "diagnoses")
            .select(SOURCE_ROW).collect()[SOURCE_ROW].sort().to_list()
        )
        self.assertEqual(rows, list(range(len(CODES))))

    def test_hook_that_reorders_rows_keeps_source_order(self):
        patient = self._dataset(_ReordersRows).get_patient("p1")
        codes = [e.code for e in patient.get_events(event_type="diagnoses")]
        self.assertEqual(codes, CODES)

    def test_an_outer_join_builds_and_keeps_source_order(self):
        # The lookup has a code no diagnosis uses; the outer join adds it as
        # a row with no source row.
        lookup = ["code,label"] + [f"{code},label-{code}" for code in CODES]
        (self.root / "codes.csv").write_text("\n".join(lookup + ["999.9,unused"]) + "\n")
        config = self.root / "config_join.yaml"
        config.write_text(JOIN_CONFIG)
        dataset = BaseDataset(
            root=str(self.root),
            tables=["diagnoses"],
            dataset_name="TieOrderJoinDataset",
            config_path=str(config),
            cache_dir=str(self.cache),
        )
        patient = dataset.get_patient("p1")
        events = patient.get_events(event_type="diagnoses")
        self.assertEqual([e.code for e in events], CODES)
        self.assertEqual([e.label for e in events], [f"label-{c}" for c in CODES])
        added = (
            dataset.global_event_df.filter(pl.col(SOURCE_ROW) == -1)
            .select("diagnoses/code").collect()["diagnoses/code"].to_list()
        )
        self.assertEqual(added, ["999.9"])

    def test_cache_key_includes_event_format(self):
        # Caches written before SOURCE_ROW existed must not be reused.
        current = self._dataset().cache_dir
        self.assertEqual(self._dataset().cache_dir, current)  # the key is deterministic
        with mock.patch(
            "pyhealth.datasets.base_dataset.EVENT_FORMAT", EVENT_FORMAT - 1
        ):
            self.assertNotEqual(self._dataset().cache_dir, current)


class TestSourceRowType(unittest.TestCase):
    def _frame(self, n: int) -> dd.DataFrame:
        pdf = pd.DataFrame({"code": [f"c{i}" for i in range(n)], SOURCE_ROW: range(n)})
        return dd.from_pandas(pdf, npartitions=2)

    def test_source_row_is_int32(self):
        frame = _source_row_as_int32(self._frame(6))
        self.assertEqual(str(frame.dtypes[SOURCE_ROW]), "int32")
        out = frame.compute(scheduler="sync")
        self.assertEqual(str(out[SOURCE_ROW].dtype), "int32")
        self.assertEqual(out[SOURCE_ROW].tolist(), list(range(6)))

    def test_delta_encoding_only_for_an_integer_column(self):
        frame = _source_row_as_int32(self._frame(4))
        self.assertEqual(
            _event_parquet_options(frame.dtypes)["column_encoding"],
            {SOURCE_ROW: "DELTA_BINARY_PACKED"},
        )
        floats = frame.astype({SOURCE_ROW: "float64"})  # e.g. a column with gaps
        self.assertEqual(_event_parquet_options(floats.dtypes), {})
        self.assertEqual(_event_parquet_options(frame.drop(columns=SOURCE_ROW).dtypes), {})

    def test_rows_without_a_source_row_get_minus_one(self):
        frame = dd.from_pandas(
            pd.DataFrame({SOURCE_ROW: [0.0, float("nan"), 1.0]}), npartitions=1
        )
        rows = _source_row_as_int32(frame).compute()[SOURCE_ROW]
        self.assertEqual(rows.dtype, "int32")
        self.assertEqual(rows.tolist(), [0, -1, 1])

    def test_too_many_rows_fail_instead_of_wrapping_around(self):
        with mock.patch("pyhealth.datasets.base_dataset._INT32_MAX", 3), \
                self.assertRaisesRegex(OverflowError, "int32"):
            _source_row_as_int32(self._frame(6)).compute(scheduler="sync")


# Conditions of one encounter, all recorded at the same time, in file order.
FHIR_CONDITIONS = ["c7", "c2", "c9", "c1", "c5", "c3"]


def _write_fhir(root: Path) -> None:
    resources = [
        {"resourceType": "Patient", "id": "p1", "birthDate": "1950-01-01", "gender": "female"},
        {
            "resourceType": "Encounter", "id": "e1",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"}, "class": {"code": "IMP"},
        },
    ] + [
        {
            "resourceType": "Condition", "id": cid,
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": f"I{i}"}]},
            "onsetDateTime": "2020-06-01T11:00:00Z",
        }
        for i, cid in enumerate(FHIR_CONDITIONS)
    ]
    (root / "fixture.ndjson").write_text("\n".join(json.dumps(r) for r in resources) + "\n")


class _FHIRDropsSourceRow(MIMIC4FHIR):
    """A FHIR preprocess hook that rebuilds the frame without the column."""

    def preprocess_condition(self, df):
        return df.drop(SOURCE_ROW)


class TestFHIRTieOrder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "fhir"
        self.root.mkdir()
        _write_fhir(self.root)
        self.cache = Path(tmp.name) / "cache"

    def _dataset(self, cls=MIMIC4FHIR) -> MIMIC4FHIR:
        return cls(root=str(self.root), glob_pattern="*.ndjson", cache_dir=str(self.cache))

    def test_tied_conditions_keep_file_order(self):
        dataset = self._dataset()
        numbered = (
            dataset.global_event_df.filter(pl.col("event_type") == "condition")
            .sort(SOURCE_ROW).select("condition/resource_id").collect()
        )
        self.assertEqual(numbered["condition/resource_id"].to_list(), FHIR_CONDITIONS)
        patient = dataset.get_patient("p1")
        ids = [e.resource_id for e in patient.get_events(event_type="condition")]
        self.assertEqual(ids, FHIR_CONDITIONS)

    def test_hook_that_drops_source_row_gets_rows_numbered_again(self):
        patient = self._dataset(_FHIRDropsSourceRow).get_patient("p1")
        ids = [e.resource_id for e in patient.get_events(event_type="condition")]
        self.assertEqual(ids, FHIR_CONDITIONS)

    def test_cache_key_includes_event_format(self):
        current = self._dataset().cache_dir
        with mock.patch("pyhealth.datasets.fhir.base.EVENT_FORMAT", EVENT_FORMAT - 1):
            self.assertNotEqual(self._dataset().cache_dir, current)


if __name__ == "__main__":
    unittest.main()
