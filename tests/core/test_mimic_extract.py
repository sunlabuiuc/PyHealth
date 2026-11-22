"""Tests for MIMICExtractDataset, on a tiny synthetic MIMIC-Extract layout."""

import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pyhealth.datasets import MIMICExtractDataset
from pyhealth.datasets.mimicextract import (
    PARAMS_JSON,
    PATIENTS_CSV,
    VITALS_LABS_CSV,
    build_mimic_extract_tables,
    prepare_mimic_extract,
)
from pyhealth.tasks import MIMICExtractMortalityHospital

HAS_TABLES = importlib.util.find_spec("tables") is not None

VARIABLES = ["Heart Rate", "Glucose"]
AGGREGATIONS = ["count", "mean", "std"]
INTIME = {
    100: "2150-03-01 08:30:00",
    200: "2150-05-02 00:00:00",
    300: "2151-01-10 12:00:00",
    301: "2151-06-01 06:00:00",
}


def _patients():
    """Four stays of three patients, in MIMIC-Extract's ``patients`` layout.

    Subject 1: 2-day stay, died in hospital. Subject 2: 1-day stay (too short
    for the default 24 + 6 h task window). Subject 3: two stays, so per-stay
    data can be checked for mixing.
    """
    rows = [
        (1, 11, 100, 2.0, 0, 1, "F", 70.5),
        (2, 22, 200, 1.0, 0, 0, "M", 55.0),
        (3, 33, 300, 8.0, 0, 0, "M", 81.2),
        (3, 34, 301, 4.0, 1, 1, "M", 81.6),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "hadm_id",
            "icustay_id",
            "los_icu",
            "mort_icu",
            "mort_hosp",
            "gender",
            "age",
        ],
    )
    df["intime"] = pd.to_datetime(df["icustay_id"].map(INTIME))
    df["outtime"] = df["intime"] + pd.to_timedelta(df["los_icu"], unit="D")
    return df.set_index(["subject_id", "hadm_id", "icustay_id"])


def _vitals_labs():
    """The wide hourly table: (variable, aggregation) columns, NaN = no data.

    Each row is (stay, hour, heart-rate count/mean/std, glucose count/mean/std).
    """
    rows = [
        (100, 0, 1, 80.0, np.nan, 0, np.nan, np.nan),
        (100, 2, 2, 90.0, 1.5, 0, np.nan, np.nan),
        (100, 30, 1, 200.0, np.nan, 0, np.nan, np.nan),  # after max_hours
        (100, 5, 0, np.nan, np.nan, 0, np.nan, np.nan),  # nothing measured
        (200, 0, 0, np.nan, np.nan, 1, 100.0, np.nan),
        (300, 0, 0, np.nan, np.nan, 1, 110.0, np.nan),
        (301, 0, 0, np.nan, np.nan, 1, 300.0, np.nan),
    ]
    stay_keys = {100: (1, 11), 200: (2, 22), 300: (3, 33), 301: (3, 34)}
    index = pd.MultiIndex.from_tuples(
        [(*stay_keys[stay], stay, hour) for stay, hour, *_ in rows],
        names=["subject_id", "hadm_id", "icustay_id", "hours_in"],
    )
    columns = pd.MultiIndex.from_product(
        [VARIABLES, AGGREGATIONS], names=["LEVEL2", "Aggregation Function"]
    )
    values = [values for _, _, *values in rows]
    return pd.DataFrame(values, index=index, columns=columns)


class TestBuildTables(unittest.TestCase):
    """Reshaping MIMIC-Extract's tables into the dataset's CSVs."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root)

    def _build(self, **kwargs):
        build_mimic_extract_tables(_patients(), _vitals_labs(), self.root, **kwargs)
        patients = pd.read_csv(self.root / PATIENTS_CSV, dtype=str)
        hourly = pd.read_csv(self.root / VITALS_LABS_CSV, dtype=str)
        return patients, hourly

    def test_patients_rows_and_columns(self):
        patients, _ = self._build()
        self.assertEqual(list(patients["icustay_id"]), ["100", "200", "300", "301"])
        self.assertEqual(list(patients["subject_id"]), ["1", "2", "3", "3"])
        for column in [
            "hadm_id",
            "intime",
            "los_icu",
            "mort_icu",
            "mort_hosp",
            "ethnicity",
            "deathtime",
        ]:
            self.assertIn(column, patients.columns)
        self.assertEqual(patients.loc[0, "intime"], "2150-03-01 08:30:00")
        # Optional columns missing from the extract are written empty.
        self.assertTrue(patients["ethnicity"].isna().all())

    def test_hourly_long_format(self):
        _, hourly = self._build()
        self.assertEqual(
            list(hourly.columns),
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "hours_in",
                "time",
                "variable",
                "mean",
                "count",
                "std",
            ],
        )
        row = hourly[(hourly["icustay_id"] == "100") & (hourly["hours_in"] == "2")]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["variable"], "heart rate")  # lower-cased
        self.assertEqual(float(row.iloc[0]["mean"]), 90.0)
        self.assertEqual(float(row.iloc[0]["std"]), 1.5)
        # intime 08:30 plus 2 hours
        self.assertEqual(row.iloc[0]["time"], "2150-03-01 10:30:00")

    def test_empty_measurements_dropped(self):
        _, hourly = self._build()
        # 6 measured values: hour 5 of stay 100 and every count-0 cell drop.
        self.assertEqual(len(hourly), 6)
        self.assertNotIn("5", set(hourly["hours_in"]))
        stay_200 = hourly[hourly["icustay_id"] == "200"]
        self.assertEqual(list(stay_200["variable"]), ["glucose"])

    def test_max_hours(self):
        _, hourly = self._build(max_hours=24)
        self.assertEqual(len(hourly), 5)
        self.assertNotIn("200.0", set(hourly["mean"]))

    def test_small_chunks_match(self):
        _, whole = self._build()
        _, chunked = self._build(chunk_rows=2)
        pd.testing.assert_frame_equal(whole, chunked)

    def test_missing_patient_column(self):
        patients = _patients().drop(columns=["mort_hosp"])
        with self.assertRaisesRegex(ValueError, "mort_hosp"):
            build_mimic_extract_tables(patients, _vitals_labs(), self.root)

    def test_level1_columns_rejected(self):
        wide = _vitals_labs()
        wide.columns = pd.MultiIndex.from_tuples(
            [("a", "b", var, "d", agg) for var, agg in wide.columns]
        )
        with self.assertRaisesRegex(ValueError, "two levels"):
            build_mimic_extract_tables(_patients(), wide, self.root)

    def test_hourly_stay_without_patient(self):
        patients = _patients().drop(index=301, level="icustay_id")
        with self.assertRaisesRegex(ValueError, "not in patients"):
            build_mimic_extract_tables(patients, _vitals_labs(), self.root)

    def test_invalid_max_hours(self):
        with self.assertRaises(ValueError):
            build_mimic_extract_tables(
                _patients(), _vitals_labs(), self.root, max_hours=0
            )


class TestPrepare(unittest.TestCase):
    """Reading all_hourly_data.h5."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            prepare_mimic_extract(self.root / "nope")

    def test_missing_h5(self):
        with self.assertRaisesRegex(FileNotFoundError, "all_hourly_data_500.h5"):
            prepare_mimic_extract(self.root, pop_size=500)

    def test_missing_pytables(self):
        (self.root / "all_hourly_data.h5").touch()
        with (
            mock.patch.dict(sys.modules, {"tables": None}),
            self.assertRaisesRegex(ImportError, "pip install tables"),
        ):
            prepare_mimic_extract(self.root)

    def _fake_hdf5(self, tables):
        """Patches PyTables and ``pd.HDFStore`` so the HDF5 read path runs
        without PyTables installed (as in CI). ``tables`` maps HDF5 keys to
        the DataFrames the fake store returns."""

        class FakeStore:
            def __init__(self, path, mode="r"):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(f"/{key}" for key in tables)

            def __getitem__(self, key):
                return tables[key]

        return (
            mock.patch.dict(sys.modules, {"tables": types.ModuleType("tables")}),
            mock.patch("pyhealth.datasets.mimicextract.pd.HDFStore", FakeStore),
        )

    def test_reads_h5_simulated(self):
        (self.root / "all_hourly_data_4.h5").touch()
        fake_tables, fake_store = self._fake_hdf5(
            {"patients": _patients(), "vitals_labs": _vitals_labs()}
        )
        with fake_tables, fake_store:
            prepare_mimic_extract(self.root, pop_size=4, max_hours=24)
        self.assertEqual(len(pd.read_csv(self.root / PATIENTS_CSV)), 4)
        self.assertEqual(len(pd.read_csv(self.root / VITALS_LABS_CSV)), 5)
        self.assertTrue((self.root / PARAMS_JSON).is_file())

    def test_missing_h5_table_simulated(self):
        (self.root / "all_hourly_data.h5").touch()
        fake_tables, fake_store = self._fake_hdf5({"patients": _patients()})
        with (
            fake_tables,
            fake_store,
            self.assertRaisesRegex(ValueError, "vitals_labs"),
        ):
            prepare_mimic_extract(self.root)

    @unittest.skipUnless(HAS_TABLES, "PyTables is not installed")
    def test_reads_h5(self):
        path = self.root / "all_hourly_data_4.h5"
        _patients().to_hdf(path, key="patients")
        _vitals_labs().to_hdf(path, key="vitals_labs")
        prepare_mimic_extract(self.root, pop_size=4, max_hours=24)
        self.assertEqual(len(pd.read_csv(self.root / PATIENTS_CSV)), 4)
        self.assertEqual(len(pd.read_csv(self.root / VITALS_LABS_CSV)), 5)
        self.assertTrue((self.root / PARAMS_JSON).is_file())

    @unittest.skipUnless(HAS_TABLES, "PyTables is not installed")
    def test_missing_h5_table(self):
        _patients().to_hdf(self.root / "all_hourly_data.h5", key="patients")
        with self.assertRaisesRegex(ValueError, "vitals_labs"):
            prepare_mimic_extract(self.root)


class TestMIMICExtractDataset(unittest.TestCase):
    """Loading the built tables with BaseDataset. Built once: it starts Dask."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.cache = Path(tempfile.mkdtemp())
        build_mimic_extract_tables(_patients(), _vitals_labs(), cls.root)
        cls.dataset = MIMICExtractDataset(root=str(cls.root), cache_dir=cls.cache)
        # Build the event cache once here (it starts Dask, a few seconds) so
        # the tests below each run in milliseconds instead of the first one
        # paying for the build.
        _ = cls.dataset.unique_patient_ids

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)
        # litdata keeps the task cache memory-mapped after set_task, and
        # Windows cannot delete mapped files, so cleanup there is best effort.
        # Elsewhere (including CI) a failed cleanup still fails the test.
        shutil.rmtree(cls.cache, ignore_errors=sys.platform == "win32")

    def test_patient_count(self):
        self.assertEqual(sorted(self.dataset.unique_patient_ids), ["1", "2", "3"])
        self.assertEqual(self.dataset.dataset_name, "mimic_extract")

    def test_stay_events(self):
        (stay,) = self.dataset.get_patient("1").get_events(event_type="patients")
        self.assertEqual(stay["icustay_id"], "100")
        self.assertEqual(stay.timestamp, datetime(2150, 3, 1, 8, 30))
        self.assertEqual(stay["mort_hosp"], 1.0)
        self.assertEqual(stay["mort_icu"], 0.0)
        self.assertEqual(stay["los_icu"], 2.0)
        self.assertAlmostEqual(stay["age"], 70.5)

    def test_hourly_events(self):
        events = self.dataset.get_patient("1").get_events(
            event_type="vitals_labs", filters=[("hours_in", "<", 24)]
        )
        self.assertEqual([e["hours_in"] for e in events], [0, 2])
        self.assertEqual(events[1]["mean"], 90.0)
        self.assertEqual(events[1]["count"], 2.0)
        self.assertEqual(events[1].timestamp, datetime(2150, 3, 1, 10, 30))

    def test_stays_do_not_mix(self):
        patient = self.dataset.get_patient("3")
        stays = patient.get_events(event_type="patients")
        self.assertEqual({s["icustay_id"] for s in stays}, {"300", "301"})
        means = {
            e["icustay_id"]: e["mean"]
            for e in patient.get_events(event_type="vitals_labs")
        }
        self.assertEqual(means, {"300": 110.0, "301": 300.0})

    def test_task_smoke(self):
        task = MIMICExtractMortalityHospital(variables=["heart rate", "glucose"])
        samples = self.dataset.set_task(task, num_workers=1)
        # Subject 2's 1-day stay is shorter than 24 + 6 hours.
        self.assertEqual(len(samples), 3)
        labels = sorted(int(s["label"]) for s in samples)
        self.assertEqual(labels, [0, 1, 1])
        self.assertEqual(len(samples[0]["features"]), 24 * 2 * 3)
        samples.close()


class TestDatasetErrors(unittest.TestCase):
    """Validation in MIMICExtractDataset.__init__ (no Dask is started)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            MIMICExtractDataset(root=str(self.root / "nope"))

    def test_no_tables_and_no_h5(self):
        with self.assertRaisesRegex(FileNotFoundError, "all_hourly_data.h5"):
            MIMICExtractDataset(root=str(self.root))

    def test_unknown_table(self):
        with self.assertRaisesRegex(ValueError, "interventions"):
            MIMICExtractDataset(root=str(self.root), tables=["interventions"])

    def test_csv_missing_columns(self):
        build_mimic_extract_tables(_patients(), _vitals_labs(), self.root)
        hourly = pd.read_csv(self.root / VITALS_LABS_CSV).drop(columns=["std"])
        hourly.to_csv(self.root / VITALS_LABS_CSV, index=False)
        with self.assertRaisesRegex(ValueError, "std"):
            MIMICExtractDataset(root=str(self.root))

    def test_changed_max_hours_rebuilds(self):
        build_mimic_extract_tables(_patients(), _vitals_labs(), self.root)
        (self.root / PARAMS_JSON).write_text('{"pop_size": null, "max_hours": 48}')
        # max_hours=24 differs, so it tries to rebuild from the (absent) HDF5.
        with self.assertRaises(FileNotFoundError):
            MIMICExtractDataset(root=str(self.root), max_hours=24)


if __name__ == "__main__":
    unittest.main()
