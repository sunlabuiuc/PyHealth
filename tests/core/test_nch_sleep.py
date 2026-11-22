"""NCHSleepDataset metadata, on small synthetic EDF and annotation files."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets import NCHSleepDataset
from pyhealth.datasets.metadata_csv import resolve_metadata_root
from pyhealth.datasets.nch_sleep import (
    read_nch_annotations,
    summarize_nch_annotations,
)
from tests.core.psg_test_utils import write_edf
from tests.core.psg_test_utils import write_tsv as _write_tsv

SFREQ = 8
SECONDS = 120
CHANNELS = ("EEG C4-M1", "Resp Airflow", "SpO2")
CSV = "nch_sleep-pyhealth.csv"

# record id -> (annotation rows as (onset, duration, description))
STUDIES = {
    "100_7": [
        (0, 30, "Sleep stage W"),
        (30, 30, "Sleep stage N2"),
        (60, 30, "Sleep stage N3"),
        (90, 30, "Sleep stage R"),
        (35.5, 12.0, "Obstructive Apnea"),
        (65.0, 15.0, "Hypopnea"),
        (95.0, 10.0, "Central Apnea"),
        (66.0, 3.0, "EEG arousal"),
    ],
    "100_12": [
        (0, 30, "Sleep stage W"),
        (30, 30, "Sleep stage N1"),
        (60, 60, "Sleep stage W"),
        (40.0, 11.0, "Obstructive Hypopnea"),
    ],
    "25_3": [
        (0, 60, "Sleep stage N2"),
        (60, 60, "Sleep stage W"),
    ],
}

SLEEP_STUDY = pd.DataFrame(
    {
        "STUDY_PAT_ID": ["100", "100", "25"],
        "SLEEP_STUDY_ID": ["7", "12", "3"],
        "SLEEP_STUDY_START_DATETIME": [
            "3/4/2017 9:30:00 PM",
            "4/10/2018 8:05:00 PM",
            "1/2/2019 10:00:00 PM",
        ],
        "SLEEP_STUDY_DURATION_DATETIME": ["0 days 09:00:00"] * 3,
        "AGE_AT_SLEEP_STUDY_DAYS": ["2900", "3300", "1200"],
    }
)

DEMOGRAPHIC = pd.DataFrame(
    {
        "STUDY_PAT_ID": ["100", "25"],
        "BIRTH_DATE": ["2009-02-01", "2015-09-20"],
        "GENDER_DESCR": ["Male", "Female"],
        "RACE_DESCR": ["White", "Black or African American"],
        "ETHNICITY_DESCR": ["Not Hispanic or Latino", "Unknown"],
    }
)


def _write_edf(path: Path) -> None:
    write_edf(path, CHANNELS, sfreq=SFREQ, seconds=SECONDS)


def _write_dataset(
    root: Path, studies=STUDIES, sleep_data="sleep_data", health_data="health_data"
) -> None:
    for record_id, rows in studies.items():
        _write_edf(root / sleep_data / f"{record_id}.edf")
        _write_tsv(root / sleep_data / f"{record_id}.tsv", rows)
    if health_data:
        (root / health_data).mkdir(parents=True, exist_ok=True)
        SLEEP_STUDY.to_csv(root / health_data / "SLEEP_STUDY.csv", index=False)
        DEMOGRAPHIC.to_csv(root / health_data / "DEMOGRAPHIC.csv", index=False)


def _prepare(root: Path) -> pd.DataFrame:
    ds = NCHSleepDataset.__new__(NCHSleepDataset)
    ds.root = str(root)
    ds.prepare_metadata()
    return pd.read_csv(root / CSV, dtype=str, keep_default_na=False)


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "nchsdb"


class TestSummarizeAnnotations(_TempDir):
    def test_counts_events_and_sleep_time(self):
        path = self.root / "a.tsv"
        _write_tsv(path, STUDIES["100_7"])

        summary = summarize_nch_annotations(path)

        # Three 30 s sleep epochs (N2, N3, R) = 90 s; 3 events in 90 s of sleep.
        self.assertEqual(summary["n_apnea"], 2)
        self.assertEqual(summary["n_hypopnea"], 1)
        self.assertAlmostEqual(summary["sleep_hours"], 0.025)
        self.assertAlmostEqual(summary["ahi"], 120.0)

    def test_no_sleep_gives_zero_ahi(self):
        path = self.root / "a.tsv"
        _write_tsv(path, [(0, 30, "Sleep stage W"), (5, 10, "Obstructive Apnea")])

        summary = summarize_nch_annotations(path)

        self.assertEqual(summary["n_apnea"], 1)
        self.assertEqual(summary["sleep_hours"], 0)
        self.assertEqual(summary["ahi"], 0)

    def test_vocabulary_matches_the_original_code(self):
        path = self.root / "a.tsv"
        rows = [(0, 30, "Sleep stage N2")] + [
            (i, 1, d)
            for i, d in enumerate(
                [
                    "Mixed Apnea",
                    "apnea",
                    "obstructive apnea",
                    "central apnea",
                    "Apnea",
                    "Mixed Hypopnea",
                    "Central Hypopnea",
                    "hypopnea",
                    "Oxygen Desaturation",
                    "Apnea Index",
                ]
            )
        ]
        _write_tsv(path, rows)

        summary = summarize_nch_annotations(path)

        self.assertEqual((summary["n_apnea"], summary["n_hypopnea"]), (5, 3))


class TestReadNCHAnnotations(_TempDir):
    def test_reads_the_three_columns(self):
        path = self.root / "a.tsv"
        _write_tsv(path, STUDIES["25_3"])

        df = read_nch_annotations(path)

        self.assertEqual(list(df.columns), ["onset", "duration", "description"])
        self.assertEqual(df.iloc[0].tolist(), [0, 60, "Sleep stage N2"])

    def test_missing_column_raises(self):
        path = self.root / "a.tsv"
        path.parent.mkdir(parents=True)
        path.write_text("onset\tdescription\n0\tSleep stage W\n")

        with self.assertRaisesRegex(ValueError, "duration"):
            read_nch_annotations(path)


class TestNCHMetadata(_TempDir):
    def test_one_row_per_study_with_health_data(self):
        _write_dataset(self.root)

        df = _prepare(self.root)

        # Numeric order of patient and study ids, not string order.
        self.assertEqual(list(df["record_id"]), ["25_3", "100_7", "100_12"])
        self.assertEqual(list(df["patient_id"]), ["25", "100", "100"])
        self.assertEqual(list(df["study_id"]), ["3", "7", "12"])
        row = df.set_index("record_id").loc["100_7"]
        self.assertEqual(
            Path(row["signal_file"]), self.root / "sleep_data" / "100_7.edf"
        )
        self.assertEqual(
            Path(row["annotation_file"]), self.root / "sleep_data" / "100_7.tsv"
        )
        self.assertEqual(row["study_start"], "2017-03-04 21:30:00")
        self.assertEqual(row["age_days"], "2900")
        self.assertEqual((row["gender"], row["race"]), ("Male", "White"))
        self.assertEqual(float(row["sfreq"]), SFREQ)
        self.assertEqual(float(row["duration_sec"]), SECONDS)
        self.assertEqual(row["channels"].split("|"), list(CHANNELS))
        self.assertEqual((row["n_apnea"], row["n_hypopnea"]), ("2", "1"))
        self.assertEqual(float(row["ahi"]), 120.0)

    def test_edf_without_annotations_is_left_out(self):
        _write_dataset(self.root)
        (self.root / "sleep_data" / "100_12.tsv").unlink()
        _write_edf(self.root / "sleep_data" / "not-a-study.edf")
        _write_tsv(self.root / "sleep_data" / "not-a-study.tsv", STUDIES["25_3"])

        with self.assertLogs("pyhealth.datasets.nch_sleep", level="WARNING") as logs:
            df = _prepare(self.root)

        self.assertEqual(list(df["record_id"]), ["25_3", "100_7"])
        output = "\n".join(logs.output)
        self.assertIn("not-a-study.edf", output)
        self.assertIn("100_12.edf", output)
        self.assertIn(f"Delete {CSV}", output)

    def test_unreadable_edf_is_left_out(self):
        _write_dataset(self.root)
        (self.root / "sleep_data" / "100_12.edf").write_bytes(b"not an edf")

        with self.assertLogs("pyhealth.datasets.nch_sleep", level="WARNING"):
            df = _prepare(self.root)

        self.assertEqual(list(df["record_id"]), ["25_3", "100_7"])

    def test_unreadable_annotations_are_left_out(self):
        _write_dataset(self.root)
        (self.root / "sleep_data" / "100_12.tsv").write_text("")
        (self.root / "sleep_data" / "25_3.tsv").write_text("start\tlabel\n0\tW\n")

        with self.assertLogs("pyhealth.datasets.nch_sleep", level="WARNING") as logs:
            df = _prepare(self.root)

        # The other studies are still indexed.
        self.assertEqual(list(df["record_id"]), ["100_7"])
        self.assertEqual(len(logs.output), 2)
        self.assertIn("25_3.edf", logs.output[0])
        self.assertIn("100_12.edf", logs.output[1])

    def test_study_missing_from_health_data_is_kept_with_empty_columns(self):
        _write_dataset(self.root, studies={**STUDIES, "999_1": STUDIES["25_3"]})

        with self.assertLogs("pyhealth.datasets.nch_sleep", level="WARNING") as logs:
            df = _prepare(self.root)

        row = df.set_index("record_id").loc["999_1"]
        self.assertEqual(
            (row["study_start"], row["age_days"], row["gender"]), ("", "", "")
        )
        self.assertIn("999_1", logs.output[0])

    def test_without_health_data_folder(self):
        _write_dataset(self.root, health_data=None)

        with self.assertLogs("pyhealth.datasets.nch_sleep", level="WARNING") as logs:
            df = _prepare(self.root)

        self.assertEqual(len(df), 3)
        self.assertTrue((df["age_days"] == "").all())
        # One warning per missing table, not one more per study.
        self.assertEqual(len(logs.output), 2)
        self.assertIn("SLEEP_STUDY.csv not found", logs.output[0])
        self.assertIn("DEMOGRAPHIC.csv not found", logs.output[1])

    def test_folder_names_are_matched_without_case(self):
        _write_dataset(self.root, sleep_data="Sleep_Data", health_data="Health_Data")

        df = _prepare(self.root)

        self.assertEqual(len(df), 3)
        self.assertEqual(df.set_index("record_id").loc["25_3", "gender"], "Female")

    def test_no_studies_writes_nothing(self):
        (self.root / "sleep_data").mkdir(parents=True)
        ds = NCHSleepDataset.__new__(NCHSleepDataset)
        ds.root = str(self.root)

        with self.assertLogs("pyhealth.datasets.nch_sleep", level="WARNING"):
            ds.prepare_metadata()

        self.assertFalse((self.root / CSV).exists())

    def test_existing_csv_is_left_untouched(self):
        _write_dataset(self.root, studies={"25_3": STUDIES["25_3"]})
        first = _prepare(self.root)
        _write_dataset(self.root)

        df = _prepare(self.root)

        self.assertEqual(list(df["record_id"]), list(first["record_id"]))

    def test_metadata_goes_to_the_home_cache_when_root_is_read_only(self):
        _write_dataset(self.root)
        home = self.root.parent / "home"
        real_to_csv = pd.DataFrame.to_csv

        def to_csv(df, path_or_buf=None, *args, **kwargs):
            # Stands in for a read-only root: chmod has no effect on Windows.
            if str(path_or_buf).startswith(str(self.root)):
                raise PermissionError(f"read-only: {path_or_buf}")
            return real_to_csv(df, path_or_buf, *args, **kwargs)

        with (
            patch.object(pd.DataFrame, "to_csv", to_csv),
            patch("pyhealth.datasets.metadata_csv.Path.home", return_value=home),
        ):
            ds = NCHSleepDataset.__new__(NCHSleepDataset)
            ds.root = str(self.root)
            ds.prepare_metadata()
            metadata_root = Path(resolve_metadata_root(self.root, "nch_sleep", CSV))

        self.assertFalse((self.root / CSV).exists())
        self.assertTrue(metadata_root.is_relative_to(home))


class TestNCHEndToEnd(_TempDir):
    def test_patients_and_study_events(self):
        _write_dataset(self.root)

        dataset = NCHSleepDataset(
            root=str(self.root), cache_dir=str(self.root.parent / "cache")
        )

        self.assertEqual(sorted(dataset.unique_patient_ids), ["100", "25"])
        studies = dataset.get_patient("100").get_events("studies")
        self.assertEqual([s.record_id for s in studies], ["100_7", "100_12"])
        self.assertEqual(
            [str(s.timestamp) for s in studies],
            ["2017-03-04 21:30:00", "2018-04-10 20:05:00"],
        )
        self.assertTrue(Path(studies[0].signal_file).is_file())
        self.assertAlmostEqual(float(studies[1].ahi), 120.0)

    def test_study_without_start_time_loads_with_empty_timestamp(self):
        _write_dataset(self.root, studies={**STUDIES, "999_1": STUDIES["25_3"]})

        dataset = NCHSleepDataset(
            root=str(self.root), cache_dir=str(self.root.parent / "cache")
        )

        (study,) = dataset.get_patient("999").get_events("studies")
        self.assertEqual(study.record_id, "999_1")
        self.assertTrue(pd.isna(study.timestamp))


if __name__ == "__main__":
    unittest.main()
