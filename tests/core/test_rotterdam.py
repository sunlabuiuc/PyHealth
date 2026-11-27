# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
"""Rotterdam source validation and loading, using four synthetic patients."""

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pyhealth.datasets import RotterdamDataset
from pyhealth.datasets.rotterdam import ROTTERDAM_COLUMNS, parse_rotterdam_record
from tests.core.survival_test_utils import threaded_csv_loading


def write_csv(root: Path, rows: list[dict[str, Any]], name: str = "rotterdam.csv"):
    """Write synthetic rows with the source header order."""
    with (root / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROTTERDAM_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def rows_fixture() -> list[dict[str, Any]]:
    """Four synthetic patients, one per recurrence/death outcome combination.

    PIDs are not contiguous, to show that IDs come from ``pid`` and not from
    row positions. Patient 12 matches the 43 source patients whose
    recurrence was censored before a later death; patient 11 has tied event
    times; patient 13 has a fractional time that must not be rounded.
    """
    covariates = {
        "year": 1985,
        "meno": 1,
        "grade": 3,
        "pgr": 35,
        "er": 291,
        "hormon": 0,
        "chemo": 0,
    }
    outcomes = [
        # pid, age, size, nodes, rtime, recur, dtime, death
        (11, 61, "<=20", 0, 900, 1, 900, 1),  # both events, tied times
        (12, 70, "20-50", 4, 1200, 0, 2100, 1),  # recurrence censored, then death
        (13, 45, ">50", 2, 365.5, 1, 3000, 0),  # recurrence only, fractional time
        (14, 52, "<=20", 0, 5000, 0, 5000, 0),  # neither event
    ]
    return [
        dict(
            pid=pid,
            age=age,
            size=size,
            nodes=nodes,
            rtime=rtime,
            recur=recur,
            dtime=dtime,
            death=death,
            **covariates,
        )
        for pid, age, size, nodes, rtime, recur, dtime, death in outcomes
    ]


def as_record(row: dict[str, Any]) -> dict[str, str]:
    """Return a row as the strings a CSV reader would produce."""
    return {k: str(v) for k, v in row.items()}


class TestRotterdamRecord(unittest.TestCase):
    def test_valid_record_values_are_returned_unchanged(self):
        row = rows_fixture()[2]
        values = parse_rotterdam_record(as_record(row), "13")
        self.assertEqual(set(values), set(ROTTERDAM_COLUMNS[1:]))
        self.assertEqual(values["size"], ">50")
        self.assertEqual(values["rtime"], 365.5)
        self.assertEqual(values["dtime"], 3000.0)
        self.assertEqual((values["recur"], values["death"]), (1.0, 0.0))
        self.assertEqual(values["age"], 45.0)

    def test_censored_recurrence_before_death_keeps_both_times(self):
        values = parse_rotterdam_record(as_record(rows_fixture()[1]), "12")
        self.assertEqual((values["rtime"], values["recur"]), (1200.0, 0.0))
        self.assertEqual((values["dtime"], values["death"]), (2100.0, 1.0))

    def test_invalid_values_name_column_and_pid(self):
        cases = [
            ("rtime", "0", "strictly positive"),
            ("dtime", "-1", "strictly positive"),
            ("rtime", "1e-100", "strictly positive"),
            ("dtime", "nan", "finite"),
            ("recur", "2", "0 or 1"),
            ("death", "0.5", "0 or 1"),
            ("meno", "-1", "0 or 1"),
            ("size", "big", "must be one of"),
            ("size", "", "missing"),
            ("age", "", "missing"),
            ("age", "0", "strictly positive"),
            ("nodes", "-1", "non-negative"),
            ("pgr", "abc", "numeric"),
            ("er", "inf", "finite"),
            ("year", "1e100", "finite"),
        ]
        for column, value, message in cases:
            record = as_record(rows_fixture()[0])
            record[column] = value
            with (
                self.subTest(column=column, value=value),
                self.assertRaisesRegex(ValueError, rf"pid '11': {column} .*{message}"),
            ):
                parse_rotterdam_record(record, "11")
        record = as_record(rows_fixture()[0])
        del record["chemo"]
        with self.assertRaisesRegex(ValueError, "pid '11': chemo is missing"):
            parse_rotterdam_record(record, "11")

    def test_recurrence_follow_up_cannot_exceed_death_follow_up(self):
        record = as_record(rows_fixture()[3])
        record["rtime"] = "5001"
        with self.assertRaisesRegex(ValueError, "pid '14': rtime .* exceeds dtime"):
            parse_rotterdam_record(record, "14")


class TestRotterdamDatasetLoading(unittest.TestCase):
    """Load the synthetic cohort once and check the cached events."""

    @classmethod
    def setUpClass(cls):
        # litdata can keep cache files open on Windows, so cleanup may fail there.
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.rows = rows_fixture()
        write_csv(root, cls.rows)
        with threaded_csv_loading():
            cls.dataset = RotterdamDataset(
                str(root), cache_dir=root / "cache", num_workers=1
            )
            cls.patient_ids = set(cls.dataset.unique_patient_ids)
            cls.patients = {
                pid: cls.dataset.get_patient(pid) for pid in cls.patient_ids
            }

    def test_patient_ids_come_from_pid(self):
        self.assertEqual(self.patient_ids, {"11", "12", "13", "14"})
        self.assertEqual(self.dataset.num_source_patients, 4)

    def test_each_patient_keeps_one_record_with_source_values(self):
        for row in self.rows:
            pid = str(row["pid"])
            with self.subTest(pid=pid):
                events = self.patients[pid].get_events("rotterdam")
                self.assertEqual(len(events), 1)
                attrs = events[0].attr_dict
                for column in ROTTERDAM_COLUMNS[1:]:
                    self.assertEqual(str(attrs[column]), str(row[column]), column)

    def test_all_patient_times_are_preserved(self):
        times = {
            pid: tuple(
                float(patient.get_events("rotterdam")[0].attr_dict[column])
                for column in ("rtime", "recur", "dtime", "death")
            )
            for pid, patient in self.patients.items()
        }
        self.assertEqual(
            times,
            {
                "11": (900.0, 1.0, 900.0, 1.0),
                "12": (1200.0, 0.0, 2100.0, 1.0),
                "13": (365.5, 1.0, 3000.0, 0.0),
                "14": (5000.0, 0.0, 5000.0, 0.0),
            },
        )


class TestRotterdamDatasetValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_csv(self.root, rows_fixture())

    def dataset(self, **kwargs: Any) -> RotterdamDataset:
        return RotterdamDataset(
            str(self.root), cache_dir=self.root / "cache", num_workers=1, **kwargs
        )

    def test_invalid_csv_fails_before_caching(self):
        for column, value in [
            ("pid", ""),
            ("rtime", 0),
            ("dtime", -5),
            ("recur", 2),
            ("death", "yes"),
            ("size", "50+"),
            ("age", ""),
            ("rtime", 6000),
        ]:
            rows = rows_fixture()
            rows[1][column] = value
            write_csv(self.root, rows)
            pattern = "pid" if column == "pid" else rf"pid '12': {column}"
            with self.subTest(column=column, value=value):
                with self.assertRaisesRegex(ValueError, pattern):
                    self.dataset()
                self.assertFalse(any((self.root / "cache").rglob("*.parquet")))

    def test_duplicate_pid_rejected_but_equal_values_allowed(self):
        rows = rows_fixture()
        rows[3] = {**rows[0], "pid": 14}  # Same values under another pid is valid.
        write_csv(self.root, rows)
        self.assertEqual(self.dataset().num_source_patients, 4)
        rows[3]["pid"] = 11
        write_csv(self.root, rows)
        with self.assertRaisesRegex(ValueError, "unique, got '11'"):
            self.dataset()

    def test_wrong_table_or_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supports only"):
            self.dataset(tables=["other"])
        import pyhealth.datasets.rotterdam as module

        text = (Path(module.__file__).parent / "configs/rotterdam.yaml").read_text()
        for old, new in [
            ('patient_id: "pid"', 'patient_id: "year"'),
            ("      - dtime\n", ""),
        ]:
            config = self.root / "bad.yaml"
            config.write_text(text.replace(old, new))
            with (
                self.subTest(change=new or f"drop {old.strip()}"),
                self.assertRaisesRegex(ValueError, "config must use pid"),
            ):
                self.dataset(config_path=str(config))

    def test_empty_missing_column_and_missing_file(self):
        write_csv(self.root, [])
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.dataset()
        (self.root / "rotterdam.csv").write_text("pid,rtime,recur\n1,10,1\n")
        with self.assertRaisesRegex(ValueError, r"missing columns: \['age'"):
            self.dataset()
        (self.root / "rotterdam.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
