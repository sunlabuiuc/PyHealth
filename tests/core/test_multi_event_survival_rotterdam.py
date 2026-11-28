# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
"""MultiEventSurvivalRotterdam features and labels, using four synthetic patients."""

import csv
import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import RotterdamDataset, get_dataloader
from pyhealth.datasets.rotterdam import ROTTERDAM_COLUMNS
from pyhealth.tasks import MultiEventSurvivalRotterdam
from tests.core.survival_test_utils import threaded_csv_loading

FEATURES = (
    "year",
    "age",
    "meno",
    "grade",
    "nodes",
    "pgr",
    "er",
    "hormon",
    "chemo",
    "size_mm",
)
FEATURES_WITHOUT_TREATMENT = tuple(f for f in FEATURES if f not in ("hormon", "chemo"))
TREATMENT_COLUMNS = [FEATURES.index("hormon"), FEATURES.index("chemo")]


def without_treatment(features: list) -> list:
    """Drop the hormon and chemo positions from a ten-feature list."""
    return [v for i, v in enumerate(features) if i not in TREATMENT_COLUMNS]


def rows_fixture() -> list[dict[str, str]]:
    """Four synthetic patients, one per recurrence/death outcome combination.

    Values are strings, as read from the CSV. Patient 22 matches the 43
    source patients whose recurrence was censored before a later death;
    patient 21 has tied event times; patient 23 has a fractional time.
    """
    rows = [
        # pid year age meno size grade nodes pgr er hormon chemo
        # rtime recur dtime death
        ("21", "1985", "61", "1", "<=20", "3", "0", "35", "291", "0", "0")
        + ("900", "1", "900", "1"),  # both events, tied times
        ("22", "1988", "70", "1", "20-50", "2", "4", "0", "12", "1", "0")
        + ("1200", "0", "2100", "1"),  # recurrence censored, then death
        ("23", "1979", "45", "0", ">50", "3", "2", "138", "0", "0", "1")
        + ("365.5", "1", "3000", "0"),  # recurrence only, fractional time
        ("24", "1992", "52", "0", "<=20", "2", "0", "60", "45", "0", "1")
        + ("5000", "0", "5000", "0"),  # neither event
    ]
    return [dict(zip(ROTTERDAM_COLUMNS, row, strict=True)) for row in rows]


# Expected task output per pid: features, [rtime, dtime], [recur, death].
EXPECTED = {
    "21": (
        [1985, 61, 1, 3, 0, 35, 291, 0, 0, 10],
        [900, 900],
        [1, 1],
    ),
    "22": (
        [1988, 70, 1, 2, 4, 0, 12, 1, 0, 35],
        [1200, 2100],
        [0, 1],
    ),
    "23": (
        [1979, 45, 0, 3, 2, 138, 0, 0, 1, 75],
        [365.5, 3000],
        [1, 0],
    ),
    "24": (
        [1992, 52, 0, 2, 0, 60, 45, 0, 1, 10],
        [5000, 5000],
        [0, 0],
    ),
}


def patient_from_rows(rows: list[dict[str, str]], pid: str = "21") -> Patient:
    """Build a Patient whose ``rotterdam`` events hold the given source rows."""
    events = [
        {
            "patient_id": pid,
            "event_type": "rotterdam",
            "timestamp": None,
            **{f"rotterdam/{k}": v for k, v in row.items() if k != "pid"},
        }
        for row in rows
    ]
    df = (
        pl.DataFrame(events)
        if events
        else pl.DataFrame(
            schema={
                "patient_id": pl.String,
                "event_type": pl.String,
                "timestamp": pl.Datetime,
            }
        )
    )
    return Patient(pid, df)


class TestMultiEventSurvivalRotterdamTask(unittest.TestCase):
    def setUp(self):
        self.task = MultiEventSurvivalRotterdam()
        self.rows = rows_fixture()

    def sample(self, row: dict[str, str]) -> dict:
        samples = self.task(patient_from_rows([row], row["pid"]))
        self.assertEqual(len(samples), 1)
        return samples[0]

    def test_schema_and_feature_order(self):
        self.assertEqual(self.task.task_name, "MultiEventSurvivalRotterdam")
        self.assertEqual(self.task.input_schema, {"features": "tensor"})
        self.assertEqual(
            self.task.output_schema,
            {"event_times": "tensor", "event_indicators": "tensor"},
        )
        self.assertTrue(self.task.use_treatment)
        self.assertEqual(self.task.feature_names, FEATURES)
        self.assertEqual(self.task.event_names, ("recurrence", "death"))

    def test_string_values_become_ten_ordered_float_features(self):
        sample = self.sample(self.rows[1])
        self.assertEqual(sample["patient_id"], "22")
        self.assertEqual(len(sample["features"]), 10)
        self.assertTrue(all(type(v) is float for v in sample["features"]))
        self.assertEqual(sample["features"], [float(v) for v in EXPECTED["22"][0]])

    def test_size_categories_map_to_millimetres(self):
        for size, millimetres in [("<=20", 10.0), ("20-50", 35.0), (">50", 75.0)]:
            row = dict(self.rows[0], size=size)
            with self.subTest(size=size):
                self.assertEqual(self.sample(row)["features"][-1], millimetres)

    def test_four_outcome_combinations_keep_source_times(self):
        for row in self.rows:
            pid = row["pid"]
            _, times, indicators = EXPECTED[pid]
            with self.subTest(pid=pid):
                patient = patient_from_rows([row], pid)
                sample = self.task(patient)[0]
                self.assertEqual(sample["event_times"], times)
                self.assertEqual(sample["event_indicators"], indicators)
                attrs = patient.get_events("rotterdam")[0].attr_dict
                self.assertEqual(
                    [attrs[c] for c in ("rtime", "recur", "dtime", "death")],
                    [row[c] for c in ("rtime", "recur", "dtime", "death")],
                )

    def test_without_treatment_drops_hormon_and_chemo_only(self):
        task = MultiEventSurvivalRotterdam(use_treatment=False)
        self.assertFalse(task.use_treatment)
        self.assertEqual(task.feature_names, FEATURES_WITHOUT_TREATMENT)
        for row in self.rows:
            pid = row["pid"]
            features, times, indicators = EXPECTED[pid]
            with self.subTest(pid=pid):
                sample = task(patient_from_rows([row], pid))[0]
                self.assertEqual(len(sample["features"]), 8)
                self.assertEqual(sample["features"], without_treatment(features))
                self.assertEqual(sample["event_times"], times)
                self.assertEqual(sample["event_indicators"], indicators)

    def test_use_treatment_must_be_bool(self):
        for value in (1, 0, "False", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                MultiEventSurvivalRotterdam(use_treatment=value)

    def test_missing_or_multiple_records_are_rejected(self):
        for rows in ([], self.rows[:2]):
            with (
                self.subTest(n=len(rows)),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                self.task(patient_from_rows(rows))

    def test_invalid_values_are_rejected_by_shared_parser(self):
        for column, value in [
            ("size", "50+"),
            ("rtime", "0"),
            ("dtime", "nan"),
            ("recur", "2"),
            ("age", ""),
            ("rtime", "901"),  # exceeds dtime
        ]:
            row = dict(self.rows[0], **{column: value})
            with (
                self.subTest(column=column, value=value),
                self.assertRaisesRegex(ValueError, rf"pid '21': {column}"),
            ):
                self.sample(row)


class TestMultiEventSurvivalRotterdamDataset(unittest.TestCase):
    """Run the default task through the real loader once, on synthetic rows."""

    @classmethod
    def setUpClass(cls):
        # litdata can keep cache files open on Windows, so cleanup may fail there.
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        with (root / "rotterdam.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ROTTERDAM_COLUMNS))
            writer.writeheader()
            writer.writerows(rows_fixture())
        with threaded_csv_loading():
            cls.dataset = RotterdamDataset(
                str(root), cache_dir=root / "cache", num_workers=1
            )
            cls.samples = cls.dataset.set_task()
            cls.no_treatment_task = MultiEventSurvivalRotterdam(use_treatment=False)
            cls.no_treatment = cls.dataset.set_task(cls.no_treatment_task)
        cls.batch = next(iter(get_dataloader(cls.samples, batch_size=4)))
        cls.no_treatment_batch = next(
            iter(get_dataloader(cls.no_treatment, batch_size=4))
        )

    def test_default_task_is_multi_event_survival(self):
        self.assertIsInstance(self.dataset.default_task, MultiEventSurvivalRotterdam)

    def test_set_task_yields_one_sample_per_patient_with_tensor_shapes(self):
        self.assertEqual(len(self.samples), 4)
        self.assertEqual(sorted(self.batch["patient_id"]), ["21", "22", "23", "24"])
        self.assertEqual(self.batch["features"].shape, (4, 10))
        self.assertEqual(self.batch["event_times"].shape, (4, 2))
        self.assertEqual(self.batch["event_indicators"].shape, (4, 2))
        for key in ("features", "event_times", "event_indicators"):
            self.assertEqual(self.batch[key].dtype, torch.float32, key)

    def test_batched_values_match_source_rows(self):
        for i, pid in enumerate(self.batch["patient_id"]):
            features, times, indicators = EXPECTED[pid]
            with self.subTest(pid=pid):
                self.assertEqual(self.batch["features"][i].tolist(), features)
                self.assertEqual(self.batch["event_times"][i].tolist(), times)
                self.assertEqual(self.batch["event_indicators"][i].tolist(), indicators)

    def test_eight_feature_variant_has_its_own_cache_and_values(self):
        self.assertNotEqual(
            self.dataset._task_cache_dir(self.no_treatment_task),
            self.dataset._task_cache_dir(self.dataset.default_task),
        )
        batch = self.no_treatment_batch
        self.assertEqual(batch["features"].shape, (4, 8))
        self.assertEqual(batch["event_times"].shape, (4, 2))
        for i, pid in enumerate(batch["patient_id"]):
            features, times, indicators = EXPECTED[pid]
            with self.subTest(pid=pid):
                self.assertEqual(
                    batch["features"][i].tolist(), without_treatment(features)
                )
                self.assertEqual(batch["event_times"][i].tolist(), times)
                self.assertEqual(batch["event_indicators"][i].tolist(), indicators)


if __name__ == "__main__":
    unittest.main()
