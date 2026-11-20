"""Tests for PBCSeqDataset and the LongitudinalLiverSurvival task."""

import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import PatientSplit, PBCSeqDataset, get_dataloader
from pyhealth.tasks import LongitudinalLiverSurvival
from pyhealth.tasks.liver_survival import DAYS_PER_YEAR, FEATURE_SETS

COLUMNS = [
    "id", "futime", "status", "trt", "age", "sex", "day", "ascites", "hepato",
    "spiders", "edema", "bili", "chol", "albumin", "alk.phos", "ast", "platelet",
    "protime", "stage",
]  # fmt: skip


def _visit(pid, futime, status, sex, day, bili, ast, stage, chol=200.0):
    """One pbcseq row; columns not under test get fixed plausible values."""
    return [pid, futime, status, 1, 50.0 + pid, sex, day, 0, 1, 0, 0.0, bili,
            chol, 3.5, 1000.0, ast, 250.0, 10.5, stage]  # fmt: skip


def _fixture_rows():
    """Five synthetic patients in the R ``pbcseq`` layout.

    1: died on day 913 (visits 0, 182, 365, 730), stage 4 then 4.
    2: censored on day 1461 (yearly visits), stage 2 then 3.
    3: transplanted on day 548, stage 3.
    4: one visit, censored on day 292, stage 1.
    5: died on day 2191; AST and cholesterol missing at day 365, stage 4.
    """
    nan = float("nan")
    return [
        _visit(1, 913, 2, "f", 0, 1.0, 100.0, 4),
        _visit(1, 913, 2, "f", 182, 1.5, 110.0, 4),
        _visit(1, 913, 2, "f", 365, 2.0, 120.0, 4),
        _visit(1, 913, 2, "f", 730, 3.0, 130.0, 4),
        _visit(2, 1461, 0, "m", 0, 0.5, 50.0, 2),
        _visit(2, 1461, 0, "m", 365, 0.6, 55.0, 2),
        _visit(2, 1461, 0, "m", 730, 0.7, 60.0, 3),
        _visit(2, 1461, 0, "m", 1095, 0.8, 65.0, 3),
        _visit(3, 548, 1, "f", 0, 4.0, 200.0, 3),
        _visit(3, 548, 1, "f", 182, 5.0, 210.0, 3),
        _visit(3, 548, 1, "f", 365, 6.0, 220.0, 3),
        _visit(4, 292, 0, "m", 0, 1.2, 80.0, 1),
        _visit(5, 2191, 2, "f", 0, 0.9, 90.0, 4),
        _visit(5, 2191, 2, "f", 365, 1.1, nan, 4, chol=nan),
        _visit(5, 2191, 2, "f", 730, 1.3, 95.0, 4),
    ]


def _write_fixture(root: Path, rows=None, columns=None) -> None:
    frame = pd.DataFrame(rows or _fixture_rows(), columns=columns or COLUMNS)
    frame.to_csv(root / "pbcseq.csv", index=False)


def _by_patient(samples):
    grouped = {}
    for sample in samples:
        grouped.setdefault(str(sample["patient_id"]), []).append(sample)
    return grouped


_TMP: Path | None = None
_DATASET: PBCSeqDataset | None = None


def setUpModule():
    """Builds the fixture dataset once for the whole module.

    Loading a PyHealth dataset converts the CSV to Parquet and builds the event
    table (about one second), so it happens here once, eagerly, instead of
    inside individual tests.
    """
    global _TMP, _DATASET
    _TMP = Path(tempfile.mkdtemp())
    _write_fixture(_TMP)
    _DATASET = PBCSeqDataset(root=str(_TMP), cache_dir=_TMP / "cache")
    _DATASET.unique_patient_ids  # noqa: B018 - load the event table now


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


class _FixtureCase(unittest.TestCase):
    """Gives each test class the shared fixture dataset."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _DATASET


class TestPBCSeqDataset(_FixtureCase):
    def test_loads_all_patients_and_visits(self):
        self.assertEqual(
            sorted(self.dataset.unique_patient_ids, key=int), list("12345")
        )
        visits = self.dataset.get_patient("1").get_events(event_type="pbcseq")
        self.assertEqual(len(visits), 4)

    def test_visits_dated_from_anchor(self):
        visits = self.dataset.get_patient("1").get_events(event_type="pbcseq")
        self.assertEqual(visits[0].timestamp, datetime(1974, 1, 1))
        self.assertEqual(visits[1].timestamp, datetime(1974, 7, 2))
        self.assertEqual(visits[1]["day"], "182")

    def test_alk_phos_renamed(self):
        frame = self.dataset.get_patient("1").get_events(
            event_type="pbcseq", return_df=True
        )
        self.assertIn("pbcseq/alk_phos", frame.columns)

    def test_default_name_and_table(self):
        self.assertEqual(self.dataset.dataset_name, "pbcseq")
        self.assertEqual(self.dataset.tables, ["pbcseq"])

    def test_missing_column_raises(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = [row[:-1] for row in _fixture_rows()]
            _write_fixture(tmp, rows=rows, columns=COLUMNS[:-1])
            dataset = PBCSeqDataset(root=str(tmp), cache_dir=tmp / "cache")
            with self.assertRaises(ValueError):
                dataset.unique_patient_ids  # noqa: B018 - triggers loading
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCountingProcessTask(_FixtureCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.task = LongitudinalLiverSurvival()
        cls.samples = [
            sample
            for pid in sorted(cls.dataset.unique_patient_ids, key=int)
            for sample in cls.task(cls.dataset.get_patient(pid))
        ]
        cls.by_patient = _by_patient(cls.samples)

    def test_schema(self):
        self.assertEqual(
            self.task.output_schema, {"survival_interval": "survival_interval"}
        )
        self.assertEqual(self.task.feature_names, FEATURE_SETS["mas4"])
        processor, options = self.task.input_schema["features"]
        self.assertEqual(processor, "tensor")
        self.assertEqual(options["normalization"], "zscore")

    def test_one_sample_per_visit(self):
        self.assertEqual(len(self.samples), 15)
        self.assertEqual(
            {pid: len(rows) for pid, rows in self.by_patient.items()},
            {"1": 4, "2": 4, "3": 3, "4": 1, "5": 3},
        )

    def test_intervals_of_patient_who_died(self):
        intervals = np.array([s["survival_interval"] for s in self.by_patient["1"]])
        expected = np.array([[0, 182, 0], [182, 365, 0], [365, 730, 0], [730, 913, 1]])
        expected = expected / [DAYS_PER_YEAR, DAYS_PER_YEAR, 1]
        np.testing.assert_allclose(intervals, expected)

    def test_transplant_censored_by_default(self):
        self.assertEqual(self.by_patient["3"][-1]["survival_interval"][2], 0)

    def test_features_from_starting_visit_in_feature_order(self):
        features = [s["features"] for s in self.by_patient["1"]]
        self.assertEqual([f[0] for f in features], [1.0, 1.5, 2.0, 3.0])  # bili
        self.assertEqual([f[2] for f in features], [100.0, 110.0, 120.0, 130.0])  # ast

    def test_missing_lab_forward_filled(self):
        self.assertEqual(
            [s["features"][2] for s in self.by_patient["5"]], [90.0, 90.0, 95.0]
        )

    def test_baseline_stage_is_first_visit_stage(self):
        self.assertEqual(self.by_patient["2"][-1]["baseline_stage"], 2)
        self.assertEqual(self.by_patient["4"][0]["baseline_stage"], 1)

    def test_landmark_time_is_interval_start(self):
        for sample in self.samples:
            self.assertEqual(sample["landmark_time"], sample["survival_interval"][0])


class TestTaskOptions(_FixtureCase):
    def _samples(self, **kwargs):
        task = LongitudinalLiverSurvival(**kwargs)
        return _by_patient(
            task(self.dataset.get_patient("1")) + task(self.dataset.get_patient("3"))
        )

    def test_landmark_mode(self):
        task = LongitudinalLiverSurvival(mode="landmark")
        self.assertEqual(task.output_schema, {"survival": "survival"})
        rows = self._samples(mode="landmark")["1"]
        durations = [row["survival"][0] for row in rows]
        np.testing.assert_allclose(
            durations, np.array([913, 731, 548, 183]) / DAYS_PER_YEAR
        )
        self.assertEqual({row["survival"][1] for row in rows}, {1})

    def test_composite_event_counts_transplant(self):
        rows = self._samples(event="death_or_transplant")["3"]
        self.assertEqual(rows[-1]["survival_interval"][2], 1)

    def test_named_and_custom_feature_sets(self):
        full = self._samples(feature_set="full")["1"][0]
        self.assertEqual(len(full["features"]), len(FEATURE_SETS["full"]))
        sex = FEATURE_SETS["full"].index("sex")
        self.assertEqual(full["features"][sex], 1.0)  # "f" -> 1.0
        custom = LongitudinalLiverSurvival(feature_set=["bili", "chol"])
        self.assertEqual(custom.feature_names, ("bili", "chol"))
        self.assertNotIn("stage", FEATURE_SETS["full"])

    def test_log_transform(self):
        raw = self._samples()["1"][0]["features"]
        logged = self._samples(log_transform=True)["1"][0]["features"]
        np.testing.assert_allclose(logged, np.log(raw))
        only_bili = self._samples(log_transform=["bili"])["1"][0]["features"]
        np.testing.assert_allclose(only_bili, [np.log(raw[0]), *raw[1:]])
        mayo = LongitudinalLiverSurvival(feature_set="mayo", log_transform=True)
        self.assertNotIn("age", mayo.log_features)
        with self.assertRaises(ValueError):
            LongitudinalLiverSurvival(log_transform=["creatinine"])

    def test_invalid_arguments(self):
        for kwargs in (
            {"mode": "history"},
            {"feature_set": "mas6"},
            {"feature_set": []},
            {"event": "graft_failure"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LongitudinalLiverSurvival(**kwargs)

    def test_unknown_feature_column_raises(self):
        task = LongitudinalLiverSurvival(feature_set=["bili", "creatinine"])
        with self.assertRaises(ValueError):
            task(self.dataset.get_patient("1"))


class TestAblationOptions(_FixtureCase):
    """Options used by the E1 (missingness) and E3 (binary label) ablations."""

    def _rows(self, pid, **kwargs):
        return LongitudinalLiverSurvival(**kwargs)(self.dataset.get_patient(pid))

    def test_binary_horizon_labels_and_censoring_drop(self):
        task = LongitudinalLiverSurvival(mode="binary_horizon", horizon=1.5)
        self.assertEqual(task.output_schema, {"label": "binary"})
        died = self._rows("1", mode="binary_horizon", horizon=1.5)
        # Years left after each visit: 2.50, 2.00, 1.50 (548 days), 0.50 (died).
        self.assertEqual([r["label"] for r in died], [0, 0, 0, 1])
        transplanted = self._rows("3", mode="binary_horizon", horizon=1.5)
        # Only the day-0 visit is followed past 1.5 years; the rest are dropped.
        self.assertEqual([r["visit_id"] for r in transplanted], ["3_0"])
        self.assertEqual(transplanted[0]["label"], 0)
        censored = self._rows("2", mode="binary_horizon", horizon=1.5)
        self.assertEqual([r["label"] for r in censored], [0, 0, 0])  # last dropped

    def test_horizon_argument_checks(self):
        with self.assertRaises(ValueError):
            LongitudinalLiverSurvival(mode="binary_horizon")
        with self.assertRaises(ValueError):
            LongitudinalLiverSurvival(mode="binary_horizon", horizon=0)
        with self.assertRaises(ValueError):
            LongitudinalLiverSurvival(horizon=2.0)

    def test_forward_fill_off_leaves_missing_values(self):
        filled = self._rows("5")
        unfilled = self._rows("5", forward_fill=False)
        self.assertEqual(filled[1]["features"][2], 90.0)  # AST carried forward
        self.assertTrue(np.isnan(unfilled[1]["features"][2]))

    def test_inject_missing_masks_follow_up_values_reproducibly(self):
        kwargs = {"inject_missing": {"ast": 0.99}, "inject_seed": 3}
        masked = self._rows("1", forward_fill=False, **kwargs)
        ast = [r["features"][2] for r in masked]
        self.assertEqual(ast[0], 100.0)  # day 0 is never masked
        self.assertTrue(all(np.isnan(v) for v in ast[1:]))
        bili = [r["features"][0] for r in masked]
        self.assertEqual(bili, [1.0, 1.5, 2.0, 3.0])  # other labs untouched
        again = [
            r["features"][2] for r in self._rows("1", forward_fill=False, **kwargs)
        ]
        np.testing.assert_array_equal(ast, again)
        filled = [r["features"][2] for r in self._rows("1", **kwargs)]
        self.assertEqual(filled, [100.0] * 4)  # forward-filled from day 0

    def test_inject_missing_argument_checks(self):
        with self.assertRaises(ValueError):
            LongitudinalLiverSurvival(inject_missing={"creatinine": 0.5})
        with self.assertRaises(ValueError):
            LongitudinalLiverSurvival(inject_missing={"ast": 1.0})


class TestSetTaskWithSplit(_FixtureCase):
    """Processors are fitted on the training patients only."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.task = LongitudinalLiverSurvival()
        cls.train, cls.test = cls.dataset.set_task(
            cls.task, split=PatientSplit(ratios=(0.6, 0.4), seed=0)
        )

    def test_patients_do_not_overlap(self):
        train_ids = {self.train[i]["patient_id"] for i in range(len(self.train))}
        test_ids = {self.test[i]["patient_id"] for i in range(len(self.test))}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(self.train) + len(self.test), 15)

    def test_normalization_uses_training_patients(self):
        batch = next(iter(get_dataloader(self.train, batch_size=len(self.train))))
        torch.testing.assert_close(
            batch["features"].mean(0), torch.zeros(4), atol=1e-5, rtol=0
        )
        self.assertEqual(batch["survival_interval"].shape, (len(self.train), 3))
        test_batch = next(iter(get_dataloader(self.test, batch_size=len(self.test))))
        self.assertFalse(
            torch.allclose(test_batch["features"].mean(0), torch.zeros(4), atol=1e-3)
        )

    def test_passthrough_keys(self):
        sample = self.train[0]
        for key in ("patient_id", "visit_id", "landmark_time", "baseline_stage"):
            self.assertIn(key, sample)


if __name__ == "__main__":
    unittest.main()
