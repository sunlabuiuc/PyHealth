"""Tests for SleepStagingDREAMT task.

All tests use in-memory fake patients with small temporary CSV files.
No real DREAMT data is required. Tests complete in milliseconds.
"""

import logging
import os
import tempfile
import shutil
import unittest
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd

from pyhealth.tasks.sleep_staging_dreamt import (
    ALL_SIGNAL_COLUMNS,
    SleepStagingDREAMT,
)

EPOCH_LEN = 30 * 64  # 1920 samples


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_csv(
    n_epochs: int,
    stages: List[str],
    tmpdir: str,
    patient_id: str = "S001",
) -> str:
    """Create a synthetic 64 Hz CSV with ``n_epochs`` epochs."""
    rng = np.random.RandomState(42)
    rows = n_epochs * EPOCH_LEN
    data = {
        "TIMESTAMP": np.arange(rows) / 64.0,
        "BVP": rng.randn(rows) * 50,
        "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
        "EDA": rng.rand(rows) * 5 + 0.1,
        "TEMP": rng.rand(rows) * 15 + 28,
        "ACC_X": rng.randn(rows) * 10,
        "ACC_Y": rng.randn(rows) * 10,
        "ACC_Z": rng.randn(rows) * 10,
        "HR": rng.rand(rows) * 30 + 60,
    }

    stage_col = []
    for i in range(n_epochs):
        stage = stages[i % len(stages)]
        stage_col.extend([stage] * EPOCH_LEN)
    data["Sleep_Stage"] = stage_col

    df = pd.DataFrame(data)
    path = os.path.join(tmpdir, f"{patient_id}_whole_df.csv")
    df.to_csv(path, index=False)
    return path


def _make_patient(
    file_path: Optional[str],
    patient_id: str = "S001",
) -> SimpleNamespace:
    """Build a mock Patient mimicking DREAMTDataset."""
    event = SimpleNamespace(file_64hz=file_path)

    def get_events(event_type=None, **kwargs):
        return [event]

    return SimpleNamespace(
        patient_id=patient_id,
        get_events=get_events,
    )


class TestInit(unittest.TestCase):
    """Task initialization tests."""

    def test_default_params(self):
        """Default init uses 5 classes and all 8 channels."""
        task = SleepStagingDREAMT()
        self.assertEqual(task.n_classes, 5)
        self.assertEqual(task.signal_columns, list(ALL_SIGNAL_COLUMNS))
        self.assertEqual(task.epoch_seconds, 30.0)
        self.assertEqual(task.sampling_rate, 64)
        self.assertTrue(task.apply_filters)
        self.assertEqual(task.epoch_len, 1920)

    def test_custom_params(self):
        """Custom init parameters are stored correctly."""
        task = SleepStagingDREAMT(
            n_classes=2,
            signal_columns=["ACC_X", "ACC_Y"],
            epoch_seconds=15.0,
            sampling_rate=32,
            apply_filters=False,
        )
        self.assertEqual(task.n_classes, 2)
        self.assertEqual(task.signal_columns, ["ACC_X", "ACC_Y"])
        self.assertEqual(task.epoch_seconds, 15.0)
        self.assertEqual(task.sampling_rate, 32)
        self.assertFalse(task.apply_filters)
        self.assertEqual(task.epoch_len, 480)

    def test_invalid_n_classes_raises(self):
        """n_classes not in {2, 3, 5} raises ValueError."""
        with self.assertRaises(ValueError):
            SleepStagingDREAMT(n_classes=4)

    def test_invalid_n_classes_other(self):
        """n_classes=1 also raises ValueError."""
        with self.assertRaises(ValueError):
            SleepStagingDREAMT(n_classes=1)

    def test_class_attributes(self):
        """Task has correct class-level attributes."""
        task = SleepStagingDREAMT()
        self.assertEqual(task.task_name, "SleepStagingDREAMT")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "multiclass"})


class TestFiveClass(unittest.TestCase):
    """5-class sleep staging tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_sample_count(self):
        """Correct number of valid epochs returned."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=5, apply_filters=False)
        samples = task(patient)
        self.assertEqual(len(samples), 5)

    def test_label_mapping(self):
        """5-class maps W=0, N1=1, N2=2, N3=3, R=4."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=5, apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        self.assertEqual(labels, [0, 1, 2, 3, 4])

    def test_signal_shape(self):
        """Signal shape is (n_channels, 1920)."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=5, apply_filters=False)
        samples = task(patient)
        self.assertEqual(samples[0]["signal"].shape, (8, 1920))
        self.assertEqual(samples[0]["signal"].dtype, np.float32)

    def test_patient_id(self):
        """Samples carry the correct patient_id."""
        csv = _make_csv(
            2, ["W", "N1"], self.tmp_path, patient_id="S042"
        )
        patient = _make_patient(csv, patient_id="S042")
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        self.assertTrue(all(s["patient_id"] == "S042" for s in samples))

    def test_epoch_indices_sequential(self):
        """epoch_index is sequential starting from 0."""
        stages = ["W", "N1", "N2", "N3", "R"] * 3
        csv = _make_csv(15, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        indices = [s["epoch_index"] for s in samples]
        self.assertEqual(indices, list(range(len(samples))))


class TestThreeClass(unittest.TestCase):
    """3-class sleep staging tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_label_mapping(self):
        """3-class maps W=0, NREM=1, REM=2."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=3, apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        self.assertEqual(labels, [0, 1, 1, 1, 2])


class TestTwoClass(unittest.TestCase):
    """2-class (wake vs sleep) tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_label_mapping(self):
        """2-class maps W=0, all sleep=1."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=2, apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        self.assertEqual(labels, [0, 1, 1, 1, 1])


class TestStageExclusion(unittest.TestCase):
    """P and Missing stage exclusion tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_p_stage_excluded(self):
        """Epochs that are entirely P are skipped; W and N1 epochs kept."""
        stages = ["P", "P", "W", "N1"]
        csv = _make_csv(4, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        self.assertEqual(len(samples), 2)
        self.assertEqual([s["label"] for s in samples], [0, 1])

    def test_artifact_inside_epoch_skips_whole_window(self):
        """Any P/Missing row inside a 30 s window drops that epoch (no splicing)."""
        rng = np.random.RandomState(0)
        rows = 2 * EPOCH_LEN
        data = {
            "TIMESTAMP": np.arange(rows) / 64.0,
            "BVP": rng.randn(rows) * 50,
            "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
            "EDA": rng.rand(rows) * 5 + 0.1,
            "TEMP": rng.rand(rows) * 15 + 28,
            "ACC_X": rng.randn(rows) * 10,
            "ACC_Y": rng.randn(rows) * 10,
            "ACC_Z": rng.randn(rows) * 10,
            "HR": rng.rand(rows) * 30 + 60,
        }
        ss = []
        for i in range(2):
            st = ["W", "P"][i]
            ss.extend([st] * EPOCH_LEN)
        data["Sleep_Stage"] = ss
        path = os.path.join(self.tmp_path, "straddle.csv")
        pd.DataFrame(data).to_csv(path, index=False)
        patient = _make_patient(path, patient_id="S_STR")
        task = SleepStagingDREAMT(n_classes=2, apply_filters=False)
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 0)

    def test_missing_stage_excluded(self):
        """The epoch that is entirely Missing is skipped; others kept."""
        stages = ["W", "Missing", "N2"]
        csv = _make_csv(3, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        self.assertEqual(len(samples), 2)
        self.assertEqual([s["label"] for s in samples], [0, 2])

    def test_all_p_returns_empty(self):
        """Patient with only P stages returns empty list."""
        stages = ["P", "P", "P"]
        csv = _make_csv(3, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        self.assertEqual(samples, [])

    def test_all_missing_returns_empty(self):
        """Patient with only Missing stages returns empty list."""
        stages = ["Missing", "Missing"]
        csv = _make_csv(2, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        self.assertEqual(samples, [])


class TestTimestampSanity(unittest.TestCase):
    """TIMESTAMP vs nominal sampling rate."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_warns_when_timestamp_implies_wrong_rate(self):
        """Log a warning if TIMESTAMP span is inconsistent with 64 Hz row count."""
        rng = np.random.RandomState(1)
        rows = EPOCH_LEN
        # Same row count as nominal 30 s @ 64 Hz, but time spans 60 s -> ~32 Hz implied
        data = {
            "TIMESTAMP": np.linspace(0.0, 60.0, rows),
            "BVP": rng.randn(rows) * 50,
            "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
            "EDA": rng.rand(rows) * 5 + 0.1,
            "TEMP": rng.rand(rows) * 15 + 28,
            "ACC_X": rng.randn(rows) * 10,
            "ACC_Y": rng.randn(rows) * 10,
            "ACC_Z": rng.randn(rows) * 10,
            "HR": rng.rand(rows) * 30 + 60,
            "Sleep_Stage": ["W"] * rows,
        }
        path = os.path.join(self.tmp_path, "bad_ts.csv")
        pd.DataFrame(data).to_csv(path, index=False)
        patient = _make_patient(path, patient_id="S_TS")
        task = SleepStagingDREAMT(apply_filters=False)
        with self.assertLogs(
            "pyhealth.tasks.sleep_staging_dreamt", level=logging.WARNING
        ) as captured:
            task(patient)
        self.assertTrue(
            any("implies" in m for m in captured.output),
            captured.output,
        )


class TestSignalSubset(unittest.TestCase):
    """Signal column selection tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_acc_only(self):
        """Selecting ACC channels gives shape (3, 1920)."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["ACC_X", "ACC_Y", "ACC_Z"],
            apply_filters=False,
        )
        samples = task(patient)
        self.assertEqual(samples[0]["signal"].shape, (3, 1920))

    def test_single_channel(self):
        """Single channel gives shape (1, 1920)."""
        stages = ["W"]
        csv = _make_csv(1, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["HR"],
            apply_filters=False,
        )
        samples = task(patient)
        self.assertEqual(samples[0]["signal"].shape, (1, 1920))

    def test_bvp_temp(self):
        """Custom subset of BVP + TEMP gives shape (2, 1920)."""
        stages = ["N3"]
        csv = _make_csv(1, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["BVP", "TEMP"],
            apply_filters=False,
        )
        samples = task(patient)
        self.assertEqual(samples[0]["signal"].shape, (2, 1920))


class TestFiltering(unittest.TestCase):
    """Signal filtering tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_filters_run_without_error(self):
        """Filters execute without raising exceptions."""
        stages = ["W", "N2", "R"]
        csv = _make_csv(3, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=True)
        samples = task(patient)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0]["signal"].shape, (8, 1920))

    def test_temp_winsorization(self):
        """TEMP values are clipped to [31, 40] after filtering."""
        stages = ["W"]
        csv = _make_csv(1, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["TEMP"],
            apply_filters=True,
        )
        samples = task(patient)
        temp_signal = samples[0]["signal"][0]
        self.assertTrue(np.all(temp_signal >= 31.0))
        self.assertTrue(np.all(temp_signal <= 40.0))

    def test_filters_disabled(self):
        """With apply_filters=False, TEMP is not clipped."""
        stages = ["W"]
        csv = _make_csv(1, stages, self.tmp_path)
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["TEMP"],
            apply_filters=False,
        )
        samples = task(patient)
        temp_signal = samples[0]["signal"][0]
        has_below = np.any(temp_signal < 31.0)
        has_above = np.any(temp_signal > 40.0)
        self.assertTrue(has_below or has_above)


class TestEdgeCases(unittest.TestCase):
    """Edge case handling tests."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_empty_patient_no_file(self):
        """Patient with no file returns empty list."""
        patient = _make_patient(None, patient_id="S_EMPTY")
        task = SleepStagingDREAMT()
        samples = task(patient)
        self.assertEqual(samples, [])

    def test_empty_patient_none_string(self):
        """Patient with file_64hz='None' returns empty list."""
        patient = _make_patient("None", patient_id="S_NONE")
        task = SleepStagingDREAMT()
        samples = task(patient)
        self.assertEqual(samples, [])

    def test_multi_patient_isolation(self):
        """Each patient's samples reference only its own id."""
        csv_a = _make_csv(
            3, ["W", "N1", "N2"], self.tmp_path, patient_id="P_A"
        )
        csv_b = _make_csv(
            2, ["N3", "R"], self.tmp_path, patient_id="P_B"
        )
        patient_a = _make_patient(csv_a, patient_id="P_A")
        patient_b = _make_patient(csv_b, patient_id="P_B")

        task = SleepStagingDREAMT(apply_filters=False)
        samples_a = task(patient_a)
        samples_b = task(patient_b)

        self.assertTrue(all(s["patient_id"] == "P_A" for s in samples_a))
        self.assertTrue(all(s["patient_id"] == "P_B" for s in samples_b))

        if samples_a:
            self.assertEqual(samples_a[0]["epoch_index"], 0)
        if samples_b:
            self.assertEqual(samples_b[0]["epoch_index"], 0)

    def test_nonexistent_file(self):
        """Nonexistent CSV path returns empty list."""
        fake_path = os.path.join(self.tmp_path, "does_not_exist.csv")
        patient = _make_patient(fake_path)
        task = SleepStagingDREAMT()
        samples = task(patient)
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()
