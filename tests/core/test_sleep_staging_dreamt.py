"""Tests for SleepStagingDREAMT task.

All tests use in-memory fake patients with small temporary CSV files.
No real DREAMT data is required. Tests complete in milliseconds.
"""

import os
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

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
    """Create a synthetic 64 Hz CSV with ``n_epochs`` epochs.

    Args:
        n_epochs: Number of 30-second epochs to generate.
        stages: Sleep stage labels, one per epoch (cycled).
        tmpdir: Directory to write the CSV into.
        patient_id: Used in the filename.

    Returns:
        Absolute path to the written CSV file.
    """
    rng = np.random.RandomState(42)
    rows = n_epochs * EPOCH_LEN
    data = {
        "TIMESTAMP": np.arange(rows) / 64.0,
        "BVP": rng.randn(rows) * 50,
        "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
        "EDA": rng.rand(rows) * 5 + 0.1,
        "TEMP": rng.rand(rows) * 15 + 28,  # range includes <31 and >40
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
    """Build a mock Patient mimicking DREAMTDataset.

    Args:
        file_path: Path to the CSV, or None for an empty patient.
        patient_id: Identifier for the mock patient.

    Returns:
        SimpleNamespace matching the DREAMT patient contract.
    """
    event = SimpleNamespace(file_64hz=file_path)
    patient = SimpleNamespace(
        patient_id=patient_id,
        get_events=lambda event_type=None: [event],
    )
    return patient


# -----------------------------------------------------------
# Tests — Initialization
# -----------------------------------------------------------


class TestInit:
    """Task initialization tests."""

    def test_default_params(self):
        """Default init uses 5 classes and all 8 channels."""
        task = SleepStagingDREAMT()
        assert task.n_classes == 5
        assert task.signal_columns == list(ALL_SIGNAL_COLUMNS)
        assert task.epoch_seconds == 30.0
        assert task.sampling_rate == 64
        assert task.apply_filters is True
        assert task.epoch_len == 1920

    def test_custom_params(self):
        """Custom init parameters are stored correctly."""
        task = SleepStagingDREAMT(
            n_classes=2,
            signal_columns=["ACC_X", "ACC_Y"],
            epoch_seconds=15.0,
            sampling_rate=32,
            apply_filters=False,
        )
        assert task.n_classes == 2
        assert task.signal_columns == ["ACC_X", "ACC_Y"]
        assert task.epoch_seconds == 15.0
        assert task.sampling_rate == 32
        assert task.apply_filters is False
        assert task.epoch_len == 480

    def test_invalid_n_classes_raises(self):
        """n_classes not in {2, 3, 5} raises ValueError."""
        with pytest.raises(ValueError, match="n_classes"):
            SleepStagingDREAMT(n_classes=4)

    def test_invalid_n_classes_other(self):
        """n_classes=1 also raises ValueError."""
        with pytest.raises(ValueError):
            SleepStagingDREAMT(n_classes=1)

    def test_class_attributes(self):
        """Task has correct class-level attributes."""
        task = SleepStagingDREAMT()
        assert task.task_name == "SleepStagingDREAMT"
        assert task.input_schema == {"signal": "tensor"}
        assert task.output_schema == {"label": "multiclass"}


# -----------------------------------------------------------
# Tests — 5-class
# -----------------------------------------------------------


class TestFiveClass:
    """5-class sleep staging tests."""

    def test_sample_count(self, tmp_path):
        """Correct number of valid epochs returned."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=5, apply_filters=False)
        samples = task(patient)
        assert len(samples) == 5

    def test_label_mapping(self, tmp_path):
        """5-class maps W=0, N1=1, N2=2, N3=3, R=4."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=5, apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        assert labels == [0, 1, 2, 3, 4]

    def test_signal_shape(self, tmp_path):
        """Signal shape is (n_channels, 1920)."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=5, apply_filters=False)
        samples = task(patient)
        assert samples[0]["signal"].shape == (8, 1920)
        assert samples[0]["signal"].dtype == np.float32

    def test_patient_id(self, tmp_path):
        """Samples carry the correct patient_id."""
        csv = _make_csv(
            2, ["W", "N1"], str(tmp_path), patient_id="S042"
        )
        patient = _make_patient(csv, patient_id="S042")
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        assert all(s["patient_id"] == "S042" for s in samples)

    def test_epoch_indices_sequential(self, tmp_path):
        """epoch_index is sequential starting from 0."""
        stages = ["W", "N1", "N2", "N3", "R"] * 3
        csv = _make_csv(15, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        indices = [s["epoch_index"] for s in samples]
        assert indices == list(range(len(samples)))


# -----------------------------------------------------------
# Tests — 3-class
# -----------------------------------------------------------


class TestThreeClass:
    """3-class sleep staging tests."""

    def test_label_mapping(self, tmp_path):
        """3-class maps W=0, NREM=1, REM=2."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=3, apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        assert labels == [0, 1, 1, 1, 2]


# -----------------------------------------------------------
# Tests — 2-class
# -----------------------------------------------------------


class TestTwoClass:
    """2-class (wake vs sleep) tests."""

    def test_label_mapping(self, tmp_path):
        """2-class maps W=0, all sleep=1."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(n_classes=2, apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        assert labels == [0, 1, 1, 1, 1]


# -----------------------------------------------------------
# Tests — Stage exclusion
# -----------------------------------------------------------


class TestStageExclusion:
    """P and Missing stage exclusion tests."""

    def test_p_stage_excluded(self, tmp_path):
        """Epochs with P (preparation) stage are dropped."""
        stages = ["P", "P", "W", "N1"]
        csv = _make_csv(4, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        # After dropping P rows, remaining data may yield fewer epochs
        # All returned labels should be valid (no P)
        assert all(lbl in {0, 1, 2, 3, 4} for lbl in labels)

    def test_missing_stage_excluded(self, tmp_path):
        """Epochs with Missing stage are dropped."""
        stages = ["W", "Missing", "N2"]
        csv = _make_csv(3, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        # Missing rows are dropped, so we may get fewer epochs
        for s in samples:
            assert s["label"] in {0, 1, 2, 3, 4}

    def test_all_p_returns_empty(self, tmp_path):
        """Patient with only P stages returns empty list."""
        stages = ["P", "P", "P"]
        csv = _make_csv(3, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        assert samples == []

    def test_all_missing_returns_empty(self, tmp_path):
        """Patient with only Missing stages returns empty list."""
        stages = ["Missing", "Missing"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=False)
        samples = task(patient)
        assert samples == []


# -----------------------------------------------------------
# Tests — Signal column subsetting
# -----------------------------------------------------------


class TestSignalSubset:
    """Signal column selection tests."""

    def test_acc_only(self, tmp_path):
        """Selecting ACC channels gives shape (3, 1920)."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["ACC_X", "ACC_Y", "ACC_Z"],
            apply_filters=False,
        )
        samples = task(patient)
        assert samples[0]["signal"].shape == (3, 1920)

    def test_single_channel(self, tmp_path):
        """Single channel gives shape (1, 1920)."""
        stages = ["W"]
        csv = _make_csv(1, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["HR"],
            apply_filters=False,
        )
        samples = task(patient)
        assert samples[0]["signal"].shape == (1, 1920)

    def test_bvp_temp(self, tmp_path):
        """Custom subset of BVP + TEMP gives shape (2, 1920)."""
        stages = ["N3"]
        csv = _make_csv(1, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["BVP", "TEMP"],
            apply_filters=False,
        )
        samples = task(patient)
        assert samples[0]["signal"].shape == (2, 1920)


# -----------------------------------------------------------
# Tests — Filtering
# -----------------------------------------------------------


class TestFiltering:
    """Signal filtering tests."""

    def test_filters_run_without_error(self, tmp_path):
        """Filters execute without raising exceptions."""
        stages = ["W", "N2", "R"]
        csv = _make_csv(3, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(apply_filters=True)
        samples = task(patient)
        assert len(samples) > 0
        assert samples[0]["signal"].shape == (8, 1920)

    def test_temp_winsorization(self, tmp_path):
        """TEMP values are clipped to [31, 40] after filtering."""
        stages = ["W"]
        csv = _make_csv(1, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["TEMP"],
            apply_filters=True,
        )
        samples = task(patient)
        temp_signal = samples[0]["signal"][0]
        assert np.all(temp_signal >= 31.0)
        assert np.all(temp_signal <= 40.0)

    def test_filters_disabled(self, tmp_path):
        """With apply_filters=False, TEMP is not clipped."""
        # Generate TEMP data that will have values outside [31, 40]
        stages = ["W"]
        csv = _make_csv(1, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStagingDREAMT(
            signal_columns=["TEMP"],
            apply_filters=False,
        )
        samples = task(patient)
        temp_signal = samples[0]["signal"][0]
        # Synthetic data has range ~[28, 43], so some should be outside
        has_below = np.any(temp_signal < 31.0)
        has_above = np.any(temp_signal > 40.0)
        assert has_below or has_above


# -----------------------------------------------------------
# Tests — Edge cases
# -----------------------------------------------------------


class TestEdgeCases:
    """Edge case handling tests."""

    def test_empty_patient_no_file(self):
        """Patient with no file returns empty list."""
        patient = _make_patient(None, patient_id="S_EMPTY")
        task = SleepStagingDREAMT()
        samples = task(patient)
        assert samples == []

    def test_empty_patient_none_string(self, tmp_path):
        """Patient with file_64hz='None' returns empty list."""
        patient = _make_patient("None", patient_id="S_NONE")
        task = SleepStagingDREAMT()
        samples = task(patient)
        assert samples == []

    def test_multi_patient_isolation(self, tmp_path):
        """Each patient's samples reference only its own id."""
        csv_a = _make_csv(
            3, ["W", "N1", "N2"], str(tmp_path), patient_id="P_A"
        )
        csv_b = _make_csv(
            2, ["N3", "R"], str(tmp_path), patient_id="P_B"
        )
        patient_a = _make_patient(csv_a, patient_id="P_A")
        patient_b = _make_patient(csv_b, patient_id="P_B")

        task = SleepStagingDREAMT(apply_filters=False)
        samples_a = task(patient_a)
        samples_b = task(patient_b)

        assert all(s["patient_id"] == "P_A" for s in samples_a)
        assert all(s["patient_id"] == "P_B" for s in samples_b)

        # epoch_index restarts at 0 for each patient
        if samples_a:
            assert samples_a[0]["epoch_index"] == 0
        if samples_b:
            assert samples_b[0]["epoch_index"] == 0

    def test_nonexistent_file(self, tmp_path):
        """Nonexistent CSV path returns empty list."""
        fake_path = os.path.join(str(tmp_path), "does_not_exist.csv")
        patient = _make_patient(fake_path)
        task = SleepStagingDREAMT()
        samples = task(patient)
        assert samples == []
