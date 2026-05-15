"""Tests for SleepWakeDetectionDREAMT task using synthetic raw signal data."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pyhealth.tasks.sleep_wake_detection_dreamt import (
    EPOCH_SAMPLES,
    FEATURE_COLUMNS,
    EXCLUDE_LABELS,
    WAKE_LABEL,
    SleepWakeDetectionDREAMT,
    extract_epoch_features,
)


def _make_synthetic_raw_csv(path: str, n_epochs: int = 5) -> None:
    """Write a synthetic raw 64Hz DREAMT CSV with n_epochs worth of data.

    Generates realistic column structure matching the real DREAMT 2.1.0
    data_64Hz files. Each epoch is EPOCH_SAMPLES (1920) rows.

    Args:
        path: File path to write the CSV to.
        n_epochs: Number of 30-second epochs to generate.
    """
    n_rows = n_epochs * EPOCH_SAMPLES
    stages = ["W", "N1", "N2", "N3", "R"]

    # Repeat each stage for a full epoch
    sleep_stages = []
    for i in range(n_epochs):
        sleep_stages.extend([stages[i % len(stages)]] * EPOCH_SAMPLES)

    df = pd.DataFrame({
        "TIMESTAMP": np.linspace(0, n_epochs * 30, n_rows),
        "BVP": np.random.randn(n_rows) * 2,
        "ACC_X": np.random.randn(n_rows) * 10 + 30,
        "ACC_Y": np.random.randn(n_rows) * 5 + 8,
        "ACC_Z": np.random.randn(n_rows) * 8 + 55,
        "TEMP": np.random.uniform(34, 37, n_rows),
        "EDA": np.random.uniform(0.01, 0.5, n_rows),
        "HR": np.random.uniform(50, 90, n_rows),
        "IBI": np.random.uniform(600, 1200, n_rows),
        "Sleep_Stage": sleep_stages,
    })
    df.to_csv(path, index=False)


def _make_synthetic_raw_csv_with_stages(path: str, stages: list) -> None:
    """Write a synthetic CSV with specific sleep stages, one epoch each.

    Args:
        path: File path to write the CSV to.
        stages: List of sleep stage strings, one per epoch.
    """
    n_rows = len(stages) * EPOCH_SAMPLES
    stage_col = []
    for s in stages:
        stage_col.extend([s] * EPOCH_SAMPLES)

    df = pd.DataFrame({
        "TIMESTAMP": np.linspace(0, len(stages) * 30, n_rows),
        "BVP": np.random.randn(n_rows),
        "ACC_X": np.random.randn(n_rows) * 10,
        "ACC_Y": np.random.randn(n_rows) * 5,
        "ACC_Z": np.random.randn(n_rows) * 8,
        "TEMP": np.random.uniform(34, 37, n_rows),
        "EDA": np.random.uniform(0.01, 0.5, n_rows),
        "HR": np.random.uniform(50, 90, n_rows),
        "IBI": np.random.uniform(600, 1200, n_rows),
        "Sleep_Stage": stage_col,
    })
    df.to_csv(path, index=False)


def _make_patient(feature_csv_path: str, patient_id: str = "S001") -> MagicMock:
    """Build a mock patient object that mimics DREAMTDataset output.

    Args:
        feature_csv_path: Path to the synthetic CSV file.
        patient_id: Patient identifier string.

    Returns:
        MagicMock patient with dreamt_sleep event attributes.
    """
    event = MagicMock()
    event.file_64hz = feature_csv_path
    event.ahi = 22.0
    event.bmi = 33.5

    patient = MagicMock()
    patient.patient_id = patient_id
    patient.get_events.return_value = [event]
    return patient


class TestSleepWakeDetectionDREAMT:

    def test_instantiation_default(self):
        """Task instantiates with correct name, schema, and default features."""
        task = SleepWakeDetectionDREAMT()
        assert task.task_name == "SleepWakeDetectionDREAMT"
        assert task.feature_columns == FEATURE_COLUMNS
        assert task.input_schema == {"features": "tensor"}
        assert task.output_schema == {"label": "binary"}

    def test_instantiation_custom_features(self):
        """Task accepts a custom feature column subset."""
        cols = ["ACC_INDEX", "HRV_HFD"]
        task = SleepWakeDetectionDREAMT(feature_columns=cols)
        assert task.feature_columns == cols

    def test_returns_correct_number_of_samples(self, tmp_path):
        """Each non-excluded epoch produces exactly one sample."""
        csv_path = str(tmp_path / "S001_whole_df.csv")
        # 5 epochs: W, N1, N2, N3, R — all valid, none excluded
        _make_synthetic_raw_csv(csv_path, n_epochs=5)
        patient = _make_patient(csv_path)

        task = SleepWakeDetectionDREAMT()
        samples = task(patient)

        assert len(samples) == 5

    def test_sample_schema(self, tmp_path):
        """Every sample contains all required keys with correct types."""
        csv_path = str(tmp_path / "S001_whole_df.csv")
        _make_synthetic_raw_csv(csv_path, n_epochs=3)
        patient = _make_patient(csv_path)

        task = SleepWakeDetectionDREAMT()
        samples = task(patient)

        for s in samples:
            assert "patient_id" in s
            assert "epoch_index" in s
            assert "ahi" in s
            assert "bmi" in s
            assert "features" in s
            assert "label" in s
            assert s["label"] in (0, 1)
            assert len(s["features"]) == len(FEATURE_COLUMNS)
            assert isinstance(s["features"], list)

    def test_binary_label_conversion(self, tmp_path):
        """Wake epoch → label 1; sleep epochs → label 0."""
        csv_path = str(tmp_path / "S001_whole_df.csv")
        _make_synthetic_raw_csv_with_stages(csv_path, ["W", "N1", "N2"])
        patient = _make_patient(csv_path)

        task = SleepWakeDetectionDREAMT()
        samples = task(patient)

        assert len(samples) == 3
        assert samples[0]["label"] == 1  # W → wake
        assert samples[1]["label"] == 0  # N1 → sleep
        assert samples[2]["label"] == 0  # N2 → sleep

    def test_preparation_and_missing_epochs_excluded(self, tmp_path):
        """P and Missing labels are dropped from output samples."""
        csv_path = str(tmp_path / "S002_whole_df.csv")
        _make_synthetic_raw_csv_with_stages(csv_path, ["P", "W", "Missing", "N2"])
        patient = _make_patient(csv_path, patient_id="S002")

        task = SleepWakeDetectionDREAMT()
        samples = task(patient)

        # Only W and N2 survive
        assert len(samples) == 2
        assert samples[0]["label"] == 1  # W → wake
        assert samples[1]["label"] == 0  # N2 → sleep

    def test_missing_file_returns_empty(self):
        """Nonexistent file path returns empty list without crashing."""
        patient = _make_patient("/nonexistent/path.csv")
        task = SleepWakeDetectionDREAMT()
        assert task(patient) == []

    def test_no_events_returns_empty(self):
        """Patient with no dreamt_sleep events returns empty list."""
        patient = MagicMock()
        patient.patient_id = "S999"
        patient.get_events.return_value = []
        task = SleepWakeDetectionDREAMT()
        assert task(patient) == []

    def test_none_file_path_returns_empty(self):
        """file_64hz = None returns empty list without crashing."""
        patient = _make_patient(None)
        task = SleepWakeDetectionDREAMT()
        assert task(patient) == []

    def test_custom_feature_subset(self, tmp_path):
        """Custom feature subset returns correct feature vector length."""
        csv_path = str(tmp_path / "S003_whole_df.csv")
        _make_synthetic_raw_csv(csv_path, n_epochs=2)
        patient = _make_patient(csv_path, patient_id="S003")

        cols = ["ACC_INDEX", "HRV_HFD"]
        task = SleepWakeDetectionDREAMT(feature_columns=cols)
        samples = task(patient)

        assert len(samples) == 2
        assert len(samples[0]["features"]) == 2

    def test_extract_epoch_features_output_length(self, tmp_path):
        """extract_epoch_features returns vector matching FEATURE_COLUMNS."""
        csv_path = str(tmp_path / "S001_whole_df.csv")
        _make_synthetic_raw_csv(csv_path, n_epochs=1)
        df = pd.read_csv(csv_path)
        epoch = df.iloc[:EPOCH_SAMPLES]

        features = extract_epoch_features(epoch)
        assert len(features) == len(FEATURE_COLUMNS)
        assert all(isinstance(f, float) for f in features)