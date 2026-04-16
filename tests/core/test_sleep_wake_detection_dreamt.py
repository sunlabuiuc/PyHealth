"""Tests for SleepWakeDetectionDREAMT task using synthetic data."""

from unittest.mock import MagicMock

import pandas as pd

from pyhealth.tasks.sleep_wake_detection_dreamt import (
    FEATURE_COLUMNS,
    SleepWakeDetectionDREAMT,
)


def _make_synthetic_feature_csv(path: str, n_epochs: int = 10) -> None:
    """Create a synthetic DREAMT feature CSV for testing."""
    data = {col: [float(i) for i in range(n_epochs)] for col in FEATURE_COLUMNS}

    stages = ["W", "N1", "N2", "N3", "R"]
    data["Sleep_Stage"] = [stages[i % len(stages)] for i in range(n_epochs)]

    pd.DataFrame(data).to_csv(path, index=False)


def _make_patient(feature_csv_path: str, patient_id: str = "S001") -> MagicMock:
    """Create a mock patient object compatible with DREAMTDataset."""
    event = MagicMock()
    event.file_64hz = feature_csv_path
    event.ahi = 22.0
    event.bmi = 33.5

    patient = MagicMock()
    patient.patient_id = patient_id
    patient.get_events.return_value = [event]
    return patient


class TestSleepWakeDetectionDREAMT:
    """
    Unit tests for the SleepWakeDetectionDREAMT task.

    This test suite validates the functionality of the SleepWakeDetectionDREAMT
    task, including:

    - Correct instantiation with default and custom feature sets.
    - Proper handling of synthetic patient data.
    - Accurate binary label conversion for wake/sleep classification.
    - Handling of missing or invalid data.

    Methods:
        test_task_name_and_schema: Validates task metadata and schema definitions.
        test_instantiation_default: Tests default initialization of the task.
        test_instantiation_custom_features: Tests initialization with custom feature columns.
        test_returns_correct_number_of_samples: Ensures the correct number of samples is returned.
        test_sample_schema: Validates the schema of returned samples.
        test_binary_label_conversion: Verifies wake/sleep label conversion logic.
        test_missing_file_returns_empty: Ensures missing files result in empty output.
        test_no_events_returns_empty: Ensures patients with no events return empty output.
        test_custom_feature_subset: Ensures subset of features is handled correctly.
        test_preparation_and_missing_epochs_excluded: Ensures invalid epochs are removed during preprocessing.
    """

    def test_task_name_and_schema(self):
        """Validate task metadata and schema definitions."""
        task = SleepWakeDetectionDREAMT()

        assert task.task_name == "SleepWakeDetectionDREAMT"
        assert "features" in task.input_schema
        assert "label" in task.output_schema
        assert task.output_schema["label"] == "binary"

    def test_instantiation_default(self):
        """Test default initialization."""
        task = SleepWakeDetectionDREAMT()
        assert task.task_name == "SleepWakeDetectionDREAMT"
        assert task.feature_columns == FEATURE_COLUMNS

    def test_instantiation_custom_features(self):
        """Test initialization with custom feature columns."""
        cols = ["ACC_INDEX", "HRV_HFD"]
        task = SleepWakeDetectionDREAMT(feature_columns=cols)
        assert task.feature_columns == cols

    def test_returns_correct_number_of_samples(self, tmp_path):
        """Ensure correct number of samples is returned."""
        csv_path = str(tmp_path / "S001_features.csv")
        _make_synthetic_feature_csv(csv_path, n_epochs=10)

        patient = _make_patient(csv_path)
        task = SleepWakeDetectionDREAMT()

        samples = task(patient)
        assert len(samples) == 10

    def test_sample_schema(self, tmp_path):
        """Validate schema of returned samples."""
        csv_path = str(tmp_path / "S001_features.csv")
        _make_synthetic_feature_csv(csv_path, n_epochs=5)

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

    def test_binary_label_conversion(self, tmp_path):
        """Verify wake/sleep label conversion logic."""
        csv_path = str(tmp_path / "S001_features.csv")
        _make_synthetic_feature_csv(csv_path, n_epochs=4)

        patient = _make_patient(csv_path)
        task = SleepWakeDetectionDREAMT()

        samples = task(patient)

        assert samples[0]["label"] == 1  # W -> wake
        assert samples[1]["label"] == 0  # N1 -> sleep

    def test_missing_file_returns_empty(self):
        """Ensure missing file results in empty output."""
        patient = _make_patient("/nonexistent/path.csv")
        task = SleepWakeDetectionDREAMT()

        assert task(patient) == []

    def test_no_events_returns_empty(self):
        """Ensure patients with no events return empty output."""
        patient = MagicMock()
        patient.patient_id = "S999"
        patient.get_events.return_value = []

        task = SleepWakeDetectionDREAMT()
        assert task(patient) == []

    def test_custom_feature_subset(self, tmp_path):
        """Ensure subset of features is handled correctly."""
        cols = ["ACC_INDEX", "HRV_HFD"]
        csv_path = str(tmp_path / "S002_features.csv")

        df = pd.DataFrame({
            "ACC_INDEX": [1.0, 2.0, 3.0],
            "HRV_HFD": [0.5, 0.6, 0.7],
            "Sleep_Stage": [0, 1, 0],
        })
        df.to_csv(csv_path, index=False)

        patient = _make_patient(csv_path, patient_id="S002")
        task = SleepWakeDetectionDREAMT(feature_columns=cols)

        samples = task(patient)

        assert len(samples) == 3
        assert len(samples[0]["features"]) == 2

    def test_preparation_and_missing_epochs_excluded(self, tmp_path):
        """Ensure invalid epochs are removed during preprocessing."""
        csv_path = str(tmp_path / "S003_features.csv")

        df = pd.DataFrame({
            **{col: [1.0] * 4 for col in FEATURE_COLUMNS},
            "Sleep_Stage": ["W", "P", "Missing", "N2"],
        })
        df.to_csv(csv_path, index=False)

        patient = _make_patient(csv_path, patient_id="S003")
        task = SleepWakeDetectionDREAMT()

        samples = task(patient)

        assert len(samples) == 2
        assert samples[0]["label"] == 1
        assert samples[1]["label"] == 0