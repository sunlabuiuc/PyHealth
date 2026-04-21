import numpy as np
import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
from pyhealth.tasks.sleep_wake_dreamt import (
    SleepWakeDetectionDREAMT,
    EPOCH_SAMPLES,
    SIGNAL_COLS,
    WAKE_LABEL,
)

def make_fake_patient(
    patient_id: str = "S001",
    ahi: float = 22.1,
    bmi: float = 33.7,
    n_epochs: int = 3,
    has_file: bool = True,
    sleep_stages: list = None,
    tmp_dir: str = None,
):
    """Creates a mock DREAMT patient object for testing."""
    if sleep_stages is None:
        sleep_stages = ["N2", "W", "N2"]

    # Build a fake overnight DataFrame
    n_rows = n_epochs * EPOCH_SAMPLES
    data = {col: np.random.randn(n_rows).astype(np.float32)
            for col in SIGNAL_COLS}

    # Assign sleep stage label at last row of each epoch
    stages = np.full(n_rows, "N2", dtype=object)
    for i, stage in enumerate(sleep_stages):
        stages[(i + 1) * EPOCH_SAMPLES - 1] = stage
    data["Sleep_Stage"] = stages

    import pandas as pd
    fake_df = pd.DataFrame(data)

    # Write to a real temp file if tmp_dir provided
    if tmp_dir and has_file:
        file_path = os.path.join(tmp_dir, f"{patient_id}_whole_df.csv")
        fake_df.to_csv(file_path, index=False)
    else:
        file_path = "/fake/path/S001_whole_df.csv" if has_file else None

    # Mock event
    event = MagicMock()
    event.file_64hz = file_path
    event.ahi = ahi
    event.bmi = bmi

    # Mock patient
    patient = MagicMock()
    patient.patient_id = patient_id
    patient.get_events.return_value = [event]

    return patient, fake_df


class TestSleepWakeDetectionDREAMT:

    def test_instantiation(self):
        """Task can be instantiated."""
        task = SleepWakeDetectionDREAMT()
        assert task.task_name == "SleepWakeDetectionDREAMT"

    def test_schema_defined(self):
        """Input and output schemas are defined correctly."""
        task = SleepWakeDetectionDREAMT()
        assert "signal" in task.input_schema
        assert "ahi" in task.input_schema
        assert "bmi" in task.input_schema
        assert "label" in task.output_schema

    def test_returns_correct_number_of_epochs(self):
        """Task returns one sample per valid epoch."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=3,
            sleep_stages=["N2", "W", "N1"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        assert len(samples) == 3

    def test_wake_label_is_1(self):
        """Wake epochs are labeled 1."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=2,
            sleep_stages=["W", "N2"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        labels = [s["label"] for s in samples]
        assert labels[0] == 1
        assert labels[1] == 0

    def test_sleep_label_is_0(self):
        """All NREM and REM epochs are labeled 0."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=4,
            sleep_stages=["N1", "N2", "N3", "R"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        assert all(s["label"] == 0 for s in samples)

    def test_missing_epochs_are_skipped(self):
        """Epochs labeled Missing are excluded from samples."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=3,
            sleep_stages=["N2", "Missing", "W"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        assert len(samples) == 2

    def test_signal_shape(self):
        """Each epoch signal is a 1D engineered feature vector."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(n_epochs=2,
                                            sleep_stages=["N2", "W"])
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        for s in samples:
            # Signal is now an engineered feature vector, not raw signal
            assert isinstance(s["signal"], np.ndarray)
            assert s["signal"].ndim == 1
            assert len(s["signal"]) > 0

    def test_clinical_metadata_attached(self):
        """AHI and BMI are correctly attached to each sample."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(
            ahi=15.5, bmi=28.3,
            n_epochs=2, sleep_stages=["N2", "W"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        for s in samples:
            assert s["ahi"] == pytest.approx(15.5)
            assert s["bmi"] == pytest.approx(28.3)

    def test_no_file_returns_empty(self):
        """Returns empty list when no file path is available."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(has_file=False)
        samples = task(patient)
        assert samples == []

    def test_no_events_returns_empty(self):
        """Returns empty list when patient has no events."""
        task = SleepWakeDetectionDREAMT()
        patient = MagicMock()
        patient.patient_id = "S999"
        patient.get_events.return_value = []
        samples = task(patient)
        assert samples == []

    def test_patient_id_in_samples(self):
        """Patient ID is correctly propagated to each sample."""
        task = SleepWakeDetectionDREAMT()
        patient, fake_df = make_fake_patient(
            patient_id="S042",
            n_epochs=2,
            sleep_stages=["W", "N2"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        assert all(s["patient_id"] == "S042" for s in samples)
    
    def test_uses_temp_directory(self):
        """Task correctly reads from a real temporary file."""
        task = SleepWakeDetectionDREAMT()
        with tempfile.TemporaryDirectory() as tmp_dir:
            patient, _ = make_fake_patient(
                patient_id="S001",
                n_epochs=3,
                sleep_stages=["N2", "W", "N1"],
                tmp_dir=tmp_dir,
            )
            # No mock needed — reads real temp file
            samples = task(patient)
            assert len(samples) == 3
            assert all("signal" in s for s in samples)
    
class TestSleepStagingDREAMT:

    def test_instantiation(self):
        """SleepStagingDREAMT can be instantiated."""
        from pyhealth.tasks.sleep_wake_dreamt import SleepStagingDREAMT
        task = SleepStagingDREAMT()
        assert task.task_name == "SleepStagingDREAMT"

    def test_schema_defined(self):
        """Input and output schemas are defined correctly."""
        from pyhealth.tasks.sleep_wake_dreamt import SleepStagingDREAMT
        task = SleepStagingDREAMT()
        assert "signal" in task.input_schema
        assert "label" in task.output_schema
        assert task.output_schema["label"] == "multiclass"

    def test_fine_labels(self):
        """Each sleep stage maps to correct integer label."""
        from pyhealth.tasks.sleep_wake_dreamt import SleepStagingDREAMT
        task = SleepStagingDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=5,
            sleep_stages=["W", "N1", "N2", "N3", "R"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        labels = [s["label"] for s in samples]
        assert labels == [0, 1, 2, 3, 4]

    def test_missing_skipped(self):
        """Missing epochs are skipped in multi-class task too."""
        from pyhealth.tasks.sleep_wake_dreamt import SleepStagingDREAMT
        task = SleepStagingDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=3,
            sleep_stages=["W", "Missing", "R"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        assert len(samples) == 2

    def test_signal_is_feature_vector(self):
        """Signal output is a 1D engineered feature vector."""
        from pyhealth.tasks.sleep_wake_dreamt import SleepStagingDREAMT
        task = SleepStagingDREAMT()
        patient, fake_df = make_fake_patient(
            n_epochs=2,
            sleep_stages=["N2", "W"]
        )
        with patch("pandas.read_csv", return_value=fake_df):
            samples = task(patient)
        for s in samples:
            assert isinstance(s["signal"], np.ndarray)
            assert s["signal"].ndim == 1
            assert len(s["signal"]) > 0
    