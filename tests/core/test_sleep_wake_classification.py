import numpy as np
import pandas as pd
from pyhealth.tasks.sleep_wake_classification import SleepWakeClassification


class FakeEvent:
    """Minimal DREAMT-like event for task tests."""

    def __init__(self, file_64hz=None):
        self.file_64hz = file_64hz


class FakePatient:
    """Minimal DREAMT-like patient with configurable events."""

    def __init__(self, patient_id: str, events=None):
        self.patient_id = patient_id
        self._events = [] if events is None else events

    def get_events(self, event_type=None):
        if event_type == "dreamt_sleep":
            return self._events
        return []

def _build_valid_record(num_rows: int = 8) -> pd.DataFrame:
    sleep_stages = ["W", "W", "N2", "N2", "REM", "REM", "W", "W"][:num_rows]
    return pd.DataFrame(
        {
            "TIMESTAMP": list(range(num_rows)),
            "BVP": np.linspace(0.1, 0.8, num_rows),
            "EDA": np.linspace(0.01, 0.08, num_rows),
            "TEMP": np.linspace(36.1, 36.5, num_rows),
            "ACC_X": np.linspace(1.0, 2.0, num_rows),
            "ACC_Y": np.linspace(0.5, 1.5, num_rows),
            "ACC_Z": np.linspace(0.2, 1.2, num_rows),
            "HR": np.linspace(60, 67, num_rows),
            "Sleep_Stage": sleep_stages,
        }
    )


def _build_patient_with_single_event(patient_id: str = "S001") -> FakePatient:
    return FakePatient(patient_id, events=[FakeEvent("unused.csv")])


def test_convert_sleep_stage_to_binary_label():
    task = SleepWakeClassification()

    assert task._convert_sleep_stage_to_binary_label("WAKE") == 1
    assert task._convert_sleep_stage_to_binary_label("W") == 1
    assert task._convert_sleep_stage_to_binary_label("N2") == 0
    assert task._convert_sleep_stage_to_binary_label("REM") == 0
    assert task._convert_sleep_stage_to_binary_label(None) is None
    assert task._convert_sleep_stage_to_binary_label("UNKNOWN") is None


def test_split_signal_into_epochs():
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    signal = np.array([0, 1, 2, 3, 4])

    epochs = task._split_signal_into_epochs(signal, sampling_rate_hz=1)

    assert len(epochs) == 2
    assert np.array_equal(epochs[0], np.array([0, 1]))
    assert np.array_equal(epochs[1], np.array([2, 3]))

def test_extract_binary_label_for_epoch():
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    record_dataframe = pd.DataFrame({"Sleep_Stage": ["W", "W", "N2", "N2"]})

    assert task._extract_binary_label_for_epoch(record_dataframe, 0, 2) == 1
    assert task._extract_binary_label_for_epoch(record_dataframe, 1, 2) == 0


def test_build_record_epoch_feature_matrix_returns_empty_when_columns_missing():
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    record_dataframe = pd.DataFrame(
        {
            "ACC_X": [1, 2, 3, 4],
            "ACC_Y": [1, 2, 3, 4],
            "ACC_Z": [1, 2, 3, 4],
            "TEMP": [36, 36, 36, 36],
            "Sleep_Stage": ["W", "W", "N2", "N2"],
        }
    )

    assert task._build_record_epoch_feature_matrix(record_dataframe) == []


def test_load_wearable_record_dataframe_returns_none_for_missing_file():
    task = SleepWakeClassification()

    assert task._load_wearable_record_dataframe(FakeEvent(file_64hz="missing.csv")) is None
    assert task._load_wearable_record_dataframe(FakeEvent(file_64hz=None)) is None


def test_task_returns_empty_when_patient_has_no_sleep_events():
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    patient = FakePatient("S001", events=[])

    assert task(patient) == []


def test_task_returns_empty_when_sleep_stage_column_is_missing(monkeypatch):
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    record_dataframe = _build_valid_record().drop(columns=["Sleep_Stage"])
    patient = _build_patient_with_single_event()

    monkeypatch.setattr(
        task,
        "_load_wearable_record_dataframe",
        lambda event: record_dataframe,
    )
    assert task(patient) == []


def test_task_skips_epochs_with_unsupported_labels(monkeypatch):
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    record_dataframe = _build_valid_record(num_rows=4)
    record_dataframe["Sleep_Stage"] = ["X", "X", "N2", "N2"]
    patient = _build_patient_with_single_event()

    monkeypatch.setattr(
        task,
        "_load_wearable_record_dataframe",
        lambda event: record_dataframe,
    )
    monkeypatch.setattr(
        task,
        "_build_record_epoch_feature_matrix",
        lambda df: [[1.0, 2.0], [3.0, 4.0]],
    )

    samples = task(patient)

    assert len(samples) == 1
    assert samples[0]["epoch_index"] == 1
    assert samples[0]["label"] == 0


def test_task_runs_full_flow_with_lightweight_feature_stub(monkeypatch):
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    record_dataframe = _build_valid_record(num_rows=8)
    patient = _build_patient_with_single_event()

    monkeypatch.setattr(
        task,
        "_load_wearable_record_dataframe",
        lambda event: record_dataframe,
    )
    monkeypatch.setattr(
        task,
        "_build_record_epoch_feature_matrix",
        lambda df: [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
    )

    samples = task(patient)

    assert len(samples) == 4
    assert all("features" in sample for sample in samples)
    assert all("label" in sample for sample in samples)
    assert all("record_id" in sample for sample in samples)
    assert samples[0]["record_id"] == "S001-event0-epoch0"
    assert samples[0]["label"] == 1
    assert samples[1]["label"] == 0
    assert samples[2]["label"] == 0
    assert samples[3]["label"] == 1


def test_task_uses_minimum_epoch_count_between_labels_and_features(monkeypatch):
    task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
    record_dataframe = _build_valid_record(num_rows=8)
    patient = _build_patient_with_single_event()

    monkeypatch.setattr(
        task,
        "_load_wearable_record_dataframe",
        lambda _: record_dataframe,
    )
    monkeypatch.setattr(
        task,
        "_build_record_epoch_feature_matrix",
        lambda _: [[1.0], [2.0]],
    )

    samples = task(patient)

    assert len(samples) == 2
    assert [sample["epoch_index"] for sample in samples] == [0, 1]
