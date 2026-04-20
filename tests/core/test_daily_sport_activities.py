from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.daily_sport_activities import DailyAndSportActivitiesDataset
from pyhealth.tasks.daily_sport_activities import (
    DailyAndSportActivitiesTask,
)


def _write_fake_signal_file(file_path: Path, shape=(125, 45), seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape).astype(np.float32)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(
        file_path,
        header=False,
        index=False,
    )

def _make_fake_dataset(root: Path):
    """
    Creates a tiny synthetic dataset with the same folder structure as the real one.

    Structure:
        root/
            a01/
                p1/
                    s01.txt
                    s02.txt
                p7/
                    s01.txt
                p8/
                    s01.txt
            a02/
                p1/
                    s01.txt
                p7/
                    s01.txt
                p8/
                    s01.txt
    """
    files = [
        ("a01", "p1", "s01.txt"),
        ("a01", "p1", "s02.txt"),
        ("a01", "p7", "s01.txt"),
        ("a01", "p8", "s01.txt"),
        ("a02", "p1", "s01.txt"),
        ("a02", "p7", "s01.txt"),
        ("a02", "p8", "s01.txt"),
    ]

    for i, (activity, subject, segment) in enumerate(files):
        path = root / activity / subject / segment
        _write_fake_signal_file(path, shape=(125, 45), seed=i)


def test_parse_data_loads_all_samples(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    samples = dataset.parse_data()

    assert len(samples) == 7
    assert isinstance(samples, list)

    first = samples[0]
    assert "record_id" in first
    assert "patient_id" in first
    assert "visit_id" in first
    assert "activity_id" in first
    assert "activity" in first
    assert "segment_id" in first
    assert "file_path" in first
    assert "signal" in first


def test_signal_shape_is_correct(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    samples = dataset.parse_data()

    for sample in samples:
        assert sample["signal"].shape == (125, 45)
        assert sample["signal"].dtype == np.float32


def test_invalid_shape_raises_error(tmp_path):
    bad_file = tmp_path / "daily_sport_activities" / "a01" / "p1" / "s01.txt"
    _write_fake_signal_file(bad_file, shape=(124, 45), seed=123)

    dataset = DailyAndSportActivitiesDataset(root=str(tmp_path / "daily_sport_activities"))

    with pytest.raises(ValueError, match="must have shape"):
        dataset.parse_data()


def test_missing_root_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        DailyAndSportActivitiesDataset(root="this/path/does/not/exist")


def test_load_data_returns_event_dataframe(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    df = dataset.load_data().compute()

    assert len(df) == 7
    assert "patient_id" in df.columns
    assert "event_type" in df.columns
    assert "timestamp" in df.columns
    assert "daily_sport_activities/file_path" in df.columns
    assert "daily_sport_activities/activity_id" in df.columns
    assert "daily_sport_activities/activity" in df.columns

    assert set(df["event_type"].unique()) == {"daily_sport_activities"}


def test_get_patient_returns_expected_events(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    patient = dataset.get_patient("p1")
    events = patient.get_events(event_type="daily_sport_activities")

    assert len(events) == 3

    event = events[0]
    assert "daily_sport_activities/file_path" in event
    assert "daily_sport_activities/activity_id" in event
    assert "daily_sport_activities/activity" in event
    assert "daily_sport_activities/visit_id" in event


def test_event_metadata_is_parsed_correctly(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    df = dataset.load_data().compute()

    row = df.iloc[0]

    assert row["patient_id"] in {"p1", "p7", "p8"}
    assert row["daily_sport_activities/activity_id"] in {"a01", "a02"}
    assert row["daily_sport_activities/activity"] in {"sitting", "standing"}
    assert row["daily_sport_activities/visit_id"] in {"s01", "s02"}
    assert row["daily_sport_activities/n_rows"] == 125
    assert row["daily_sport_activities/n_cols"] == 45


def test_set_task_generates_samples(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    task = DailyAndSportActivitiesTask(
        window_size=50,
        stride=25,
        normalize=True,
    )
    samples = dataset.set_task(task)

    assert len(samples) > 0

    sample = samples[0]
    assert "patient_id" in sample
    assert "visit_id" in sample
    assert "record_id" in sample
    assert "signal" in sample
    assert "label" in sample

    assert sample["signal"].shape == (50, 45)
    assert isinstance(sample["label"], (int, np.integer))


def test_task_selected_features_reduces_dimension(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    task = DailyAndSportActivitiesTask(
        window_size=50,
        stride=25,
        selected_features=[0, 1, 2, 3],
    )
    samples = dataset.set_task(task)

    assert len(samples) > 0
    assert samples[0]["signal"].shape == (50, 4)


def test_task_invalid_feature_index_raises(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    task = DailyAndSportActivitiesTask(
        window_size=50,
        stride=25,
        selected_features=[999],
    )

    with pytest.raises(ValueError, match="out of bounds"):
        dataset.set_task(task)


def test_task_window_too_large_raises(tmp_path):
    data_root = tmp_path / "daily_sport_activities"
    _make_fake_dataset(data_root)

    dataset = DailyAndSportActivitiesDataset(root=str(data_root))
    task = DailyAndSportActivitiesTask(
        window_size=200,
        stride=25,
    )

    with pytest.raises(ValueError):
        dataset.set_task(task)
