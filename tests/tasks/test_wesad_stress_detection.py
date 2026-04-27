from __future__ import annotations
import os
import pickle
import numpy as np
from pyhealth.tasks.wesad_stress_detection import wesad_stress_detection_fn


def test_wesad_stress_detection_fn_basic(tmp_path):
    root = str(tmp_path / "task_data")
    os.makedirs(root, exist_ok=True)

    patient_id = "S1"
    subject_path = os.path.join(root, f"{patient_id}.pkl")

    # 20 seconds at 4 Hz = 80 samples
    # First 10 sec baseline, next 10 sec stress
    eda = np.linspace(0.0, 1.0, 80)
    label = np.array([1] * 40 + [2] * 40)

    with open(subject_path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": 4}, f)

    record = [
        {
            "load_from_path": root,
            "patient_id": patient_id,
            "signal_file": f"{patient_id}.pkl",
            "save_to_path": root,
        }
    ]

    samples = wesad_stress_detection_fn(
        record=record,
        window_sec=10,
        shift_sec=10,
        stress_label=2,
        baseline_label=1,
        keep_baseline_only=True,
    )

    assert len(samples) == 2

    first = samples[0]
    second = samples[1]

    assert first["patient_id"] == patient_id
    assert first["visit_id"] == patient_id
    assert "epoch_path" in first
    assert os.path.exists(first["epoch_path"])

    assert first["label"] == 0
    assert second["label"] == 1

    with open(first["epoch_path"], "rb") as f:
        epoch = pickle.load(f)

    assert "signal" in epoch
    assert "label" in epoch
    assert "fs" in epoch
    assert len(epoch["signal"]) == 40
    assert epoch["fs"] == 4


def test_wesad_stress_detection_fn_skips_mixed_nonbaseline_window(tmp_path):
    root = str(tmp_path / "task_data_mixed")
    os.makedirs(root, exist_ok=True)

    patient_id = "S2"
    subject_path = os.path.join(root, f"{patient_id}.pkl")

    # 15 seconds at 4 Hz = 60 samples
    # window 0: baseline, window 1: mixed non-baseline/non-stress
    eda = np.linspace(0.0, 1.0, 60)
    label = np.array([1] * 20 + [3] * 20 + [1] * 20)

    with open(subject_path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": 4}, f)

    record = [
        {
            "load_from_path": root,
            "patient_id": patient_id,
            "signal_file": f"{patient_id}.pkl",
            "save_to_path": root,
        }
    ]

    samples = wesad_stress_detection_fn(
        record=record,
        window_sec=5,
        shift_sec=5,
        stress_label=2,
        baseline_label=1,
        keep_baseline_only=True,
    )

    # baseline, mixed-other, baseline
    assert len(samples) == 2
    assert all(sample["label"] == 0 for sample in samples)

def test_wesad_stress_detection_fn_short_signal_returns_empty(tmp_path):
    root = str(tmp_path / "task_short")
    os.makedirs(root, exist_ok=True)

    patient_id = "S_short"
    subject_path = os.path.join(root, f"{patient_id}.pkl")

    # 5 seconds at 4 Hz = 20 samples, but window is 10 sec => 40 samples needed
    eda = np.linspace(0.0, 1.0, 20)
    label = np.ones(20)

    with open(subject_path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": 4}, f)

    record = [
        {
            "load_from_path": root,
            "patient_id": patient_id,
            "signal_file": f"{patient_id}.pkl",
            "save_to_path": root,
        }
    ]

    samples = wesad_stress_detection_fn(
        record=record,
        window_sec=10,
        shift_sec=10,
    )

    assert samples == []


def test_wesad_stress_detection_fn_invalid_window_sec_raises(tmp_path):
    root = str(tmp_path / "task_bad_window")
    os.makedirs(root, exist_ok=True)

    patient_id = "S_bad_window"
    subject_path = os.path.join(root, f"{patient_id}.pkl")

    eda = np.linspace(0.0, 1.0, 80)
    label = np.ones(80)

    with open(subject_path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": 4}, f)

    record = [
        {
            "load_from_path": root,
            "patient_id": patient_id,
            "signal_file": f"{patient_id}.pkl",
            "save_to_path": root,
        }
    ]

    try:
        wesad_stress_detection_fn(record=record, window_sec=0, shift_sec=5)
        assert False, "Expected ValueError for non-positive window_sec"
    except ValueError as e:
        assert "positive" in str(e).lower()


def test_wesad_stress_detection_fn_invalid_shift_sec_raises(tmp_path):
    root = str(tmp_path / "task_bad_shift")
    os.makedirs(root, exist_ok=True)

    patient_id = "S_bad_shift"
    subject_path = os.path.join(root, f"{patient_id}.pkl")

    eda = np.linspace(0.0, 1.0, 80)
    label = np.ones(80)

    with open(subject_path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": 4}, f)

    record = [
        {
            "load_from_path": root,
            "patient_id": patient_id,
            "signal_file": f"{patient_id}.pkl",
            "save_to_path": root,
        }
    ]

    try:
        wesad_stress_detection_fn(record=record, window_sec=10, shift_sec=0)
        assert False, "Expected ValueError for non-positive shift_sec"
    except ValueError as e:
        assert "positive" in str(e).lower()


def test_wesad_stress_detection_fn_keep_baseline_only_false_keeps_other_windows(tmp_path):
    root = str(tmp_path / "task_keep_false")
    os.makedirs(root, exist_ok=True)

    patient_id = "S_keep_false"
    subject_path = os.path.join(root, f"{patient_id}.pkl")

    # 3 windows of 5 sec each at 4 Hz:
    # baseline, other-nonstress, stress
    eda = np.linspace(0.0, 1.0, 60)
    label = np.array([1] * 20 + [3] * 20 + [2] * 20)

    with open(subject_path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": 4}, f)

    record = [
        {
            "load_from_path": root,
            "patient_id": patient_id,
            "signal_file": f"{patient_id}.pkl",
            "save_to_path": root,
        }
    ]

    samples = wesad_stress_detection_fn(
        record=record,
        window_sec=5,
        shift_sec=5,
        stress_label=2,
        baseline_label=1,
        keep_baseline_only=False,
    )

    assert len(samples) == 3
    assert samples[0]["label"] == 0
    assert samples[1]["label"] == 0
    assert samples[2]["label"] == 1