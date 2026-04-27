from __future__ import annotations
import os
import pickle
import numpy as np
from pyhealth.datasets.wesad_nonstationary import WESADNonstationaryDataset


def _write_subject(root: str, subject_id: str, eda, label, fs: int = 4) -> None:
    path = os.path.join(root, f"{subject_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {
                "eda": np.asarray(eda, dtype=float),
                "label": np.asarray(label, dtype=int),
                "fs": fs,
            },
            f,
        )


def test_wesad_nonstationary_dataset_basic(tmp_path):
    root = str(tmp_path / "wesad")
    os.makedirs(root, exist_ok=True)

    _write_subject(
        root=root,
        subject_id="S1",
        eda=np.linspace(0.0, 1.0, 80),
        label=np.ones(80),
        fs=4,
    )
    _write_subject(
        root=root,
        subject_id="S2",
        eda=np.linspace(1.0, 2.0, 80),
        label=np.full(80, 2),
        fs=4,
    )

    dataset = WESADNonstationaryDataset(
        root=root,
        augmentation_mode="none",
        refresh_cache=True,
    )

    assert len(dataset.patients) == 2
    assert "S1" in dataset.patients
    assert "S2" in dataset.patients

    record = dataset.patients["S1"][0]
    assert record["patient_id"] == "S1"
    assert "load_from_path" in record
    assert "signal_file" in record
    assert "save_to_path" in record

    processed_path = os.path.join(record["load_from_path"], record["signal_file"])
    assert os.path.exists(processed_path)

    with open(processed_path, "rb") as f:
        subject = pickle.load(f)

    assert "eda" in subject
    assert "label" in subject
    assert "fs" in subject
    assert len(subject["eda"]) == 80
    assert len(subject["label"]) == 80
    assert subject["fs"] == 4


def test_wesad_nonstationary_dataset_augmentation_changes_signal(tmp_path):
    root = str(tmp_path / "wesad_aug")
    os.makedirs(root, exist_ok=True)

    eda = np.linspace(0.0, 1.0, 100)
    label = np.ones(100)

    _write_subject(root=root, subject_id="S1", eda=eda, label=label, fs=4)

    dataset = WESADNonstationaryDataset(
        root=root,
        augmentation_mode="learned",
        change_type="mean",
        magnitude=0.5,
        duration_ratio=0.25,
        refresh_cache=True,
        random_state=42,
    )

    record = dataset.patients["S1"][0]
    processed_path = os.path.join(record["load_from_path"], record["signal_file"])

    with open(processed_path, "rb") as f:
        subject = pickle.load(f)

    augmented = subject["eda"]
    assert augmented.shape == eda.shape
    assert not np.allclose(augmented, eda)

def test_wesad_nonstationary_dataset_missing_eda_raises(tmp_path):
    root = str(tmp_path / "wesad_missing_eda")
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, "S1.pkl")
    with open(path, "wb") as f:
        pickle.dump({"label": np.ones(10), "fs": 4}, f)

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for missing 'eda'"
    except ValueError as e:
        assert "eda" in str(e)


def test_wesad_nonstationary_dataset_missing_label_raises(tmp_path):
    root = str(tmp_path / "wesad_missing_label")
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, "S1.pkl")
    with open(path, "wb") as f:
        pickle.dump({"eda": np.linspace(0.0, 1.0, 10), "fs": 4}, f)

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for missing 'label'"
    except ValueError as e:
        assert "label" in str(e)


def test_wesad_nonstationary_dataset_missing_fs_raises(tmp_path):
    root = str(tmp_path / "wesad_missing_fs")
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, "S1.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {"eda": np.linspace(0.0, 1.0, 10), "label": np.ones(10)},
            f,
        )

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for missing 'fs'"
    except ValueError as e:
        assert "fs" in str(e)


def test_wesad_nonstationary_dataset_length_mismatch_raises(tmp_path):
    root = str(tmp_path / "wesad_length_mismatch")
    os.makedirs(root, exist_ok=True)

    _write_subject(
        root=root,
        subject_id="S1",
        eda=np.linspace(0.0, 1.0, 10),
        label=np.ones(8),
        fs=4,
    )

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for length mismatch"
    except ValueError as e:
        assert "mismatch" in str(e).lower()


def test_wesad_nonstationary_dataset_invalid_eda_dimension_raises(tmp_path):
    root = str(tmp_path / "wesad_bad_eda_dim")
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, "S1.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {
                "eda": np.ones((2, 10)),
                "label": np.ones(10),
                "fs": 4,
            },
            f,
        )

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for non-1D eda"
    except ValueError as e:
        assert "1d" in str(e).lower()


def test_wesad_nonstationary_dataset_invalid_label_dimension_raises(tmp_path):
    root = str(tmp_path / "wesad_bad_label_dim")
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, "S1.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {
                "eda": np.linspace(0.0, 1.0, 10),
                "label": np.ones((2, 10)),
                "fs": 4,
            },
            f,
        )

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for non-1D label"
    except ValueError as e:
        assert "1d" in str(e).lower()


def test_wesad_nonstationary_dataset_invalid_fs_raises(tmp_path):
    root = str(tmp_path / "wesad_bad_fs")
    os.makedirs(root, exist_ok=True)

    _write_subject(
        root=root,
        subject_id="S1",
        eda=np.linspace(0.0, 1.0, 10),
        label=np.ones(10),
        fs=0,
    )

    try:
        WESADNonstationaryDataset(root=root, refresh_cache=True)
        assert False, "Expected ValueError for non-positive fs"
    except ValueError as e:
        assert "positive" in str(e).lower()


def test_wesad_nonstationary_dataset_dev_mode_limits_subjects(tmp_path):
    root = str(tmp_path / "wesad_dev")
    os.makedirs(root, exist_ok=True)

    for i in range(3):
        _write_subject(
            root=root,
            subject_id=f"S{i+1}",
            eda=np.linspace(0.0, 1.0, 20),
            label=np.ones(20),
            fs=4,
        )

    dataset = WESADNonstationaryDataset(
        root=root,
        dev=True,
        refresh_cache=True,
    )

    assert len(dataset.patients) == 2