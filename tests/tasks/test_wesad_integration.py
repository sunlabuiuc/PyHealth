from __future__ import annotations
import os
import pickle
import numpy as np
from pyhealth.datasets import WESADNonstationaryDataset
from pyhealth.tasks import wesad_stress_detection_fn


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


def test_wesad_dataset_set_task_integration(tmp_path):
    root = str(tmp_path / "wesad_integration")
    os.makedirs(root, exist_ok=True)

    _write_subject(
        root=root,
        subject_id="S1",
        eda=np.linspace(0.0, 1.0, 80),
        label=np.array([1] * 40 + [2] * 40),
        fs=4,
    )

    dataset = WESADNonstationaryDataset(
        root=root,
        augmentation_mode="none",
        refresh_cache=True,
    )

    sample_dataset = dataset.set_task(
        wesad_stress_detection_fn,
        window_sec=10,
        shift_sec=10,
        stress_label=2,
        baseline_label=1,
        keep_baseline_only=True,
    )

    assert hasattr(sample_dataset, "samples")
    assert len(sample_dataset.samples) == 2
    assert sample_dataset.samples[0]["label"] == 0
    assert sample_dataset.samples[1]["label"] == 1

def test_wesad_dataset_set_task_multiple_patients(tmp_path):
    root = str(tmp_path / "wesad_multi")
    os.makedirs(root, exist_ok=True)

    _write_subject(
        root=root,
        subject_id="S1",
        eda=np.linspace(0.0, 1.0, 80),
        label=np.array([1] * 40 + [2] * 40),
        fs=4,
    )
    _write_subject(
        root=root,
        subject_id="S2",
        eda=np.linspace(1.0, 2.0, 80),
        label=np.array([1] * 40 + [2] * 40),
        fs=4,
    )

    dataset = WESADNonstationaryDataset(
        root=root,
        augmentation_mode="none",
        refresh_cache=True,
    )

    sample_dataset = dataset.set_task(
        wesad_stress_detection_fn,
        window_sec=10,
        shift_sec=10,
        stress_label=2,
        baseline_label=1,
        keep_baseline_only=True,
    )

    assert hasattr(sample_dataset, "samples")
    assert len(sample_dataset.samples) == 4
    assert set(sample_dataset.patient_to_index.keys()) == {"S1", "S2"}
    assert len(sample_dataset.patient_to_index["S1"]) == 2
    assert len(sample_dataset.patient_to_index["S2"]) == 2