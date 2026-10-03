"""
These are Synthetic unit tests for ECGMultiLabelCardiologyTask.
They are implemented for the CS598DLH SP26 Final Project

Authored by Jonathan Gong, Misael Lazaro, and Sydney Robeson
NetIDs: jgong11, misaell2, sel9

This task is inspired by Nonaka & Seita (2021)
"In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"
Paper link: https://proceedings.mlr.press/v149/nonaka21a.html

These tests use only synthetic pseudo-data and remain lightweight enough for
quick execution. They also include a small CNN smoke test showing that
task-processed samples can be adapted into a PyHealth sample dataset.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from scipy.io import savemat

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CNN
from pyhealth.tasks.ecg_classification import ECGMultiLabelCardiologyTask


def _write_record(
    root: Path,
    stem: str,
    signal: np.ndarray,
    dx_codes: List[str],
    sex: str = "Female",
    age: str = "45",
) -> Dict[str, Any]:
    """Writes one synthetic ECG record pair (.mat + .hea).

    Args:
        root: Directory to write files into.
        stem: Record basename without extension.
        signal: ECG array shaped (leads, timesteps).
        dx_codes: Diagnostic label codes to write in the header.
        sex: Header sex metadata.
        age: Header age metadata.

    Returns:
        A visit dictionary matching the task input contract.
    """
    mat_path = root / f"{stem}.mat"
    hea_path = root / f"{stem}.hea"

    savemat(mat_path, {"val": signal.astype(np.float32)})

    header = "\n".join(
        [
            f"{stem} 12 {signal.shape[1]} 500",
            f"#Dx: {','.join(dx_codes)}",
            f"#Sex: {sex}",
            f"#Age: {age}",
        ]
    )
    hea_path.write_text(header, encoding="utf-8")

    return {
        "load_from_path": str(root),
        "patient_id": f"patient_{stem}",
        "signal_file": mat_path.name,
        "label_file": hea_path.name,
    }


def _labels_from_multihot(
    sample: Dict[str, Any],
    labels: List[str],
) -> Dict[str, Any]:
    """Converts a multi-hot label vector into a list of active labels."""
    converted = dict(sample)
    label_vector = np.asarray(sample["label"]).astype(np.float32)

    active = [
        labels[idx]
        for idx, value in enumerate(label_vector.tolist())
        if float(value) > 0.5
    ]
    converted["label"] = active
    return converted


@pytest.fixture
def labels() -> List[str]:
    """Small synthetic SNOMED-style label vocabulary."""
    return ["164889003", "164890007", "426783006"]


@pytest.fixture
def task(labels: List[str]) -> ECGMultiLabelCardiologyTask:
    """Returns a task with tiny window settings for fast tests.

    window_size = sampling_rate * epoch_sec = 10 * 2 = 20
    step_size   = sampling_rate * shift     = 10 * 1 = 10
    """
    return ECGMultiLabelCardiologyTask(
        labels=labels,
        epoch_sec=2,
        shift=1,
        sampling_rate=10,
    )


def test_meta(labels: List[str]) -> None:
    """Tests task metadata and initialization."""
    task = ECGMultiLabelCardiologyTask(
        labels=labels,
        epoch_sec=2,
        shift=1,
        sampling_rate=10,
    )

    assert task.task_name == "ECGMultiLabelCardiologyTask"
    assert task.input_schema == {"signal": "tensor"}
    assert task.output_schema == {"label": "multilabel"}
    assert task.labels == labels
    assert task.label_to_index == {
        "164889003": 0,
        "164890007": 1,
        "426783006": 2,
    }


def test_single_visit(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests windowing, signal slicing, and multi-hot label generation."""
    signal = np.arange(12 * 40, dtype=np.float32).reshape(12, 40)
    visit = _write_record(
        tmp_path,
        stem="rec1",
        signal=signal,
        dx_codes=["164889003", "426783006"],
        sex="Male",
        age="60",
    )

    samples = task(visit)

    assert len(samples) == 3

    first = samples[0]
    assert first["patient_id"] == "patient_rec1"
    assert first["visit_id"] == "rec1"
    assert first["record_id"] == 1
    assert first["signal"].shape == (12, 20)
    assert first["signal"].dtype == np.float32
    assert np.array_equal(
        first["label"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
    assert first["Sex"] == ["Male"]
    assert first["Age"] == ["60"]

    second = samples[1]
    assert second["record_id"] == 2
    assert np.array_equal(second["signal"], signal[:, 10:30].astype(np.float32))

    third = samples[2]
    assert third["record_id"] == 3
    assert np.array_equal(third["signal"], signal[:, 20:40].astype(np.float32))


def test_visit_list(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests list input normalization across multiple visits."""
    visit_1 = _write_record(
        tmp_path,
        stem="rec_a",
        signal=np.ones((12, 30), dtype=np.float32),
        dx_codes=["164889003"],
    )
    visit_2 = _write_record(
        tmp_path,
        stem="rec_b",
        signal=np.ones((12, 25), dtype=np.float32) * 2,
        dx_codes=["164890007", "999999999"],
    )

    samples = task([visit_1, visit_2])

    assert len(samples) == 3
    assert np.array_equal(
        samples[0]["label"],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.array_equal(
        samples[-1]["label"],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_missing_keys(task: ECGMultiLabelCardiologyTask) -> None:
    """Tests validation for malformed visit dictionaries."""
    bad_visit = {
        "patient_id": "p1",
        "signal_file": "missing_root.mat",
    }

    samples = task(bad_visit)
    assert samples == []


def test_short_signal(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests edge case where the signal is too short for one window."""
    short_signal = np.zeros((12, 19), dtype=np.float32)
    visit = _write_record(
        tmp_path,
        stem="short_rec",
        signal=short_signal,
        dx_codes=["164889003"],
    )

    samples = task(visit)
    assert samples == []


def test_exact_window(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests exact-boundary windowing behavior."""
    exact_signal = np.arange(12 * 20, dtype=np.float32).reshape(12, 20)
    visit = _write_record(
        tmp_path,
        stem="exact_rec",
        signal=exact_signal,
        dx_codes=["164890007"],
    )

    samples = task(visit)

    assert len(samples) == 1
    assert samples[0]["record_id"] == 1
    assert samples[0]["signal"].shape == (12, 20)
    assert np.array_equal(
        samples[0]["label"],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_nondiv_windows(
    tmp_path: Path,
    task: ECGMultiLabelCardiologyTask,
) -> None:
    """Tests sliding-window count when length is not evenly divisible."""
    signal = np.arange(12 * 35, dtype=np.float32).reshape(12, 35)
    visit = _write_record(
        tmp_path,
        stem="nondiv",
        signal=signal,
        dx_codes=["426783006"],
    )

    samples = task(visit)

    assert len(samples) == 2
    assert np.array_equal(samples[0]["signal"], signal[:, 0:20].astype(np.float32))
    assert np.array_equal(samples[1]["signal"], signal[:, 10:30].astype(np.float32))


def test_bad_mat(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests graceful handling of unreadable .mat files."""
    mat_path = tmp_path / "broken.mat"
    hea_path = tmp_path / "broken.hea"

    mat_path.write_text("not a valid matlab file", encoding="utf-8")
    hea_path.write_text(
        "\n".join(
            [
                "broken 12 100 500",
                "#Dx: 164889003",
                "#Sex: Female",
                "#Age: 55",
            ]
        ),
        encoding="utf-8",
    )

    visit = {
        "load_from_path": str(tmp_path),
        "patient_id": "patient_broken",
        "signal_file": mat_path.name,
        "label_file": hea_path.name,
    }

    samples = task(visit)
    assert samples == []


def test_missing_header(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests permissive handling of missing header files."""
    mat_path = tmp_path / "missing_header.mat"
    signal = np.ones((12, 40), dtype=np.float32)
    savemat(mat_path, {"val": signal})

    visit = {
        "load_from_path": str(tmp_path),
        "patient_id": "patient_missing_header",
        "signal_file": mat_path.name,
        "label_file": "missing_header.hea",
    }

    samples = task(visit)

    assert len(samples) == 3
    for sample in samples:
        assert sample["signal"].shape == (12, 20)
        assert sample["Sex"] == []
        assert sample["Age"] == []
        assert np.array_equal(
            sample["label"],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )


def test_missing_signal(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests graceful handling of missing signal files."""
    hea_path = tmp_path / "missing_signal.hea"
    hea_path.write_text(
        "\n".join(
            [
                "missing_signal 12 100 500",
                "#Dx: 164889003",
                "#Sex: Female",
                "#Age: 50",
            ]
        ),
        encoding="utf-8",
    )

    visit = {
        "load_from_path": str(tmp_path),
        "patient_id": "patient_missing_signal",
        "signal_file": "missing_signal.mat",
        "label_file": hea_path.name,
    }

    samples = task(visit)
    assert samples == []


def test_encode_unknown(task: ECGMultiLabelCardiologyTask) -> None:
    """Tests multi-hot label generation with unknown labels present."""
    encoded = task._encode_labels(["164890007", "not_in_vocab", "426783006"])
    expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    assert np.array_equal(encoded, expected)


def test_encode_dupes(task: ECGMultiLabelCardiologyTask) -> None:
    """Tests duplicate labels still produce binary multi-hot outputs."""
    encoded = task._encode_labels(["164889003", "164889003", "164890007"])
    expected = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    assert np.array_equal(encoded, expected)


def test_header_parse(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests partial-header parsing."""
    header_path = tmp_path / "partial.hea"
    header_path.write_text(
        "\n".join(
            [
                "partial 12 100 500",
                "#Dx: 164889003,426783006",
            ]
        ),
        encoding="utf-8",
    )

    metadata = task._parse_header_metadata(str(header_path))

    assert metadata["dx_codes"] == ["164889003", "426783006"]
    assert metadata["sex"] == []
    assert metadata["age"] == []


def test_empty_dx(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests empty diagnosis metadata produces all-zero labels."""
    signal = np.ones((12, 40), dtype=np.float32)
    mat_path = tmp_path / "empty_dx.mat"
    hea_path = tmp_path / "empty_dx.hea"

    savemat(mat_path, {"val": signal})
    hea_path.write_text(
        "\n".join(
            [
                "empty_dx 12 40 500",
                "#Dx:",
                "#Sex: Female",
                "#Age: 33",
            ]
        ),
        encoding="utf-8",
    )

    visit = {
        "load_from_path": str(tmp_path),
        "patient_id": "patient_empty_dx",
        "signal_file": mat_path.name,
        "label_file": hea_path.name,
    }

    samples = task(visit)

    assert len(samples) == 3
    for sample in samples:
        assert np.array_equal(
            sample["label"],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )


def test_immutable(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests that the task does not mutate the input visit dictionary."""
    visit = _write_record(
        tmp_path,
        stem="immutable",
        signal=np.ones((12, 30), dtype=np.float32),
        dx_codes=["164889003"],
    )
    original = deepcopy(visit)

    _ = task(visit)

    assert visit == original


def test_sample_shape(tmp_path: Path, task: ECGMultiLabelCardiologyTask) -> None:
    """Tests emitted sample keys, shapes, and dtypes across multiple visits."""
    visit_1 = _write_record(
        tmp_path,
        stem="shape_a",
        signal=np.ones((12, 40), dtype=np.float32),
        dx_codes=["164889003"],
        sex="Male",
        age="40",
    )
    visit_2 = _write_record(
        tmp_path,
        stem="shape_b",
        signal=np.ones((12, 20), dtype=np.float32),
        dx_codes=["164890007", "426783006"],
        sex="Female",
        age="41",
    )

    samples = task([visit_1, visit_2])

    assert len(samples) == 4
    for sample in samples:
        assert set(sample.keys()) == {
            "patient_id",
            "visit_id",
            "record_id",
            "signal",
            "label",
            "Sex",
            "Age",
        }
        assert sample["signal"].shape == (12, 20)
        assert sample["signal"].dtype == np.float32
        assert sample["label"].shape == (3,)
        assert sample["label"].dtype == np.float32
        assert isinstance(sample["Sex"], list)
        assert isinstance(sample["Age"], list)


def test_four_label_cfg(tmp_path: Path) -> None:
    """Show AF/I-AVB/LBBB/RBBB task label configuration."""
    four_labels = ["AF", "I-AVB", "LBBB", "RBBB"]
    task = ECGMultiLabelCardiologyTask(
        labels=four_labels,
        epoch_sec=2,
        shift=1,
        sampling_rate=10,
    )

    visit = _write_record(
        tmp_path,
        stem="four_cfg",
        signal=np.ones((12, 40), dtype=np.float32),
        dx_codes=["AF", "RBBB"],
    )

    samples = task(visit)

    assert len(samples) == 3
    for sample in samples:
        assert sample["label"].shape == (4,)
        assert np.array_equal(
            sample["label"],
            np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )


def test_four_label_all_on(tmp_path: Path) -> None:
    """Shows that all four configured labels can be active together."""
    four_labels = ["AF", "I-AVB", "LBBB", "RBBB"]
    task = ECGMultiLabelCardiologyTask(
        labels=four_labels,
        epoch_sec=2,
        shift=1,
        sampling_rate=10,
    )

    visit = _write_record(
        tmp_path,
        stem="four_all",
        signal=np.ones((12, 20), dtype=np.float32),
        dx_codes=["AF", "I-AVB", "LBBB", "RBBB"],
    )

    samples = task(visit)

    assert len(samples) == 1
    assert np.array_equal(
        samples[0]["label"],
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_four_label_order(tmp_path: Path) -> None:
    """Shows that output positions follow the configured four-label order."""
    four_labels = ["AF", "I-AVB", "LBBB", "RBBB"]
    task = ECGMultiLabelCardiologyTask(
        labels=four_labels,
        epoch_sec=2,
        shift=1,
        sampling_rate=10,
    )

    visit = _write_record(
        tmp_path,
        stem="four_order",
        signal=np.ones((12, 20), dtype=np.float32),
        dx_codes=["I-AVB", "LBBB"],
    )

    samples = task(visit)

    assert len(samples) == 1
    assert np.array_equal(
        samples[0]["label"],
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_cnn_smoke(
    tmp_path: Path,
    task: ECGMultiLabelCardiologyTask,
    labels: List[str],
) -> None:
    """Smoke-tests CNN on task-processed synthetic ECG samples."""
    visit_1 = _write_record(
        tmp_path,
        stem="cnn_a",
        signal=np.random.randn(12, 40).astype(np.float32),
        dx_codes=["164889003"],
    )
    visit_2 = _write_record(
        tmp_path,
        stem="cnn_b",
        signal=np.random.randn(12, 30).astype(np.float32),
        dx_codes=["164890007", "426783006"],
    )

    raw_samples = task([visit_1, visit_2])
    assert len(raw_samples) >= 3

    adapted_samples = [
        _labels_from_multihot(sample, labels) for sample in raw_samples
    ]

    dataset = create_sample_dataset(
        samples=adapted_samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multilabel"},
        dataset_name="synthetic_ecg",
        task_name="ecg_multilabel",
        in_memory=True,
    )

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = CNN(
        dataset=dataset,
        embedding_dim=16,
        hidden_dim=8,
        num_layers=1,
    )
    model.train()

    results = model(**batch)

    assert "loss" in results
    assert "y_prob" in results
    assert "y_true" in results
    assert "logit" in results

    assert results["y_prob"].shape[0] == 2
    assert results["y_true"].shape[0] == 2
    assert results["logit"].shape[0] == 2
    assert results["logit"].shape[1] == len(labels)

    loss = results["loss"]
    assert loss.ndim == 0
    assert np.isfinite(loss.detach().cpu().item())

    loss.backward()

    has_grad = any(
        param.grad is not None
        for param in model.parameters()
        if param.requires_grad
    )
    assert has_grad