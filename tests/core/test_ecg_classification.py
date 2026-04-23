"""Tests for ECGBinaryClassification task using synthetic patient objects.

Synthetic signals are generated with numpy — no real PTB-XL data required.
All tests complete in milliseconds.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_event(
    filename: str = "/fake/record",
    mi_label: int = 1,
    hyp_label: int = 0,
    sttc_label: int = 0,
    cd_label: int = 0,
    ecg_id: str = "1",
) -> MagicMock:
    """Return a mock Event whose attributes match PTB-XL metadata columns."""
    event = MagicMock()
    data: Dict[str, Any] = {
        "filename": filename,
        "ecg_id": ecg_id,
        "mi_label": mi_label,
        "hyp_label": hyp_label,
        "sttc_label": sttc_label,
        "cd_label": cd_label,
    }
    event.__getitem__ = lambda self, key: data[key]
    return event


def _make_fake_patient(events: List[MagicMock], patient_id: str = "P001") -> MagicMock:
    patient = MagicMock()
    patient.patient_id = patient_id
    patient.get_events.return_value = events
    return patient


def _fake_signal(leads: int = 12, length: int = 1000) -> np.ndarray:
    return np.random.randn(leads, length).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestECGBinaryClassificationInit:
    def test_valid_task_label(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        for label in ("MI", "HYP", "STTC", "CD"):
            task = ECGBinaryClassification(task_label=label)
            assert task.task_label == label

    def test_invalid_task_label_raises(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        with pytest.raises(ValueError, match="task_label must be one of"):
            ECGBinaryClassification(task_label="INVALID")

    def test_default_task_label_is_mi(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification()
        assert task.task_label == "MI"

    def test_schemas(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification()
        assert task.input_schema == {"ecg": "tensor"}
        assert task.output_schema == {"label": "binary"}


class TestECGBinaryClassificationCall:
    def test_returns_list_of_dicts(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification(task_label="MI", target_length=500)
        events = [_make_fake_event(mi_label=1), _make_fake_event(mi_label=0, ecg_id="2")]
        patient = _make_fake_patient(events)

        fake_signal = _fake_signal(12, 600)
        with patch.object(task, "_load_signal", return_value=fake_signal):
            samples = task(patient)

        assert isinstance(samples, list)
        assert len(samples) == 2

    def test_sample_keys(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification()
        events = [_make_fake_event()]
        patient = _make_fake_patient(events)

        with patch.object(task, "_load_signal", return_value=_fake_signal()):
            samples = task(patient)

        assert set(samples[0].keys()) >= {"patient_id", "ecg_id", "ecg", "label"}

    def test_ecg_shape_after_truncation(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification(target_length=500)
        events = [_make_fake_event()]
        patient = _make_fake_patient(events)

        # Signal longer than target → should truncate
        with patch.object(task, "_load_signal", return_value=_fake_signal(12, 800)):
            samples = task(patient)

        assert samples[0]["ecg"].shape == (12, 500)

    def test_ecg_shape_after_padding(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification(target_length=1000)
        events = [_make_fake_event()]
        patient = _make_fake_patient(events)

        # Signal shorter than target → should zero-pad
        with patch.object(task, "_load_signal", return_value=_fake_signal(12, 600)):
            samples = task(patient)

        assert samples[0]["ecg"].shape == (12, 1000)
        # Padded region should be zeros
        assert (samples[0]["ecg"][:, 600:] == 0).all()

    def test_label_correct_for_mi(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification(task_label="MI")
        events = [_make_fake_event(mi_label=1), _make_fake_event(mi_label=0, ecg_id="2")]
        patient = _make_fake_patient(events)

        with patch.object(task, "_load_signal", return_value=_fake_signal()):
            samples = task(patient)

        assert samples[0]["label"] == 1
        assert samples[1]["label"] == 0

    def test_label_correct_for_hyp(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification(task_label="HYP")
        events = [_make_fake_event(hyp_label=1)]
        patient = _make_fake_patient(events)

        with patch.object(task, "_load_signal", return_value=_fake_signal()):
            samples = task(patient)

        assert samples[0]["label"] == 1

    def test_failed_signal_load_skipped(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification()
        events = [_make_fake_event(ecg_id="1"), _make_fake_event(ecg_id="2")]
        patient = _make_fake_patient(events)

        # First load fails, second succeeds
        with patch.object(
            task, "_load_signal", side_effect=[None, _fake_signal()]
        ):
            samples = task(patient)

        assert len(samples) == 1
        assert samples[0]["ecg_id"] == "2"

    def test_normalize_produces_unit_std(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        rng = np.random.default_rng(0)
        signal = rng.normal(loc=5.0, scale=3.0, size=(12, 500)).astype(np.float32)
        normed = ECGBinaryClassification._normalize(signal)

        np.testing.assert_allclose(normed.mean(axis=1), 0.0, atol=1e-5)
        np.testing.assert_allclose(normed.std(axis=1), 1.0, atol=1e-5)

    def test_empty_patient_returns_empty_list(self):
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification()
        patient = _make_fake_patient([])
        samples = task(patient)
        assert samples == []
