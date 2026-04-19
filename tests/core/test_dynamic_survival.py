# tests/core/test_dynamic_survival.py

"""
Unit tests for DynamicSurvivalTask.

This module verifies:
- Sample generation through dataset.set_task()
- Correct label behavior (event and censor)
- Feature extraction pipeline
- Edge cases (empty patient)
- Core engine functionality (anchors and labels)

All tests use synthetic data and run quickly.
"""

from datetime import datetime, timedelta
import numpy as np
import pytest
from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask


# ----------------------
# Mock Classes
# ----------------------

class MockEvent:
    """Simple mock event object."""

    def __init__(self, code, timestamp, vocabulary):
        self.code = code
        self.timestamp = timestamp
        self.vocabulary = vocabulary


class MockVisit:
    """Mock visit containing EHR events."""

    def __init__(self, time, diagnosis=None, procedure=None, drug=None):
        self.encounter_time = time
        self.event_list_dict = {
            "DIAGNOSES_ICD": [
                MockEvent(c, time, "ICD9CM") for c in (diagnosis or [])
            ],
            "PROCEDURES_ICD": [
                MockEvent(c, time, "ICD9PROC") for c in (procedure or [])
            ],
            "PRESCRIPTIONS": [
                MockEvent(c, time, "NDC") for c in (drug or [])
            ],
        }


class MockPatient:
    """Mock patient object."""

    def __init__(self, pid, visits_data, death_time=None):
        self.patient_id = pid
        self.visits = {
            f"v{i}": MockVisit(**v) for i, v in enumerate(visits_data)
        }
        self.death_datetime = death_time


class MockDataset:
    """Minimal dataset wrapper for testing."""

    def __init__(self, patients=None):
        patients = patients or []
        self.patients = {p.patient_id: p for p in patients}

    def set_task(self, task):
        """Apply task to all patients."""
        samples = []
        for patient in self.patients.values():
            out = task(patient)
            if out:
                samples.extend(out)
        return samples


# ----------------------
# Helper
# ----------------------

def create_patients(n=10):
    """
    Creates a small deterministic synthetic dataset for unit tests.

    Characteristics:
    - Fixed visit times
    - Predictable event/censor patterns
    - Designed for fast, reproducible testing

    NOTE: This is intentionally simple and NOT meant for modeling experiments.
    """
    patients = []
    for i in range(n):
        visits = [{"time": t, "feature": np.zeros(1)} for t in range(5, 50, 5)]

        patients.append({
            "patient_id": f"p{i}",
            "visits": visits,
            "outcome_time": 60 if i % 2 == 0 else None,
            "censor_time": 55 if i % 2 == 1 else None,
        })
    return patients


# ----------------------
# Test: Event Case
# ----------------------

def test_dynamic_survival_event():
    """Test sample generation when event occurs."""

    base_time = datetime(2025, 4, 1)

    patient = MockPatient(
        pid="P1",
        death_time=base_time + timedelta(days=2),
        visits_data=[
            {"time": base_time, "diagnosis": ["4019"]},
            {"time": base_time + timedelta(days=1),
             "diagnosis": ["4101"]},
        ],
    )

    dataset = MockDataset([patient])

    task = DynamicSurvivalTask(
        dataset,
        horizon=5,
        observation_window=1,
        anchor_interval=1,
    )

    samples = dataset.set_task(task)

    assert len(samples) > 0

    s = samples[0]

    assert s["x"].ndim == 2
    assert s["y"].shape == (5,)
    assert s["mask"].shape == (5,)

    assert s["x"].dtype == np.float32
    assert s["y"].dtype == np.float32
    assert s["mask"].dtype == np.float32

    # Event at delta=1
    assert s["y"][1] == 1.0
    assert s["mask"][2] == 0.0

    # At most one event (DSA constraint)
    assert np.sum(s["y"]) <= 1
    # Mask must be binary
    assert np.all((s["mask"] == 0) | (s["mask"] == 1))


# ----------------------
# Test: Censor Case
# ----------------------

def test_dynamic_survival_censor():
    """Test behavior for censored patient."""

    base_time = datetime(2025, 4, 1)

    patient = MockPatient(
        pid="P2",
        death_time=None,
        visits_data=[
            {"time": base_time, "diagnosis": ["25000"]},
            {"time": base_time + timedelta(days=1),
             "diagnosis": ["4019"]},
        ],
    )

    dataset = MockDataset([patient])

    task = DynamicSurvivalTask(
        dataset,
        horizon=5,
        observation_window=1,
        anchor_interval=1,
    )

    samples = dataset.set_task(task)

    # Always returns list
    assert isinstance(samples, list)

    # Censor case may produce zero samples (valid)
    if len(samples) == 0:
        return

    s = samples[0]

    # Mask should contain zeros due to censoring
    assert np.any(s["mask"] == 0)


# ----------------------
# Test: Empty Patient
# ----------------------

def test_empty_patient():
    """Test patient with no visits."""

    patient = MockPatient(
        pid="P3",
        death_time=None,
        visits_data=[],
    )

    dataset = MockDataset([patient])

    task = DynamicSurvivalTask(
        dataset,
        horizon=5,
        observation_window=1,
        anchor_interval=1,
    )

    samples = dataset.set_task(task)

    assert len(samples) == 0


# ----------------------
# Test: Label Generation
# ----------------------

def test_generate_survival_label_basic():
    """Test correctness of survival label generation."""

    task = DynamicSurvivalTask(MockDataset(), horizon=5)

    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=12,
    )

    assert y[2] == 1
    assert mask[3] == 0


# ----------------------
# Test: Anchor Generation
# ----------------------

def test_generate_anchors_basic():
    """Test anchor generation logic."""

    task = DynamicSurvivalTask(MockDataset(), observation_window=1)

    anchors = task.engine.generate_anchors(
        event_times=[0, 1],
        outcome_time=3,
    )

    assert len(anchors) > 0


# ----------------------
# Test: End-to-End (PyHealth object patients)
# ----------------------

def test_end_to_end_pipeline_object_patients():
    """Test full pipeline with multiple PyHealth-style patients."""

    base_time = datetime(2025, 4, 1)

    patients = [
        MockPatient(
            pid="P1",
            death_time=base_time + timedelta(days=2),
            visits_data=[
                {"time": base_time, "diagnosis": ["4019"]},
                {"time": base_time + timedelta(days=1),
                 "diagnosis": ["4101"]},
            ],
        ),
        MockPatient(
            pid="P2",
            death_time=None,
            visits_data=[
                {"time": base_time, "diagnosis": ["25000"]},
                {"time": base_time + timedelta(days=1),
                 "diagnosis": ["4019"]},
            ],
        ),
    ]

    dataset = MockDataset(patients)

    task = DynamicSurvivalTask(
        dataset,
        horizon=5,
        observation_window=1,
        anchor_interval=1,
    )

    samples = dataset.set_task(task)

    assert len(samples) > 0

    for s in samples:
        assert s["x"].ndim == 2
        assert s["y"].shape[0] == 5
        assert s["mask"].shape[0] == 5
        assert np.sum(s["y"]) <= 1
        assert np.all((s["mask"] == 0) | (s["mask"] == 1))


# ----------------------
# Test: Multiple patients (dict-based)
# ----------------------

def test_multiple_patients_processing():
    task = DynamicSurvivalTask(MockDataset())
    patients = create_patients(25)

    all_samples = []
    for p in patients:
        all_samples.extend(task.engine.process_patient(p))

    assert len(all_samples) > 0


def test_censoring_mask_fixed():
    task = DynamicSurvivalTask(MockDataset(), horizon=5)
    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=None,
        censor_time=12,  # delta = 2
    )

    # Convention: censor_time is the last observed event-free step.
    # Steps 0..delta are included in the risk set (mask=1).
    # Steps delta+1.. are excluded (mask=0).
    # This mirrors the event case where mask[delta+1:] = 0.
    assert np.all(mask[:3] == 1)   # steps 0,1,2 included
    assert np.all(mask[3:] == 0)   # steps 3,4 excluded
    assert np.sum(y) == 0          # no event recorded for censored patient

def test_censoring_mask_single():
    task = DynamicSurvivalTask(MockDataset(), horizon=5, anchor_strategy='single')
    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=None,
        censor_time=12
    )

    assert np.all(mask[:1] == 1)
    assert np.all(mask[1:] == 0)


def test_single_anchor_strategy():
    task = DynamicSurvivalTask(MockDataset(), anchor_strategy="single")

    anchors = task.engine.generate_anchors([5, 10], outcome_time=20)

    assert len(anchors) == 1
    assert anchors[0] == 20


def test_empty_events():
    task = DynamicSurvivalTask(MockDataset())
    patient = {
        "patient_id": "p",
        "visits": [],
    }

    samples = task.engine.process_patient(patient)

    assert samples == []


def test_output_format():
    task = DynamicSurvivalTask(MockDataset(), observation_window=5, horizon=5)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": t, "feature": np.zeros(1)} for t in [5, 10, 15]],
        "outcome_time": 20,
    }

    samples = task.engine.process_patient(patient)

    assert len(samples) > 0, "No samples were generated for the patient"
    s = samples[0]
    assert "x" in s and "y" in s and "mask" in s
    assert isinstance(s["x"], np.ndarray)
    assert isinstance(s["y"], np.ndarray)
    assert isinstance(s["mask"], np.ndarray)


def test_event_before_anchor():
    task = DynamicSurvivalTask(MockDataset(), horizon=5)

    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=8,
    )

    assert np.all(mask == 0)


def test_event_within_horizon():
    task = DynamicSurvivalTask(MockDataset(), horizon=5)

    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=12,
    )

    # delta = 2
    assert y[2] == 1
    assert np.sum(y) == 1
    assert np.all(mask[:3] == 1)
    assert np.all(mask[3:] == 0)


def test_event_outside_horizon():
    task = DynamicSurvivalTask(MockDataset(), horizon=5)

    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=20,
    )

    assert np.sum(y) == 0
    assert np.all(mask == 1)


def test_no_valid_anchors():
    task = DynamicSurvivalTask(MockDataset(), observation_window=100)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": 1, "feature": np.zeros(1)}, {"time": 2, "feature": np.zeros(1)}],
        "outcome_time": 3,
    }

    samples = task.engine.process_patient(patient)

    assert samples == []


def test_label_shape_consistency():
    task = DynamicSurvivalTask(MockDataset(), horizon=7)

    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=15,
    )

    assert y.shape == (7,)
    assert mask.shape == (7,)


def test_full_pipeline_shapes():
    task = DynamicSurvivalTask(MockDataset(), horizon=6)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": t, "feature": np.zeros(1)} for t in range(5, 50, 5)],
        "outcome_time": 60,
    }

    samples = task.engine.process_patient(patient)

    for s in samples:
        assert s["y"].shape[0] == 6
        assert s["mask"].shape[0] == 6
        assert s["x"].ndim == 2


def test_anchor_with_no_observation_window():
    task = DynamicSurvivalTask(MockDataset(), observation_window=10)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": 5, "feature": np.zeros(1)}],  # before window
        "outcome_time": 20,
    }

    samples = task.engine.process_patient(patient)

    assert isinstance(samples, list)


def test_anchor_respects_censor_time():
    task = DynamicSurvivalTask(MockDataset(), anchor_interval=5)

    anchors = task.engine.generate_anchors(
        event_times=[5, 10, 15],
        outcome_time=None,
        censor_time=20,
    )

    assert all(a < 20 for a in anchors)


# ----------------------
# Test: End-to-End (dict patients)
# ----------------------

def test_end_to_end_pipeline_dict_patients():
    task = DynamicSurvivalTask(MockDataset())

    patients = create_patients(5)
    samples = []

    for p in patients:
        samples.extend(task.engine.process_patient(p))

    assert len(samples) > 0

    for s in samples:
        assert s["x"].shape[0] > 0
        assert s["y"].sum() <= 1
        assert np.all((s["mask"] == 0) | (s["mask"] == 1))
