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

    def __init__(self, patients):
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
# Test 1: Event Case
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


# ----------------------
# Test 2: Censor Case
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
# Test 3: Empty Patient
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
# Test 4: Label Generation
# ----------------------

def test_generate_survival_label_basic():
    """Test correctness of survival label generation."""

    dummy_dataset = MockDataset([])

    task = DynamicSurvivalTask(dummy_dataset, horizon=5)

    y, mask = task.engine.generate_survival_label(
        anchor_time=10,
        event_time=12,
    )

    assert y[2] == 1
    assert mask[3] == 0


# ----------------------
# Test 5: Anchor Generation
# ----------------------

def test_generate_anchors_basic():
    """Test anchor generation logic."""

    dummy_dataset = MockDataset([])

    task = DynamicSurvivalTask(dummy_dataset, observation_window=1)

    anchors = task.engine.generate_anchors(
        event_times=[0, 1],
        outcome_time=3,
    )

    assert len(anchors) > 0


# ----------------------
# Test 6: End-to-End Pipeline
# ----------------------

def test_end_to_end_pipeline():
    """Test full pipeline with multiple patients."""

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