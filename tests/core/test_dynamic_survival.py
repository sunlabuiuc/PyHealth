# tests/core/test_dynamic_survival.py

import numpy as np
import pytest
from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask


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
        visits = [{"time": t} for t in range(5, 50, 5)]

        patients.append({
            "patient_id": f"p{i}",
            "visits": visits,
            "outcome_time": 60 if i % 2 == 0 else None,
            "censor_time": 55 if i % 2 == 1 else None,
        })
    return patients


def test_multiple_patients_processing():
    task = DynamicSurvivalTask()
    patients = create_patients(25)

    all_samples = []
    for p in patients:
        all_samples.extend(task(p))

    assert len(all_samples) > 0


def test_censoring_mask():
    task = DynamicSurvivalTask(horizon=5)
    y, mask = task.generate_survival_label(
        anchor_time=10,
        event_time=None,
        censor_time=12,
    )

    assert np.all(mask[:2] == 1)
    assert np.all(mask[2:] == 0)


def test_single_anchor_strategy():
    task = DynamicSurvivalTask(anchor_strategy="single")

    anchors = task.generate_anchors([5, 10], outcome_time=20)

    assert len(anchors) == 1
    assert anchors[0] == 20


def test_empty_events():
    task = DynamicSurvivalTask()
    patient = {
        "patient_id": "p",
        "visits": [],
    }

    samples = task(patient)

    assert samples == []


def test_output_format():
    task = DynamicSurvivalTask(observation_window=5, horizon=5)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": t} for t in [5, 10, 15]],
        "outcome_time": 20,
    }

    samples = task(patient)

    assert len(samples) > 0, "No samples were generated for the patient"
    s = samples[0]
    assert "x" in s and "y" in s and "mask" in s
    assert isinstance(s["x"], np.ndarray)
    assert isinstance(s["y"], np.ndarray)
    assert isinstance(s["mask"], np.ndarray)


def test_event_before_anchor():
    task = DynamicSurvivalTask(horizon=5)

    y, mask = task.generate_survival_label(
        anchor_time=10,
        event_time=8,
    )

    assert np.all(mask == 0)


def test_event_within_horizon():
    task = DynamicSurvivalTask(horizon=5)

    y, mask = task.generate_survival_label(
        anchor_time=10,
        event_time=12,
    )

    # delta = 2
    assert y[2] == 1
    assert np.sum(y) == 1
    assert np.all(mask[:3] == 1)
    assert np.all(mask[3:] == 0)


def test_event_outside_horizon():
    task = DynamicSurvivalTask(horizon=5)

    y, mask = task.generate_survival_label(
        anchor_time=10,
        event_time=20,
    )

    assert np.sum(y) == 0
    assert np.all(mask == 1)


def test_no_valid_anchors():
    task = DynamicSurvivalTask(observation_window=100)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": 1}, {"time": 2}],
        "outcome_time": 3,
    }

    samples = task(patient)

    assert samples == []


def test_label_shape_consistency():
    task = DynamicSurvivalTask(horizon=7)

    y, mask = task.generate_survival_label(
        anchor_time=10,
        event_time=15,
    )

    assert y.shape == (7,)
    assert mask.shape == (7,)


def test_full_pipeline_shapes():
    task = DynamicSurvivalTask(horizon=6)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": t} for t in range(5, 50, 5)],
        "outcome_time": 60,
    }

    samples = task(patient)

    for s in samples:
        assert s["y"].shape[0] == 6
        assert s["mask"].shape[0] == 6
        assert s["x"].ndim == 2


def test_anchor_with_no_observation_window():
    task = DynamicSurvivalTask(observation_window=10)

    patient = {
        "patient_id": "p1",
        "visits": [{"time": 5}],  # before window
        "outcome_time": 20,
    }

    samples = task(patient)

    assert isinstance(samples, list)


def test_anchor_respects_censor_time():
    task = DynamicSurvivalTask(anchor_interval=5)

    anchors = task.generate_anchors(
        event_times=[5, 10, 15],
        outcome_time=None,
        censor_time=20,
    )

    assert all(a < 20 for a in anchors)


def test_end_to_end_pipeline():
    task = DynamicSurvivalTask()

    patients = create_patients(5)
    samples = []

    for p in patients:
        samples.extend(task(p))

    assert len(samples) > 0

    for s in samples:
        assert s["x"].shape[0] > 0
        assert s["y"].sum() <= 1
        assert np.all((s["mask"] == 0) | (s["mask"] == 1))
