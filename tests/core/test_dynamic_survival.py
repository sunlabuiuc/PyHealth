# Authors: Skyler Lehto (lehto2@illinois.edu), Ryan Bradley (ryancb3@illinois.edu), Weonah Choi (weonahc2@illinois.edu)
# Paper: Dynamic Survival Analysis for Early Event Prediction (Yèche et al., 2024)
# Link: https://arxiv.org/abs/2403.12818
# Description: Unit tests for DynamicSurvivalTask using synthetic data.

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

import json
import shutil
import tempfile
import unittest
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
# Test Suite
# ----------------------

class TestDynamicSurvivalTask(unittest.TestCase):

    def test_dynamic_survival_event(self):
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

    def test_dynamic_survival_censor(self):
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

        assert isinstance(samples, list)

        # Censor case may produce zero samples (valid)
        if len(samples) == 0:
            return

        s = samples[0]

        # Mask should contain zeros due to censoring
        assert np.any(s["mask"] == 0)

    def test_empty_patient(self):
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

    def test_generate_survival_label_basic(self):
        """Test correctness of survival label generation."""
        task = DynamicSurvivalTask(MockDataset(), horizon=5)

        y, mask = task.engine.generate_survival_label(
            anchor_time=10,
            event_time=12,
        )

        assert y[2] == 1
        assert mask[3] == 0

    def test_generate_anchors_basic(self):
        """Test anchor generation logic."""
        task = DynamicSurvivalTask(MockDataset(), observation_window=1)

        anchors = task.engine.generate_anchors(
            event_times=[0, 1],
            outcome_time=3,
        )

        assert len(anchors) > 0

    def test_end_to_end_pipeline_object_patients(self):
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

    def test_multiple_patients_processing(self):
        """Test engine processes a batch of dict-based patients without errors."""
        task = DynamicSurvivalTask(MockDataset())
        patients = create_patients(25)

        all_samples = []
        for p in patients:
            all_samples.extend(task.engine.process_patient(p))

        assert len(all_samples) > 0

    def test_censoring_mask_fixed(self):
        """Test censoring mask is correctly truncated after the censor step (fixed strategy)."""
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

    def test_censoring_mask_single(self):
        """generate_survival_label behavior is independent of anchor_strategy."""
        task = DynamicSurvivalTask(MockDataset(), horizon=5, anchor_strategy="single")
        y, mask = task.engine.generate_survival_label(
            anchor_time=10,
            event_time=None,
            censor_time=12,  # delta = 2, same convention as fixed
        )

        # anchor_strategy does not affect label generation — only anchor placement does.
        # With delta=2: steps 0,1,2 included; steps 3,4 excluded.
        assert np.all(mask[:3] == 1)
        assert np.all(mask[3:] == 0)
        assert np.sum(y) == 0

    def test_single_anchor_strategy(self):
        """Single anchor strategy produces exactly one anchor."""
        task = DynamicSurvivalTask(
            MockDataset(), anchor_strategy="single", observation_window=1
        )

        anchors = task.engine.generate_anchors([5, 10], outcome_time=20)

        assert len(anchors) == 1

    def test_empty_events(self):
        """Test that a patient with no visits produces no samples."""
        task = DynamicSurvivalTask(MockDataset())
        patient = {
            "patient_id": "p",
            "visits": [],
        }

        samples = task.engine.process_patient(patient)

        assert samples == []

    def test_output_format(self):
        """Test that output samples contain x, y, and mask as numpy arrays."""
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

    def test_event_before_anchor(self):
        """Test that an event occurring before the anchor zeroes the entire mask."""
        task = DynamicSurvivalTask(MockDataset(), horizon=5)

        y, mask = task.engine.generate_survival_label(
            anchor_time=10,
            event_time=8,
        )

        assert np.all(mask == 0)

    def test_event_within_horizon(self):
        """Test label and mask values when the event falls inside the horizon."""
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

    def test_event_outside_horizon(self):
        """Test that an event beyond the horizon produces all-zero y and all-one mask."""
        task = DynamicSurvivalTask(MockDataset(), horizon=5)

        y, mask = task.engine.generate_survival_label(
            anchor_time=10,
            event_time=20,
        )

        assert np.sum(y) == 0
        assert np.all(mask == 1)

    def test_no_valid_anchors(self):
        """Test that an observation window larger than patient history yields no samples."""
        task = DynamicSurvivalTask(MockDataset(), observation_window=100)

        patient = {
            "patient_id": "p1",
            "visits": [{"time": 1, "feature": np.zeros(1)}, {"time": 2, "feature": np.zeros(1)}],
            "outcome_time": 3,
        }

        samples = task.engine.process_patient(patient)

        assert samples == []

    def test_label_shape_consistency(self):
        """Test that y and mask shapes match the configured horizon."""
        task = DynamicSurvivalTask(MockDataset(), horizon=7)

        y, mask = task.engine.generate_survival_label(
            anchor_time=10,
            event_time=15,
        )

        assert y.shape == (7,)
        assert mask.shape == (7,)

    def test_full_pipeline_shapes(self):
        """Test output array shapes across all samples from a multi-visit patient."""
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

    def test_anchor_with_no_observation_window(self):
        """Test that a patient with visits only before the window still returns a list."""
        task = DynamicSurvivalTask(MockDataset(), observation_window=10)

        patient = {
            "patient_id": "p1",
            "visits": [{"time": 5, "feature": np.zeros(1)}],  # before window
            "outcome_time": 20,
        }

        samples = task.engine.process_patient(patient)

        assert isinstance(samples, list)

    def test_anchor_respects_censor_time(self):
        """Test that no anchor is placed at or after the censor time."""
        task = DynamicSurvivalTask(MockDataset(), anchor_interval=5)

        anchors = task.engine.generate_anchors(
            event_times=[5, 10, 15],
            outcome_time=None,
            censor_time=20,
        )

        assert all(a < 20 for a in anchors)

    def test_end_to_end_pipeline_dict_patients(self):
        """Test full pipeline with synthetic dict-based patients, validating output constraints."""
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

    def test_uses_temporary_directory(self):
        """Verify task output can be written to and cleaned up from a temp directory."""
        task = DynamicSurvivalTask(MockDataset(), horizon=5, observation_window=5)

        patient = {
            "patient_id": "p_tmp",
            "visits": [{"time": t, "feature": np.zeros(1)} for t in range(5, 30, 5)],
            "outcome_time": 35,
        }

        samples = task.engine.process_patient(patient)
        assert len(samples) > 0

        tmp_dir = tempfile.mkdtemp()
        try:
            out_path = tmp_dir + "/samples.json"
            with open(out_path, "w") as f:
                json.dump([s["visit_id"] for s in samples], f)

            with open(out_path) as f:
                contents = json.load(f)

            assert len(contents) == len(samples)
        finally:
            shutil.rmtree(tmp_dir)

    def test_large_mock_patient_cohort(self):
        """Test pipeline with 15 MockPatient objects covering mixed event/censor cases."""
        base_time = datetime(2025, 1, 1)

        patients = []
        for i in range(15):
            visits_data = [
                {"time": base_time + timedelta(days=d), "diagnosis": [str(1000 + i)]}
                for d in range(0, 20, 2)
            ]
            # Alternate event / censored patients
            death_time = base_time + timedelta(days=25) if i % 2 == 0 else None
            patients.append(MockPatient(pid=f"MP{i}", visits_data=visits_data, death_time=death_time))

        dataset = MockDataset(patients)
        task = DynamicSurvivalTask(dataset, horizon=10, observation_window=5, anchor_interval=3)
        samples = dataset.set_task(task)

        assert len(samples) > 0

        for s in samples:
            assert s["x"].ndim == 2
            assert s["y"].shape == (10,)
            assert s["mask"].shape == (10,)
            assert np.sum(s["y"]) <= 1
            assert np.all((s["mask"] == 0) | (s["mask"] == 1))

        for s in samples:
            assert s["x"].shape[0] > 0
            assert s["y"].sum() <= 1
            assert np.all((s["mask"] == 0) | (s["mask"] == 1))

    def test_feature_flags_use_proc_false(self):
        """Test that disabling procedure codes still produces valid samples."""
        base_time = datetime(2025, 4, 1)

        patient = MockPatient(
            pid="P_flags",
            death_time=base_time + timedelta(days=3),
            visits_data=[
                {"time": base_time, "diagnosis": ["4019"], "procedure": ["0011"]},
                {"time": base_time + timedelta(days=1), "diagnosis": ["4101"]},
            ],
        )

        dataset = MockDataset([patient])

        task = DynamicSurvivalTask(
            dataset,
            horizon=5,
            observation_window=1,
            anchor_interval=1,
            use_proc=False,
        )

        samples = dataset.set_task(task)

        assert isinstance(samples, list)
        if len(samples) > 0:
            assert samples[0]["x"].ndim == 2
            assert samples[0]["y"].shape == (5,)


if __name__ == "__main__":
    unittest.main()
