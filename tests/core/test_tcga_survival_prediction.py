"""Synthetic unit tests for TCGASurvivalPrediction.

The task only calls ``patient.get_events(...)`` and indexes the result with
string keys, so a tiny stub is sufficient. The FakePatient stub keeps the tests fast and focused on the task's
__call__ logic; the end-to-end pipeline with a real Patient is covered
by test_dataset_to_model_pipeline in test_tcga_rna_seq.py.
"""

import json

import numpy as np
import pytest

from pyhealth.tasks import TCGASurvivalPrediction


class FakePatient:
    """Minimal stand-in for ``pyhealth.data.Patient`` used by the task."""

    def __init__(self, patient_id: str, events: dict) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: str):
        return self._events.get(event_type, [])


def make_patient(pid="P1", cohort="BRCA", time=120.0, event=1,
                 n_genes=256, seed=0):
    """Fake patient with a single ``tcga_rna_seq`` event, matching the
    real dataset's event structure."""
    rng = np.random.default_rng(seed)
    event_attrs = {
        "cohort": f"TCGA-{cohort}",
        "duration": time,
        "event": event,
        "embedding": json.dumps(rng.standard_normal(n_genes).astype(np.float32).tolist()),
    }
    return FakePatient(pid, {"tcga_rna_seq": [event_attrs]})


def test_task_produces_correctly_shaped_sample():
    task = TCGASurvivalPrediction()
    samples = task(make_patient())
    assert len(samples) == 1
    s = samples[0]
    assert s["duration"] == [120.0]
    assert s["event"] == [1.0]
    assert isinstance(s["cohort"], list) and isinstance(s["cohort"][0], float)
    assert s["embedding"].shape == (256,)


def test_task_rejects_invalid_duration():
    task = TCGASurvivalPrediction()
    with pytest.raises(ValueError, match="duration"):
        task(make_patient(time=-1.0))
    with pytest.raises(ValueError, match="duration"):
        task(make_patient(time=0.0))
    with pytest.raises(ValueError, match="duration"):
        task(make_patient(time=float("nan")))


def test_task_rejects_invalid_event():
    task = TCGASurvivalPrediction()
    with pytest.raises(ValueError, match="[Ee]vent must be 0 or 1"):
        task(make_patient(event=0.5))
    with pytest.raises(ValueError, match="[Ee]vent must be 0 or 1"):
        task(make_patient(event=2))


def test_task_rejects_wrong_embedding_dimension():
    task = TCGASurvivalPrediction()
    with pytest.raises(ValueError, match="256"):
        task(make_patient(n_genes=128))


def test_task_rejects_multiple_events():
    task = TCGASurvivalPrediction()
    p = make_patient()
    # Duplicate the event to trigger the guard
    p._events["tcga_rna_seq"].append(dict(p._events["tcga_rna_seq"][0]))
    with pytest.raises(ValueError, match="[Ee]xactly one"):
        task(p)


def test_task_rejects_unknown_cohort():
    task = TCGASurvivalPrediction()
    with pytest.raises(ValueError, match="Unknown TCGA cohort"):
        task(make_patient(cohort="XYZ"))


def test_task_returns_empty_for_missing_events():
    p = FakePatient("P2", {})
    assert TCGASurvivalPrediction()(p) == []


def test_cohort_mapping_is_reversible():
    for cohort, idx in TCGASurvivalPrediction.COHORT_TO_INT.items():
        assert TCGASurvivalPrediction.INT_TO_COHORT[idx] == cohort


def test_cohort_int_is_stable_for_known_cohorts():
    # BRCA, LUAD, KIRC are the three cohorts used in the example;
    # verify their indices are stable so example results are reproducible.
    idx = TCGASurvivalPrediction.COHORT_TO_INT
    assert idx["BRCA"] == 2
    assert idx["LUAD"] == 16
    assert idx["KIRC"] == 11

def test_int_to_cohort_instance_attribute():
    task = TCGASurvivalPrediction()
    assert task.int_to_cohort[2] == "BRCA"
    assert task.int_to_cohort[16] == "LUAD"
    # Custom mapping works too
    custom = TCGASurvivalPrediction(cohort_to_int={"foo": 0, "bar": 1})
    assert custom.int_to_cohort[0] == "foo"
    assert custom.cohort_to_int == {"foo": 0, "bar": 1}
