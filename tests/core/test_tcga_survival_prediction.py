"""Synthetic unit tests for TCGASurvivalPrediction.

The task only calls ``patient.get_events(...)`` and indexes the result with
string keys, so a tiny stub is sufficient. Constructing a real ``Patient`` in
PyHealth 2.1 requires a polars DataFrame from a real dataset; that is not
appropriate for a fast synthetic unit test.
"""

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


def make_patient(pid="P1", cohort="BRCA", time=120.0, event=1, n_genes=64):
    clinical = {
        "survival_time": time,
        "event": event,
        "cohort": f"TCGA-{cohort}",
    }
    expression = {
        "gene_expression": np.random.randn(n_genes).astype(np.float32),
    }
    return FakePatient(pid, {"clinical": [clinical], "expression": [expression]})


def test_task_produces_correctly_shaped_sample():
    task = TCGASurvivalPrediction()
    samples = task(make_patient())
    assert len(samples) == 1
    s = samples[0]
    assert s["survival_time"] == [120.0]
    assert s["event"] == [1.0]
    assert isinstance(s["cohort"], list) and isinstance(s["cohort"][0], float)
    assert s["gene_expression"].shape == (64,)


def test_unknown_cohort_raises():
    task = TCGASurvivalPrediction()
    with pytest.raises(ValueError, match="Unknown TCGA cohort"):
        task(make_patient(cohort="XYZ"))


def test_missing_events_returns_empty():
    p = FakePatient("P2", {})
    assert TCGASurvivalPrediction()(p) == []