"""Synthetic unit tests for TCGASurvivalPrediction."""

import json

import numpy as np
import pytest

from pyhealth.tasks import TCGASurvivalPrediction


class FakePatient:
    """Minimal stand-in for ``pyhealth.data.Patient``."""

    def __init__(self, patient_id: str, events: dict) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: str):
        return self._events.get(event_type, [])


def make_patient(pid="P1", cohort="BRCA", time=120.0, event=1,
                 n_genes=256, seed=0):
    rng = np.random.default_rng(seed)
    event_attrs = {
        "cohort": f"TCGA-{cohort}",
        "duration": time,   # CSV column name (dataset's internal field)
        "event": event,
        "embedding": json.dumps(
            rng.standard_normal(n_genes).astype(np.float32).tolist()
        ),
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
    idx = TCGASurvivalPrediction.COHORT_TO_INT
    assert idx["BRCA"] == 2
    assert idx["LUAD"] == 16
    assert idx["KIRC"] == 11