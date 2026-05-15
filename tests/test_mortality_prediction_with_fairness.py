"""Tests for MortalityPredictionWithFairnessMIMIC3 and audit_predictions.

Uses synthetic in-memory data only — no real or demo datasets, no network
calls. Whole suite runs in under 5 seconds.
"""
from datetime import datetime

import numpy as np
import pytest

from pyhealth.tasks.fairness_utils import audit_predictions
from pyhealth.tasks.mortality_prediction_with_fairness import (
    MortalityPredictionWithFairnessMIMIC3,
    _bin_age,
    _compute_age_years,
    _normalize_admission_type,
    _normalize_ethnicity,
    _surgical_status,
)


# --------- helpers ---------


class _Event:
    """Minimal event stand-in with arbitrary attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Patient:
    """Minimal Patient stand-in with a get_events(event_type, filters=None) API."""

    def __init__(self, patient_id, events_by_type, gender="M"):
        self.patient_id = patient_id
        self._events = events_by_type  # dict: event_type -> List[_Event]
        self.gender = gender
        self.birth_datetime = datetime(1950, 1, 1)

    def get_events(self, event_type, filters=None):
        events = self._events.get(event_type, [])
        if not filters:
            return events
        for key, op, val in filters:
            if op == "==":
                events = [e for e in events if getattr(e, key, None) == val]
        return events


def _make_patient(
    pid="P1",
    n_visits=2,
    dead_at_last=True,
    admission_type="EMERGENCY",
    insurance="Medicare",
    ethnicity="WHITE",
    first_careunit="MICU",
    gender="M",
    dx=("D1",),
    rx=("R1",),
    px=("P1",),
):
    visits = []
    for i in range(n_visits):
        die = 1 if (dead_at_last and i == n_visits - 1) else 0
        visits.append(
            _Event(
                hadm_id=f"H{i+1}",
                hospital_expire_flag=die,
                timestamp=datetime(2020, 1, i + 1),
                admittime=datetime(2020, 1, i + 1),
                admission_type=admission_type,
                insurance=insurance,
                ethnicity=ethnicity,
            )
        )
    by_type = {
        "admissions": visits,
        "patients": [_Event(gender=gender, dob=datetime(1950, 1, 1))],
        "icustays": [
            _Event(hadm_id=v.hadm_id, first_careunit=first_careunit) for v in visits
        ],
        "diagnoses_icd": [
            _Event(hadm_id=v.hadm_id, icd9_code=c) for v in visits for c in dx
        ],
        "procedures_icd": [
            _Event(hadm_id=v.hadm_id, icd9_code=c) for v in visits for c in px
        ],
        "prescriptions": [
            _Event(hadm_id=v.hadm_id, ndc=c) for v in visits for c in rx
        ],
    }
    return _Patient(pid, by_type, gender=gender)


# --------- helper function tests ---------


def test_bin_age_buckets():
    assert _bin_age(30) == "<50"
    assert _bin_age(50) == "50-65"
    assert _bin_age(64.9) == "50-65"
    assert _bin_age(70) == "65-75"
    assert _bin_age(84) == "75-85"
    assert _bin_age(85) == ">85"
    assert _bin_age(300) == ">85"  # MIMIC-shifted
    assert _bin_age(None) == "unknown"


def test_ethnicity_normalization():
    assert _normalize_ethnicity("WHITE")["ethnicity_4"] == "WHITE"
    assert _normalize_ethnicity("BLACK/AFRICAN AMERICAN")["ethnicity_4"] == "BLACK"
    assert _normalize_ethnicity("UNKNOWN/NOT SPECIFIED")["ethnicity_4"] == "UNK"
    assert _normalize_ethnicity("HISPANIC OR LATINO")["ethnicity_4"] == "OTHER"
    assert _normalize_ethnicity("ASIAN")["ethnicity_W"] == "NON-WHITE"
    assert _normalize_ethnicity(None)["ethnicity_4"] == "UNK"


def test_surgical_status():
    assert _surgical_status("SICU") == "Surgical"
    assert _surgical_status("CSRU") == "Surgical"
    assert _surgical_status("MICU") == "Non-surgical"
    assert _surgical_status(None) == "Non-surgical"


def test_admission_type_normalization():
    assert _normalize_admission_type("ELECTIVE") == "elective"
    assert _normalize_admission_type("EMERGENCY") == "emergency"
    assert _normalize_admission_type("URGENT") == "emergency"
    assert _normalize_admission_type(None) == "emergency"


def test_compute_age_handles_mimic_shifted_dob():
    p = _Patient("X", {}, gender="M")
    # Simulate MIMIC's "age > 89" shift: DOB in year 1700
    p.birth_datetime = datetime(1700, 1, 1)
    assert _compute_age_years(p, datetime(2100, 6, 15)) == 400.0  # raw year diff
    # _bin_age clips to 90
    assert _bin_age(_compute_age_years(p, datetime(2100, 6, 15))) == ">85"


# --------- task class tests ---------


def test_task_class_schema():
    task = MortalityPredictionWithFairnessMIMIC3()
    assert task.task_name == "MortalityPredictionWithFairnessMIMIC3"
    assert set(task.input_schema.keys()) == {"conditions", "procedures", "drugs"}
    assert task.output_schema == {"mortality": "binary"}


def test_task_basic_sample_has_expected_keys():
    task = MortalityPredictionWithFairnessMIMIC3()
    patient = _make_patient(dead_at_last=True, gender="F")
    samples = task(patient)
    assert len(samples) == 1
    expected = {
        "hadm_id", "patient_id", "conditions", "procedures", "drugs",
        "mortality", "sex", "age_group", "ethnicity_4", "ethnicity_W",
        "insurance_type", "surgical_status", "admission_type",
    }
    assert set(samples[0].keys()) == expected
    assert samples[0]["mortality"] == 1
    assert samples[0]["sex"] == "F"


def test_task_drops_visits_missing_codes():
    task = MortalityPredictionWithFairnessMIMIC3()
    # patient with empty procedures
    patient = _make_patient(px=())
    assert task(patient) == []


def test_task_drops_single_visit_patient():
    task = MortalityPredictionWithFairnessMIMIC3()
    patient = _make_patient(n_visits=1)
    assert task(patient) == []


def test_task_carries_all_cohort_attributes():
    task = MortalityPredictionWithFairnessMIMIC3()
    patient = _make_patient(
        admission_type="ELECTIVE",
        insurance="Medicaid",
        ethnicity="HISPANIC OR LATINO",
        first_careunit="SICU",
        gender="M",
    )
    s = task(patient)[0]
    assert s["admission_type"] == "elective"
    assert s["insurance_type"] == "Medicaid"
    assert s["ethnicity_4"] == "OTHER"
    assert s["ethnicity_W"] == "NON-WHITE"
    assert s["surgical_status"] == "Surgical"
    assert s["sex"] == "M"


# --------- audit utility tests ---------


def _synthetic_audit_samples(n=80, seed=0):
    rng = np.random.default_rng(seed)
    samples, probs, labels = [], [], []
    for i in range(n):
        is_advantaged = i % 2 == 0
        label = int(rng.random() < 0.3)
        if is_advantaged:
            prob = 0.1 + 0.8 * label + rng.normal(0, 0.05)
        else:
            prob = 0.4 + rng.normal(0, 0.2)
        prob = float(np.clip(prob, 0, 1))
        samples.append({
            "hadm_id": f"V{i}",
            "patient_id": f"P{i}",
            "mortality": label,
            "sex": "F" if is_advantaged else "M",
            "age_group": "50-65",
            "ethnicity_4": "WHITE" if is_advantaged else "BLACK",
            "ethnicity_W": "WHITE" if is_advantaged else "NON-WHITE",
            "insurance_type": "Private" if is_advantaged else "Medicaid",
            "surgical_status": "Non-surgical",
            "admission_type": "emergency",
        })
        probs.append(prob)
        labels.append(label)
    return samples, probs, labels


def test_audit_returns_dataframe_with_expected_columns():
    s, p, y = _synthetic_audit_samples()
    audit = audit_predictions(s, p, y, n_bootstrap=20)
    for col in ("grouping", "category", "metric", "median_cohort",
                "median_rest", "delta", "pvalue", "significantly_worse"):
        assert col in audit.columns
    assert len(audit) > 0


def test_audit_detects_disparity_in_fabricated_data():
    s, p, y = _synthetic_audit_samples(n=80, seed=1)
    audit = audit_predictions(s, p, y, n_bootstrap=50, significance_level=0.05)
    auroc = audit[audit["metric"] == "auroc"]
    assert not auroc.empty
    assert auroc["delta"].max() > 0.05


def test_audit_handles_missing_groupings():
    s, p, y = _synthetic_audit_samples(n=40)
    audit = audit_predictions(
        s, p, y, n_bootstrap=10,
        groupings=("sex", "totally_made_up_grouping"),
    )
    assert set(audit["grouping"].unique()) == {"sex"}


def test_audit_skips_tiny_cohorts():
    s, p, y = _synthetic_audit_samples(n=40)
    s[0]["age_group"] = "cohort-of-one"
    audit = audit_predictions(s, p, y, n_bootstrap=10)
    assert "cohort-of-one" not in audit["category"].values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
