"""
Name: Ranjithkumar Rajendran
NetID: rr54
Paper: KEEP (CHIL 2025) — Elhussein et al.

Unit tests for ERReadmissionMIMIC4 using synthetic
MockPatient data (3 core + 4 edge-case scenarios).
"""
import datetime
import polars as pl
from pyhealth.data import Patient
from pyhealth.tasks.mimic4_er_readmission import (
    ERReadmissionMIMIC4,
)


def _patient(pid, events):
    """Build a mock Patient from a list of dicts."""
    df = pl.DataFrame(events)
    return Patient(patient_id=pid, data_source=df)


# ----------------------------------------------------------------
# Core scenarios
# ----------------------------------------------------------------

def test_er_positive_readmission():
    """ER admit → readmitted within 10 days → label 1."""
    task = ERReadmissionMIMIC4(
        window=datetime.timedelta(days=30)
    )
    p = _patient("P1", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(
                2026, 1, 1, 12
            ),
            "event_type": "diagnoses_icd",
            "admissions/hadm_id": None,
            "admissions/admission_location": None,
            "admissions/dischtime": None,
            "diagnoses_icd/hadm_id": "H1",
            "diagnoses_icd/icd_code": "401.9",
            "diagnoses_icd/icd_version": "9",
        },
        {
            "timestamp": datetime.datetime(2026, 1, 10),
            "event_type": "admissions",
            "admissions/hadm_id": "H2",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-11 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    samples = task(p)
    assert len(samples) == 1
    assert samples[0]["readmission"] == 1
    assert isinstance(samples[0]["conditions"], list)
    assert samples[0]["conditions"] == ["9_401.9"]


def test_er_negative_readmission():
    """ER admit → next admit after 40 days → label 0."""
    task = ERReadmissionMIMIC4(
        window=datetime.timedelta(days=30)
    )
    p = _patient("P2", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(
                2026, 1, 1, 12
            ),
            "event_type": "diagnoses_icd",
            "admissions/hadm_id": None,
            "admissions/admission_location": None,
            "admissions/dischtime": None,
            "diagnoses_icd/hadm_id": "H1",
            "diagnoses_icd/icd_code": "250.00",
            "diagnoses_icd/icd_version": "9",
        },
        {
            "timestamp": datetime.datetime(2026, 3, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H2",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-03-05 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    samples = task(p)
    assert len(samples) == 1
    assert samples[0]["readmission"] == 0
    assert samples[0]["conditions"] == ["9_250.00"]


def test_non_er_admission_skipped():
    """Non-ER admit → should produce no samples."""
    task = ERReadmissionMIMIC4()
    p = _patient("P3", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "PHYSICIAN REFERRAL"
            ),
            "admissions/dischtime": (
                "2026-01-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(
                2026, 1, 1, 12
            ),
            "event_type": "diagnoses_icd",
            "admissions/hadm_id": None,
            "admissions/admission_location": None,
            "admissions/dischtime": None,
            "diagnoses_icd/hadm_id": "H1",
            "diagnoses_icd/icd_code": "428.0",
            "diagnoses_icd/icd_version": "9",
        },
        {
            "timestamp": datetime.datetime(2026, 1, 10),
            "event_type": "admissions",
            "admissions/hadm_id": "H2",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-11 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    samples = task(p)
    assert len(samples) == 0


# ----------------------------------------------------------------
# Edge-case scenarios
# ----------------------------------------------------------------

def test_single_admission_returns_empty():
    """Only one admission → impossible to determine
    readmission → return []."""
    task = ERReadmissionMIMIC4()
    p = _patient("P4", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    assert task(p) == []


def test_er_no_diagnoses_skipped():
    """ER admission exists but has zero diagnoses →
    that visit should be skipped."""
    task = ERReadmissionMIMIC4()
    p = _patient("P5", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(2026, 1, 10),
            "event_type": "admissions",
            "admissions/hadm_id": "H2",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-11 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    # No diagnoses_icd events → should return []
    assert task(p) == []


def test_custom_window_boundary():
    """Readmission exactly at window boundary (7 days)
    with a 7-day window should NOT be labelled 1
    because the comparison is strict less-than."""
    task = ERReadmissionMIMIC4(
        window=datetime.timedelta(days=7)
    )
    p = _patient("P6", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-02 00:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(
                2026, 1, 1, 6
            ),
            "event_type": "diagnoses_icd",
            "admissions/hadm_id": None,
            "admissions/admission_location": None,
            "admissions/dischtime": None,
            "diagnoses_icd/hadm_id": "H1",
            "diagnoses_icd/icd_code": "J18.9",
            "diagnoses_icd/icd_version": "10",
        },
        {
            # Exactly 7 days after discharge
            "timestamp": datetime.datetime(2026, 1, 9),
            "event_type": "admissions",
            "admissions/hadm_id": "H2",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-10 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    samples = task(p)
    assert len(samples) == 1
    # 7 days == window → strict "<" → label 0
    assert samples[0]["readmission"] == 0
    assert samples[0]["conditions"] == ["10_J18.9"]


def test_multiple_er_visits():
    """Patient with 3 ER admissions → should produce
    2 samples (one per non-last admission)."""
    task = ERReadmissionMIMIC4()
    p = _patient("P7", [
        {
            "timestamp": datetime.datetime(2026, 1, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H1",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(
                2026, 1, 1, 6
            ),
            "event_type": "diagnoses_icd",
            "admissions/hadm_id": None,
            "admissions/admission_location": None,
            "admissions/dischtime": None,
            "diagnoses_icd/hadm_id": "H1",
            "diagnoses_icd/icd_code": "I10",
            "diagnoses_icd/icd_version": "10",
        },
        {
            "timestamp": datetime.datetime(2026, 1, 10),
            "event_type": "admissions",
            "admissions/hadm_id": "H2",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-01-11 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
        {
            "timestamp": datetime.datetime(
                2026, 1, 10, 6
            ),
            "event_type": "diagnoses_icd",
            "admissions/hadm_id": None,
            "admissions/admission_location": None,
            "admissions/dischtime": None,
            "diagnoses_icd/hadm_id": "H2",
            "diagnoses_icd/icd_code": "E11.9",
            "diagnoses_icd/icd_version": "10",
        },
        {
            "timestamp": datetime.datetime(2026, 2, 1),
            "event_type": "admissions",
            "admissions/hadm_id": "H3",
            "admissions/admission_location": (
                "EMERGENCY ROOM"
            ),
            "admissions/dischtime": (
                "2026-02-02 12:00:00"
            ),
            "diagnoses_icd/hadm_id": None,
            "diagnoses_icd/icd_code": None,
            "diagnoses_icd/icd_version": None,
        },
    ])
    samples = task(p)
    assert len(samples) == 2
    assert samples[0]["conditions"] == ["10_I10"]
    assert samples[1]["conditions"] == ["10_E11.9"]
    # H1→H2 is 8 days → readmit
    assert samples[0]["readmission"] == 1
    # H2→H3 is 21 days → readmit
    assert samples[1]["readmission"] == 1
