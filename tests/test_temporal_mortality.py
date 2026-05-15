"""
Tests for the TemporalMortalityPredictionEICU task.

These tests use tiny synthetic patient objects and do not require real eICU data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from pyhealth.tasks.temporal_mortality import TemporalMortalityPredictionEICU


@dataclass
class FakeEvent:
    """Simple event container for synthetic tests."""
    attrs: Dict[str, object]

    def __getattr__(self, item):
        if item in self.attrs:
            return self.attrs[item]
        raise AttributeError(item)


class FakePatient:
    """Minimal patient stub implementing the methods used by the task."""

    def __init__(
        self,
        patient_id: str,
        patient_events: Optional[List[FakeEvent]] = None,
        diagnosis_events: Optional[List[FakeEvent]] = None,
        physicalexam_events: Optional[List[FakeEvent]] = None,
        medication_events: Optional[List[FakeEvent]] = None,
    ) -> None:
        self.patient_id = patient_id
        self._events = {
            "patient": patient_events or [],
            "diagnosis": diagnosis_events or [],
            "physicalexam": physicalexam_events or [],
            "medication": medication_events or [],
        }

    def get_events(self, event_type: str, filters=None, return_df: bool = False):
        assert return_df is False
        events = self._events.get(event_type, [])
        if not filters:
            return events

        filtered = []
        for event in events:
            keep = True
            for field, op, value in filters:
                event_value = getattr(event, field)
                if op == "==" and str(event_value) != str(value):
                    keep = False
                    break
            if keep:
                filtered.append(event)
        return filtered


def make_valid_patient() -> FakePatient:
    """Creates a patient with two stays so one sample can be generated."""
    patient_events = [
        FakeEvent(
            {
                "patientunitstayid": "stay_1",
                "hospitaldischargestatus": "Alive",
                "hospitaldischargeyear": 2014,
            }
        ),
        FakeEvent(
            {
                "patientunitstayid": "stay_2",
                "hospitaldischargestatus": "Expired",
                "hospitaldischargeyear": 2015,
            }
        ),
    ]

    diagnosis_events = [
        FakeEvent({"patientunitstayid": "stay_1", "icd9code": "038.9"}),
        FakeEvent({"patientunitstayid": "stay_2", "icd9code": "518.81"}),
    ]
    physicalexam_events = [
        FakeEvent({"patientunitstayid": "stay_1", "physicalexampath": "cardiovascular"}),
        FakeEvent({"patientunitstayid": "stay_2", "physicalexampath": "pulmonary"}),
    ]
    medication_events = [
        FakeEvent({"patientunitstayid": "stay_1", "drugname": "vancomycin"}),
        FakeEvent({"patientunitstayid": "stay_2", "drugname": "norepinephrine"}),
    ]

    return FakePatient(
        patient_id="patient_1",
        patient_events=patient_events,
        diagnosis_events=diagnosis_events,
        physicalexam_events=physicalexam_events,
        medication_events=medication_events,
    )


def test_temporal_mortality_generates_samples():
    task = TemporalMortalityPredictionEICU()
    patient = make_valid_patient()

    samples = list(task(patient))

    assert len(samples) == 1
    sample = samples[0]

    assert sample["patient_id"] == "patient_1"
    assert sample["visit_id"] == "stay_1"
    assert "conditions" in sample
    assert "procedures" in sample
    assert "drugs" in sample
    assert "mortality" in sample
    assert "discharge_year" in sample
    assert "stay_order" in sample
    assert "split_group" in sample


def test_temporal_mortality_label_is_binary():
    task = TemporalMortalityPredictionEICU()
    patient = make_valid_patient()

    sample = list(task(patient))[0]
    assert sample["mortality"] in [0, 1]


def test_temporal_mortality_split_group_is_valid():
    task = TemporalMortalityPredictionEICU()
    patient = make_valid_patient()

    sample = list(task(patient))[0]
    assert sample["split_group"] in ["early", "late"]


def test_temporal_mortality_skips_visits_without_required_features():
    task = TemporalMortalityPredictionEICU()

    patient = FakePatient(
        patient_id="patient_empty",
        patient_events=[
            FakeEvent(
                {
                    "patientunitstayid": "stay_x",
                    "hospitaldischargestatus": "Alive",
                    "hospitaldischargeyear": 2014,
                }
            ),
            FakeEvent(
                {
                    "patientunitstayid": "stay_y",
                    "hospitaldischargestatus": "Alive",
                    "hospitaldischargeyear": 2015,
                }
            ),
        ],
        diagnosis_events=[],
        physicalexam_events=[],
        medication_events=[],
    )

    samples = list(task(patient))
    assert samples == []


def test_temporal_mortality_requires_multiple_visits():
    task = TemporalMortalityPredictionEICU()

    patient = FakePatient(
        patient_id="single_visit_patient",
        patient_events=[
            FakeEvent(
                {
                    "patientunitstayid": "stay_only",
                    "hospitaldischargestatus": "Alive",
                    "hospitaldischargeyear": 2014,
                }
            )
        ],
        diagnosis_events=[
            FakeEvent({"patientunitstayid": "stay_only", "icd9code": "486"})
        ],
        physicalexam_events=[
            FakeEvent({"patientunitstayid": "stay_only", "physicalexampath": "respiratory"})
        ],
        medication_events=[
            FakeEvent({"patientunitstayid": "stay_only", "drugname": "ceftriaxone"})
        ],
    )

    samples = list(task(patient))
    assert samples == []