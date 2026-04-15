from dataclasses import dataclass
from datetime import datetime

from pyhealth.tasks.temporal_hospital_mortality_prediction import (
    TemporalHospitalMortalityPredictionMIMIC3,
)


@dataclass
class DummyEvent:
    hadm_id: int | None = None
    timestamp: datetime | None = None
    hospital_expire_flag: int | None = None
    icd9_code: str | None = None
    drug: str | None = None
    drug_name: str | None = None


class DummyPatient:
    def __init__(self) -> None:
        self.patient_id = "p1"
        self._events = {
            "admissions": [
                DummyEvent(
                    hadm_id=100,
                    timestamp=datetime(2005, 1, 1),
                    hospital_expire_flag=1,
                )
            ],
            "diagnoses_icd": [DummyEvent(hadm_id=100, icd9_code="4019")],
            "procedures_icd": [DummyEvent(hadm_id=100, icd9_code="3893")],
            "prescriptions": [DummyEvent(hadm_id=100, drug="aspirin")],
        }

    def get_events(self, event_type, filters=None):
        events = self._events.get(event_type, [])
        if not filters:
            return events
        out = []
        for event in events:
            keep = True
            for attr, op, value in filters:
                if op == "==" and getattr(event, attr) != value:
                    keep = False
            if keep:
                out.append(event)
        return out


def test_temporal_task_generates_sample() -> None:
    task = TemporalHospitalMortalityPredictionMIMIC3()
    patient = DummyPatient()
    samples = task(patient)

    assert len(samples) == 1
    sample = samples[0]
    assert sample["mortality"] == 1
    assert sample["conditions"] == ["4019"]
    assert sample["procedures"] == ["3893"]
    assert sample["drugs"] == ["aspirin"]
    assert "admission_year" in sample
    assert "admission_year_raw" in sample


def test_temporal_task_requires_modalities() -> None:
    task = TemporalHospitalMortalityPredictionMIMIC3(require_all_modalities=True)
    patient = DummyPatient()
    patient._events["prescriptions"] = []
    samples = task(patient)
    assert samples == []
