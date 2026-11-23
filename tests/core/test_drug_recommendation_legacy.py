"""Behavior checks for legacy drug-recommendation entry points."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyhealth.tasks.drug_recommendation import (
    DrugRecommendationEICU,
    drug_recommendation_mimic3_fn,
    drug_recommendation_mimic4_fn,
    drug_recommendation_omop_fn,
)


class LegacyVisit:
    def __init__(self, visit_id, codes):
        self.visit_id = visit_id
        self.codes = codes

    def get_code_list(self, table):
        return list(self.codes.get(table, []))


class LegacyPatient:
    patient_id = "patient"

    def __init__(self, visits):
        self.visits = visits

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, index):
        return self.visits[index]


@pytest.mark.parametrize(
    ("task_fn", "tables", "history_key", "maps_ndc"),
    [
        (
            drug_recommendation_mimic3_fn,
            ("DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"),
            "drugs_hist",
            True,
        ),
        (
            drug_recommendation_mimic4_fn,
            ("diagnoses_icd", "procedures_icd", "prescriptions"),
            "drugs_hist",
            True,
        ),
        (
            drug_recommendation_omop_fn,
            ("condition_occurrence", "procedure_occurrence", "drug_exposure"),
            "drugs_all",
            False,
        ),
    ],
)
def test_legacy_tasks_filter_visits_and_exclude_current_drugs_from_history(
    task_fn, tables, history_key, maps_ndc
):
    def visit(visit_id, condition, procedure, drug):
        return LegacyVisit(visit_id, dict(zip(tables, (condition, procedure, drug))))

    visits = [
        visit("v1", ["c1"], ["p1"], ["d1"]),
        visit("excluded", ["c2"], [], ["d2"]),
        visit("v3", ["c3"], ["p3"], ["d3"]),
    ]
    with patch(
        "pyhealth.tasks.drug_recommendation._map_ndc_list_to_atc3",
        side_effect=lambda codes: ["atc-" + c for c in codes],
    ) as mapper:
        samples = task_fn(LegacyPatient(visits))
        assert task_fn(LegacyPatient(visits[:2])) == []
    first_drug, last_drug = ("atc-d1", "atc-d3") if maps_ndc else ("d1", "d3")
    assert [sample["visit_id"] for sample in samples] == ["v1", "v3"]
    assert samples[0][history_key] == [[]]
    assert samples[1][history_key] == [[first_drug], []]
    assert samples[1]["conditions"] == [["c1"], ["c3"]]
    assert samples[1]["procedures"] == [["p1"], ["p3"]]
    assert samples[1]["drugs"] == [last_drug]
    assert bool(mapper.call_count) is maps_ndc


class EICUPatient:
    patient_id = "patient"

    def __init__(self, stays, events):
        self.stays = [SimpleNamespace(patientunitstayid=s) for s in stays]
        self.events = events

    def get_events(self, event_type, filters=None):
        if event_type == "patient":
            return self.stays
        stay_id = filters[0][2]
        return self.events.get((event_type, stay_id), [])


def test_eicu_history_excludes_current_medication_and_incomplete_stay():
    events = {}
    for stay in ("1", "3"):
        events[("diagnosis", stay)] = [SimpleNamespace(icd9code="c" + stay)]
        events[("physicalexam", stay)] = [SimpleNamespace(physicalexampath="p" + stay)]
        events[("medication", stay)] = [SimpleNamespace(drugname="d" + stay)]
    task = DrugRecommendationEICU()
    samples = task(EICUPatient(["1", "2", "3"], events))
    assert [sample["visit_id"] for sample in samples] == ["1", "3"]
    assert samples[1]["conditions"] == [["c1"], ["c3"]]
    assert samples[1]["drugs_hist"] == [["d1"], []]
    assert samples[1]["drugs"] == ["d3"]
    assert task(EICUPatient(["1"], events)) == []
    assert task(EICUPatient(["1", "2"], events)) == []
