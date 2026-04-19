# -*- coding: utf-8 -*-
"""Tests for BHCSummarizationTask and HallucinationDetectionTask.

Covers Section 4.1 (summarization) and Section 4.7 (hallucination detection)
tasks from:

    Hegselmann et al. "A Data-Centric Approach To Generate Faithful and
    High Quality Patient Summaries with Large Language Models." CHIL 2024.

Run with:

    pytest tests/test_mimic4_note_tasks.py -v
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pyhealth.tasks.mimic4_note_tasks import (
    BHCSummarizationTask,
    HallucinationDetectionTask,
)

def make_event(
    brief_hospital_course: str,
    summary: str,
    has_hallucination: int = -1,
) -> MagicMock:
    """Build a synthetic discharge note event."""
    event = MagicMock()
    event.brief_hospital_course = brief_hospital_course
    event.summary = summary
    event.has_hallucination = has_hallucination
    return event


def make_visit(visit_id: str, events: list) -> MagicMock:
    """Build a synthetic visit containing discharge events."""
    visit = MagicMock()
    visit.visit_id = visit_id
    visit.get_event_list.return_value = events
    return visit


def make_patient(patient_id: str, visits: dict) -> MagicMock:
    """Build a synthetic patient with a dict of visits."""
    patient = MagicMock()
    patient.patient_id = patient_id
    patient.visits = visits
    return patient


# -------------------
# Synthetic patients 
# -------------------

PATIENT_1 = make_patient(
    "p001",
    {
        "v001": make_visit(
            "v001",
            [
                make_event(
                    brief_hospital_course=(
                        "Patient presented with chest pain and shortness "
                        "of breath. Admitted for pneumonia."
                    ),
                    summary=(
                        "You were admitted for a chest infection. "
                        "You received antibiotics and improved."
                    ),
                    has_hallucination=0,
                )
            ],
        )
    },
)

PATIENT_2 = make_patient(
    "p002",
    {
        "v002": make_visit(
            "v002",
            [
                make_event(
                    brief_hospital_course=(
                        "Patient with hypertension admitted for stroke. "
                        "MRI confirmed left hemisphere infarct."
                    ),
                    summary=(
                        "You were admitted for a mild fracture of the "
                        "left clavicle."
                    ),
                    has_hallucination=1,
                )
            ],
        )
    },
)

PATIENT_3 = make_patient(
    "p003",
    {
        "v003": make_visit(
            "v003",
            [
                make_event(
                    brief_hospital_course=(
                        "Post-op day 2 after appendectomy. "
                        "Vital signs stable. Pain controlled."
                    ),
                    summary=(
                        "You had your appendix removed. "
                        "You were given pain medications."
                    ),
                    has_hallucination=0,
                )
            ],
        )
    },
)

PATIENT_4 = make_patient(
    "p004",
    {
        "v004": make_visit(
            "v004",
            [
                make_event(
                    brief_hospital_course="",  # empty — should be skipped
                    summary="You were admitted for chest pain.",
                    has_hallucination=0,
                )
            ],
        )
    },
)

PATIENT_5 = make_patient(
    "p005",
    {
        "v005": make_visit(
            "v005",
            [
                make_event(
                    brief_hospital_course=(
                        "Patient with atrial fibrillation on anticoagulation."
                    ),
                    summary="",  # empty — should be skipped
                    has_hallucination=0,
                )
            ],
        )
    },
)


# -------------------------------
# Synthetic summarization output 
# -------------------------------

SYNTHETIC_GENERATED_ROWS = [
    {
        "bhc": (
            "Patient presented with chest pain. Admitted for ACS. "
            "Treated with aspirin and heparin."
        ),
        "target_summary": (
            "You were admitted for chest pain. "
            "You received blood thinners."
        ),
        "predicted_summary_S": (
            "You were admitted to the hospital for chest pain and "
            "received medications to treat your heart."
        ),
        "generated_words": 18,
    },
    {
        "bhc": (
            "Patient with DMII, HTN admitted for pneumonia. "
            "Started on IV antibiotics. Improved and discharged."
        ),
        "target_summary": (
            "You were treated for a lung infection with antibiotics."
        ),
        "predicted_summary_S": (
            "You were admitted for a lung infection and treated with "
            "antibiotics. Your condition improved."
        ),
        "generated_words": 16,
    },
    {
        "bhc": (
            "Post-op appendectomy patient. Pain controlled with Tylenol. "
            "Tolerating diet. Discharged home."
        ),
        "target_summary": "You had your appendix removed successfully.",
        "predicted_summary_S": (
            "You had surgery to remove your appendix and recovered well."
        ),
        "generated_words": 12,
    },
    {
        "bhc": (
            "Patient with atrial fibrillation. Rate controlled with "
            "metoprolol. Anticoagulation continued."
        ),
        "target_summary": (
            "You were treated for an irregular heartbeat with medications."
        ),
        "predicted_summary_S": (
            "You were admitted for an irregular heartbeat and started on "
            "medications to control your heart rate."
        ),
        "generated_words": 20,
    },
    {
        "bhc": (
            "Diabetic patient with HbA1c 9.2. Insulin regimen adjusted. "
            "Glucose controlled prior to discharge."
        ),
        "target_summary": (
            "Your blood sugar was high and we adjusted your insulin."
        ),
        "predicted_summary_S": (
            "Your blood sugar levels were high and we adjusted your "
            "diabetes medications."
        ),
        "generated_words": 14,
    },
]


# ----------------------------------------
# BHCSummarizationTask tests (Section 4.1)
# ----------------------------------------


class TestBHCSummarizationTask:
    """Tests for BHCSummarizationTask."""

    def test_task_name(self):
        """task_name is set correctly."""
        task = BHCSummarizationTask()
        assert task.task_name == "BHCSummarizationMIMIC4Note"

    def test_input_schema(self):
        """input_schema contains context as str."""
        task = BHCSummarizationTask()
        assert "context" in task.input_schema
        assert task.input_schema["context"] == "str"

    def test_output_schema(self):
        """output_schema contains summary as str."""
        task = BHCSummarizationTask()
        assert "summary" in task.output_schema
        assert task.output_schema["summary"] == "str"

    def test_call_returns_list(self):
        """__call__ returns a list."""
        task = BHCSummarizationTask()
        result = task(PATIENT_1)
        assert isinstance(result, list)

    def test_call_correct_sample_count(self):
        """__call__ returns one sample per valid event."""
        task = BHCSummarizationTask()
        assert len(task(PATIENT_1)) == 1
        assert len(task(PATIENT_2)) == 1

    def test_required_keys_present(self):
        """Each sample contains all required keys."""
        task = BHCSummarizationTask()
        sample = task(PATIENT_1)[0]
        assert {"patient_id", "visit_id", "context", "summary"}.issubset(
            sample.keys()
        )

    def test_context_text_correct(self):
        """context is populated from brief_hospital_course."""
        task = BHCSummarizationTask()
        sample = task(PATIENT_1)[0]
        assert "chest pain" in sample["context"]

    def test_summary_text_correct(self):
        """summary is populated from event summary."""
        task = BHCSummarizationTask()
        sample = task(PATIENT_1)[0]
        assert "admitted" in sample["summary"]

    def test_empty_context_skipped(self):
        """Events with empty brief_hospital_course are skipped."""
        task = BHCSummarizationTask()
        assert len(task(PATIENT_4)) == 0

    def test_empty_summary_skipped(self):
        """Events with empty summary are skipped."""
        task = BHCSummarizationTask()
        assert len(task(PATIENT_5)) == 0

    def test_patient_id_preserved(self):
        """patient_id is correctly carried through."""
        task = BHCSummarizationTask()
        assert task(PATIENT_1)[0]["patient_id"] == "p001"

    def test_visit_id_preserved(self):
        """visit_id is correctly carried through."""
        task = BHCSummarizationTask()
        assert task(PATIENT_1)[0]["visit_id"] == "v001"

    # -------------------------
    # 4.1 output quality tests 
    # -------------------------

    def test_generated_columns_present(self):
        """Generated output has required bhc, target, predicted columns."""
        for row in SYNTHETIC_GENERATED_ROWS:
            assert "bhc" in row
            assert "target_summary" in row
            assert "predicted_summary_S" in row

    def test_bhc_is_string(self):
        """BHC field is always a string."""
        for row in SYNTHETIC_GENERATED_ROWS:
            assert isinstance(row["bhc"], str)

    def test_target_summary_is_string(self):
        """Target summary is always a string."""
        for row in SYNTHETIC_GENERATED_ROWS:
            assert isinstance(row["target_summary"], str)

    def test_predicted_summary_is_string(self):
        """Predicted summary is always a string."""
        for row in SYNTHETIC_GENERATED_ROWS:
            assert isinstance(row["predicted_summary_S"], str)

    def test_predicted_not_empty(self):
        """At least 90% of predicted summaries are non-empty."""
        non_empty = sum(
            1 for r in SYNTHETIC_GENERATED_ROWS
            if len(r["predicted_summary_S"]) > 0
        )
        assert non_empty / len(SYNTHETIC_GENERATED_ROWS) >= 0.9

    def test_predicted_length_reasonable(self):
        """Average generated summary length is under 200 words."""
        lengths = [
            len(r["predicted_summary_S"].split())
            for r in SYNTHETIC_GENERATED_ROWS
        ]
        assert np.mean(lengths) < 200

    def test_word_count_column_present(self):
        """generated_words column is present in output."""
        for row in SYNTHETIC_GENERATED_ROWS:
            assert "generated_words" in row

    def test_word_count_is_numeric(self):
        """generated_words values are numeric."""
        for row in SYNTHETIC_GENERATED_ROWS:
            assert isinstance(row["generated_words"], (int, float, np.integer))


# ----------------------------------------------
# HallucinationDetectionTask tests (Section 4.7)
# ----------------------------------------------


class TestHallucinationDetectionTask:
    """Tests for HallucinationDetectionTask."""

    def test_task_name(self):
        """task_name is set correctly."""
        task = HallucinationDetectionTask()
        assert task.task_name == "HallucinationDetectionMIMIC4Note"

    def test_input_schema(self):
        """input_schema contains context and summary as str."""
        task = HallucinationDetectionTask()
        assert task.input_schema["context"] == "str"
        assert task.input_schema["summary"] == "str"

    def test_output_schema(self):
        """output_schema contains label as binary."""
        task = HallucinationDetectionTask()
        assert task.output_schema["label"] == "binary"

    def test_call_returns_list(self):
        """__call__ returns a list."""
        task = HallucinationDetectionTask()
        assert isinstance(task(PATIENT_1), list)

    def test_correct_sample_count(self):
        """__call__ returns one sample per valid event."""
        task = HallucinationDetectionTask()
        assert len(task(PATIENT_1)) == 1

    def test_required_keys_present(self):
        """Each sample contains all required keys."""
        task = HallucinationDetectionTask()
        sample = task(PATIENT_1)[0]
        assert {"patient_id", "visit_id", "context", "summary", "label"}.issubset(
            sample.keys()
        )

    def test_label_faithful_is_zero(self):
        """Faithful summary (has_hallucination=0) gets label=0."""
        task = HallucinationDetectionTask()
        assert task(PATIENT_1)[0]["label"] == 0

    def test_label_hallucinated_is_one(self):
        """Hallucinated summary (has_hallucination=1) gets label=1."""
        task = HallucinationDetectionTask()
        assert task(PATIENT_2)[0]["label"] == 1

    def test_label_in_valid_range(self):
        """Label is always -1, 0, or 1."""
        task = HallucinationDetectionTask()
        for patient in [PATIENT_1, PATIENT_2, PATIENT_3]:
            for sample in task(patient):
                assert sample["label"] in (-1, 0, 1)

    def test_empty_context_skipped(self):
        """Events with empty context are skipped."""
        task = HallucinationDetectionTask()
        assert len(task(PATIENT_4)) == 0

    def test_empty_summary_skipped(self):
        """Events with empty summary are skipped."""
        task = HallucinationDetectionTask()
        assert len(task(PATIENT_5)) == 0

    def test_default_label_minus_one(self):
        """Default label is -1 when no annotation available."""
        task = HallucinationDetectionTask()
        unannotated = make_patient(
            "p_anon",
            {
                "v_anon": make_visit(
                    "v_anon",
                    [
                        make_event(
                            brief_hospital_course="Patient admitted for surgery.",
                            summary="You had surgery.",
                        )
                    ],
                )
            },
        )
        # Remove has_hallucination to simulate missing annotation
        unannotated.visits["v_anon"].get_event_list.return_value[
            0
        ].has_hallucination = -1
        sample = task(unannotated)[0]
        assert sample["label"] == -1

    def test_custom_default_label(self):
        """Custom default_label is respected."""
        task = HallucinationDetectionTask(default_label=0)
        assert task.default_label == 0

    def test_patient_id_preserved(self):
        """patient_id is correctly carried through."""
        task = HallucinationDetectionTask()
        assert task(PATIENT_2)[0]["patient_id"] == "p002"

    def test_visit_id_preserved(self):
        """visit_id is correctly carried through."""
        task = HallucinationDetectionTask()
        assert task(PATIENT_2)[0]["visit_id"] == "v002"

    def test_context_matches_bhc(self):
        """context field contains BHC text."""
        task = HallucinationDetectionTask()
        sample = task(PATIENT_2)[0]
        assert "stroke" in sample["context"]

    def test_summary_matches_event(self):
        """summary field contains DI text."""
        task = HallucinationDetectionTask()
        sample = task(PATIENT_2)[0]
        assert "clavicle" in sample["summary"]