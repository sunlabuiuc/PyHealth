"""Tests for :class:`~pyhealth.tasks.MedLingoJargonExpansionTask` (no network)."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from pyhealth.data import Patient
from pyhealth.tasks import MedLingoJargonExpansionTask


def _patient_from_row(
    pid: str,
    word1: str,
    word2: str,
    question: str,
    answer: str,
) -> Patient:
    df = pl.DataFrame(
        {
            "patient_id": [pid],
            "event_type": ["questions"],
            "timestamp": [datetime(2020, 1, 1)],
            "questions/word1": [word1],
            "questions/word2": [word2],
            "questions/question": [question],
            "questions/answer": [answer],
        }
    )
    return Patient(pid, df)


def test_one_shot_uses_csv_question_verbatim():
    task = MedLingoJargonExpansionTask(shot_mode="one_shot")
    q = "ICL_DEMO_ONLY_XYZ What is MI?"
    p = _patient_from_row("0", "MI", "STEMI", q, "myocardial infarction")
    out = task(p)
    assert len(out) == 1
    assert out[0]["prompt"] == q
    assert out[0]["answer"] == "myocardial infarction"
    assert out[0]["id"] == "0"


def test_zero_shot_ignores_question_field():
    task = MedLingoJargonExpansionTask(shot_mode="zero_shot")
    p = _patient_from_row(
        "1",
        "foo",
        "bar",
        "ICL_DEMO_ONLY_XYZ never use this in zero-shot",
        "plain",
    )
    out = task(p)
    assert len(out) == 1
    assert "ICL_DEMO_ONLY_XYZ" not in out[0]["prompt"]
    assert "foo" in out[0]["prompt"] and "bar" in out[0]["prompt"]


def test_zero_shot_and_one_shot_differ_on_same_row():
    p = _patient_from_row("2", "a", "b", "full released question", "lbl")
    z = MedLingoJargonExpansionTask(shot_mode="zero_shot")(p)[0]["prompt"]
    o = MedLingoJargonExpansionTask(shot_mode="one_shot")(p)[0]["prompt"]
    assert z != o
    assert o == "full released question"


def test_invalid_shot_mode():
    with pytest.raises(ValueError, match="shot_mode"):
        MedLingoJargonExpansionTask(shot_mode="bad")


def test_empty_answer_drops_sample():
    task = MedLingoJargonExpansionTask(shot_mode="one_shot")
    p = _patient_from_row("3", "a", "b", "q", "")
    assert task(p) == []


def test_zero_shot_requires_both_words():
    task = MedLingoJargonExpansionTask(shot_mode="zero_shot")
    p = _patient_from_row("4", "", "b", "q", "ans")
    assert task(p) == []


def test_task_name_includes_shot_mode():
    assert (
        MedLingoJargonExpansionTask(shot_mode="zero_shot").task_name
        == "MedLingoJargonExpansionTask/zero_shot"
    )
    assert (
        MedLingoJargonExpansionTask(shot_mode="one_shot").task_name
        == "MedLingoJargonExpansionTask/one_shot"
    )


def test_wrong_event_count_returns_empty():
    df = pl.DataFrame(
        {
            "patient_id": ["5", "5"],
            "event_type": ["questions", "questions"],
            "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "questions/word1": ["a", "b"],
            "questions/word2": ["c", "d"],
            "questions/question": ["q1", "q2"],
            "questions/answer": ["x", "y"],
        }
    )
    p = Patient("5", df)
    assert MedLingoJargonExpansionTask(shot_mode="one_shot")(p) == []
