"""Synthetic tests for :class:`~pyhealth.datasets.MedLingoDataset` (no real MedLingo)."""

from __future__ import annotations

import pandas as pd
import pytest

from pyhealth.datasets import MedLingoDataset
from pyhealth.tasks import MedLingoJargonExpansionTask

THREE_ROWS = [
    {
        "word1": "MI",
        "word2": "STEMI",
        "question": "Q0?",
        "answer": "heart attack",
    },
    {
        "word1": "HTN",
        "word2": "BP",
        "question": "Q1?",
        "answer": "high blood pressure",
    },
    {
        "word1": "DM",
        "word2": "A1c",
        "question": "Q2?",
        "answer": "diabetes",
    },
]


def _write_questions_csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def medlingo_three_patients(tmp_path_factory):
    """One parquet build shared by load + default-task tests."""
    base = tmp_path_factory.mktemp("medlingo_mod")
    root = base / "data"
    root.mkdir()
    cache = base / "cache"
    cache.mkdir()
    _write_questions_csv(root / "questions.csv", THREE_ROWS)
    return MedLingoDataset(root=str(root), cache_dir=str(cache), num_workers=1)


def test_medlingo_default_task_raw_sample(medlingo_three_patients):
    """Raw task output (no ``set_task`` / litdata)."""
    ds = medlingo_three_patients
    assert isinstance(ds.default_task, MedLingoJargonExpansionTask)
    assert ds.default_task.shot_mode == "one_shot"
    raw = ds.default_task(ds.get_patient("0"))
    assert len(raw) == 1
    assert raw[0]["prompt"] == "Q0?"
    assert raw[0]["answer"] == "heart attack"


def test_medlingo_loads_rows_as_patients(medlingo_three_patients):
    ds = medlingo_three_patients
    assert len(ds.unique_patient_ids) == 3
    p0 = ds.get_patient("0")
    evs = p0.get_events(event_type="questions")
    assert len(evs) == 1
    assert evs[0].word1 == "MI"
    assert evs[0].answer == "heart attack"


def test_medlingo_missing_column_raises(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_questions_csv(
        root / "questions.csv",
        [{"word1": "a", "word2": "b", "question": "q"}],
    )
    ds = MedLingoDataset(root=str(root), cache_dir=tmp_path / "c", num_workers=1)
    with pytest.raises(ValueError, match="missing required column"):
        _ = ds.unique_patient_ids


def test_medlingo_z_case_insensitive_columns(tmp_path):
    """Runs after module-scoped tests (name) so a second CSV build is isolated."""
    root = tmp_path / "data"
    root.mkdir()
    cache = tmp_path / "cache"
    _write_questions_csv(
        root / "questions.csv",
        [
            {
                "Word1": "a",
                "WORD2": "b",
                "Question": "Q?",
                "ANSWER": "ans",
            },
        ],
    )
    ds = MedLingoDataset(root=str(root), cache_dir=cache, num_workers=1)
    p = ds.get_patient("0")
    ev = p.get_events(event_type="questions")[0]
    assert ev.word1 == "a"
    assert ev.answer == "ans"
