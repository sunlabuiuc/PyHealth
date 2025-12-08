# test_mimic3_llm_diagnosis.py
import pandas as pd
import pytest
from pathlib import Path

from pyhealth.tasks.mimic3_llm_diagnosis import MIMIC3LLMDiagnosisTask

class DummyLLM:
    """A lightweight fake LLM for testing."""
    def classify(self, prompt: str):
        # Always return YES with confidence 0.9 for simplicity
        return "YES", 0.9

@pytest.fixture
def dummy_notes_csv(tmp_path: Path):
    """Create a temporary NOTEEVENTS.csv with minimal content."""
    csv_path = tmp_path / "NOTEEVENTS.csv"
    df = pd.DataFrame({
        "SUBJECT_ID": [1, 1, 2],
        "TEXT": [
            "Patient has CHF and hypertension.",
            "Follow-up note: stable condition.",
            "Patient denies chest pain."
        ]
    })
    df.to_csv(csv_path, index=False)
    return csv_path

def test_aggregate_by_patient(dummy_notes_csv):
    task = MIMIC3LLMDiagnosisTask(
        notes_path=str(dummy_notes_csv),
        diagnosis="heart failure",
        model_name="google/flan-t5-large"
    )
    # Replace heavy LLM with dummy
    task.model = DummyLLM()
    task.tokenizer = None  # not used in dummy

    patients = task.aggregate_by_patient()
    assert "PATIENT_NOTES" in patients.columns
    assert patients.shape[0] == 2  # two patients aggregated

def test_classify_patient(dummy_notes_csv):
    task = MIMIC3LLMDiagnosisTask(
        notes_path=str(dummy_notes_csv),
        diagnosis="heart failure",
        model_name="google/flan-t5-large"
    )
    # Replace heavy LLM with dummy
    task.model = DummyLLM()
    task.tokenizer = None

    patient_notes = "Patient has CHF and hypertension."
    result = task.classify_patient(patient_notes)
    assert result["diagnosis"] == "heart failure"
    assert result["verdict"] in ["YES", "NO"]
    assert 0.0 <= result["confidence"] <= 1.0

def test_run_task(dummy_notes_csv):
    task = MIMIC3LLMDiagnosisTask(
        notes_path=str(dummy_notes_csv),
        diagnosis="heart failure",
        model_name="google/flan-t5-large"
    )
    # Replace heavy LLM with dummy
    task.model = DummyLLM()
    task.tokenizer = None

    results = task.run(sample_size=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    for res in results:
        assert "patient_id" in res
        assert "diagnosis" in res
        assert "verdict" in res
        assert "confidence" in res
