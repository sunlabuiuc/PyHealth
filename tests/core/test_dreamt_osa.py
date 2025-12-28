import numpy as np  # not strictly needed, but kept for consistency
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import DREAMTOSAClassification


def test_dreamtosa_ahi_severity_4class():
    # Minimal event-level DataFrame (no splits → default split="train")
    df = pl.DataFrame(
        {
            "timestamp": [1],
            "patient_id": ["P1"],
            "visit_id": ["V1"],
            "event_type": ["dreamt_sleep"],
        }
    )

    # Create patient from raw event DataFrame
    patient = Patient(
        patient_id="P1",
        data_source=df,
    )

    # Add DREAMT metadata attributes expected by the task
    # (These are normally populated by DREAMTDataset / BaseDataset)
    patient.ahi = 22.5  # moderate OSA → class 2
    patient.age = 50
    patient.gender = "F"
    patient.bmi = 30.1
    patient.mean_sao2 = 94.2
    patient.arousal_index = 10.5
    patient.medical_history = "HTN"
    patient.sleep_disorders = "OSA"

    # Initialize task
    task = DREAMTOSAClassification(task="ahi_severity_4class")

    # Run task
    samples = task(patient)

    # One sample per patient
    assert len(samples) == 1
    sample = samples[0]

    # Check keys
    assert "feature" in sample
    assert "label" in sample
    assert "split" in sample
    assert "patient_id" in sample

    # Check label mapping: AHI=22.5 → 2 (moderate)
    assert sample["label"] == 2

    # Check feature dict contents
    features = sample["feature"]
    assert isinstance(features, dict)
    for key in [
        "age",
        "gender",
        "bmi",
        "mean_sao2",
        "arousal_index",
        "medical_history",
        "sleep_disorders",
    ]:
        assert key in features

    # With no splits table, default split is "train"
    assert sample["split"] == "train"
    assert sample["patient_id"] == "P1"


def test_dreamtosa_ahi_binary_15_with_splits():
    # Build event-level DataFrame including a "splits" event
    df = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "patient_id": ["P2", "P2"],
            "visit_id": ["V1", "V1"],
            "event_type": ["dreamt_sleep", "splits"],
            "split": [None, "val"],  # only used for the "splits" row
        }
    )

    patient = Patient(
        patient_id="P2",
        data_source=df,
    )

    # Add DREAMT metadata
    patient.ahi = 18.0  # >= 15 → label 1
    patient.age = 60
    patient.gender = "M"
    patient.bmi = 28.0
    patient.mean_sao2 = 93.0
    patient.arousal_index = 15.0
    patient.medical_history = "DM2"
    patient.sleep_disorders = "OSA"

    task = DREAMTOSAClassification(task="ahi_binary_15")
    samples = task(patient)

    assert len(samples) == 1
    sample = samples[0]

    # Binary label from AHI >= 15
    assert sample["label"] == 1

    # Split should be picked up from the "splits" event
    assert sample["split"] == "val"
    assert sample["patient_id"] == "P2"

    features = sample["feature"]
    assert isinstance(features, dict)
    assert features["age"] == 60
    assert features["gender"] == "M"
