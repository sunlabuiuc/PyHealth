import numpy as np
import polars as pl
from pyhealth.data import Patient
from pyhealth.tasks import sepsis_ehr_task


def test_sepsis_ehr_task():
    # Build test flat event-level DataFrame
    df = pl.DataFrame({
        "timestamp": [1, 2, 3],
        "patient_id": ["P1", "P1", "P1"],
        "visit_id": ["V1", "V1", "V1"],
        "event_type": ["ehr", "ehr", "ehr"],   
        "table": ["ehr", "ehr", "ehr"],        
        "heart_rate": [80, 82, 85],
        "spo2": [96, 95, 97],
        "glucose": [120, 118, 110],
        "label": [1, 1, 1],
    })

    # Create patient from raw event DataFrame
    patient = Patient(
        patient_id="P1",
        data_source=df
    )

    # Run task
    samples = sepsis_ehr_task(patient)

    assert len(samples) == 1
    sample = samples[0]

    assert "ehr_features_mean" in sample
    assert "y" in sample
    assert isinstance(sample["ehr_features_mean"], np.ndarray)
    assert sample["y"] == 1
    assert sample["ehr_features_mean"].shape[0] == 3  # 3 vital signs
