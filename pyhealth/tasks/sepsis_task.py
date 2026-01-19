"""
EHR-only sepsis classification task for PyHealth. This task computes mean vital 
sign values per visit and assigns a binary sepsis labelf ro classification.

Attributes:
    patient (Patient):
        A PyHealth Patient object whose `data_source` attribute is a
        Polars DataFrame containing the patient's full event history.
Returns:
    List[Dict[str, Any]]:
            A list of samples. Each sample has the structure:

            {
                "patient_id": <str>,
                "visit_id": <str>,
                "ehr_features_mean": numpy.ndarray of shape (num_features,),
                "y": <int binary label>
            }
"""
import numpy as np
from typing import Dict, Any, List
from pyhealth.data import Patient


def sepsis_ehr_task(patient: Patient) -> List[Dict[str, Any]]:
    # Retrieve patient's event dataframe
    df = patient.data_source

    # Filter rows where event_type == "ehr"
    ehr_df = df.filter(df["event_type"] == "ehr")
    if ehr_df.is_empty():
        return []

    samples = []

    # Unique visits for this patient
    visit_ids = ehr_df["visit_id"].unique().to_list()

    # Explicit whitelist of vital sign columns
    vital_cols = ["heart_rate", "spo2", "glucose"]

    for vid in visit_ids:
        visit_df = ehr_df.filter(ehr_df["visit_id"] == vid)

        # Select only the known vital sign columns
        available = [c for c in vital_cols if c in visit_df.columns]
        x = visit_df.select(available).mean().to_numpy().astype(float).flatten()

        y = int(visit_df["label"][0])

        samples.append({
            "patient_id": patient.patient_id,
            "visit_id": vid,
            "ehr_features_mean": x,
            "y": y,
        })

    return samples
