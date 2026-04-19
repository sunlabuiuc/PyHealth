"""Mortality prediction task for the Synthea synthetic EHR dataset.

Author: Justin Xu

Paper: Raphael Poulain, Mehak Gupta, and Rahmatollah Beheshti.
    "CEHR-GAN-BERT: Incorporating Temporal Information from Structured EHR
    Data to Improve Prediction Tasks." MLHC 2022, Section A.2 (Mortality-Disch).
    https://proceedings.mlr.press/v182/poulain22a.html

Description: Predicts whether a patient dies within a configurable window
    (default 365 days) after discharge from their latest inpatient encounter.
    Patients who die during the anchor encounter are excluded.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from .base_task import BaseTask


class MortalityPredictionSynthea(BaseTask):
    """Post-discharge mortality prediction on Synthea data.

    Cohort construction follows Poulain et al. MLHC 2022, Section A.2:

    1. Identify all **inpatient** encounters for the patient.
    2. Use the **latest** inpatient encounter as the anchor visit.
    3. Exclude patients who died **during** the anchor encounter
       (death date <= encounter stop time).
    4. Label = 1 if the patient died within ``prediction_window_days``
       of discharge; 0 otherwise.
    5. Collect condition / procedure / medication codes from all
       encounters up to and including the anchor.
    6. Require at least one feature code.

    Returns **one sample per patient**.

    Args:
        prediction_window_days: Number of days after discharge within
            which death counts as a positive label.  Default ``365``.

    Examples:
        >>> from pyhealth.datasets import SyntheaDataset
        >>> from pyhealth.tasks import MortalityPredictionSynthea
        >>> dataset = SyntheaDataset(
        ...     root="/path/to/synthea/csv",
        ...     tables=["conditions", "medications", "procedures"],
        ... )
        >>> task = MortalityPredictionSynthea(prediction_window_days=365)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MortalityPredictionSynthea"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "medications": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(self, prediction_window_days: int = 365, **kwargs):
        self.prediction_window_days = prediction_window_days
        super().__init__(**kwargs)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient and return at most one sample."""
        # ----------------------------------------------------------
        # 1. Get patient-level death date from the patients table
        # ----------------------------------------------------------
        patient_events = patient.get_events(event_type="patients")
        if not patient_events:
            return []

        patient_info = patient_events[0]
        death_date_raw = getattr(patient_info, "DEATHDATE", None)

        death_date = None
        if death_date_raw and str(death_date_raw).strip():
            try:
                death_date = datetime.strptime(
                    str(death_date_raw).strip()[:10], "%Y-%m-%d"
                )
            except (ValueError, TypeError):
                death_date = None

        # ----------------------------------------------------------
        # 2. Get all inpatient encounters, sorted by timestamp
        # ----------------------------------------------------------
        encounters = patient.get_events(event_type="encounters")
        inpatient = [
            e for e in encounters
            if getattr(e, "ENCOUNTERCLASS", "") == "inpatient"
        ]
        if not inpatient:
            return []

        # Sort by timestamp (START) ascending; take the latest
        inpatient.sort(key=lambda e: e.timestamp)
        anchor = inpatient[-1]

        # ----------------------------------------------------------
        # 3. Parse anchor stop time and exclude died-during-encounter
        # ----------------------------------------------------------
        stop_raw = getattr(anchor, "STOP", None)
        if not stop_raw or not str(stop_raw).strip():
            return []

        try:
            anchor_stop = datetime.strptime(
                str(stop_raw).strip(), "%Y-%m-%dT%H:%M:%SZ"
            )
        except (ValueError, TypeError):
            return []

        if death_date is not None and death_date <= anchor_stop:
            return []  # died during or before discharge

        # ----------------------------------------------------------
        # 4. Determine mortality label
        # ----------------------------------------------------------
        if death_date is not None:
            days_after_discharge = (death_date - anchor_stop).days
            mortality_label = int(
                days_after_discharge <= self.prediction_window_days
            )
        else:
            mortality_label = 0

        # ----------------------------------------------------------
        # 5. Collect feature codes from encounters <= anchor
        # ----------------------------------------------------------
        anchor_ts = anchor.timestamp

        conditions_events = patient.get_events(
            event_type="conditions", end=anchor_stop
        )
        procedures_events = patient.get_events(
            event_type="procedures", end=anchor_stop
        )
        medications_events = patient.get_events(
            event_type="medications", end=anchor_stop
        )

        conditions = [
            str(e.CODE) for e in conditions_events if getattr(e, "CODE", None)
        ]
        procedures = [
            str(e.CODE) for e in procedures_events if getattr(e, "CODE", None)
        ]
        medications = [
            str(e.CODE) for e in medications_events if getattr(e, "CODE", None)
        ]

        # Must have at least one feature code
        if len(conditions) + len(procedures) + len(medications) == 0:
            return []

        # ----------------------------------------------------------
        # 6. Build and return the single sample
        # ----------------------------------------------------------
        return [
            {
                "visit_id": str(getattr(anchor, "Id", "")),
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "medications": medications,
                "mortality": mortality_label,
            }
        ]
