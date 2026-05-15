"""Hurtful Words Task - Bias Reproduction Study.

Task for the "Hurtful Words" (Zhang et al. 2020) reproduction study using MIMIC-III dataset.
Quantifies bias in clinical BERT models through demographic stratification.

The task extracts clinical notes and demographics (gender, ethnicity) for mortality prediction,
creating intersectional group labels for fairness analysis.

Examples:
    >>> from pyhealth.datasets import MIMIC3Dataset
    >>> from pyhealth.tasks import HurtfulWordsMortalityTask
    >>> dataset = MIMIC3Dataset(
    ...     root="/path/to/mimic-iii/1.4",
    ...     tables=["noteevents", "admissions", "patients"],
    ... )
    >>> task = HurtfulWordsMortalityTask()
    >>> samples = dataset.set_task(task)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import math

from .base_task import BaseTask


class HurtfulWordsMortalityTask(BaseTask):
    """Task for predicting mortality using clinical notes with demographic tracking.

    This task extracts clinical notes from NOTEEVENTS and patient demographics
    (gender, ethnicity) to create intersectional group labels for fairness analysis
    in the Hurtful Words bias reproduction study.

    The task predicts in-hospital mortality for the next visit based on current
    clinical information, enabling fairness evaluation across demographic groups.
    """

    task_name: str = "HurtfulWordsMortalityTask"
    input_schema: Dict[str, str] = {
        "clinical_notes": "text",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient for the mortality prediction task.

        Extracts clinical notes from NOTEEVENTS, demographics from PATIENTS and
        ADMISSIONS tables, and creates intersectional group labels.

        Args:
            patient: Patient object with events accessible via get_events()

        Returns:
            List of sample dictionaries, each containing:
                - clinical_notes: Concatenated text from NOTEEVENTS
                - mortality: Binary label (0=survived, 1=died)
                - gender: Patient gender (M/F or UNKNOWN)
                - ethnicity: Patient ethnicity from admission record
                - intersectional_group: "{GENDER}_{ETHNICITY}" label
                - age: Age at admission
                - insurance: Insurance type
                - patient_id: De-identified patient ID
                - hadm_id: Hospital admission ID
        """
        samples = []

        # Get all admissions; skip if fewer than 2 (need current + next for label)
        visits = patient.get_events(event_type="admissions")
        if len(visits) <= 1:
            return []

        # Get patient demographics (static across visits)
        patient_events = patient.get_events(event_type="patients")
        if not patient_events:
            return []

        demo = patient_events[0]
        gender = str(demo.attr_dict.get("gender") or "UNKNOWN").strip()
        if not gender or gender in ["", "None"]:
            gender = "UNKNOWN"

        # Compute age from date of birth
        dob_raw = demo.attr_dict.get("dob")
        birth_dt = None
        if isinstance(dob_raw, datetime):
            birth_dt = dob_raw
        elif dob_raw is not None:
            try:
                birth_dt = datetime.fromisoformat(str(dob_raw))
            except Exception:
                birth_dt = None

        def compute_age(ts: Optional[datetime]) -> Optional[int]:
            """Compute age in years from timestamp."""
            if birth_dt is None or ts is None:
                return None
            age = int((ts - birth_dt).days // 365.25)
            return age if age >= 0 else None

        # Sort visits by timestamp
        visits = sorted(visits, key=lambda e: e.timestamp if e.timestamp else datetime.min)

        # Create samples: current visit -> predict mortality in next visit
        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Get mortality label from next visit
            mortality_label = int(next_visit.hospital_expire_flag) if next_visit.hospital_expire_flag in [0, 1, "0", "1"] else 0

            # Extract demographics for this visit
            ethnicity = str(visit.attr_dict.get("ethnicity") or "UNKNOWN").strip()
            if not ethnicity or ethnicity in ["", "None"]:
                ethnicity = "UNKNOWN"

            insurance = str(visit.attr_dict.get("insurance") or "UNKNOWN").strip()
            if not insurance or insurance in ["", "None"]:
                insurance = "UNKNOWN"

            # Compute age at this visit
            age = compute_age(visit.timestamp) if visit.timestamp else None
            if age is None or age < 0:
                continue

            # Create intersectional group label
            intersectional_group = f"{gender}_{ethnicity}"

            # Extract clinical notes from NOTEEVENTS
            notes = patient.get_events(
                event_type="noteevents",
                filters=[("hadm_id", "==", visit.hadm_id)]
            )
            clinical_notes = ""
            for note in notes:
                note_text = note.attr_dict.get("text") or ""
                if note_text:
                    clinical_notes += note_text + " "
            clinical_notes = clinical_notes.strip()

            # Skip if no notes available
            if not clinical_notes:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "hadm_id": visit.hadm_id,
                    "clinical_notes": clinical_notes,
                    "mortality": mortality_label,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "intersectional_group": intersectional_group,
                    "age": age,
                    "insurance": insurance,
                }
            )

        return samples
