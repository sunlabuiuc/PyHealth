# Authors: Skyler Lehto (lehto2@illinois.edu),
#          Ryan Bradley (ryancb3@illinois.edu),
#          Weonah Choi (weonahc2@illinois.edu)
# Paper: Dynamic Survival Analysis for Early Event Prediction (Yèche et al., 2024)
# Link: https://arxiv.org/abs/2403.12818
# Description: Shared mock EHR classes used by dynamic survival example scripts.

"""Lightweight mock EHR objects for dynamic survival example scripts.

These classes stand in for real PyHealth patient/visit/event objects so
the example and ablation scripts can run without a local MIMIC download.
Imported by dynamic_survival_ablation.py and mimic_dynamic_survival_gru.py.
"""


class MockEvent:
    """A single coded EHR event (diagnosis, procedure, or prescription)."""

    def __init__(self, code, timestamp, vocabulary):
        """
        Args:
            code: Clinical code string (e.g. ICD-9, NDC).
            timestamp: Event datetime.
            vocabulary: Vocabulary name (e.g. "ICD9CM").
        """
        self.code = code
        self.timestamp = timestamp
        self.vocabulary = vocabulary


class MockVisit:
    """A single patient visit containing diagnosis events."""

    def __init__(self, time, diagnosis=None):
        """
        Args:
            time: Visit datetime used as encounter_time.
            diagnosis: Optional list of ICD-9 diagnosis code strings.
        """
        self.encounter_time = time
        self.event_list_dict = {
            "DIAGNOSES_ICD": [
                MockEvent(c, time, "ICD9CM") for c in (diagnosis or [])
            ],
            "PROCEDURES_ICD": [],
            "PRESCRIPTIONS": [],
        }


class MockPatient:
    """A patient with an ordered dict of MockVisit objects."""

    def __init__(self, pid, visits_data, death_time=None):
        """
        Args:
            pid: Unique patient identifier string.
            visits_data: List of dicts passed as kwargs to MockVisit.
            death_time: Optional datetime of death; None if censored.
        """
        self.patient_id = pid
        self.visits = {
            f"v{i}": MockVisit(**v) for i, v in enumerate(visits_data)
        }
        self.death_datetime = death_time


class MockDataset:
    """Minimal dataset wrapper that applies a task to all patients."""

    def __init__(self, patients):
        """
        Args:
            patients: List of MockPatient objects.
        """
        self.patients = {p.patient_id: p for p in patients}

    def set_task(self, task):
        """Apply task to every patient and return the collected samples.

        Called the same way as a real PyHealth dataset so example scripts
        are structurally identical to production usage.

        Args:
            task: A callable task (e.g. DynamicSurvivalTask) that accepts
                a patient object and returns a list of sample dicts.

        Returns:
            List of sample dicts from all patients combined.
        """
        samples = []
        for p in self.patients.values():
            out = task(p)
            if out:
                samples.extend(out)
        return samples
