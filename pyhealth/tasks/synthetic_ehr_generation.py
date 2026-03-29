"""
Task for generating synthetic EHR data.

This module contains tasks for training generative models on Electronic Health Records.
The tasks process patient visit sequences to create samples suitable for training
autoregressive models that can generate synthetic patient histories.
"""

from typing import Any, Dict, List

import polars as pl

from pyhealth.data import Patient
from .base_task import BaseTask


class SyntheticEHRGenerationMIMIC3(BaseTask):
    """Task for synthetic EHR generation using MIMIC-III dataset.

    This task prepares patient visit sequences for training autoregressive generative
    models. Each sample represents a patient's complete history of diagnoses across
    multiple visits, formatted as a nested sequence suitable for sequence-to-sequence
    modeling.

    The task creates samples where:
    - Input: Historical visit sequences (all visits except potentially the last)
    - Output: Full visit sequences (for teacher forcing during training)

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - visit_codes: Nested list of diagnosis codes per visit
        output_schema (Dict[str, str]): The schema for output data:
            - future_codes: Nested list of all diagnosis codes (for generation)

    Args:
        min_visits (int): Minimum number of visits required per patient. Default is 2.
        max_visits (int): Maximum number of visits to include per patient.
                         If None, includes all visits. Default is None.
    """

    task_name: str = "SyntheticEHRGenerationMIMIC3"
    input_schema: Dict[str, str] = {
        "visit_codes": "nested_sequence",
    }
    output_schema: Dict[str, str] = {
        "future_codes": "nested_sequence",
    }

    def __init__(self, min_visits: int = 2, max_visits: int = None):
        """Initialize the synthetic EHR generation task.

        Args:
            min_visits: Minimum number of visits required per patient
            max_visits: Maximum number of visits to include (None = no limit)
        """
        self.min_visits = min_visits
        self.max_visits = max_visits

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create synthetic EHR generation samples.

        For generative modeling, we create one sample per patient containing their
        complete visit history. Each visit contains diagnosis codes.

        Args:
            patient: Patient object with get_events method

        Returns:
            List containing a single sample with patient_id and nested visit sequences
        """
        samples = []

        # Get all admissions sorted chronologically
        admissions = patient.get_events(event_type="admissions")

        # Filter by minimum visits
        if len(admissions) < self.min_visits:
            return []

        # Limit to max_visits if specified
        if self.max_visits is not None:
            admissions = admissions[:self.max_visits]

        # Collect diagnosis codes for each visit
        visit_sequences = []
        valid_visit_count = 0

        for admission in admissions:
            # Get diagnosis codes using hadm_id
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )

            if diagnoses_icd is None or len(diagnoses_icd) == 0:
                continue

            conditions = (
                diagnoses_icd.select(pl.col("diagnoses_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Filter out empty or null codes
            conditions = [c for c in conditions if c]

            if len(conditions) > 0:
                visit_sequences.append(conditions)
                valid_visit_count += 1

        # Check if we have enough valid visits
        if valid_visit_count < self.min_visits:
            return []

        # Create a single sample with the full patient history
        # For autoregressive generation, both input and output are the same sequence
        sample = {
            "patient_id": patient.patient_id,
            "visit_codes": visit_sequences,
            "future_codes": visit_sequences,  # Same as input for teacher forcing
        }

        samples.append(sample)
        return samples


class SyntheticEHRGenerationMIMIC4(BaseTask):
    """Task for synthetic EHR generation using MIMIC-IV dataset.

    This task prepares patient visit sequences for training autoregressive generative
    models on MIMIC-IV data. Each sample represents a patient's complete history of
    diagnoses across multiple visits.

    The task creates samples where:
    - Input: Historical visit sequences (all visits except potentially the last)
    - Output: Full visit sequences (for teacher forcing during training)

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for input data:
            - visit_codes: Nested list of diagnosis codes per visit
        output_schema (Dict[str, str]): The schema for output data:
            - future_codes: Nested list of all diagnosis codes (for generation)

    Args:
        min_visits (int): Minimum number of visits required per patient. Default is 2.
        max_visits (int): Maximum number of visits to include per patient.
                         If None, includes all visits. Default is None.
    """

    task_name: str = "SyntheticEHRGenerationMIMIC4"
    input_schema: Dict[str, str] = {
        "visit_codes": "nested_sequence",
    }
    output_schema: Dict[str, str] = {
        "future_codes": "nested_sequence",
    }

    def __init__(self, min_visits: int = 2, max_visits: int = None):
        """Initialize the synthetic EHR generation task.

        Args:
            min_visits: Minimum number of visits required per patient
            max_visits: Maximum number of visits to include (None = no limit)
        """
        self.min_visits = min_visits
        self.max_visits = max_visits

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create synthetic EHR generation samples.

        For generative modeling, we create one sample per patient containing their
        complete visit history. Each visit contains diagnosis codes.

        Args:
            patient: Patient object with get_events method

        Returns:
            List containing a single sample with patient_id and nested visit sequences
        """
        samples = []

        # Get all admissions sorted chronologically
        admissions = patient.get_events(event_type="admissions")

        # Filter by minimum visits
        if len(admissions) < self.min_visits:
            return []

        # Limit to max_visits if specified
        if self.max_visits is not None:
            admissions = admissions[:self.max_visits]

        # Collect diagnosis codes for each visit
        visit_sequences = []
        valid_visit_count = 0

        for admission in admissions:
            # Get diagnosis codes using hadm_id
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )

            if diagnoses_icd is None or len(diagnoses_icd) == 0:
                continue

            conditions = (
                diagnoses_icd.select(pl.col("diagnoses_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Filter out empty or null codes
            conditions = [c for c in conditions if c]

            if len(conditions) > 0:
                visit_sequences.append(conditions)
                valid_visit_count += 1

        # Check if we have enough valid visits
        if valid_visit_count < self.min_visits:
            return []

        # Create a single sample with the full patient history
        sample = {
            "patient_id": patient.patient_id,
            "visit_codes": visit_sequences,
            "future_codes": visit_sequences,  # Same as input for teacher forcing
        }

        samples.append(sample)
        return samples
