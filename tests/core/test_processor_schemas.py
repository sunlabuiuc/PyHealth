#!/usr/bin/env python3
"""
Test demonstrating how to use both string aliases and direct processor
classes in task schemas with real MIMIC-III data.
"""

import unittest
import logging
import polars as pl
import datetime
from typing import Dict, List

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.processors import TextProcessor, MultiLabelProcessor
from pyhealth.tasks.medical_coding import MIMIC3ICD9Coding
from pyhealth.tasks.base_task import BaseTask
from pyhealth.data.data import Patient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMIC3ICD9CodingStringSchema(BaseTask):
    """Medical coding task for MIMIC-III using ICD-9 codes.

    This task uses clinical notes to predict ICD-9 codes for a patient.

    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """

    task_name: str = "mimic3_icd9_coding"
    input_schema: Dict[str, str] = {"text": "text"}
    output_schema: Dict[str, str] = {"icd_codes": "multilabel"}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        filtered_df = df.filter(
            pl.col("patient_id").is_in(
                df.filter(pl.col("event_type") == "noteevents")
                .select("patient_id")
                .unique()
                .to_series()
            )
        )
        return filtered_df

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract the clinical notes and ICD-9 codes.

        Args:
            patient: Patient object containing events

        Returns:
            List of samples, each containing text and ICD codes
        """
        samples = []
        admissions = patient.get_events(event_type="admissions")
        for admission in admissions:

            text = ""
            icd_codes = set()

            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            # Get clinical notes
            notes = patient.get_events(
                event_type="noteevents", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            text = ""

            for note in notes:
                text += " " + note.text

            diagnoses_icd = [event.icd9_code for event in diagnoses_icd]
            procedures_icd = [event.icd9_code for event in procedures_icd]
            icd_codes = list(set(diagnoses_icd + procedures_icd))

            if text == "" or len(icd_codes) < 1:
                continue

            samples.append(
                {"patient_id": patient.patient_id, "text": text, "icd_codes": icd_codes}
            )

        return samples


class MIMIC3ICD9CodingHybridSchema(BaseTask):
    """Medical coding task for MIMIC-III using ICD-9 codes.

    This task uses clinical notes to predict ICD-9 codes for a patient.

    Args:
        task_name: Name of the task
        input_schema: Definition of the input data schema
        output_schema: Definition of the output data schema
    """

    task_name: str = "mimic3_icd9_coding"
    input_schema: Dict[str, str] = {"text": TextProcessor}
    output_schema: Dict[str, str] = {"icd_codes": "multilabel"}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        filtered_df = df.filter(
            pl.col("patient_id").is_in(
                df.filter(pl.col("event_type") == "noteevents")
                .select("patient_id")
                .unique()
                .to_series()
            )
        )
        return filtered_df

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process a patient and extract the clinical notes and ICD-9 codes.

        Args:
            patient: Patient object containing events

        Returns:
            List of samples, each containing text and ICD codes
        """
        samples = []
        admissions = patient.get_events(event_type="admissions")
        for admission in admissions:

            text = ""
            icd_codes = set()

            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            # Get clinical notes
            notes = patient.get_events(
                event_type="noteevents", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            text = ""

            for note in notes:
                text += " " + note.text

            diagnoses_icd = [event.icd9_code for event in diagnoses_icd]
            procedures_icd = [event.icd9_code for event in procedures_icd]
            icd_codes = list(set(diagnoses_icd + procedures_icd))

            if text == "" or len(icd_codes) < 1:
                continue

            samples.append(
                {"patient_id": patient.patient_id, "text": text, "icd_codes": icd_codes}
            )

        return samples


class TestProcessorSchemas(unittest.TestCase):
    """Test processor schema configurations with synthetic MIMIC-III data."""

    @classmethod
    def setUpClass(cls):
        """Set up the synthetic MIMIC-III dataset for all tests."""
        # Use synthetic MIMIC-III dataset from Google Cloud Storage
        cls.dataset_root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"

        # Load dataset with tables needed for medical coding
        tables = ["diagnoses_icd", "procedures_icd", "noteevents"]

        try:
            cls.dataset = MIMIC3Dataset(
                root=cls.dataset_root,
                tables=tables,
                dev=False,
                # dev=True,  # Consistently use dev mode for faster testing
            )
            logger.info("Successfully loaded synthetic MIMIC-III dataset in dev mode")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load synthetic MIMIC-III dataset: {e}")

    def test_string_aliases(self):
        """Test using string aliases (backward compatibility)."""
        logger.info("Testing string aliases schema")

        task = MIMIC3ICD9CodingStringSchema()
        sample_dataset = self.dataset.set_task(task)

        if len(sample_dataset) == 0:
            self.skipTest("No samples generated from synthetic dataset")

        # The SampleDataset is already built with processors
        self.assertGreater(len(sample_dataset), 0)

        # Verify processors were created correctly
        self.assertIn("text", sample_dataset.input_processors)
        self.assertIn("icd_codes", sample_dataset.output_processors)

        # Test a sample
        sample = sample_dataset[0]
        self.assertIn("text", sample)
        self.assertIn("icd_codes", sample)

        logger.info(f"String schema test passed with {len(sample_dataset)} samples")

    def test_mixed_approach(self):
        """Test using string aliases (backward compatibility)."""
        logger.info("Testing mixed approach schema")

        task = MIMIC3ICD9CodingHybridSchema()
        sample_dataset = self.dataset.set_task(task)

        if len(sample_dataset) == 0:
            self.skipTest("No samples generated from synthetic dataset")

        # The SampleDataset is already built with processors
        self.assertGreater(len(sample_dataset), 0)

        # Verify processors were created correctly
        self.assertIn("text", sample_dataset.input_processors)
        self.assertIn("icd_codes", sample_dataset.output_processors)

        # Test a sample
        sample = sample_dataset[0]
        self.assertIn("text", sample)
        self.assertIn("icd_codes", sample)

        logger.info(f"Hybrid schema test passed with {len(sample_dataset)} samples")

    def test_direct_classes(self):
        """Test using direct processor classes."""
        logger.info("Testing direct processor classes schema")

        task = MIMIC3ICD9Coding()  # Uses direct processor classes
        sample_dataset = self.dataset.set_task(task)

        # We expect real samples from the synthetic dataset
        self.assertGreater(
            len(sample_dataset),
            0,
            "Expected samples from synthetic dataset - check task implementation",
        )

        # Verify processors were created correctly
        self.assertIn("text", sample_dataset.input_processors)
        self.assertIn("icd_codes", sample_dataset.output_processors)

        # Verify processor types
        self.assertIsInstance(sample_dataset.input_processors["text"], TextProcessor)
        self.assertIsInstance(
            sample_dataset.output_processors["icd_codes"], MultiLabelProcessor
        )

        # Test a sample and verify it contains expected data
        sample = sample_dataset[0]
        self.assertIn("text", sample)
        self.assertIn("icd_codes", sample)

        # Check that the sample contains actual text and ICD codes
        self.assertIsInstance(sample["text"], str)
        self.assertGreater(len(sample["text"]), 0)

        # Check ICD codes (after processing should be tensor-like)
        self.assertTrue(hasattr(sample["icd_codes"], "__len__"))

        logger.info(f"Direct classes test passed with {len(sample_dataset)} samples")


if __name__ == "__main__":
    unittest.main()
