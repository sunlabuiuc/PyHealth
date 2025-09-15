#!/usr/bin/env python3
"""
Test demonstrating how to use direct processor classes in task schemas
with real MIMIC-III data using actual medical coding tasks.
"""

import unittest
import logging

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.processors import TextProcessor, MultiLabelProcessor
from pyhealth.tasks.medical_coding import MIMIC3ICD9Coding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def test_direct_classes(self):
        """Test using direct processor classes."""
        print("\n" + "=" * 80)
        print("TEST: Direct Processor Classes Schema")
        print("=" * 80)
        logger.info(
            "🧪 STARTING: test_direct_classes() - "
            "Testing direct processor class references"
        )

        print("📋 Testing task: MIMIC3ICD9Coding")
        print("📋 Input schema: {'text': TextProcessor} (direct class)")
        print("📋 Output schema: {'icd_codes': MultiLabelProcessor} " "(direct class)")

        # First, let's debug what's available in our dataset
        print("\n🔍 DEBUGGING: Examining dataset contents...")

        # Get a sample patient to see what data is available
        patient_ids = self.dataset.unique_patient_ids
        if not patient_ids:
            self.skipTest("No patients found in synthetic dataset")

        sample_patient = self.dataset.get_patient(patient_ids[1])
        all_events = sample_patient.get_events()
        print(f"   - Sample patient has {len(all_events)} events")

        # Check what event types are available
        event_types = set([event.event_type for event in all_events])
        print(f"   - Available event types: {event_types}")

        # Check for specific data we need
        noteevents = sample_patient.get_events(event_type="noteevents")
        diagnoses = sample_patient.get_events(event_type="diagnoses_icd")
        print(
            f"   - Patient {sample_patient.patient_id}: "
            f"{len(noteevents)} notes, {len(diagnoses)} diagnoses"
        )

        # If we have notes, let's see what they contain
        if noteevents:
            note = noteevents[0]
            print(f"   - Sample note attributes: {list(note.attr_dict.keys())}")
            if hasattr(note, "text") and note.text:
                print(f"   - Note text length: {len(note.text)}")
            elif "text" in note.attr_dict:
                print(f"   - Note text length: {len(note.attr_dict['text'])}")

        # If we have diagnoses, let's see what they contain
        if diagnoses:
            diag = diagnoses[0]
            print(
                f"   - Sample diagnosis attributes: " f"{list(diag.attr_dict.keys())}"
            )
            if hasattr(diag, "icd9_code"):
                print(f"   - Diagnosis ICD9 code: {diag.icd9_code}")

        print("\n🔄 PROCESSING: Creating task and calling dataset.set_task()...")
        task = MIMIC3ICD9Coding()  # Uses direct processor classes
        sample_dataset = self.dataset.set_task(task)

        # We expect real samples from the synthetic dataset
        self.assertGreater(
            len(sample_dataset),
            0,
            "Expected samples from synthetic dataset - " "check task implementation",
        )
        print(f"✅ Generated {len(sample_dataset)} samples from dataset")

        print("\n🔧 VALIDATION: Checking processor creation and types...")
        # Verify processors were created correctly
        self.assertIn("text", sample_dataset.input_processors)
        self.assertIn("icd_codes", sample_dataset.output_processors)

        # Verify processor types
        self.assertIsInstance(sample_dataset.input_processors["text"], TextProcessor)
        self.assertIsInstance(
            sample_dataset.output_processors["icd_codes"], MultiLabelProcessor
        )
        print("✅ Processors created successfully:")
        print(
            f"   - Input processor 'text': "
            f"{type(sample_dataset.input_processors['text']).__name__}"
        )
        print(
            f"   - Output processor 'icd_codes': "
            f"{type(sample_dataset.output_processors['icd_codes']).__name__}"
        )

        print("\n📊 SAMPLE TESTING: Examining processed sample data...")
        # Test a sample and verify it contains expected data
        sample = sample_dataset[0]
        self.assertIn("text", sample)
        self.assertIn("icd_codes", sample)

        # Check that the sample contains actual text and ICD codes
        self.assertIsInstance(sample["text"], str)
        self.assertGreater(len(sample["text"]), 0)
        print(f"✅ Sample text length: {len(sample['text'])}")
        print(f"✅ Sample text preview: '{sample['text'][:100]}...'")

        # Check ICD codes (after processing should be tensor-like)
        self.assertTrue(hasattr(sample["icd_codes"], "__len__"))
        print("✅ Sample has processed ICD codes")

        logger.info(
            f"🎉 SUCCESS: test_direct_classes() passed with "
            f"{len(sample_dataset)} samples"
        )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    unittest.main()
