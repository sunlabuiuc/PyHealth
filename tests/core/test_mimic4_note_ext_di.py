"""Unit tests for MimicIVNoteExtDIDataset and PatientSummaryGeneration.

Tests use synthetic data in test-resources/core/mimic4_note_ext_di/.
No real MIMIC data is required.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MimicIVNoteExtDIDataset
from pyhealth.tasks import PatientSummaryGeneration


class TestMimicIVNoteExtDIDatasetFromCSV(unittest.TestCase):
    """Test loading from a pre-built CSV (the standard YAML path)."""

    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic4_note_ext_di"
        )
        cls.cache_dir = tempfile.TemporaryDirectory()
        # Load directly from CSV — root points to directory with summaries.csv
        cls.dataset = MimicIVNoteExtDIDataset.__new__(MimicIVNoteExtDIDataset)
        # Bypass the JSONL conversion by calling BaseDataset.__init__ directly
        from pyhealth.datasets.base_dataset import BaseDataset

        config_path = str(
            Path(__file__).parent.parent.parent
            / "pyhealth"
            / "datasets"
            / "configs"
            / "mimic4_note_ext_di.yaml"
        )
        cls.dataset.variant = "test"
        BaseDataset.__init__(
            cls.dataset,
            root=str(cls.root),
            tables=["summaries"],
            dataset_name="mimic4_note_ext_di",
            config_path=config_path,
            cache_dir=cls.cache_dir.name,
        )
        cls.task = PatientSummaryGeneration()
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 5)

    def test_patient_has_events(self):
        patient = self.dataset.get_patient("0")
        events = patient.get_events(event_type="summaries")
        self.assertEqual(len(events), 1)

    def test_event_has_text_and_summary(self):
        patient = self.dataset.get_patient("0")
        event = patient.get_events(event_type="summaries")[0]
        self.assertIn("text", event)
        self.assertIn("summary", event)
        self.assertTrue(event.text.startswith("Brief Hospital Course:"))

    def test_task_sample_count(self):
        self.assertEqual(len(self.samples), 5)

    def test_task_sample_keys(self):
        sample = self.samples[0]
        self.assertIn("id", sample)
        self.assertIn("text", sample)
        self.assertIn("summary", sample)

    def test_task_sample_content(self):
        sample = self.samples[0]
        self.assertIsInstance(sample["text"], str)
        self.assertIsInstance(sample["summary"], str)
        self.assertGreater(len(sample["text"]), 50)
        self.assertGreater(len(sample["summary"]), 50)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, PatientSummaryGeneration)


class TestMimicIVNoteExtDIDatasetFromJSONL(unittest.TestCase):
    """Test loading from JSONL files (the normal user path)."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.cache_dir = tempfile.TemporaryDirectory()

        # Create a fake PhysioNet directory structure with JSONL files
        bhc_dir = os.path.join(
            cls.tmpdir.name, "mimic-iv-note-ext-di-bhc", "dataset"
        )
        os.makedirs(bhc_dir, exist_ok=True)

        # Write synthetic JSONL data
        records = [
            {
                "text": "Brief Hospital Course: Patient A "
                "presented with fever.",
                "summary": "You came to the hospital with a "
                "fever. You were treated with antibiotics.",
            },
            {
                "text": "Brief Hospital Course: Patient B "
                "had a fall.",
                "summary": "You were admitted after a fall. "
                "X-rays showed no fractures.",
            },
            {
                "text": "Brief Hospital Course: Patient C "
                "had chest pain.",
                "summary": "You came in with chest pain. "
                "Tests showed your heart is healthy.",
            },
        ]

        jsonl_path = os.path.join(bhc_dir, "train.json")
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        cls.dataset = MimicIVNoteExtDIDataset(
            root=cls.tmpdir.name,
            variant="bhc_train",
            cache_dir=cls.cache_dir.name,
        )
        cls.samples = cls.dataset.set_task()

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.cache_dir.cleanup()
        cls.tmpdir.cleanup()

    def test_jsonl_loads_correctly(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_jsonl_samples(self):
        self.assertEqual(len(self.samples), 3)

    def test_jsonl_sample_content(self):
        sample = self.samples[0]
        self.assertIn("text", sample)
        self.assertIn("summary", sample)
        self.assertIsInstance(sample["text"], str)


class TestMimicIVNoteExtDIDatasetErrors(unittest.TestCase):
    """Test error handling."""

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            MimicIVNoteExtDIDataset(
                root="/tmp/nonexistent",
                variant="invalid_variant",
            )

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                MimicIVNoteExtDIDataset(
                    root=tmpdir,
                    variant="bhc_train",
                )


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and edge cases."""

    def test_all_patients_produce_samples(self):
        """Every patient should produce exactly one sample."""
        root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic4_note_ext_di"
        )
        cache_dir = tempfile.TemporaryDirectory()
        from pyhealth.datasets.base_dataset import BaseDataset

        dataset = MimicIVNoteExtDIDataset.__new__(
            MimicIVNoteExtDIDataset
        )
        dataset.variant = "test"
        config_path = str(
            Path(__file__).parent.parent.parent
            / "pyhealth"
            / "datasets"
            / "configs"
            / "mimic4_note_ext_di.yaml"
        )
        BaseDataset.__init__(
            dataset,
            root=str(root),
            tables=["summaries"],
            dataset_name="mimic4_note_ext_di",
            config_path=config_path,
            cache_dir=cache_dir.name,
        )
        task = PatientSummaryGeneration()
        samples = dataset.set_task(task)
        n_patients = len(dataset.unique_patient_ids)
        self.assertEqual(len(samples), n_patients)
        samples.close()

    def test_sample_text_not_empty(self):
        """No sample should have empty text or summary."""
        root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic4_note_ext_di"
        )
        cache_dir = tempfile.TemporaryDirectory()
        from pyhealth.datasets.base_dataset import BaseDataset

        dataset = MimicIVNoteExtDIDataset.__new__(
            MimicIVNoteExtDIDataset
        )
        dataset.variant = "test"
        config_path = str(
            Path(__file__).parent.parent.parent
            / "pyhealth"
            / "datasets"
            / "configs"
            / "mimic4_note_ext_di.yaml"
        )
        BaseDataset.__init__(
            dataset,
            root=str(root),
            tables=["summaries"],
            dataset_name="mimic4_note_ext_di",
            config_path=config_path,
            cache_dir=cache_dir.name,
        )
        task = PatientSummaryGeneration()
        samples = dataset.set_task(task)
        for sample in samples:
            self.assertGreater(len(sample["text"]), 0)
            self.assertGreater(len(sample["summary"]), 0)
        samples.close()

    def test_available_variants(self):
        """All expected variant names should be recognized."""
        from pyhealth.datasets.mimic4_note_ext_di import (
            _VARIANT_FILE_MAP,
        )

        expected = {
            "bhc_all", "bhc_train", "bhc_valid", "bhc_test",
            "bhc_train_100", "original", "cleaned",
            "cleaned_improved",
        }
        self.assertTrue(expected.issubset(set(_VARIANT_FILE_MAP)))


class TestPatientSummaryGenerationTask(unittest.TestCase):
    """Test the task class independently."""

    def test_task_name(self):
        task = PatientSummaryGeneration()
        self.assertEqual(task.task_name, "PatientSummaryGeneration")

    def test_input_schema(self):
        task = PatientSummaryGeneration()
        self.assertEqual(task.input_schema, {"text": "text"})

    def test_output_schema(self):
        task = PatientSummaryGeneration()
        self.assertEqual(task.output_schema, {"summary": "text"})


if __name__ == "__main__":
    unittest.main()
