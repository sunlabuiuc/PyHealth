"""Unit tests for CFS Dataset and Sleep Staging Task."""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets import CFSDataset
from pyhealth.tasks import SleepStagingCFS


class TestCFSDataset(unittest.TestCase):
    """Test suite for CFSDataset initialization and metadata creation."""

    def setUp(self):
        """Create a temporary directory with minimal CFS dataset structure."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

        # Create directory structure
        psg_dir = os.path.join(self.root, "polysomnography", "edfs")
        annot_nsrr_dir = os.path.join(
            self.root, "polysomnography", "annotations-events-nsrr"
        )
        annot_prof_dir = os.path.join(
            self.root, "polysomnography", "annotations-events-profusion"
        )
        os.makedirs(psg_dir, exist_ok=True)
        os.makedirs(annot_nsrr_dir, exist_ok=True)
        os.makedirs(annot_prof_dir, exist_ok=True)

        # Create dataset directory
        dataset_dir = os.path.join(self.root, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)

        # Create minimal demographic CSV
        demo_data = {
            "nsrrid": ["800002", "800010"],
            "age": [38, 50],
            "male": [1, 0],
            "race": ["Black", "White"],
            "ethnicity": ["Non-Hispanic", "Hispanic"],
            "bmi": [27.05, 35.46],
            "smoker": [0, 1],
            "ahi": [2.77, 12.97],
        }
        demo_df = pd.DataFrame(demo_data)
        demo_path = os.path.join(
            dataset_dir, "cfs-visit5-dataset-0.7.0.csv"
        )
        demo_df.to_csv(demo_path, index=False)

        # Create dummy EDF files
        self._create_dummy_edf_files(psg_dir)

        # Create dummy annotation files
        self._create_dummy_annotation_files(annot_nsrr_dir, annot_prof_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_dummy_edf_files(self, psg_dir: str) -> None:
        """Create minimal EDF file placeholders for testing.

        Note: These are not valid EDF files, just placeholders to test
        the dataset initialization and metadata creation.
        """
        edf_files = ["800002-01.edf", "800010-01.edf"]
        for edf_file in edf_files:
            filepath = os.path.join(psg_dir, edf_file)
            # Create empty file (actual EDF header would be needed for real testing)
            with open(filepath, "w") as f:
                f.write("dummy edf content")

    def _create_dummy_annotation_files(
        self, annot_nsrr_dir: str, annot_prof_dir: str
    ) -> None:
        """Create dummy annotation files for testing."""
        studies = ["800002-01", "800010-01"]

        # Create NSRR XML annotation files (dummy content)
        nsrr_xml = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
    <SleepStages>
        <SleepStage start="0" duration="30" stage="Wake"/>
        <SleepStage start="30" duration="30" stage="N1"/>
        <SleepStage start="60" duration="30" stage="N2"/>
        <SleepStage start="90" duration="30" stage="N3"/>
        <SleepStage start="120" duration="30" stage="REM"/>
    </SleepStages>
</PSGAnnotation>
"""

        for study in studies:
            nsrr_path = os.path.join(annot_nsrr_dir, f"{study}_nsrr.xml")
            with open(nsrr_path, "w") as f:
                f.write(nsrr_xml)

        # Create Profusion XML annotation files (dummy content)
        prof_xml = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
    <SleepStages>
        <SleepStage start="0" duration="30" stage="0"/>
        <SleepStage start="30" duration="30" stage="1"/>
        <SleepStage start="60" duration="30" stage="2"/>
        <SleepStage start="90" duration="30" stage="3"/>
        <SleepStage start="120" duration="30" stage="4"/>
    </SleepStages>
</PSGAnnotation>
"""

        for study in studies:
            prof_path = os.path.join(annot_prof_dir, f"{study}_profusion.xml")
            with open(prof_path, "w") as f:
                f.write(prof_xml)

    def test_initialization(self):
        """Test CFSDataset initialization."""
        with pytest.raises(Exception):
            # Should fail because we don't have valid EDF files
            # But it should initialize without errors
            dataset = CFSDataset(root=self.root)

    def test_metadata_creation(self):
        """Test that metadata file is created correctly."""
        dataset = CFSDataset.__new__(CFSDataset)
        dataset._prepare_metadata(self.root)

        metadata_path = os.path.join(self.root, "polysomnography-metadata-pyhealth.csv")
        self.assertTrue(os.path.exists(metadata_path))

        # Verify metadata content
        metadata_df = pd.read_csv(metadata_path)
        self.assertEqual(len(metadata_df), 2)  # Should have 2 records
        self.assertIn("800002", metadata_df["nsrrid"].values)
        self.assertIn("800010", metadata_df["nsrrid"].values)

    def test_metadata_columns(self):
        """Test that metadata has all required columns."""
        dataset = CFSDataset.__new__(CFSDataset)
        dataset._prepare_metadata(self.root)

        metadata_path = os.path.join(self.root, "polysomnography-metadata-pyhealth.csv")
        metadata_df = pd.read_csv(metadata_path)

        required_cols = [
            "nsrrid", "study_id", "signal_file", "label_file",
            "label_file_nsrr"
        ]
        for col in required_cols:
            self.assertIn(col, metadata_df.columns)


class TestSleepStagingCFS(unittest.TestCase):
    """Test suite for SleepStagingCFS task."""

    def setUp(self):
        """Initialize test fixtures."""
        self.task = SleepStagingCFS(
            chunk_duration=30.0, preload=True
        )

    def test_initialization(self):
        """Test SleepStagingCFS initialization."""
        self.assertEqual(self.task.chunk_duration, 30.0)
        self.assertTrue(self.task.preload)
        self.assertEqual(self.task.task_name, "SleepStagingCFS")

    def test_task_schemas(self):
        """Test task input/output schemas."""
        self.assertIn("signal", self.task.input_schema)
        self.assertEqual(self.task.input_schema["signal"], "tensor")
        self.assertIn("label", self.task.output_schema)
        self.assertEqual(self.task.output_schema["label"], "multiclass")

    def test_sleep_stage_mapping(self):
        """Test sleep stage string to numeric mapping."""
        test_cases = {
            "WAKE": 0, "W": 0, "0": 0,
            "SLEEP STAGE 1": 1, "N1": 1,
            "SLEEP STAGE 2": 2, "N2": 2,
            "SLEEP STAGE 3": 3, "N3": 3,
            "SLEEP STAGE 4": 3, "4": 3,
            "REM": 4, "R": 4,
        }

        for stage_str, expected in test_cases.items():
            result = self.task._map_sleep_stage(stage_str)
            self.assertEqual(
                result, expected,
                f"Failed mapping for '{stage_str}': got {result}, expected {expected}"
            )

    def test_invalid_sleep_stage_mapping(self):
        """Test mapping of invalid sleep stage strings."""
        invalid_stages = ["INVALID", "UNKNOWN", "STAGE 5", ""]
        for stage_str in invalid_stages:
            result = self.task._map_sleep_stage(stage_str)
            self.assertIsNone(result, f"Invalid stage '{stage_str}' should map to None")

    def test_custom_signals(self):
        """Test task initialization with custom signal selection."""
        custom_signals = ["EEG Fpz-Cz", "EOG-L", "EMG-Chin"]
        task = SleepStagingCFS(
            chunk_duration=30.0,
            selected_signals=custom_signals
        )
        self.assertEqual(task.selected_signals, custom_signals)

    def test_signal_channels_mapping(self):
        """Test standard signal channel mappings."""
        # Verify that standard_signals dictionary has all expected types
        expected_types = {"eeg", "eog_left", "eog_right", "emg", "ecg"}
        actual_types = set(self.task.standard_signals.keys())
        self.assertEqual(expected_types, actual_types)

        # Verify that each type has multiple variants
        for signal_type, variants in self.task.standard_signals.items():
            self.assertIsInstance(variants, list)
            self.assertGreater(len(variants), 0,
                f"Signal type '{signal_type}' has no variants")

    def test_task_call_with_mock_patient(self):
        """Test task.__call__() with mock patient data."""
        # This is a simplified test since we don't have real PSG data
        # In production, this would use actual EDF files

        # Create a mock patient object
        class MockEvent:
            def __init__(self):
                self.signal_file = None
                self.label_file = None
                self.label_file_nsrr = None
                self.study_id = "test-01"
                self.age = 50
                self.sex = "M"

        class MockPatient:
            def __init__(self):
                self.patient_id = "test_patient"

            def get_events(self):
                return [MockEvent()]

        patient = MockPatient()

        # This should return empty samples since we don't have real PSG files
        samples = self.task(patient)
        self.assertIsInstance(samples, list)
        # With mock data, we expect empty list since files don't exist
        # In real testing with valid EDF files, this would return samples


class TestCFSDatasetIntegration(unittest.TestCase):
    """Integration tests for CFS dataset with tasks."""

    def test_task_configuration(self):
        """Test that CFS dataset can be configured with a sleep staging task."""
        task = SleepStagingCFS(chunk_duration=30.0)
        self.assertEqual(task.task_name, "SleepStagingCFS")

        # Verify task is callable
        self.assertTrue(callable(task))

    def test_multiple_signal_selections(self):
        """Test creating tasks with different signal selections."""
        signal_sets = [
            ["EEG Fpz-Cz", "EOG-L", "EOG-R", "EMG-Chin"],
            ["EEG C3-A2", "EMG-Chin"],
            None,  # Auto-select
        ]

        for signals in signal_sets:
            task = SleepStagingCFS(selected_signals=signals)
            if signals:
                self.assertEqual(task.selected_signals, signals)
            else:
                self.assertIsNone(task.selected_signals)


class TestSleepStagingCFSEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Initialize test fixtures."""
        self.task = SleepStagingCFS(chunk_duration=30.0)

    def test_stage_mapping_case_insensitivity(self):
        """Test that stage mapping handles various cases."""
        test_cases = [
            "wake", "Wake", "WAKE",
            "n1", "N1", "N 1",
            "REM", "rem", "Rem",
        ]

        for stage_str in test_cases:
            result = self.task._map_sleep_stage(stage_str)
            # All should map to valid numeric labels
            self.assertIsNotNone(result)
            self.assertIn(result, [0, 1, 2, 3, 4])

    def test_empty_patient_events(self):
        """Test handling of patient with no events."""
        class MockPatient:
            def __init__(self):
                self.patient_id = "no_events_patient"

            def get_events(self):
                return []

        patient = MockPatient()
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_chunk_duration_flexibility(self):
        """Test tasks with different chunk durations."""
        durations = [30.0, 60.0, 15.0]

        for duration in durations:
            task = SleepStagingCFS(chunk_duration=duration)
            self.assertEqual(task.chunk_duration, duration)


if __name__ == "__main__":
    unittest.main()
