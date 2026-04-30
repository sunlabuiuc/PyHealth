"""Unit tests for CFS Dataset and Sleep Staging Task.

Uses synthetic data only for fast test execution.
All tests complete in milliseconds with no real EDF files or external data.

Dataset link:
    Dataset must be requested from the NSRR at https://sleepdata.org/datasets/cfs

Dataset paper: (please cite if you use this dataset)
    Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D,
    Redline S. The National Sleep Research Resource: towards a sleep data commons.
    J Am Med Inform Assoc. 2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064.
    PMID: 29860441; PMCID: PMC6188513.

    Redline S, Tishler PV, Tosteson TD, Williamson J, Kump K, Browner I, Ferrette V,
    Krejci P. The familial aggregation of obstructive sleep apnea. Am J Respir Crit 
    Care Med. 1995 Mar;151(3 Pt 1):682-7.
    doi: 10.1164/ajrccm/151.3_Pt_1.682. PMID: 7881656.

Please include the following text in the Acknowledgements:
    The Cleveland Family Study (CFS) was supported by grants from the National Institutes
    of Health (HL46380, M01 RR00080-39, T32-HL07567, RO1-46380).
    The National Sleep ResearchResource was supported by the National Heart, Lung, and
    Blood Institute (R24 HL114473, 75N92019R002).

Authors:
    Austin Jarrett (ajj7@illinois.edu)
    Justin Cheok (jcheok2@illinois.edu)
    Jimmy Scray (escray2@illinois.edu)
"""
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pyhealth.datasets import CFSDataset
from pyhealth.tasks import SleepStagingCFS


# ============================================================================
# Synthetic Data Generators (for fast unit testing)
# ============================================================================

def create_synthetic_cfs_samples(
    n_samples: int = 10,
    n_channels: int = 5,
    signal_length: int = 3000,
    include_ppg: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic CFS-format polysomnography samples.

    Creates lightweight synthetic PSG samples with realistic signal dimensions
    but no actual EDF file I/O. Used for fast unit testing.

    Args:
        n_samples (int): Number of samples to generate. Default is 10.
        n_channels (int): Number of signal channels. Default is 5 (EEG, EOG-L,
            EOG-R, EMG, ECG).
        signal_length (int): Number of samples per signal. Default is 3000
            (~100Hz for 30 seconds).
        include_ppg (bool): Whether to include PPG signal. Default is False.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        List of dicts, each containing:
            - 'eeg': array shape (256,)
            - 'eog_left': array shape (256,)
            - 'eog_right': array shape (256,)
            - 'emg': array shape (128,)
            - 'ecg': array shape (256,)
            - 'ppg': array shape (256,) [only if include_ppg=True]
            - 'label': int in range [0, 4] (Wake, N1, N2, N3, REM)
    """
    rng = np.random.RandomState(seed)
    samples = []

    for i in range(n_samples):
        sample = {
            "eeg": rng.randn(256).astype(np.float32),
            "eog_left": rng.randn(256).astype(np.float32),
            "eog_right": rng.randn(256).astype(np.float32),
            "emg": rng.randn(128).astype(np.float32),
            "ecg": rng.randn(256).astype(np.float32),
            "label": i % 5,  # Cycle through 5 sleep stages
        }

        if include_ppg:
            sample["ppg"] = rng.randn(256).astype(np.float32)

        samples.append(sample)

    return samples


def create_synthetic_patient(
    patient_id: str = "test_patient_001",
    n_psg_events: int = 1,
    age: int = 50,
    male: int = 1,
    seed: int = 42,
) -> Any:
    """Create a mock patient object with synthetic PSG events.

    Args:
        patient_id (str): Patient ID. Default is "test_patient_001".
        n_psg_events (int): Number of PSG events. Default is 1.
        age (int): Patient age. Default is 50.
        male (int): Gender (1=male, 0=female). Default is 1.
        seed (int): Random seed. Default is 42.

    Returns:
        Mock patient object with get_events() method.
    """
    patient = MagicMock()
    patient.patient_id = patient_id

    # Create demographics event
    demo_event = MagicMock()
    demo_event.age = age
    demo_event.male = male

    # Create PSG events with mock signal_file and label_file_nsrr
    psg_events = []
    for i in range(n_psg_events):
        psg_event = MagicMock()
        psg_event.signal_file = f"/mock/path/{patient_id}-0{i}.edf"
        psg_event.label_file_nsrr = f"/mock/path/{patient_id}-0{i}_nsrr.xml"
        psg_event.study_id = f"{patient_id}-0{i}"
        psg_events.append(psg_event)

    def get_events(event_type=None):
        if event_type == "demographics":
            return [demo_event]
        elif event_type == "polysomnography":
            return psg_events
        return []

    patient.get_events = get_events
    return patient


# ============================================================================
# Test Classes
# ============================================================================

class TestCFSDataset(unittest.TestCase):
    """Test suite for CFSDataset initialization and metadata creation.

    Tests basic dataset initialization, metadata validation, and directory
    structure. Uses minimal synthetic directory structures to verify metadata
    preparation without real EDF files.
    """

    def setUp(self):
        """Create a temporary directory with minimal CFS dataset structure.

        Sets up directory paths for polysomnography data and annotations
        to test metadata creation without actual EDF files.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

        # Create required directory structure
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

        # Create dataset directory for demographics
        dataset_dir = os.path.join(self.root, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)

        # Create minimal synthetic demographic CSV
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

        # Create placeholder files
        self._create_dummy_edf_files(psg_dir)
        self._create_dummy_annotation_files(annot_nsrr_dir, annot_prof_dir)

    def tearDown(self):
        """Clean up temporary directory after test completion."""
        self.temp_dir.cleanup()

    def _create_dummy_edf_files(self, psg_dir: str) -> None:
        """Create minimal EDF file placeholders.

        These are not valid EDF files, just empty placeholders to test
        metadata preparation without actual EDF processing.

        Args:
            psg_dir (str): Path to PSG directory.
        """
        edf_files = ["800002-01.edf", "800010-01.edf"]
        for edf_file in edf_files:
            filepath = os.path.join(psg_dir, edf_file)
            with open(filepath, "w") as f:
                f.write("placeholder")

    def _create_dummy_annotation_files(
        self, annot_nsrr_dir: str, annot_prof_dir: str
    ) -> None:
        """Create placeholder annotation files.

        Args:
            annot_nsrr_dir (str): Path to NSRR annotation directory.
            annot_prof_dir (str): Path to Profusion annotation directory.
        """
        studies = ["800002-01", "800010-01"]

        nsrr_xml = (
            '<?xml version="1.0"?><PSGAnnotation>'
            '<SleepStages><SleepStage start="0" duration="30"/>'
            "</SleepStages></PSGAnnotation>"
        )

        for study in studies:
            nsrr_path = os.path.join(annot_nsrr_dir, f"{study}_nsrr.xml")
            with open(nsrr_path, "w") as f:
                f.write(nsrr_xml)

            prof_path = os.path.join(annot_prof_dir, f"{study}_profusion.xml")
            with open(prof_path, "w") as f:
                f.write(nsrr_xml)

    def test_metadata_creation(self):
        """Test that metadata file is created correctly.

        Verifies that CFSDataset metadata preparation creates the expected
        metadata CSV file with correct structure.
        """
        dataset = CFSDataset.__new__(CFSDataset)
        dataset._prepare_metadata(self.root)

        metadata_path = os.path.join(
            self.root, "polysomnography-metadata-pyhealth.csv"
        )
        self.assertTrue(os.path.exists(metadata_path))

        metadata_df = pd.read_csv(metadata_path)
        self.assertEqual(len(metadata_df), 2)

    def test_metadata_columns(self):
        """Test that metadata has all required columns.

        Verifies the metadata CSV contains expected column names for
        patient IDs, study IDs, and file paths.
        """
        dataset = CFSDataset.__new__(CFSDataset)
        dataset._prepare_metadata(self.root)

        metadata_path = os.path.join(
            self.root, "polysomnography-metadata-pyhealth.csv"
        )
        metadata_df = pd.read_csv(metadata_path)

        required_cols = ["nsrrid", "study_id", "signal_file"]
        for col in required_cols:
            self.assertIn(col, metadata_df.columns)


class TestSleepStagingCFSBasic(unittest.TestCase):
    """Test suite for basic SleepStagingCFS task initialization and schema.

    Tests task initialization with various configurations and validates
    input/output schema definitions.
    """

    def setUp(self):
        """Initialize task instances for testing."""
        np.random.seed(42)

    def test_initialization_default(self):
        """Test SleepStagingCFS initialization with default parameters.

        Verifies that task initializes with correct default values for
        chunk duration, preload flag, and PPG setting.
        """
        task = SleepStagingCFS()
        self.assertEqual(task.chunk_duration, 30.0)
        self.assertFalse(task.preload)
        self.assertFalse(task.include_ppg)
        self.assertEqual(task.task_name, "SleepStagingCFS")

    def test_initialization_with_ppg(self):
        """Test SleepStagingCFS initialization with PPG enabled.

        Verifies that include_ppg parameter correctly enables PPG in
        the task configuration.
        """
        task = SleepStagingCFS(include_ppg=True, ppg_samples=256)
        self.assertTrue(task.include_ppg)
        self.assertEqual(task.ppg_samples, 256)
        self.assertIn("ppg", task.input_schema)

    def test_initialization_without_ppg(self):
        """Test SleepStagingCFS initialization without PPG.

        Verifies that PPG is not included in schema when disabled.
        """
        task = SleepStagingCFS(include_ppg=False)
        self.assertFalse(task.include_ppg)
        self.assertNotIn("ppg", task.input_schema)

    def test_input_schema(self):
        """Test input schema definition.

        Verifies that input schema contains expected signal modalities.
        """
        task = SleepStagingCFS(include_ppg=False)
        expected_signals = {"eeg", "eog_left", "eog_right", "emg", "ecg"}
        actual_signals = set(task.input_schema.keys())
        self.assertEqual(expected_signals, actual_signals)

    def test_input_schema_with_ppg(self):
        """Test input schema with PPG enabled.

        Verifies PPG is included in schema when enabled.
        """
        task = SleepStagingCFS(include_ppg=True)
        self.assertIn("ppg", task.input_schema)
        self.assertEqual(task.input_schema["ppg"], "tensor")

    def test_output_schema(self):
        """Test output schema definition.

        Verifies output schema specifies multiclass label format.
        """
        task = SleepStagingCFS()
        self.assertIn("label", task.output_schema)
        self.assertEqual(task.output_schema["label"], "multiclass")

    def test_custom_chunk_duration(self):
        """Test task initialization with custom chunk duration.

        Verifies that chunk_duration parameter is respected.
        """
        durations = [15.0, 30.0, 60.0]
        for duration in durations:
            task = SleepStagingCFS(chunk_duration=duration)
            self.assertEqual(task.chunk_duration, duration)

    def test_custom_sample_counts(self):
        """Test task initialization with custom sample counts.

        Verifies that target sample counts for each modality are
        configurable.
        """
        task = SleepStagingCFS(
            eeg_samples=512,
            eog_samples=512,
            emg_samples=256,
            ecg_samples=512,
            ppg_samples=512,
        )
        self.assertEqual(task.eeg_samples, 512)
        self.assertEqual(task.eog_samples, 512)
        self.assertEqual(task.emg_samples, 256)
        self.assertEqual(task.ecg_samples, 512)
        self.assertEqual(task.ppg_samples, 512)

    def test_selected_signals(self):
        """Test task initialization with custom signal selection.

        Verifies that selected_signals parameter correctly specifies
        which channels to extract.
        """
        custom_signals = ["EEG Fpz-Cz", "EOG-L", "EOG-R", "EMG-Chin", "ECG"]
        task = SleepStagingCFS(selected_signals=custom_signals)
        self.assertEqual(task.selected_signals, custom_signals)


class TestSleepStagingCFSLabelMapping(unittest.TestCase):
    """Test suite for sleep stage label mapping.

    Tests comprehensive label mapping from annotation strings to numeric
    labels, covering AASM format, numeric format, and CFS-specific formats.
    """

    def setUp(self):
        """Initialize task for label mapping tests."""
        self.task = SleepStagingCFS()

    def test_wake_stage_mapping(self):
        """Test mapping of Wake stage variants.

        Verifies all common Wake representations map to 0.
        """
        wake_variants = ["WAKE", "W", "0", "wake", "Wake"]
        for variant in wake_variants:
            result = self.task._map_sleep_stage(variant)
            self.assertEqual(
                result, 0,
                f"Failed to map '{variant}' to Wake (0), got {result}"
            )

    def test_n1_stage_mapping(self):
        """Test mapping of N1 (Stage 1) variants.

        Verifies all representations of Light Sleep Stage 1 map to 1.
        """
        n1_variants = [
            "N1", "STAGE 1", "SLEEP STAGE 1", "STAGE 1 SLEEP",
            "n1", "stage 1", "sleep stage 1"
        ]
        for variant in n1_variants:
            result = self.task._map_sleep_stage(variant)
            self.assertEqual(
                result, 1,
                f"Failed to map '{variant}' to N1 (1), got {result}"
            )

    def test_n2_stage_mapping(self):
        """Test mapping of N2 (Stage 2) variants.

        Verifies all representations of Light Sleep Stage 2 map to 2.
        """
        n2_variants = [
            "N2", "STAGE 2", "SLEEP STAGE 2", "STAGE 2 SLEEP",
            "n2", "stage 2", "sleep stage 2"
        ]
        for variant in n2_variants:
            result = self.task._map_sleep_stage(variant)
            self.assertEqual(
                result, 2,
                f"Failed to map '{variant}' to N2 (2), got {result}"
            )

    def test_n3_stage_mapping(self):
        """Test mapping of N3 (Deep Sleep Stage 3) variants.

        Verifies all representations of Deep Sleep map to 3, including
        legacy Stage 4 notation.
        """
        n3_variants = [
            "N3", "STAGE 3", "SLEEP STAGE 3", "STAGE 3 SLEEP",
            "STAGE 4", "SLEEP STAGE 4", "3", "4",
            "n3", "stage 3", "sleep stage 3"
        ]
        for variant in n3_variants:
            result = self.task._map_sleep_stage(variant)
            self.assertEqual(
                result, 3,
                f"Failed to map '{variant}' to N3 (3), got {result}"
            )

    def test_rem_stage_mapping(self):
        """Test mapping of REM sleep variants.

        Verifies all REM sleep representations map to 4.
        """
        rem_variants = [
            "REM", "R", "SLEEP STAGE REM", "SLEEP STAGE R",
            "REM SLEEP", "rem", "r", "sleep stage rem"
        ]
        for variant in rem_variants:
            result = self.task._map_sleep_stage(variant)
            self.assertEqual(
                result, 4,
                f"Failed to map '{variant}' to REM (4), got {result}"
            )

    def test_invalid_stage_mapping(self):
        """Test mapping of invalid sleep stage strings.

        Verifies that invalid stages map to None.
        """
        invalid_stages = ["INVALID", "UNKNOWN", "STAGE 5", "", "MOVEMENT"]
        for stage_str in invalid_stages:
            result = self.task._map_sleep_stage(stage_str)
            self.assertIsNone(
                result,
                f"Invalid stage '{stage_str}' should map to None, got {result}"
            )

    def test_whitespace_handling(self):
        """Test that stage mapping handles leading/trailing whitespace.

        Verifies mapping works even with extra whitespace.
        """
        variants_with_space = ["  WAKE  ", "  N1  ", "  REM  "]
        expected_results = [0, 1, 4]
        for variant, expected in zip(variants_with_space, expected_results):
            result = self.task._map_sleep_stage(variant)
            self.assertEqual(
                result, expected,
                f"Failed to map '{variant}' with whitespace"
            )


class TestSleepStagingCFSSignalProcessing(unittest.TestCase):
    """Test suite for signal processing operations.

    Tests signal resampling, modality extraction, and channel handling.
    """

    def setUp(self):
        """Initialize task for signal processing tests."""
        np.random.seed(42)
        self.task = SleepStagingCFS(
            eeg_samples=256,
            eog_samples=256,
            emg_samples=128,
            ecg_samples=256,
        )

    def test_resample_signal_no_change(self):
        """Test resampling when target length equals input length.

        Verifies that resampling a signal to its current length returns
        the same signal (up to floating point precision).
        """
        signal = np.random.randn(256).astype(np.float32)
        resampled = self.task._resample_signal(signal, 256)
        np.testing.assert_array_almost_equal(resampled, signal, decimal=5)

    def test_resample_signal_downsample(self):
        """Test downsampling a signal to half its length.

        Verifies that downsampling produces output of correct shape.
        """
        signal = np.random.randn(512)
        target_length = 256
        resampled = self.task._resample_signal(signal, target_length)
        self.assertEqual(resampled.shape[0], target_length)
        self.assertEqual(resampled.dtype, np.float32)

    def test_resample_signal_upsample(self):
        """Test upsampling a signal to double its length.

        Verifies that upsampling produces output of correct shape.
        """
        signal = np.random.randn(128)
        target_length = 256
        resampled = self.task._resample_signal(signal, target_length)
        self.assertEqual(resampled.shape[0], target_length)
        self.assertEqual(resampled.dtype, np.float32)

    def test_extract_modalities_all_present(self):
        """Test extracting modalities when all channels are present.

        Verifies correct extraction and resampling when all standard
        signal types are available.
        """
        # Create multi-channel signal: EEG, EOG-L, EOG-R, EMG, ECG
        signal_data = np.random.randn(5, 1000)
        channel_names = ["EEG Fpz-Cz", "EOG-L", "EOG-R", "EMG-Chin", "ECG"]

        modalities = self.task._extract_and_resample_modalities(
            signal_data, channel_names
        )

        # Verify all modalities present
        expected_modalities = {"eeg", "eog_left", "eog_right", "emg", "ecg"}
        actual_modalities = set(modalities.keys())
        self.assertEqual(expected_modalities, actual_modalities)

        # Verify shapes
        self.assertEqual(modalities["eeg"].shape, (256,))
        self.assertEqual(modalities["eog_left"].shape, (256,))
        self.assertEqual(modalities["eog_right"].shape, (256,))
        self.assertEqual(modalities["emg"].shape, (128,))
        self.assertEqual(modalities["ecg"].shape, (256,))

    def test_extract_modalities_with_ppg(self):
        """Test extracting modalities including PPG.

        Verifies PPG extraction and resampling when enabled.
        """
        task = SleepStagingCFS(include_ppg=True, ppg_samples=256)

        # Create multi-channel signal with PPG
        signal_data = np.random.randn(6, 1000)
        channel_names = [
            "EEG Fpz-Cz", "EOG-L", "EOG-R", "EMG-Chin", "ECG", "PlethWV"
        ]

        modalities = task._extract_and_resample_modalities(
            signal_data, channel_names
        )

        # Verify PPG present
        self.assertIn("ppg", modalities)
        self.assertEqual(modalities["ppg"].shape, (256,))

    def test_extract_modalities_missing_ppg_zero_padded(self):
        """Test that missing PPG is zero-padded.

        Verifies that when PPG is enabled but not present in data,
        a zero-padded array is created.
        """
        task = SleepStagingCFS(include_ppg=True, ppg_samples=256)

        # Create signal without PPG
        signal_data = np.random.randn(5, 1000)
        channel_names = ["EEG Fpz-Cz", "EOG-L", "EOG-R", "EMG-Chin", "ECG"]

        modalities = task._extract_and_resample_modalities(
            signal_data, channel_names
        )

        # PPG should be present but all zeros
        self.assertIn("ppg", modalities)
        self.assertEqual(modalities["ppg"].shape, (256,))
        np.testing.assert_array_equal(modalities["ppg"], np.zeros(256))

    def test_extract_modalities_missing_modality_zero_padded(self):
        """Test that missing modalities are zero-padded.

        Verifies graceful handling when a standard modality is not
        available in the EDF file.
        """
        # Create signal missing EMG
        signal_data = np.random.randn(4, 1000)
        channel_names = ["EEG Fpz-Cz", "EOG-L", "EOG-R", "ECG"]

        modalities = self.task._extract_and_resample_modalities(
            signal_data, channel_names
        )

        # EMG should be zero-padded
        self.assertIn("emg", modalities)
        self.assertEqual(modalities["emg"].shape, (128,))
        np.testing.assert_array_equal(modalities["emg"], np.zeros(128))

    def test_extract_modalities_multiple_same_type(self):
        """Test extraction when multiple channels of same type present.

        Verifies that multiple channels of the same type (e.g., multiple
        EEG channels) are averaged together.
        """
        # Create signal with 2 EEG channels
        signal_data = np.array([
            np.ones(1000),  # EEG channel 1
            np.ones(1000) * 2,  # EEG channel 2
            np.zeros(1000),  # EOG-L
            np.zeros(1000),  # EOG-R
            np.zeros(1000),  # EMG
            np.zeros(1000),  # ECG
        ])
        channel_names = [
            "EEG C3-A2", "EEG C4-A1", "EOG-L", "EOG-R", "EMG-Chin", "ECG"
        ]

        modalities = self.task._extract_and_resample_modalities(
            signal_data, channel_names
        )

        # EEG should be average of the two channels
        # (1 + 2) / 2 = 1.5, after resampling all values should be ~1.5
        self.assertTrue(
            np.all(modalities["eeg"] > 1.0) and np.all(modalities["eeg"] < 2.0)
        )


class TestSleepStagingCFSEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error handling.

    Tests unusual but valid inputs and boundary conditions.
    """

    def setUp(self):
        """Initialize task for edge case testing."""
        self.task = SleepStagingCFS()

    def test_call_with_empty_events(self):
        """Test task.__call__() with patient having no events.

        Verifies that task returns empty sample list for patient with
        no polysomnography events.
        """
        patient = create_synthetic_patient(n_psg_events=0)
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_call_with_mock_patient_no_real_data(self):
        """Test task.__call__() with mock patient (no actual EDF files).

        Verifies that task handles cases where signal files don't exist.
        Since mock files don't exist, task should handle gracefully.
        """
        patient = create_synthetic_patient(n_psg_events=1)
        # This will attempt to read non-existent files and handle errors
        samples = self.task(patient)
        # Should return empty list since files don't exist
        self.assertIsInstance(samples, list)

    def test_case_insensitive_stage_mapping(self):
        """Test that stage mapping is case-insensitive.

        Verifies various case combinations map correctly.
        """
        test_cases = [
            ("wake", 0), ("Wake", 0), ("WAKE", 0),
            ("n1", 1), ("N1", 1), ("n2", 2), ("N2", 2),
            ("REM", 4), ("rem", 4), ("Rem", 4),
        ]

        for stage_str, expected in test_cases:
            result = self.task._map_sleep_stage(stage_str)
            self.assertEqual(
                result, expected,
                f"Case-insensitive mapping failed for '{stage_str}'"
            )

    def test_multiple_chunk_durations(self):
        """Test that different chunk durations are handled.

        Verifies task initialization with various chunk durations.
        """
        durations = [15.0, 20.0, 30.0, 60.0, 120.0]
        for duration in durations:
            task = SleepStagingCFS(chunk_duration=duration)
            self.assertEqual(task.chunk_duration, duration)

    def test_signal_channels_mapping_completeness(self):
        """Test that standard_signals covers all modality types.

        Verifies that all required modality types have variants.
        """
        required_types = {"eeg", "eog_left", "eog_right", "emg", "ecg"}
        actual_types = set(self.task.standard_signals.keys())
        self.assertTrue(
            required_types.issubset(actual_types),
            f"Missing modality types: {required_types - actual_types}"
        )

        # Verify each type has at least one variant
        for signal_type, variants in self.task.standard_signals.items():
            self.assertGreater(
                len(variants), 0,
                f"Signal type '{signal_type}' has no variants"
            )


class TestSleepStagingCFSIntegration(unittest.TestCase):
    """Integration tests combining multiple components.

    Tests interactions between task configuration, schema, and label
    mapping in realistic scenarios.
    """

    def test_task_schema_consistency_without_ppg(self):
        """Test that schema is consistent when PPG is disabled.

        Verifies input schema matches expected modalities.
        """
        task = SleepStagingCFS(include_ppg=False)
        expected_signals = {"eeg", "eog_left", "eog_right", "emg", "ecg"}
        actual_signals = set(task.input_schema.keys())
        self.assertEqual(expected_signals, actual_signals)

    def test_task_schema_consistency_with_ppg(self):
        """Test that schema is consistent when PPG is enabled.

        Verifies PPG is included in input schema when enabled.
        """
        task = SleepStagingCFS(include_ppg=True)
        expected_signals = {
            "eeg", "eog_left", "eog_right", "emg", "ecg", "ppg"
        }
        actual_signals = set(task.input_schema.keys())
        self.assertEqual(expected_signals, actual_signals)

    def test_all_labels_representable(self):
        """Test that all valid sleep stages have consistent mapping.

        Verifies that each sleep stage (0-4) is reachable through label
        mapping from realistic annotation strings.
        """
        # Create realistic annotation variants for all stages
        label_variants = {
            0: ["Wake", "WAKE", "W"],
            1: ["N1", "Stage 1 Sleep", "SLEEP STAGE 1"],
            2: ["N2", "Stage 2 Sleep", "SLEEP STAGE 2"],
            3: ["N3", "Stage 3 Sleep", "SLEEP STAGE 4"],
            4: ["REM", "REM Sleep", "SLEEP STAGE REM"],
        }

        task = SleepStagingCFS()
        for expected_label, variants in label_variants.items():
            for variant in variants:
                result = task._map_sleep_stage(variant)
                self.assertEqual(
                    result, expected_label,
                    f"Failed mapping '{variant}' to {expected_label}"
                )

    def test_multiple_signal_selections_valid(self):
        """Test task initialization with different signal selections.

        Verifies that various custom signal selections are accepted.
        """
        signal_sets = [
            ["EEG Fpz-Cz", "EOG-L", "EOG-R", "EMG-Chin", "ECG"],
            ["EEG C3-A2", "EMG-Chin"],
            None,  # Auto-select
        ]

        for signals in signal_sets:
            task = SleepStagingCFS(selected_signals=signals)
            if signals:
                self.assertEqual(task.selected_signals, signals)
            else:
                self.assertIsNone(task.selected_signals)


# ============================================================================
# Synthetic Data Tests
# ============================================================================

class TestSyntheticDataGeneration(unittest.TestCase):
    """Test suite for synthetic data generation functions.

    Tests the helper functions used to generate synthetic PSG samples
    for unit testing.
    """

    def test_create_synthetic_cfs_samples_default(self):
        """Test synthetic sample generation with defaults.

        Verifies that generated samples have correct structure and
        dimensions.
        """
        samples = create_synthetic_cfs_samples(n_samples=5)

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertIn("eeg", sample)
            self.assertIn("eog_left", sample)
            self.assertIn("eog_right", sample)
            self.assertIn("emg", sample)
            self.assertIn("ecg", sample)
            self.assertIn("label", sample)
            self.assertNotIn("ppg", sample)

            self.assertEqual(sample["eeg"].shape, (256,))
            self.assertEqual(sample["eog_left"].shape, (256,))
            self.assertEqual(sample["emg"].shape, (128,))
            self.assertIn(sample["label"], [0, 1, 2, 3, 4])

    def test_create_synthetic_cfs_samples_with_ppg(self):
        """Test synthetic sample generation with PPG enabled.

        Verifies that PPG is included when requested.
        """
        samples = create_synthetic_cfs_samples(
            n_samples=5, include_ppg=True
        )

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertIn("ppg", sample)
            self.assertEqual(sample["ppg"].shape, (256,))

    def test_create_synthetic_cfs_samples_reproducibility(self):
        """Test that same seed produces same samples.

        Verifies reproducibility of synthetic data generation.
        """
        samples1 = create_synthetic_cfs_samples(n_samples=5, seed=42)
        samples2 = create_synthetic_cfs_samples(n_samples=5, seed=42)

        for s1, s2 in zip(samples1, samples2):
            for key in ["eeg", "eog_left", "eog_right", "emg", "ecg"]:
                np.testing.assert_array_equal(s1[key], s2[key])
            self.assertEqual(s1["label"], s2["label"])

    def test_create_synthetic_cfs_samples_label_distribution(self):
        """Test that labels are distributed across all sleep stages.

        Verifies that synthetic samples cover all 5 sleep stages.
        """
        samples = create_synthetic_cfs_samples(n_samples=10)
        labels = [s["label"] for s in samples]
        unique_labels = set(labels)
        # With 10 samples cycling through 5 stages, expect all to be present
        self.assertEqual(unique_labels, {0, 1, 2, 3, 4})

    def test_create_synthetic_patient_structure(self):
        """Test synthetic patient object structure.

        Verifies mock patient has required methods and attributes.
        """
        patient = create_synthetic_patient(
            patient_id="test_001", n_psg_events=2
        )

        self.assertEqual(patient.patient_id, "test_001")
        self.assertTrue(callable(patient.get_events))

        # Test get_events with different event types
        demo_events = patient.get_events(event_type="demographics")
        self.assertEqual(len(demo_events), 1)
        self.assertEqual(demo_events[0].age, 50)
        self.assertEqual(demo_events[0].male, 1)

        psg_events = patient.get_events(event_type="polysomnography")
        self.assertEqual(len(psg_events), 2)

    def test_create_synthetic_patient_demographics(self):
        """Test that synthetic patient demographics are configurable.

        Verifies custom age and gender can be set.
        """
        patient = create_synthetic_patient(
            patient_id="custom_001", age=75, male=0
        )

        demo_events = patient.get_events(event_type="demographics")
        self.assertEqual(demo_events[0].age, 75)
        self.assertEqual(demo_events[0].male, 0)


if __name__ == "__main__":
    unittest.main()
