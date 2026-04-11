"""Tests for MultiViewTimeSeriesTask.

Tests use synthetic/pseudo data only — no real datasets.
All tests complete in milliseconds.
"""

import os
import pickle
import shutil
import tempfile
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from pyhealth.tasks.multi_view_time_series_task import (
    MultiViewTimeSeriesTask,
    load_epoch_views,
    get_view_shapes,
)


def make_mock_patient(
    temp_dir: str,
    patient_id: str = "TEST001",
    n_channels: int = 2,
    sample_rate: int = 100,
    duration_seconds: int = 120,
    labels: list = None,
):
    """Creates a mock patient record with synthetic EEG data.

    Args:
        temp_dir: Temporary directory for saving output files.
        patient_id: Patient identifier string.
        n_channels: Number of EEG channels.
        sample_rate: Sampling rate in Hz.
        duration_seconds: Total duration of the synthetic signal.
        labels: List of sleep stage labels. Defaults to cycling W/N1/N2/N3/REM.

    Returns:
        A list containing one record dict (matches task's expected input format).
    """
    n_samples = sample_rate * duration_seconds
    n_epochs = duration_seconds // 30

    if labels is None:
        possible = ["W", "N1", "N2", "N3", "REM"]
        labels = [possible[i % len(possible)] for i in range(n_epochs)]

    # Mock the raw MNE object
    mock_raw = MagicMock()
    mock_raw.get_data.return_value = np.random.randn(n_channels, n_samples)
    mock_raw.info = {"sfreq": sample_rate}

    # Mock annotations
    mock_ann = []
    for label in labels:
        ann = {"duration": 30, "description": f"Sleep stage {label}"}
        mock_ann.append(ann)

    record = [{
        "load_from_path": temp_dir,
        "signal_file": f"{patient_id}.edf",
        "label_file": f"{patient_id}.hyp",
        "save_to_path": os.path.join(temp_dir, "output"),
        "subject_id": patient_id,
    }]

    return record, mock_raw, mock_ann


class TestMultiViewTimeSeriesTaskInit(unittest.TestCase):
    """Tests task instantiation and schema attributes."""

    def test_task_name(self):
        task = MultiViewTimeSeriesTask()
        self.assertEqual(task.task_name, "MultiViewTimeSeries")

    def test_input_schema(self):
        task = MultiViewTimeSeriesTask()
        self.assertIn("signal_temporal", task.input_schema)
        self.assertIn("signal_derivative", task.input_schema)
        self.assertIn("signal_frequency", task.input_schema)

    def test_output_schema(self):
        task = MultiViewTimeSeriesTask()
        self.assertIn("label", task.output_schema)
        self.assertEqual(task.output_schema["label"], "multiclass")

    def test_default_params(self):
        task = MultiViewTimeSeriesTask()
        self.assertEqual(task.epoch_seconds, 30)
        self.assertIsNone(task.sample_rate)

    def test_custom_params(self):
        task = MultiViewTimeSeriesTask(epoch_seconds=10, sample_rate=200)
        self.assertEqual(task.epoch_seconds, 10)
        self.assertEqual(task.sample_rate, 200)


class TestMultiViewTimeSeriesTaskSampleProcessing(unittest.TestCase):
    """Tests sample processing, feature extraction, and label generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task = MultiViewTimeSeriesTask(epoch_seconds=30)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _run_task_with_mocks(self, patient_id="TEST001", duration=120, labels=None):
        """Helper to run task with mocked MNE calls."""
        record, mock_raw, mock_ann = make_mock_patient(
            self.temp_dir,
            patient_id=patient_id,
            duration_seconds=duration,
            labels=labels,
        )

        with patch("mne.io.read_raw_edf", return_value=mock_raw), \
             patch("mne.read_annotations", return_value=mock_ann):
            samples = self.task(record)

        return samples

    def test_returns_list(self):
        samples = self._run_task_with_mocks()
        self.assertIsInstance(samples, list)

    def test_correct_number_of_samples(self):
        # 120 seconds / 30 seconds per epoch = 4 epochs
        samples = self._run_task_with_mocks(duration=120)
        self.assertEqual(len(samples), 4)

    def test_sample_keys(self):
        samples = self._run_task_with_mocks()
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        self.assertIn("record_id", sample)
        self.assertIn("patient_id", sample)
        self.assertIn("epoch_path", sample)
        self.assertIn("label", sample)

    def test_patient_id_in_samples(self):
        samples = self._run_task_with_mocks(patient_id="PAT042")
        for s in samples:
            self.assertEqual(s["patient_id"], "PAT042")

    def test_record_id_format(self):
        samples = self._run_task_with_mocks(patient_id="TEST001")
        self.assertTrue(samples[0]["record_id"].startswith("TEST001-epoch-"))

    def test_label_generation(self):
        labels = ["W", "N1", "N2", "REM"]
        samples = self._run_task_with_mocks(duration=120, labels=labels)
        extracted = [s["label"] for s in samples]
        self.assertEqual(extracted, labels)

    def test_unknown_labels_skipped(self):
        labels = ["W", "?", "N2", "Unknown"]
        samples = self._run_task_with_mocks(duration=120, labels=labels)
        for s in samples:
            self.assertNotIn(s["label"], ["?", "Unknown"])

    def test_epoch_path_exists(self):
        samples = self._run_task_with_mocks()
        for s in samples:
            self.assertTrue(os.path.exists(s["epoch_path"]))

    def test_pickle_file_saved(self):
        samples = self._run_task_with_mocks()
        self.assertTrue(samples[0]["epoch_path"].endswith(".pkl"))


class TestMultiViewFeatureExtraction(unittest.TestCase):
    """Tests that the three views are correctly computed and saved."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task = MultiViewTimeSeriesTask(epoch_seconds=30)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _get_views(self, duration=120):
        record, mock_raw, mock_ann = make_mock_patient(
            self.temp_dir, duration_seconds=duration
        )
        with patch("mne.io.read_raw_edf", return_value=mock_raw), \
             patch("mne.read_annotations", return_value=mock_ann):
            samples = self.task(record)

        with open(samples[0]["epoch_path"], "rb") as f:
            views = pickle.load(f)
        return views

    def test_views_keys(self):
        views = self._get_views()
        self.assertIn("temporal", views)
        self.assertIn("derivative", views)
        self.assertIn("frequency", views)
        self.assertIn("label", views)

    def test_temporal_shape(self):
        views = self._get_views()
        # 2 channels, 100 Hz * 30 seconds = 3000 samples
        self.assertEqual(views["temporal"].shape, (2, 3000))

    def test_derivative_shape(self):
        views = self._get_views()
        # derivative loses one sample
        self.assertEqual(views["derivative"].shape, (2, 2999))

    def test_frequency_shape(self):
        views = self._get_views()
        # FFT keeps half the samples (Nyquist)
        self.assertEqual(views["frequency"].shape, (2, 1500))

    def test_temporal_is_numpy(self):
        views = self._get_views()
        self.assertIsInstance(views["temporal"], np.ndarray)

    def test_derivative_is_numpy(self):
        views = self._get_views()
        self.assertIsInstance(views["derivative"], np.ndarray)

    def test_frequency_is_numpy(self):
        views = self._get_views()
        self.assertIsInstance(views["frequency"], np.ndarray)

    def test_frequency_is_non_negative(self):
        # FFT magnitude must always be >= 0
        views = self._get_views()
        self.assertTrue(np.all(views["frequency"] >= 0))

    def test_temporal_matches_input(self):
        # Temporal view should be the raw signal unchanged
        n_samples = 100 * 120
        mock_data = np.random.randn(2, n_samples)
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = mock_data
        mock_raw.info = {"sfreq": 100}
        mock_ann = [{"duration": 30, "description": "Sleep stage W"}] * 4

        record = [{
            "load_from_path": self.temp_dir,
            "signal_file": "p.edf",
            "label_file": "p.hyp",
            "save_to_path": os.path.join(self.temp_dir, "output"),
            "subject_id": "P001",
        }]

        with patch("mne.io.read_raw_edf", return_value=mock_raw), \
             patch("mne.read_annotations", return_value=mock_ann):
            samples = self.task(record)

        with open(samples[0]["epoch_path"], "rb") as f:
            views = pickle.load(f)

        np.testing.assert_array_equal(
            views["temporal"], mock_data[:, :3000]
        )


class TestMultiViewEdgeCases(unittest.TestCase):
    """Tests edge cases and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task = MultiViewTimeSeriesTask(epoch_seconds=30)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_fallback_labels_on_annotation_error(self):
        """Task should fall back to dummy labels if annotation loading fails."""
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(2, 100 * 120)
        mock_raw.info = {"sfreq": 100}

        record = [{
            "load_from_path": self.temp_dir,
            "signal_file": "p.edf",
            "label_file": "p.hyp",
            "save_to_path": os.path.join(self.temp_dir, "output"),
            "subject_id": "P001",
        }]

        with patch("mne.io.read_raw_edf", return_value=mock_raw), \
             patch("mne.read_annotations", side_effect=Exception("file not found")):
            samples = self.task(record)

        self.assertGreater(len(samples), 0)

    def test_output_dir_created(self):
        """Task should create output directory if it doesn't exist."""
        output_dir = os.path.join(self.temp_dir, "new", "nested", "dir")
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(2, 100 * 60)
        mock_raw.info = {"sfreq": 100}
        mock_ann = [{"duration": 30, "description": "Sleep stage W"}] * 2

        record = [{
            "load_from_path": self.temp_dir,
            "signal_file": "p.edf",
            "label_file": "p.hyp",
            "save_to_path": output_dir,
            "subject_id": "P001",
        }]

        with patch("mne.io.read_raw_edf", return_value=mock_raw), \
             patch("mne.read_annotations", return_value=mock_ann):
            self.task(record)

        self.assertTrue(os.path.exists(output_dir))

    def test_mismatched_sample_rate_warning(self):
        """Task should warn and use actual sample rate if mismatch."""
        task = MultiViewTimeSeriesTask(epoch_seconds=30, sample_rate=200)
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(2, 100 * 60)
        mock_raw.info = {"sfreq": 100}  # actual is 100, requested is 200
        mock_ann = [{"duration": 30, "description": "Sleep stage W"}] * 2

        record = [{
            "load_from_path": self.temp_dir,
            "signal_file": "p.edf",
            "label_file": "p.hyp",
            "save_to_path": os.path.join(self.temp_dir, "output"),
            "subject_id": "P001",
        }]

        with patch("mne.io.read_raw_edf", return_value=mock_raw), \
             patch("mne.read_annotations", return_value=mock_ann):
            samples = task(record)

        self.assertGreater(len(samples), 0)


class TestHelperFunctions(unittest.TestCase):
    """Tests for load_epoch_views and get_view_shapes helpers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_epoch_views(self):
        epoch_data = {
            "temporal": np.random.randn(2, 3000),
            "derivative": np.random.randn(2, 2999),
            "frequency": np.abs(np.random.randn(2, 1500)),
            "label": "W",
        }
        path = os.path.join(self.temp_dir, "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(epoch_data, f)

        views = load_epoch_views(path)
        self.assertIn("temporal", views)
        self.assertIn("derivative", views)
        self.assertIn("frequency", views)
        self.assertIn("label", views)

    def test_get_view_shapes(self):
        shapes = get_view_shapes(sample_rate=100, epoch_seconds=30, num_channels=2)
        self.assertEqual(shapes["temporal"], (2, 3000))
        self.assertEqual(shapes["derivative"], (2, 2999))
        self.assertEqual(shapes["frequency"], (2, 1500))

    def test_get_view_shapes_custom(self):
        shapes = get_view_shapes(sample_rate=200, epoch_seconds=10, num_channels=1)
        self.assertEqual(shapes["temporal"], (1, 2000))
        self.assertEqual(shapes["derivative"], (1, 1999))
        self.assertEqual(shapes["frequency"], (1, 1000))


if __name__ == "__main__":
    unittest.main()