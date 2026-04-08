"""Tests for the EEG-GCNN dataset and task classes.

All tests use synthetic data — no real EEG files are required.
Tests complete in milliseconds.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pyhealth.tasks.eeg_gcnn_nd_detection import (
    BIPOLAR_CHANNELS,
    DEFAULT_BANDS,
    NUM_CHANNELS,
    EEGGCNNDiseaseDetection,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def task():
    """Default task instance."""
    return EEGGCNNDiseaseDetection()


@pytest.fixture
def task_spatial():
    """Task with spatial-only adjacency."""
    return EEGGCNNDiseaseDetection(adjacency_type="spatial")


@pytest.fixture
def task_none():
    """Task with identity adjacency."""
    return EEGGCNNDiseaseDetection(adjacency_type="none")


@pytest.fixture
def synthetic_bipolar_window():
    """Synthetic 10-second bipolar window at 250 Hz (8, 2500)."""
    rng = np.random.RandomState(42)
    return rng.randn(NUM_CHANNELS, 2500)


@pytest.fixture
def synthetic_bipolar_signal():
    """Synthetic 30-second bipolar signal at 250 Hz (8, 7500)."""
    rng = np.random.RandomState(42)
    return rng.randn(NUM_CHANNELS, 7500)


def _make_mock_raw(n_seconds=30, sfreq=250):
    """Create a mock MNE Raw object with all required TUAB channels."""
    n_samples = int(n_seconds * sfreq)
    rng = np.random.RandomState(0)

    ch_names_needed = set()
    for anode, cathode, _ in BIPOLAR_CHANNELS:
        ch_names_needed.add(anode)
        ch_names_needed.add(cathode)
    ch_names = sorted(ch_names_needed)
    n_ch = len(ch_names)

    data = rng.randn(n_ch, n_samples) * 1e-5  # realistic EEG scale

    raw = MagicMock()
    raw.ch_names = ch_names
    raw.get_data.return_value = data
    raw.filter.return_value = raw
    raw.notch_filter.return_value = raw
    raw.resample.return_value = raw
    return raw


# ---------------------------------------------------------------
# Task initialization tests
# ---------------------------------------------------------------

class TestTaskInit:

    def test_default_params(self, task):
        assert task.resample_rate == 250
        assert task.highpass_freq == 1.0
        assert task.notch_freq == 50.0
        assert task.window_sec == 10
        assert task.adjacency_type == "combined"
        assert task.connectivity_measure == "coherence"
        assert len(task.bands) == 6

    def test_custom_bands(self):
        custom = {"alpha": (8.0, 12.0), "theta": (4.0, 8.0)}
        task = EEGGCNNDiseaseDetection(bands=custom)
        assert len(task.bands) == 2

    def test_invalid_adjacency_type(self):
        with pytest.raises(ValueError, match="adjacency_type"):
            EEGGCNNDiseaseDetection(adjacency_type="invalid")

    def test_invalid_connectivity(self):
        with pytest.raises(ValueError, match="connectivity_measure"):
            EEGGCNNDiseaseDetection(connectivity_measure="plv")

    def test_task_schemas(self, task):
        assert "node_features" in task.input_schema
        assert "adj_matrix" in task.input_schema
        assert "label" in task.output_schema
        assert task.output_schema["label"] == "binary"
        assert task.task_name == "eeg_gcnn_nd_detection"


# ---------------------------------------------------------------
# PSD feature extraction tests
# ---------------------------------------------------------------

class TestPSDExtraction:

    def test_shape(self, task, synthetic_bipolar_window):
        psd = task._extract_psd_features(synthetic_bipolar_window)
        assert psd.shape == (NUM_CHANNELS, len(DEFAULT_BANDS))

    def test_finite_values(self, task, synthetic_bipolar_window):
        psd = task._extract_psd_features(synthetic_bipolar_window)
        assert np.all(np.isfinite(psd))

    def test_custom_bands_shape(self, synthetic_bipolar_window):
        custom = {"alpha": (8.0, 12.0)}
        task = EEGGCNNDiseaseDetection(bands=custom)
        psd = task._extract_psd_features(synthetic_bipolar_window)
        assert psd.shape == (NUM_CHANNELS, 1)


# ---------------------------------------------------------------
# Adjacency matrix tests
# ---------------------------------------------------------------

class TestAdjacency:

    def test_spatial_shape(self, task):
        adj = task._build_spatial_adjacency()
        assert adj.shape == (NUM_CHANNELS, NUM_CHANNELS)

    def test_spatial_diagonal(self, task):
        adj = task._build_spatial_adjacency()
        np.testing.assert_array_equal(np.diag(adj), np.ones(NUM_CHANNELS))

    def test_spatial_positive(self, task):
        adj = task._build_spatial_adjacency()
        assert np.all(adj >= 0)

    def test_none_is_identity(self, task_none, synthetic_bipolar_window):
        adj = task_none._build_adjacency(synthetic_bipolar_window)
        np.testing.assert_array_equal(adj, np.eye(NUM_CHANNELS))

    def test_spatial_adjacency_type(
        self, task_spatial, synthetic_bipolar_window
    ):
        adj = task_spatial._build_adjacency(synthetic_bipolar_window)
        assert adj.shape == (NUM_CHANNELS, NUM_CHANNELS)
        # Should NOT be identity — off-diagonal elements > 0
        assert np.any(adj[0, 1:] > 0)


# ---------------------------------------------------------------
# Bipolar channel computation tests
# ---------------------------------------------------------------

class TestBipolarComputation:

    def test_compute_bipolar(self):
        raw = _make_mock_raw(n_seconds=10, sfreq=250)
        bipolar = EEGGCNNDiseaseDetection._compute_bipolar(raw)
        assert bipolar.shape == (NUM_CHANNELS, 2500)

    def test_bipolar_is_difference(self):
        raw = _make_mock_raw(n_seconds=1, sfreq=250)
        data = raw.get_data()
        ch_map = {name: idx for idx, name in enumerate(raw.ch_names)}
        bipolar = EEGGCNNDiseaseDetection._compute_bipolar(raw)

        anode, cathode, _ = BIPOLAR_CHANNELS[0]
        expected = data[ch_map[anode]] - data[ch_map[cathode]]
        np.testing.assert_array_almost_equal(bipolar[0], expected)


# ---------------------------------------------------------------
# End-to-end __call__ test with mocked I/O
# ---------------------------------------------------------------

class TestTaskCall:

    def test_call_produces_samples(self):
        """Verify __call__ produces correct samples with mocked EEG I/O."""
        task = EEGGCNNDiseaseDetection(
            adjacency_type="none",
            window_sec=10,
        )

        mock_raw = _make_mock_raw(n_seconds=30, sfreq=250)

        # Build a mock patient with one TUAB event
        event = MagicMock()
        event.signal_file = "/fake/path.edf"
        event.label = 0

        patient = MagicMock()
        patient.patient_id = "test_001"
        patient.get_events.side_effect = (
            lambda table: [event] if table == "tuab" else []
        )

        with patch.object(task, "_read_eeg", return_value=mock_raw), \
             patch.object(task, "_preprocess", return_value=mock_raw):
            samples = task(patient)

        # 30s / 10s = 3 windows
        assert len(samples) == 3
        for s in samples:
            assert s["patient_id"] == "test_001"
            assert isinstance(s["node_features"], torch.Tensor)
            assert s["node_features"].shape == (8, 6)
            assert isinstance(s["adj_matrix"], torch.Tensor)
            assert s["adj_matrix"].shape == (8, 8)
            assert s["label"] == 0

    def test_call_skips_bad_file(self):
        """Verify __call__ gracefully skips unreadable files."""
        task = EEGGCNNDiseaseDetection(adjacency_type="none")

        event = MagicMock()
        event.signal_file = "/bad/path.edf"
        event.label = 0

        patient = MagicMock()
        patient.patient_id = "test_002"
        patient.get_events.side_effect = (
            lambda table: [event] if table == "tuab" else []
        )

        with patch.object(
            task, "_read_eeg", side_effect=ValueError("corrupt file")
        ):
            samples = task(patient)

        assert len(samples) == 0

    def test_call_both_sources(self):
        """Verify samples from both TUAB and LEMON are collected."""
        task = EEGGCNNDiseaseDetection(
            adjacency_type="none", window_sec=10
        )

        mock_raw = _make_mock_raw(n_seconds=10, sfreq=250)

        tuab_event = MagicMock()
        tuab_event.signal_file = "/fake/tuab.edf"
        tuab_event.label = 0

        lemon_event = MagicMock()
        lemon_event.signal_file = "/fake/lemon.vhdr"
        lemon_event.label = 1

        patient = MagicMock()
        patient.patient_id = "test_003"

        def get_events(table):
            if table == "tuab":
                return [tuab_event]
            elif table == "lemon":
                return [lemon_event]
            return []

        patient.get_events.side_effect = get_events

        with patch.object(task, "_read_eeg", return_value=mock_raw), \
             patch.object(task, "_preprocess", return_value=mock_raw):
            samples = task(patient)

        assert len(samples) == 2
        labels = {s["label"] for s in samples}
        assert labels == {0, 1}


# ---------------------------------------------------------------
# Dataset metadata tests
# ---------------------------------------------------------------

class TestDatasetMetadata:

    def test_prepare_tuab_csv(self):
        """Verify TUAB CSV is generated from synthetic directory structure."""
        from pyhealth.datasets.eeg_gcnn import EEGGCNNDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic TUAB directory structure
            subj_dir = (
                Path(tmpdir)
                / "train"
                / "normal"
                / "01_tcp_ar"
                / "000"
                / "00000001"
            )
            subj_dir.mkdir(parents=True)
            (subj_dir / "00000001_00000001_01.edf").touch()
            (subj_dir / "00000001_00000002_01.edf").touch()

            # Instantiate dataset just to trigger prepare_metadata
            # We can't fully instantiate BaseDataset without CSV content,
            # so we test prepare_metadata directly.
            ds = EEGGCNNDataset.__new__(EEGGCNNDataset)
            ds.root = tmpdir
            ds.prepare_metadata()

            csv_path = Path(tmpdir) / "eeg_gcnn-tuab-pyhealth.csv"
            assert csv_path.exists()

            import pandas as pd

            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert "patient_id" in df.columns
            assert "signal_file" in df.columns
            assert "label" in df.columns
            assert all(df["label"] == 0)
            assert all(df["source"] == "tuab")

    def test_prepare_lemon_csv(self):
        """Verify LEMON CSV is generated from synthetic directory structure."""
        from pyhealth.datasets.eeg_gcnn import EEGGCNNDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            subj_dir = Path(tmpdir) / "lemon" / "sub-010002"
            subj_dir.mkdir(parents=True)
            (subj_dir / "sub-010002.vhdr").touch()

            ds = EEGGCNNDataset.__new__(EEGGCNNDataset)
            ds.root = tmpdir
            ds.prepare_metadata()

            csv_path = Path(tmpdir) / "eeg_gcnn-lemon-pyhealth.csv"
            assert csv_path.exists()

            import pandas as pd

            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df.iloc[0]["label"] == 1
            assert df.iloc[0]["source"] == "lemon"


# ---------------------------------------------------------------
# Constants / module-level tests
# ---------------------------------------------------------------

class TestConstants:

    def test_bipolar_channel_count(self):
        assert len(BIPOLAR_CHANNELS) == 8

    def test_default_bands_count(self):
        assert len(DEFAULT_BANDS) == 6

    def test_num_channels(self):
        assert NUM_CHANNELS == 8


# ---------------------------------------------------------------
# Functional and combined adjacency tests
# (mne_connectivity is mocked so no real EEG dependency)
# ---------------------------------------------------------------

def _make_mock_mne_connectivity(n_channels: int = 8, n_freqs: int = 10):
    """Return a (mock mne_connectivity module, mock connectivity result).

    ``conn.get_data(output="dense")`` returns shape
    ``(n_channels, n_channels, n_freqs)`` with values in [0, 1].
    """
    rng = np.random.RandomState(7)
    conn_data = rng.uniform(0.0, 1.0, (n_channels, n_channels, n_freqs))

    mock_conn_result = MagicMock()
    mock_conn_result.get_data.return_value = conn_data

    mock_mne_conn_module = MagicMock()
    mock_mne_conn_module.spectral_connectivity_epochs.return_value = (
        mock_conn_result
    )
    return mock_mne_conn_module, mock_conn_result


class TestFunctionalAdjacency:
    """Tests for _build_functional_adjacency and combined adjacency mode.

    mne_connectivity is injected via sys.modules so the tests run in
    milliseconds with no network or file I/O.
    """

    @pytest.fixture(autouse=True)
    def _patch_mne_conn(self, synthetic_bipolar_window):
        """Inject a mock mne_connectivity for every test in this class."""
        mock_module, self._mock_result = _make_mock_mne_connectivity()
        with patch.dict(sys.modules, {"mne_connectivity": mock_module}):
            self._mock_module = mock_module
            yield

    # --- _build_functional_adjacency ---

    def test_functional_shape(self, synthetic_bipolar_window):
        task = EEGGCNNDiseaseDetection(adjacency_type="functional")
        adj = task._build_functional_adjacency(synthetic_bipolar_window)
        assert adj.shape == (NUM_CHANNELS, NUM_CHANNELS)

    def test_functional_diagonal_is_one(self, synthetic_bipolar_window):
        task = EEGGCNNDiseaseDetection(adjacency_type="functional")
        adj = task._build_functional_adjacency(synthetic_bipolar_window)
        np.testing.assert_array_equal(np.diag(adj), np.ones(NUM_CHANNELS))

    def test_functional_symmetric(self, synthetic_bipolar_window):
        task = EEGGCNNDiseaseDetection(adjacency_type="functional")
        adj = task._build_functional_adjacency(synthetic_bipolar_window)
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_functional_values_finite(self, synthetic_bipolar_window):
        task = EEGGCNNDiseaseDetection(adjacency_type="functional")
        adj = task._build_functional_adjacency(synthetic_bipolar_window)
        assert np.all(np.isfinite(adj))

    def test_functional_coherence_method_passed(self, synthetic_bipolar_window):
        """spectral_connectivity_epochs must be called with method='coh'."""
        task = EEGGCNNDiseaseDetection(
            adjacency_type="functional", connectivity_measure="coherence"
        )
        task._build_functional_adjacency(synthetic_bipolar_window)
        call_kwargs = (
            self._mock_module.spectral_connectivity_epochs.call_args[1]
        )
        assert call_kwargs["method"] == "coh"

    def test_functional_wpli_method_passed(self, synthetic_bipolar_window):
        """spectral_connectivity_epochs must be called with method='wpli'."""
        task = EEGGCNNDiseaseDetection(
            adjacency_type="functional", connectivity_measure="wpli"
        )
        task._build_functional_adjacency(synthetic_bipolar_window)
        call_kwargs = (
            self._mock_module.spectral_connectivity_epochs.call_args[1]
        )
        assert call_kwargs["method"] == "wpli"

    # --- combined adjacency via _build_adjacency ---

    def test_combined_shape(self, synthetic_bipolar_window):
        task = EEGGCNNDiseaseDetection(adjacency_type="combined")
        adj = task._build_adjacency(synthetic_bipolar_window)
        assert adj.shape == (NUM_CHANNELS, NUM_CHANNELS)

    def test_combined_diagonal_is_one(self, synthetic_bipolar_window):
        task = EEGGCNNDiseaseDetection(adjacency_type="combined")
        adj = task._build_adjacency(synthetic_bipolar_window)
        np.testing.assert_array_equal(np.diag(adj), np.ones(NUM_CHANNELS))

    def test_combined_is_mean_of_spatial_and_functional(
        self, synthetic_bipolar_window
    ):
        """Combined adjacency = (spatial + functional) / 2, diag set to 1."""
        task = EEGGCNNDiseaseDetection(adjacency_type="combined")
        combined = task._build_adjacency(synthetic_bipolar_window)

        functional = task._build_functional_adjacency(synthetic_bipolar_window)
        spatial = task._spatial_adj

        expected = (spatial + functional) / 2.0
        np.fill_diagonal(expected, 1.0)
        np.testing.assert_array_almost_equal(combined, expected)

    def test_combined_off_diagonal_differs_from_identity(
        self, synthetic_bipolar_window
    ):
        """Combined adjacency must not collapse to identity matrix."""
        task = EEGGCNNDiseaseDetection(adjacency_type="combined")
        adj = task._build_adjacency(synthetic_bipolar_window)
        off_diag = adj[~np.eye(NUM_CHANNELS, dtype=bool)]
        assert np.any(off_diag != 0.0)
