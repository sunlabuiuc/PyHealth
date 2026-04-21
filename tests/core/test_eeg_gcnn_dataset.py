"""Tests for the EEG-GCNN pipeline — raw path and FigShare path.

Covers:
  - EEGGCNNDiseaseDetection  (raw EEG task: PSD, adjacency, bipolar, __call__)
  - EEGGCNNRawDataset        (metadata CSV generation + full precompute_features)
  - EEGGCNNDataset           (FigShare prepare_metadata, alpha variants)
  - EEGGCNNClassification    (task schema, excluded_bands, sample shapes)

Most tests use synthetic data — no real EEG recordings are required.
TestRawDatasetPrecompute uses the sample files in
examples/eeg_gcnn/sample_raw_data/ (committed to the repo).
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from joblib import dump

from pyhealth.datasets.eeg_gcnn import EEGGCNNDataset
from pyhealth.tasks.eeg_gcnn_classification import (
    BAND_NAMES,
    EEGGCNNClassification,
)
from pyhealth.tasks.eeg_gcnn_disease_detection import (
    BIPOLAR_CHANNELS,
    DEFAULT_BANDS,
    NUM_CHANNELS,
    EEGGCNNDiseaseDetection,
)


# ===========================================================================
# Shared helpers — raw path
# ===========================================================================

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


# ===========================================================================
# Shared helpers — FigShare path
# ===========================================================================

# Minimal 10-10 coordinate TSV — only the 8 reference electrodes needed.
_COORDS_TSV = (
    "label\tx\ty\tz\n"
    "F5\t-0.55\t0.67\t0.50\n"
    "F6\t 0.55\t0.67\t0.50\n"
    "C5\t-0.71\t0.00\t0.71\n"
    "C6\t 0.71\t0.00\t0.71\n"
    "P5\t-0.55\t-0.67\t0.50\n"
    "P6\t 0.55\t-0.67\t0.50\n"
    "O1\t-0.30\t-0.95\t0.10\n"
    "O2\t 0.30\t-0.95\t0.10\n"
)

_N_WINDOWS = 4   # 4 windows across 2 patients
_N_PATIENTS = 2


def _write_synthetic_root(root: str) -> None:
    """Write all five required FigShare files to *root* using synthetic data."""
    rng = np.random.default_rng(0)

    # psd_features_data_X  — shape (N, 48)
    X = rng.random((_N_WINDOWS, 48)).astype(np.float32)
    dump(X, os.path.join(root, "psd_features_data_X"))

    # labels_y  — alternating class labels
    y = np.array(["diseased", "healthy", "diseased", "healthy"])
    dump(y, os.path.join(root, "labels_y"))

    # master_metadata_index.csv  — 2 patients, 2 windows each
    pd.DataFrame({
        "patient_ID": ["p001", "p001", "p002", "p002"],
    }).to_csv(os.path.join(root, "master_metadata_index.csv"), index=False)

    # spec_coh_values.npy  — shape (N, 64)
    coh = rng.random((_N_WINDOWS, 64)).astype(np.float32)
    np.save(os.path.join(root, "spec_coh_values.npy"), coh)

    # standard_1010.tsv.txt
    Path(os.path.join(root, "standard_1010.tsv.txt")).write_text(_COORDS_TSV)


@dataclass
class _DummyEvent:
    node_features_path: str
    adj_matrix_path: str
    window_idx: int
    label: int


class _DummyPatient:
    def __init__(self, patient_id: str, events: List[_DummyEvent]) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_DummyEvent]:
        if event_type == "eeg_windows":
            return self._events
        return []


# ===========================================================================
# Pytest fixtures — raw path
# ===========================================================================

@pytest.fixture
def task():
    """Default EEGGCNNDiseaseDetection instance."""
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


# ===========================================================================
# EEGGCNNDiseaseDetection — task initialisation
# ===========================================================================

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
        t = EEGGCNNDiseaseDetection(bands=custom)
        assert len(t.bands) == 2

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
        assert task.task_name == "EEGGCNNDiseaseDetection"


# ===========================================================================
# EEGGCNNDiseaseDetection — PSD feature extraction
# ===========================================================================

class TestPSDExtraction:

    def test_shape(self, task, synthetic_bipolar_window):
        psd = task._extract_psd_features(synthetic_bipolar_window)
        assert psd.shape == (NUM_CHANNELS, len(DEFAULT_BANDS))

    def test_finite_values(self, task, synthetic_bipolar_window):
        psd = task._extract_psd_features(synthetic_bipolar_window)
        assert np.all(np.isfinite(psd))

    def test_custom_bands_shape(self, synthetic_bipolar_window):
        custom = {"alpha": (8.0, 12.0)}
        t = EEGGCNNDiseaseDetection(bands=custom)
        psd = t._extract_psd_features(synthetic_bipolar_window)
        assert psd.shape == (NUM_CHANNELS, 1)


# ===========================================================================
# EEGGCNNDiseaseDetection — adjacency matrix
# ===========================================================================

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

    def test_spatial_adjacency_type(self, task_spatial, synthetic_bipolar_window):
        adj = task_spatial._build_adjacency(synthetic_bipolar_window)
        assert adj.shape == (NUM_CHANNELS, NUM_CHANNELS)
        # Should NOT be identity — off-diagonal elements > 0
        assert np.any(adj[0, 1:] > 0)


# ===========================================================================
# EEGGCNNDiseaseDetection — bipolar channel computation
# ===========================================================================

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


# ===========================================================================
# EEGGCNNDiseaseDetection — end-to-end __call__ with mocked I/O
# ===========================================================================

class TestTaskCall:

    def test_call_produces_samples(self):
        """Verify __call__ produces correct samples with mocked EEG I/O."""
        t = EEGGCNNDiseaseDetection(adjacency_type="none", window_sec=10)
        mock_raw = _make_mock_raw(n_seconds=30, sfreq=250)

        event = MagicMock()
        event.signal_file = "/fake/path.edf"
        event.label = 0

        patient = MagicMock()
        patient.patient_id = "test_001"
        patient.get_events.side_effect = (
            lambda table: [event] if table == "tuab" else []
        )

        with patch.object(t, "_read_eeg", return_value=mock_raw), \
             patch.object(t, "_preprocess", return_value=mock_raw):
            samples = t(patient)

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
        t = EEGGCNNDiseaseDetection(adjacency_type="none")

        event = MagicMock()
        event.signal_file = "/bad/path.edf"
        event.label = 0

        patient = MagicMock()
        patient.patient_id = "test_002"
        patient.get_events.side_effect = (
            lambda table: [event] if table == "tuab" else []
        )

        with patch.object(t, "_read_eeg", side_effect=ValueError("corrupt file")):
            samples = t(patient)

        assert len(samples) == 0

    def test_call_both_sources(self):
        """Verify samples from both TUAB and LEMON are collected."""
        t = EEGGCNNDiseaseDetection(adjacency_type="none", window_sec=10)
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

        with patch.object(t, "_read_eeg", return_value=mock_raw), \
             patch.object(t, "_preprocess", return_value=mock_raw):
            samples = t(patient)

        assert len(samples) == 2
        labels = {s["label"] for s in samples}
        assert labels == {0, 1}


# ===========================================================================
# EEGGCNNRawDataset — metadata CSV generation
# ===========================================================================

class TestDatasetMetadata:

    def test_prepare_tuab_csv(self):
        """Verify TUAB CSV is generated from synthetic directory structure."""
        from pyhealth.datasets.eeg_gcnn_raw import EEGGCNNRawDataset

        with tempfile.TemporaryDirectory() as tmpdir:
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

            ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
            ds.root = tmpdir
            ds.prepare_metadata()

            csv_path = Path(tmpdir) / "eeg_gcnn-tuab-pyhealth.csv"
            assert csv_path.exists()

            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert "patient_id" in df.columns
            assert "signal_file" in df.columns
            assert "label" in df.columns
            assert all(df["label"] == 0)
            assert all(df["source"] == "tuab")

    def test_prepare_lemon_csv(self):
        """Verify LEMON CSV is generated from synthetic directory structure."""
        from pyhealth.datasets.eeg_gcnn_raw import EEGGCNNRawDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            subj_dir = Path(tmpdir) / "lemon" / "sub-010002"
            subj_dir.mkdir(parents=True)
            (subj_dir / "sub-010002.vhdr").touch()

            ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
            ds.root = tmpdir
            ds.prepare_metadata()

            csv_path = Path(tmpdir) / "eeg_gcnn-lemon-pyhealth.csv"
            assert csv_path.exists()

            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df.iloc[0]["label"] == 1
            assert df.iloc[0]["source"] == "lemon"


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestConstants:

    def test_bipolar_channel_count(self):
        assert len(BIPOLAR_CHANNELS) == 8

    def test_default_bands_count(self):
        assert len(DEFAULT_BANDS) == 6

    def test_num_channels(self):
        assert NUM_CHANNELS == 8


# ===========================================================================
# EEGGCNNDataset — prepare_metadata (FigShare path)
# ===========================================================================

class TestEEGGCNNDatasetPrepareMetadata(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        _write_synthetic_root(self.root)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_prepare_metadata_creates_csv(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        csv_path = os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        self.assertTrue(os.path.exists(csv_path))

    def test_prepare_metadata_csv_row_count(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        df = pd.read_csv(
            os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        )
        self.assertEqual(len(df), _N_WINDOWS)

    def test_prepare_metadata_csv_columns(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        df = pd.read_csv(
            os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        )
        for col in ("patient_id", "window_idx", "label",
                    "node_features_path", "adj_matrix_path"):
            self.assertIn(col, df.columns)

    def test_prepare_metadata_npy_shapes(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        df = pd.read_csv(
            os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        )
        for _, row in df.iterrows():
            nf = np.load(row["node_features_path"])
            am = np.load(row["adj_matrix_path"])
            self.assertEqual(nf.shape, (8, 6))
            self.assertEqual(am.shape, (8, 8))

    def test_prepare_metadata_idempotent(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        # Second call must not raise
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)

    def test_prepare_metadata_different_alpha_creates_separate_csv(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.0)
        EEGGCNNDataset.prepare_metadata(self.root, alpha=1.0)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.root, "eeg_gcnn_windows_alpha0.00.csv")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.root, "eeg_gcnn_windows_alpha1.00.csv")
            )
        )

    def test_missing_required_file_raises(self) -> None:
        os.remove(os.path.join(self.root, "labels_y"))
        with self.assertRaises(FileNotFoundError):
            EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)

    def test_default_task_returns_classification_instance(self) -> None:
        ds = EEGGCNNDataset.__new__(EEGGCNNDataset)
        self.assertIsInstance(ds.default_task, EEGGCNNClassification)


# ===========================================================================
# EEGGCNNClassification — task schema and sample generation
# ===========================================================================

class TestEEGGCNNClassification(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = self._tmpdir.name

        rng = np.random.default_rng(1)

        self._events: List[_DummyEvent] = []
        for i in range(2):
            nf = rng.random((8, 6)).astype(np.float32)
            am = rng.random((8, 8)).astype(np.float32)
            nf_path = os.path.join(tmp, f"{i}_nf.npy")
            am_path = os.path.join(tmp, f"{i}_am.npy")
            np.save(nf_path, nf)
            np.save(am_path, am)
            self._events.append(
                _DummyEvent(
                    node_features_path=nf_path,
                    adj_matrix_path=am_path,
                    window_idx=i,
                    label=i % 2,
                )
            )

        self._patient = _DummyPatient("p001", self._events)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_schema_attributes(self) -> None:
        task = EEGGCNNClassification()
        self.assertEqual(task.task_name, "EEGGCNNClassification")
        self.assertEqual(task.input_schema["node_features"], "tensor")
        self.assertEqual(task.input_schema["adj_matrix"], "tensor")
        self.assertEqual(task.output_schema["label"], "binary")

    def test_call_returns_one_sample_per_window(self) -> None:
        task = EEGGCNNClassification()
        samples = task(self._patient)
        self.assertEqual(len(samples), 2)

    def test_call_sample_keys(self) -> None:
        task = EEGGCNNClassification()
        sample = task(self._patient)[0]
        for key in ("patient_id", "window_idx", "node_features",
                    "adj_matrix", "label"):
            self.assertIn(key, sample)

    def test_call_sample_shapes(self) -> None:
        task = EEGGCNNClassification()
        for sample in task(self._patient):
            self.assertEqual(sample["node_features"].shape, (8, 6))
            self.assertEqual(sample["adj_matrix"].shape, (8, 8))

    def test_call_patient_id_propagated(self) -> None:
        task = EEGGCNNClassification()
        for sample in task(self._patient):
            self.assertEqual(sample["patient_id"], "p001")

    def test_excluded_band_zeros_correct_column(self) -> None:
        band = "delta"          # index 0 in BAND_NAMES
        task = EEGGCNNClassification(excluded_bands=[band])
        for sample in task(self._patient):
            np.testing.assert_array_equal(
                sample["node_features"][:, 0],
                np.zeros(8, dtype=np.float32),
            )
            self.assertFalse(np.all(sample["node_features"][:, 1] == 0))

    def test_multiple_excluded_bands(self) -> None:
        task = EEGGCNNClassification(excluded_bands=["delta", "theta"])
        for sample in task(self._patient):
            np.testing.assert_array_equal(
                sample["node_features"][:, 0], np.zeros(8, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                sample["node_features"][:, 1], np.zeros(8, dtype=np.float32)
            )

    def test_invalid_band_raises(self) -> None:
        with self.assertRaises(ValueError):
            EEGGCNNClassification(excluded_bands=["not_a_band"])

    def test_empty_patient_returns_empty_list(self) -> None:
        task = EEGGCNNClassification()
        empty_patient = _DummyPatient("p_empty", [])
        self.assertEqual(task(empty_patient), [])

    def test_all_band_names_are_valid(self) -> None:
        for band in BAND_NAMES:
            EEGGCNNClassification(excluded_bands=[band])


# ===========================================================================
# EEGGCNNRawDataset — full precompute_features with real sample files
# ===========================================================================

# Path to the committed sample raw data (relative to repo root).
_REPO_ROOT = Path(__file__).parent.parent.parent
_SAMPLE_RAW_DATA = _REPO_ROOT / "examples" / "eeg_gcnn" / "sample_raw_data"


@unittest.skipUnless(
    _SAMPLE_RAW_DATA.exists(),
    "Sample raw data not found — skipping precompute integration tests",
)
class TestRawDatasetPrecompute(unittest.TestCase):
    """Integration tests for EEGGCNNRawDataset.precompute_features().

    Uses the real (small) EDF/BrainVision files committed under
    examples/eeg_gcnn/sample_raw_data/.  Processes max 1 TUAB + 1 LEMON
    subject so the suite stays fast (< 30 s on a laptop).
    """

    def setUp(self) -> None:
        from pyhealth.datasets.eeg_gcnn_raw import EEGGCNNRawDataset

        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name

        # Bypass BaseDataset.__init__ — we only need .root + precompute_features
        self.ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
        self.ds.root = str(_SAMPLE_RAW_DATA)
        self.ds.precompute_features(
            output_dir=self.output_dir,
            max_tuab=1,
            max_lemon=1,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # --- output file existence ---

    def test_creates_psd_features_file(self) -> None:
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "psd_features_data_X"))
        )

    def test_creates_labels_file(self) -> None:
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "labels_y"))
        )

    def test_creates_metadata_csv(self) -> None:
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "master_metadata_index.csv")
            )
        )

    def test_creates_coherence_file(self) -> None:
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "spec_coh_values.npy"))
        )

    def test_creates_electrode_coords_file(self) -> None:
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "standard_1010.tsv.txt")
            )
        )

    # --- output shapes and content ---

    def test_psd_features_shape(self) -> None:
        """psd_features_data_X must be (N, 48) — 8 channels × 6 bands."""
        import joblib
        X = joblib.load(os.path.join(self.output_dir, "psd_features_data_X"))
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[1], 48)
        self.assertGreater(X.shape[0], 0)

    def test_coherence_shape(self) -> None:
        """spec_coh_values must be (N, 64) — 8×8 electrode pairs."""
        coh = np.load(os.path.join(self.output_dir, "spec_coh_values.npy"))
        self.assertEqual(coh.ndim, 2)
        self.assertEqual(coh.shape[1], 64)

    def test_consistent_window_counts(self) -> None:
        """N must be consistent across psd_features_data_X, labels_y, coherence, and CSV."""
        import joblib
        X = joblib.load(os.path.join(self.output_dir, "psd_features_data_X"))
        y = joblib.load(os.path.join(self.output_dir, "labels_y"))
        coh = np.load(os.path.join(self.output_dir, "spec_coh_values.npy"))
        meta = pd.read_csv(
            os.path.join(self.output_dir, "master_metadata_index.csv")
        )
        N = X.shape[0]
        self.assertEqual(y.shape[0], N)
        self.assertEqual(coh.shape[0], N)
        self.assertEqual(len(meta), N)

    def test_labels_are_valid_strings(self) -> None:
        """labels_y must contain only 'diseased' and/or 'healthy'."""
        import joblib
        y = joblib.load(os.path.join(self.output_dir, "labels_y"))
        for label in y:
            self.assertIn(label, {"diseased", "healthy"})

    def test_both_classes_present(self) -> None:
        """With 1 TUAB + 1 LEMON subject both classes must appear."""
        import joblib
        y = joblib.load(os.path.join(self.output_dir, "labels_y"))
        classes = set(y)
        self.assertIn("diseased", classes)
        self.assertIn("healthy", classes)

    def test_psd_features_finite(self) -> None:
        """No NaN or Inf in the PSD feature array."""
        import joblib
        X = joblib.load(os.path.join(self.output_dir, "psd_features_data_X"))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_metadata_has_patient_id_column(self) -> None:
        meta = pd.read_csv(
            os.path.join(self.output_dir, "master_metadata_index.csv")
        )
        self.assertIn("patient_ID", meta.columns)


if __name__ == "__main__":
    unittest.main()
