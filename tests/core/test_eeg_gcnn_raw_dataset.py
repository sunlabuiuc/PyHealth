"""Tests for EEGGCNNRawDataset and EEGGCNNDiseaseDetection.

EEGGCNNRawDataset internally uses EEGGCNNDiseaseDetection to process raw
EEG recordings, so this file covers both classes together.

Most tests use synthetic data — no real EEG recordings are required.
TestRawDatasetPrecompute uses the sample files committed under
examples/eeg_gcnn/sample_raw_data/.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyhealth.datasets.eeg_gcnn_raw import (
    EEGGCNNRawDataset,
    _TUAB_FIRST,
    _TUAB_SECOND,
    _LEMON_FIRST,
    _LEMON_SECOND,
)

import numpy as np
import pandas as pd
import pytest
import torch

from pyhealth.tasks.eeg_gcnn_disease_detection import (
    BIPOLAR_CHANNELS,
    DEFAULT_BANDS,
    NUM_CHANNELS,
    EEGGCNNDiseaseDetection,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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

    data = rng.randn(n_ch, n_samples) * 1e-5

    raw = MagicMock()
    raw.ch_names = ch_names
    raw.get_data.return_value = data
    raw.filter.return_value = raw
    raw.notch_filter.return_value = raw
    raw.resample.return_value = raw
    return raw


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task():
    return EEGGCNNDiseaseDetection()


@pytest.fixture
def task_spatial():
    return EEGGCNNDiseaseDetection(adjacency_type="spatial")


@pytest.fixture
def task_none():
    return EEGGCNNDiseaseDetection(adjacency_type="none")


@pytest.fixture
def synthetic_bipolar_window():
    rng = np.random.RandomState(42)
    return rng.randn(NUM_CHANNELS, 2500)


@pytest.fixture
def synthetic_bipolar_signal():
    rng = np.random.RandomState(42)
    return rng.randn(NUM_CHANNELS, 7500)


# ---------------------------------------------------------------------------
# EEGGCNNDiseaseDetection — task initialisation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EEGGCNNDiseaseDetection — PSD feature extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EEGGCNNDiseaseDetection — adjacency matrix
# ---------------------------------------------------------------------------

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
        assert np.any(adj[0, 1:] > 0)


# ---------------------------------------------------------------------------
# EEGGCNNDiseaseDetection — bipolar channel computation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EEGGCNNDiseaseDetection — end-to-end __call__ with mocked I/O
# ---------------------------------------------------------------------------

class TestTaskCall:

    def test_call_produces_samples(self):
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

        assert len(samples) == 3
        for s in samples:
            assert s["patient_id"] == "test_001"
            assert isinstance(s["node_features"], torch.Tensor)
            assert s["node_features"].shape == (8, 6)
            assert isinstance(s["adj_matrix"], torch.Tensor)
            assert s["adj_matrix"].shape == (8, 8)
            assert s["label"] == 0

    def test_call_skips_bad_file(self):
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


# ---------------------------------------------------------------------------
# EEGGCNNRawDataset — metadata CSV generation (synthetic dirs)
# ---------------------------------------------------------------------------

class TestDatasetMetadata:

    def test_prepare_tuab_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subj_dir = (
                Path(tmpdir)
                / "train" / "normal" / "01_tcp_ar" / "000" / "00000001"
            )
            subj_dir.mkdir(parents=True)
            (subj_dir / "00000001_00000001_01.edf").touch()
            (subj_dir / "00000001_00000002_01.edf").touch()

            ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
            ds.root = tmpdir
            with patch("pathlib.Path.home", return_value=Path(tmpdir) / "fakehome"):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            subj_dir = Path(tmpdir) / "lemon" / "sub-010002"
            subj_dir.mkdir(parents=True)
            (subj_dir / "sub-010002.vhdr").touch()

            ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
            ds.root = tmpdir
            with patch("pathlib.Path.home", return_value=Path(tmpdir) / "fakehome"):
                ds.prepare_metadata()

            csv_path = Path(tmpdir) / "eeg_gcnn-lemon-pyhealth.csv"
            assert csv_path.exists()

            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df.iloc[0]["label"] == 1
            assert df.iloc[0]["source"] == "lemon"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_bipolar_channel_count(self):
        assert len(BIPOLAR_CHANNELS) == 8

    def test_default_bands_count(self):
        assert len(DEFAULT_BANDS) == 6

    def test_num_channels(self):
        assert NUM_CHANNELS == 8


# ---------------------------------------------------------------------------
# EEGGCNNRawDataset — _fix_vhdr_references
# ---------------------------------------------------------------------------

class TestFixVhdrReferences:

    def test_patches_stale_references(self, tmp_path):
        sub_dir = tmp_path / "sub-032301"
        sub_dir.mkdir()
        (sub_dir / "sub-032301.eeg").touch()
        (sub_dir / "sub-032301.vmrk").touch()
        vhdr = sub_dir / "sub-032301.vhdr"
        vhdr.write_text(
            "[Common Infos]\nDataFile=sub-010002.eeg\nMarkerFile=sub-010002.vmrk\n"
        )
        EEGGCNNRawDataset._fix_vhdr_references(vhdr)
        content = vhdr.read_text()
        assert "DataFile=sub-032301.eeg" in content
        assert "MarkerFile=sub-032301.vmrk" in content

    def test_no_op_when_already_correct(self, tmp_path):
        sub_dir = tmp_path / "sub-032301"
        sub_dir.mkdir()
        (sub_dir / "sub-032301.eeg").touch()
        (sub_dir / "sub-032301.vmrk").touch()
        vhdr = sub_dir / "sub-032301.vhdr"
        original = "[Common Infos]\nDataFile=sub-032301.eeg\nMarkerFile=sub-032301.vmrk\n"
        vhdr.write_text(original)
        EEGGCNNRawDataset._fix_vhdr_references(vhdr)
        assert vhdr.read_text() == original

    def test_no_op_when_no_companion_files(self, tmp_path):
        sub_dir = tmp_path / "sub-032301"
        sub_dir.mkdir()
        vhdr = sub_dir / "sub-032301.vhdr"
        original = "[Common Infos]\nDataFile=sub-010002.eeg\nMarkerFile=sub-010002.vmrk\n"
        vhdr.write_text(original)
        EEGGCNNRawDataset._fix_vhdr_references(vhdr)
        assert vhdr.read_text() == original


# ---------------------------------------------------------------------------
# EEGGCNNRawDataset — precompute_features with synthetic data
# ---------------------------------------------------------------------------

def _make_precompute_raw(source="tuab", n_seconds=30, sfreq=250.0):
    """Synthetic MNE-like raw with the correct bipolar channel names."""
    n_samples = int(n_seconds * sfreq)
    rng = np.random.default_rng(42)
    if source == "tuab":
        ch_names = list(dict.fromkeys(_TUAB_FIRST + _TUAB_SECOND))
    else:
        ch_names = list(dict.fromkeys(_LEMON_FIRST + _LEMON_SECOND))
    data = (rng.random((len(ch_names), n_samples)) * 1e-5).astype(np.float64)
    raw = MagicMock()
    raw.ch_names = ch_names
    raw.get_data.return_value = data
    raw.info = {"sfreq": sfreq}
    return raw


class TestPrecomputeFeaturesSynthetic:

    def _make_ds(self, root):
        ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
        ds.root = str(root)
        return ds

    def _make_tuab_dir(self, root, n_subjects=2):
        edf_dir = root / "tuab" / "train" / "normal" / "01_tcp_ar"
        edf_dir.mkdir(parents=True)
        for i in range(n_subjects):
            (edf_dir / f"subject{i:03d}_s001_t000.edf").touch()

    def _make_lemon_dir(self, root, n_subjects=2):
        lemon_dir = root / "lemon"
        lemon_dir.mkdir(parents=True)
        for i in range(n_subjects):
            sub = lemon_dir / f"sub-03230{i}"
            sub.mkdir()
            (sub / f"sub-03230{i}.vhdr").touch()

    def test_output_files_created(self, tmp_path):
        self._make_tuab_dir(tmp_path)
        ds = self._make_ds(tmp_path)
        mock_raw = _make_precompute_raw("tuab")
        with patch.object(EEGGCNNRawDataset, "_load_raw", return_value=mock_raw):
            ds.precompute_features(output_dir=str(tmp_path / "out"))
        out = tmp_path / "out"
        for fname in (
            "psd_features_data_X", "labels_y",
            "master_metadata_index.csv", "spec_coh_values.npy",
            "standard_1010.tsv.txt",
        ):
            assert (out / fname).exists(), f"Missing output file: {fname}"

    def test_output_shapes(self, tmp_path):
        self._make_tuab_dir(tmp_path, n_subjects=2)
        ds = self._make_ds(tmp_path)
        mock_raw = _make_precompute_raw("tuab", n_seconds=30)
        with patch.object(EEGGCNNRawDataset, "_load_raw", return_value=mock_raw):
            ds.precompute_features(output_dir=str(tmp_path / "out"))
        from joblib import load as jload
        X = jload(tmp_path / "out" / "psd_features_data_X")
        coh = np.load(tmp_path / "out" / "spec_coh_values.npy")
        # 2 subjects × 3 windows (30 s / 10 s) = 6 windows
        assert X.shape == (6, 48)    # 8 channels × 6 bands
        assert coh.shape == (6, 64)  # 8 × 8 edges

    def test_max_tuab_limits_subjects(self, tmp_path):
        self._make_tuab_dir(tmp_path, n_subjects=4)
        ds = self._make_ds(tmp_path)
        mock_raw = _make_precompute_raw("tuab", n_seconds=10)
        with patch.object(EEGGCNNRawDataset, "_load_raw", return_value=mock_raw):
            ds.precompute_features(output_dir=str(tmp_path / "out"), max_tuab=2)
        from joblib import load as jload
        y = jload(tmp_path / "out" / "labels_y")
        assert len(y) == 2  # 2 subjects × 1 window each

    def test_max_lemon_limits_subjects(self, tmp_path):
        self._make_lemon_dir(tmp_path, n_subjects=4)
        ds = self._make_ds(tmp_path)
        mock_raw = _make_precompute_raw("lemon", n_seconds=10)
        with patch.object(EEGGCNNRawDataset, "_load_raw", return_value=mock_raw):
            ds.precompute_features(output_dir=str(tmp_path / "out"), max_lemon=2)
        from joblib import load as jload
        y = jload(tmp_path / "out" / "labels_y")
        assert len(y) == 2

    def test_labels_correct_for_each_source(self, tmp_path):
        self._make_tuab_dir(tmp_path, n_subjects=1)
        self._make_lemon_dir(tmp_path, n_subjects=1)
        ds = self._make_ds(tmp_path)

        def _mock_load(path, source, sfreq):
            return _make_precompute_raw(source, n_seconds=10)

        with patch.object(EEGGCNNRawDataset, "_load_raw", side_effect=_mock_load):
            ds.precompute_features(output_dir=str(tmp_path / "out"))
        from joblib import load as jload
        y = jload(tmp_path / "out" / "labels_y")
        assert "diseased" in y
        assert "healthy" in y

    def test_skips_bad_recording_and_continues(self, tmp_path):
        self._make_tuab_dir(tmp_path, n_subjects=2)
        ds = self._make_ds(tmp_path)
        mock_raw = _make_precompute_raw("tuab", n_seconds=10)
        calls = {"n": 0}

        def _mock_load(path, source, sfreq):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("corrupt file")
            return mock_raw

        with patch.object(EEGGCNNRawDataset, "_load_raw", side_effect=_mock_load):
            ds.precompute_features(output_dir=str(tmp_path / "out"))
        from joblib import load as jload
        y = jload(tmp_path / "out" / "labels_y")
        assert len(y) == 1  # only second subject succeeded

    def test_empty_data_raises(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with pytest.raises(RuntimeError, match="No windows extracted"):
            ds.precompute_features(output_dir=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# EEGGCNNRawDataset — full precompute_features with real sample files
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_SAMPLE_RAW_DATA = _REPO_ROOT / "examples" / "eeg_gcnn" / "sample_raw_data"


@unittest.skipUnless(
    _SAMPLE_RAW_DATA.exists(),
    "Sample raw data not found — skipping precompute integration tests",
)
class TestRawDatasetPrecompute(unittest.TestCase):
    """Integration tests for EEGGCNNRawDataset.precompute_features().

    Uses the real EDF/BrainVision sample files committed under
    examples/eeg_gcnn/sample_raw_data/. Processes max 1 TUAB + 1 LEMON
    subject to stay fast (< 30 s on a laptop).
    """

    def setUp(self) -> None:
        from pyhealth.datasets.eeg_gcnn_raw import EEGGCNNRawDataset

        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name

        self.ds = EEGGCNNRawDataset.__new__(EEGGCNNRawDataset)
        self.ds.root = str(_SAMPLE_RAW_DATA)
        self.ds.precompute_features(
            output_dir=self.output_dir,
            max_tuab=1,
            max_lemon=1,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_creates_psd_features_file(self) -> None:
        import os
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "psd_features_data_X"))
        )

    def test_creates_labels_file(self) -> None:
        import os
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "labels_y"))
        )

    def test_creates_metadata_csv(self) -> None:
        import os
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "master_metadata_index.csv")
            )
        )

    def test_creates_coherence_file(self) -> None:
        import os
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "spec_coh_values.npy"))
        )

    def test_creates_electrode_coords_file(self) -> None:
        import os
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "standard_1010.tsv.txt")
            )
        )

    def test_psd_features_shape(self) -> None:
        import os
        import joblib
        X = joblib.load(os.path.join(self.output_dir, "psd_features_data_X"))
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[1], 48)
        self.assertGreater(X.shape[0], 0)

    def test_coherence_shape(self) -> None:
        import os
        coh = np.load(os.path.join(self.output_dir, "spec_coh_values.npy"))
        self.assertEqual(coh.ndim, 2)
        self.assertEqual(coh.shape[1], 64)

    def test_consistent_window_counts(self) -> None:
        import os
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
        import os
        import joblib
        y = joblib.load(os.path.join(self.output_dir, "labels_y"))
        for label in y:
            self.assertIn(label, {"diseased", "healthy"})

    def test_both_classes_present(self) -> None:
        import os
        import joblib
        y = joblib.load(os.path.join(self.output_dir, "labels_y"))
        self.assertIn("diseased", set(y))
        self.assertIn("healthy", set(y))

    def test_psd_features_finite(self) -> None:
        import os
        import joblib
        X = joblib.load(os.path.join(self.output_dir, "psd_features_data_X"))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_metadata_has_patient_id_column(self) -> None:
        import os
        meta = pd.read_csv(
            os.path.join(self.output_dir, "master_metadata_index.csv")
        )
        self.assertIn("patient_ID", meta.columns)


if __name__ == "__main__":
    unittest.main()
