"""Tests for Wav2SleepDataset and Wav2SleepStaging."""

import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from pyhealth.models.base_model import BaseModel
from pyhealth.models.wav2sleep import (
    DilatedConvBlock,
    EpochMixer,
    FEATURE_DIM,
    ResidualBlock,
    SequenceMixer,
    SignalEncoder,
    Wav2Sleep,
)
from pyhealth.tasks.wav2sleep_staging import Wav2SleepStaging


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def make_xml_annotation(path: Path, stages: list[int]) -> None:
    """Write a minimal profusion-style XML annotation file."""
    root = ET.Element("CMPStudyConfig")
    stage_container = ET.SubElement(root, "SleepStages")
    for s in stages:
        el = ET.SubElement(stage_container, "SleepStage")
        el.text = str(s)
    ET.ElementTree(root).write(str(path))


def make_wsc_annotation(path: Path, stages: list[int]) -> None:
    """Write a minimal WSC-style TSV annotation file."""
    df = pd.DataFrame({"User-Defined Stage": stages})
    df.to_csv(str(path), sep="\t", index=False)


def make_mock_raw(ch_names: list[str], sfreq: float = 256.0, n_samples: int = 7680):
    """Return a mock MNE Raw object with synthetic data."""
    mock_raw = MagicMock()
    mock_raw.ch_names = ch_names
    mock_raw.info = {"sfreq": sfreq}
    mock_raw.get_data.return_value = np.random.randn(len(ch_names), n_samples).astype(
        np.float32
    )
    return mock_raw


def build_tmp_root(root: Path) -> None:
    """
    Builds a minimal synthetic PSG root directory for dataset tests.

    shhs/
      polysomnography/
        edfs/shhs1/
          patient-001.edf   (empty placeholder)
        annotations-events-profusion/shhs1/
          patient-001-profusion.xml

    mesa/
      polysomnography/
        edfs/
          patient-002.edf
        annotations-events-profusion/
          patient-002-profusion.xml

    wsc/
      polysomnography/
        patient-003.edf
        patient-003.stg.txt
    """
    # SHHS (has subdirectory layer)
    shhs_edf = root / "shhs" / "polysomnography" / "edfs" / "shhs1"
    shhs_lbl = (
        root / "shhs" / "polysomnography" / "annotations-events-profusion" / "shhs1"
    )
    shhs_edf.mkdir(parents=True)
    shhs_lbl.mkdir(parents=True)
    (shhs_edf / "patient-001.edf").touch()
    make_xml_annotation(shhs_lbl / "patient-001-profusion.xml", [0, 1, 2, 5, 9])

    # MESA (flat layout, has PPG)
    mesa_edf = root / "mesa" / "polysomnography" / "edfs"
    mesa_lbl = root / "mesa" / "polysomnography" / "annotations-events-profusion"
    mesa_edf.mkdir(parents=True)
    mesa_lbl.mkdir(parents=True)
    (mesa_edf / "patient-002.edf").touch()
    make_xml_annotation(mesa_lbl / "patient-002-profusion.xml", [0, 0, 5, 3])

    # WSC (flat layout, TSV annotations)
    wsc_psg = root / "wsc" / "polysomnography"
    wsc_psg.mkdir(parents=True)
    (wsc_psg / "patient-003.edf").touch()
    make_wsc_annotation(wsc_psg / "patient-003.stg.txt", [0, 1, 2, 5])


# ─────────────────────────────────────────────
# Dataset Tests
# ─────────────────────────────────────────────


class TestWav2SleepDatasetMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_root = Path(self.tmp_dir) / "psg_root"
        self.tmp_root.mkdir()
        build_tmp_root(self.tmp_root)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _make_dataset_instance(self):
        """Return a Wav2SleepDataset with __init__ bypassed."""
        from pyhealth.datasets.wav2sleep import Wav2SleepDataset

        with patch.object(Wav2SleepDataset, "__init__", lambda self, **kw: None):
            ds = Wav2SleepDataset.__new__(Wav2SleepDataset)
        return ds

    def test_prepare_metadata_creates_csv(self):
        """prepare_metadata should produce a CSV at root/wav2sleep-metadata.csv."""
        ds = self._make_dataset_instance()
        ds.prepare_metadata(str(self.tmp_root))
        self.assertTrue((self.tmp_root / "wav2sleep-metadata.csv").exists())

    def test_metadata_columns(self):
        """CSV should have patient_id, source_dataset, edf_path, label_path."""
        ds = self._make_dataset_instance()
        ds.prepare_metadata(str(self.tmp_root))
        df = pd.read_csv(self.tmp_root / "wav2sleep-metadata.csv")
        self.assertTrue(
            {"patient_id", "source_dataset", "edf_path", "label_path"}.issubset(
                df.columns
            )
        )

    def test_metadata_patient_count(self):
        """Should find exactly 3 patients across the 3 synthetic datasets."""
        ds = self._make_dataset_instance()
        ds.prepare_metadata(str(self.tmp_root))
        df = pd.read_csv(self.tmp_root / "wav2sleep-metadata.csv")
        self.assertEqual(len(df), 3)

    def test_metadata_skips_missing_label(self):
        """If an EDF has no matching annotation, it should be skipped."""
        (self.tmp_root / "mesa" / "polysomnography" / "edfs" / "orphan.edf").touch()
        ds = self._make_dataset_instance()
        ds.prepare_metadata(str(self.tmp_root))
        df = pd.read_csv(self.tmp_root / "wav2sleep-metadata.csv")
        self.assertNotIn("orphan", df["patient_id"].values)

    def test_get_edf_and_label_dirs_flat(self):
        """Flat datasets (mesa) should return a single (edf_dir, label_dir) pair."""
        ds = self._make_dataset_instance()
        pairs = ds.get_edf_and_label_dirs(self.tmp_root / "mesa")
        self.assertEqual(len(pairs), 1)

    def test_get_edf_and_label_dirs_subdirs(self):
        """Datasets with subdirectory layers (shhs) should return one pair per subdir."""
        ds = self._make_dataset_instance()
        pairs = ds.get_edf_and_label_dirs(self.tmp_root / "shhs")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].name, "shhs1")

    def test_invalid_root_raises(self):
        """Passing a nonexistent root should raise FileNotFoundError."""
        from pyhealth.datasets.wav2sleep import Wav2SleepDataset

        with self.assertRaises(FileNotFoundError):
            Wav2SleepDataset(root=str(self.tmp_root / "does_not_exist"))


# ─────────────────────────────────────────────
# Task Tests — annotation parsing
# ─────────────────────────────────────────────


class TestLoadStages(unittest.TestCase):
    def setUp(self):
        self.task = Wav2SleepStaging()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_xml_wake_maps_correctly(self):
        """XML stage 0 → category 0 (Wake)."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [0])
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(result[0], 0)

    def test_xml_n1_n2_map_to_light(self):
        """XML stages 1 and 2 → category 1 (Light)."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [1, 2])
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 1)

    def test_xml_n3_maps_to_deep(self):
        """XML stages 3 and 4 → category 2 (Deep)."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [3, 4])
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 2)

    def test_xml_rem_maps_correctly(self):
        """XML stage 5 → category 3 (REM)."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [5])
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(result[0], 3)

    def test_xml_unscored_maps_to_minus_one(self):
        """XML stage 9 → category -1 (Unscored)."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [9])
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(result[0], -1)

    def test_wsc_stage_mapping(self):
        """WSC TSV stages map correctly via WSC_STAGE_MAP."""
        p = self.tmp_path / "p.stg.txt"
        make_wsc_annotation(p, [0, 1, 5, 7])
        result = self.task.load_stages(str(p), "wsc")
        self.assertEqual(list(result[:4]), [0, 1, 3, -1])

    def test_output_padded_to_1200(self):
        """Short recordings should be padded to T=1200 epochs."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [0, 5])
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(len(result), 1200)
        self.assertEqual(result[1199], -1)

    def test_output_truncated_to_1200(self):
        """Recordings longer than 1200 epochs should be truncated."""
        p = self.tmp_path / "p.xml"
        make_xml_annotation(p, [0] * 1500)
        result = self.task.load_stages(str(p), "shhs")
        self.assertEqual(len(result), 1200)


# ─────────────────────────────────────────────
# Task Tests — signal preprocessing
# ─────────────────────────────────────────────


class TestPadOrTruncate(unittest.TestCase):
    def setUp(self):
        self.task = Wav2SleepStaging()

    def test_truncates_signal(self):
        arr = np.ones(200)
        result = self.task._pad_or_truncate(arr, is_label=False, target_length=100)
        self.assertEqual(len(result), 100)

    def test_pads_signal_with_zeros(self):
        arr = np.ones(50)
        result = self.task._pad_or_truncate(arr, is_label=False, target_length=100)
        self.assertEqual(len(result), 100)
        self.assertEqual(result[99], 0.0)

    def test_pads_labels_with_minus_one(self):
        arr = np.array([0, 1, 2], dtype=np.int8)
        result = self.task._pad_or_truncate(arr, is_label=True, target_length=10)
        self.assertEqual(result[9], -1)

    def test_exact_length_unchanged(self):
        arr = np.arange(100)
        result = self.task._pad_or_truncate(arr, is_label=False, target_length=100)
        self.assertEqual(len(result), 100)
        np.testing.assert_array_equal(result, arr)


class TestPreprocessSignal(unittest.TestCase):
    def setUp(self):
        self.task = Wav2SleepStaging()

    def test_none_signal_returns_zeros(self):
        """Missing signals (None) should return a zero-filled array."""
        result = self.task.preprocess_signal("ECG", None, 256.0)
        self.assertTrue(np.all(result == 0))
        self.assertEqual(result.dtype, np.float32)

    def test_output_is_unit_normalized(self):
        """Output should have mean ≈ 0 and std ≈ 1."""
        signal = np.random.randn(7680).astype(np.float32)
        result = self.task.preprocess_signal("ECG", signal, 256.0)
        self.assertAlmostEqual(result.mean(), 0.0, delta=0.05)
        self.assertAlmostEqual(result.std(), 1.0, delta=0.05)

    def test_ecg_output_length(self):
        """ECG output should have T * k = 1200 * 1024 samples."""
        signal = np.random.randn(7680).astype(np.float32)
        result = self.task.preprocess_signal("ECG", signal, 256.0)
        self.assertEqual(len(result), 1200 * 1024)

    def test_thx_output_length(self):
        """THX output should have T * k = 1200 * 256 samples."""
        signal = np.random.randn(7680).astype(np.float32)
        result = self.task.preprocess_signal("THX", signal, 256.0)
        self.assertEqual(len(result), 1200 * 256)


# ─────────────────────────────────────────────
# Task Tests — load_signals (mocked MNE)
# ─────────────────────────────────────────────


class TestLoadSignals(unittest.TestCase):
    def setUp(self):
        self.task = Wav2SleepStaging()

    def test_shhs_loads_no_ppg(self):
        """SHHS has no PPG — availability_mask should mark PPG as missing."""
        mock_raw = make_mock_raw(["ECG", "THOR RES", "ABDO RES"])
        with patch("mne.io.read_raw_edf", return_value=mock_raw):
            signals, mask = self.task.load_signals("fake.edf", "shhs")
        ppg_idx = list(signals.keys()).index("PPG")
        self.assertTrue(mask[ppg_idx])

    def test_mesa_loads_ppg(self):
        """MESA has PPG — availability_mask should mark PPG as present."""
        mock_raw = make_mock_raw(["EKG", "Pleth", "Thor", "Abdo"])
        with patch("mne.io.read_raw_edf", return_value=mock_raw):
            signals, mask = self.task.load_signals("fake.edf", "mesa")
        ppg_idx = list(signals.keys()).index("PPG")
        self.assertFalse(mask[ppg_idx])

    def test_signal_keys(self):
        """Returned signals dict must contain ECG, PPG, THX, ABD."""
        mock_raw = make_mock_raw(["ECG", "THOR RES", "ABDO RES"])
        with patch("mne.io.read_raw_edf", return_value=mock_raw):
            signals, _ = self.task.load_signals("fake.edf", "shhs")
        self.assertEqual(set(signals.keys()), {"ECG", "PPG", "THX", "ABD"})

    def test_availability_mask_length(self):
        """Availability mask should have one entry per signal (4 total)."""
        mock_raw = make_mock_raw(["ECG", "THOR RES", "ABDO RES"])
        with patch("mne.io.read_raw_edf", return_value=mock_raw):
            _, mask = self.task.load_signals("fake.edf", "shhs")
        self.assertEqual(len(mask), 4)


class TestPatientEvents(unittest.TestCase):
    """Tests that Wav2SleepStaging.__call__ correctly parses patient events."""

    def setUp(self):
        self.task = Wav2SleepStaging()

    def _make_event(self, edf_path: str, label_path: str, source_dataset: str):
        """Build a mock event with the fields __call__ accesses."""
        event = MagicMock()
        event.edf_path = edf_path
        event.label_path = label_path
        event.source_dataset = source_dataset
        return event

    def _make_patient(self, events: list) -> MagicMock:
        """Build a mock Patient whose get_events() returns the given events."""
        patient = MagicMock()
        patient.patient_id = "test-patient"
        patient.get_events.return_value = events
        return patient

    def test_call_returns_one_sample_per_event(self):
        """__call__ should produce exactly one sample per event on the patient."""
        fake_signals = {
            "ECG": np.zeros(1200 * 1024, dtype=np.float32),
            "PPG": np.zeros(1200 * 1024, dtype=np.float32),
            "THX": np.zeros(1200 * 256, dtype=np.float32),
            "ABD": np.zeros(1200 * 256, dtype=np.float32),
        }
        fake_mask = [False, True, False, False]
        fake_stages = np.zeros(1200, dtype=np.int8)

        event = self._make_event("fake.edf", "fake.xml", "shhs")
        patient = self._make_patient([event])

        with (
            patch.object(
                self.task, "load_signals", return_value=(fake_signals, fake_mask)
            ),
            patch.object(self.task, "load_stages", return_value=fake_stages),
        ):
            samples = self.task(patient)

        self.assertEqual(len(samples), 1)

    def test_sample_contains_required_keys(self):
        """Each sample should contain patient_id, all signals, availability_mask, stages."""
        fake_signals = {
            "ECG": np.zeros(1200 * 1024, dtype=np.float32),
            "PPG": np.zeros(1200 * 1024, dtype=np.float32),
            "THX": np.zeros(1200 * 256, dtype=np.float32),
            "ABD": np.zeros(1200 * 256, dtype=np.float32),
        }
        fake_mask = [False, False, False, False]
        fake_stages = np.zeros(1200, dtype=np.int8)

        event = self._make_event("fake.edf", "fake.xml", "mesa")
        patient = self._make_patient([event])

        with (
            patch.object(
                self.task, "load_signals", return_value=(fake_signals, fake_mask)
            ),
            patch.object(self.task, "load_stages", return_value=fake_stages),
        ):
            samples = self.task(patient)

        expected_keys = {
            "patient_id",
            "ECG",
            "PPG",
            "THX",
            "ABD",
            "availability_mask",
            "stages",
        }
        self.assertEqual(set(samples[0].keys()), expected_keys)

    def test_sample_patient_id_matches(self):
        """patient_id in each sample should match the patient object."""
        fake_signals = {
            k: np.zeros(10, dtype=np.float32) for k in ["ECG", "PPG", "THX", "ABD"]
        }
        fake_mask = [False, False, False, False]
        fake_stages = np.zeros(1200, dtype=np.int8)

        event = self._make_event("fake.edf", "fake.xml", "shhs")
        patient = self._make_patient([event])

        with (
            patch.object(
                self.task, "load_signals", return_value=(fake_signals, fake_mask)
            ),
            patch.object(self.task, "load_stages", return_value=fake_stages),
        ):
            samples = self.task(patient)

        self.assertEqual(samples[0]["patient_id"], "test-patient")

    def test_event_fields_forwarded_correctly(self):
        """load_signals and load_stages should be called with the event's path and dataset."""
        fake_signals = {
            k: np.zeros(10, dtype=np.float32) for k in ["ECG", "PPG", "THX", "ABD"]
        }
        fake_mask = [False, False, False, False]
        fake_stages = np.zeros(1200, dtype=np.int8)

        event = self._make_event("path/to/recording.edf", "path/to/labels.xml", "mesa")
        patient = self._make_patient([event])

        with (
            patch.object(
                self.task, "load_signals", return_value=(fake_signals, fake_mask)
            ) as mock_signals,
            patch.object(
                self.task, "load_stages", return_value=fake_stages
            ) as mock_stages,
        ):
            self.task(patient)

        mock_signals.assert_called_once_with("path/to/recording.edf", "mesa")
        mock_stages.assert_called_once_with("path/to/labels.xml", "mesa")

    def test_multiple_events_produce_multiple_samples(self):
        """A patient with multiple events should produce one sample per event."""
        fake_signals = {
            k: np.zeros(10, dtype=np.float32) for k in ["ECG", "PPG", "THX", "ABD"]
        }
        fake_mask = [False, False, False, False]
        fake_stages = np.zeros(1200, dtype=np.int8)

        events = [
            self._make_event(f"recording_{i}.edf", f"labels_{i}.xml", "shhs")
            for i in range(3)
        ]
        patient = self._make_patient(events)

        with (
            patch.object(
                self.task, "load_signals", return_value=(fake_signals, fake_mask)
            ),
            patch.object(self.task, "load_stages", return_value=fake_stages),
        ):
            samples = self.task(patient)

        self.assertEqual(len(samples), 3)


# ─────────────────────────────────────────────
# Model Tests — sub-components
# ─────────────────────────────────────────────


class TestResidualBlock(unittest.TestCase):
    def test_output_shape(self):
        """Output should be (batch, c_out, length // pool_size)."""
        block = ResidualBlock(c_in=1, c_out=16, pool_size=2)
        x = torch.randn(2, 1, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_same_channel_skip_is_identity(self):
        """c_in == c_out should use an Identity skip connection."""
        block = ResidualBlock(c_in=16, c_out=16)
        self.assertIsInstance(block.skip, nn.Identity)

    def test_different_channel_skip_is_conv(self):
        """c_in != c_out should use a Conv1d skip connection."""
        block = ResidualBlock(c_in=1, c_out=16)
        self.assertIsInstance(block.skip, nn.Conv1d)


class TestSignalEncoder(unittest.TestCase):
    def test_invalid_sample_rate_raises(self):
        """Sample rates other than 1024 or 256 should raise ValueError."""
        with self.assertRaises(ValueError):
            SignalEncoder(signal_sample_rate=512)

    def test_ecg_output_feature_dim(self):
        """ECG encoder dense layer should project to FEATURE_DIM."""
        encoder = SignalEncoder(signal_sample_rate=1024)
        self.assertEqual(encoder.dense.out_features, FEATURE_DIM)

    def test_thx_output_feature_dim(self):
        """THX encoder dense layer should project to FEATURE_DIM."""
        encoder = SignalEncoder(signal_sample_rate=256)
        self.assertEqual(encoder.dense.out_features, FEATURE_DIM)


class TestDilatedConvBlock(unittest.TestCase):
    def test_output_shape_preserved(self):
        """Dilated conv block should preserve sequence length and feature dim."""
        block = DilatedConvBlock(dilation=2, kernel_size=7)
        x = torch.randn(2, FEATURE_DIM, 50)
        out = block(x)
        self.assertEqual(out.shape, (2, FEATURE_DIM, 50))


class TestEpochMixer(unittest.TestCase):
    def setUp(self):
        self.T = 4
        self.batch = 2
        self.mixer = EpochMixer()

    def test_output_shape(self):
        """Output should collapse modality dim, preserving (batch, T, feature_dim)."""
        x = torch.randn(self.batch, self.T, 4, FEATURE_DIM)
        mask = torch.zeros(self.batch, 4, dtype=torch.bool)
        self.mixer.eval()
        with torch.no_grad():
            out = self.mixer(x, mask)
        self.assertEqual(out.shape, (self.batch, self.T, FEATURE_DIM))

    def test_attention_mask_cls_always_unmasked(self):
        """CLS token (index 0) should never be masked."""
        mask = torch.ones(self.batch, 4, dtype=torch.bool)
        attn_mask = self.mixer._build_attention_mask(self.batch, self.T, mask)
        self.assertFalse(attn_mask[:, 0].any())

    def test_attention_mask_at_least_one_modality_visible(self):
        """Even if all modalities are masked, at least one should be forced visible."""
        mask = torch.ones(self.batch, 4, dtype=torch.bool)
        attn_mask = self.mixer._build_attention_mask(self.batch, self.T, mask)
        modality_cols = attn_mask[:, 1:]
        self.assertFalse(modality_cols.all())


class TestSequenceMixer(unittest.TestCase):
    def test_output_shape(self):
        """Sequence mixer should preserve (batch, T, feature_dim)."""
        mixer = SequenceMixer()
        x = torch.randn(2, 10, FEATURE_DIM)
        out = mixer(x)
        self.assertEqual(out.shape, (2, 10, FEATURE_DIM))


# ─────────────────────────────────────────────
# Model Tests — Wav2Sleep
# ─────────────────────────────────────────────


class _FakeEncoder(nn.Module):
    """Minimal nn.Module stand-in for SignalEncoder in forward pass tests."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self._output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._output


class TestWav2Sleep(unittest.TestCase):
    """Tests for the full Wav2Sleep model.

    Signal encoders are mocked to return pre-computed (batch, T, feature_dim)
    tensors, keeping tests fast while still exercising the epoch mixer,
    sequence mixer, and classifier.
    """

    T = 4
    BATCH = 2

    def _make_model(self, modalities=None):
        with patch.object(
            BaseModel, "__init__", lambda self, **kw: nn.Module.__init__(self)
        ):
            return Wav2Sleep(dataset=MagicMock(), modalities=modalities)

    def _mock_encoders(self, model):
        """Replace signal encoders with lightweight fakes that return synthetic tensors."""
        for m in model.selected_modalities:
            model.signal_encoders[m] = _FakeEncoder(
                torch.randn(self.BATCH, self.T, FEATURE_DIM)
            )

    def _make_batch(self, model, with_labels=False):
        """Build a minimal input batch (real signal content is irrelevant — encoders are mocked)."""
        batch = {m: torch.zeros(self.BATCH, 1) for m in model.selected_modalities}
        batch["availability_mask"] = torch.zeros(self.BATCH, 4, dtype=torch.bool)
        if with_labels:
            batch["stages"] = torch.randint(0, 4, (self.BATCH, self.T))
        return batch

    def test_default_modalities(self):
        """Default model should include all four modalities."""
        model = self._make_model()
        self.assertEqual(
            set(model.selected_modalities.keys()), {"ECG", "PPG", "THX", "ABD"}
        )

    def test_subset_modalities(self):
        """Model should only include the modalities it was initialised with."""
        model = self._make_model(modalities=["ECG", "THX"])
        self.assertEqual(set(model.selected_modalities.keys()), {"ECG", "THX"})

    def test_forward_output_keys_without_labels(self):
        """Without ground-truth stages, output should have y_prob and y_hat only."""
        model = self._make_model()
        self._mock_encoders(model)
        model.eval()
        with torch.no_grad():
            output = model(**self._make_batch(model))
        self.assertIn("y_prob", output)
        self.assertIn("y_hat", output)
        self.assertNotIn("loss", output)

    def test_forward_output_shapes(self):
        """y_prob should be (batch, T, 4) and y_hat should be (batch, T)."""
        model = self._make_model()
        self._mock_encoders(model)
        model.eval()
        with torch.no_grad():
            output = model(**self._make_batch(model))
        self.assertEqual(output["y_prob"].shape, (self.BATCH, self.T, 4))
        self.assertEqual(output["y_hat"].shape, (self.BATCH, self.T))

    def test_forward_with_labels_returns_loss(self):
        """Providing ground-truth stages should add a scalar loss to the output."""
        model = self._make_model()
        self._mock_encoders(model)
        model.eval()
        with torch.no_grad():
            output = model(**self._make_batch(model, with_labels=True))
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"].item(), float)

    def test_y_prob_sums_to_one(self):
        """Softmax output should sum to 1 across the class dimension."""
        model = self._make_model()
        self._mock_encoders(model)
        model.eval()
        with torch.no_grad():
            output = model(**self._make_batch(model))
        sums = output["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_gradient_computation(self):
        """Loss should be differentiable — classifier weights should receive gradients."""
        model = self._make_model()
        self._mock_encoders(model)
        model.train()
        output = model(**self._make_batch(model, with_labels=True))
        output["loss"].backward()
        self.assertIsNotNone(model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
