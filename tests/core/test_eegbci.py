import os
import unittest
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.tasks.eegbci import (
    EEGBCI_LABELS,
    label_family_for_run,
    numeric_label_for_task,
    run_type_for_run,
    task_label_for_event,
)


class TestEEGBCIHelpers(unittest.TestCase):
    def test_run_type_for_run(self):
        self.assertEqual(run_type_for_run(3), "motor_execution_left_right")
        self.assertEqual(run_type_for_run(4), "motor_imagery_left_right")
        self.assertEqual(run_type_for_run(5), "motor_execution_fists_feet")
        self.assertEqual(run_type_for_run(6), "motor_imagery_fists_feet")
        self.assertEqual(run_type_for_run(14), "motor_imagery_fists_feet")

    def test_task_label_for_event_is_run_aware(self):
        self.assertEqual(task_label_for_event(3, "T0"), "rest")
        self.assertEqual(task_label_for_event(3, "T1"), "execute_left_fist")
        self.assertEqual(task_label_for_event(3, "T2"), "execute_right_fist")
        self.assertEqual(task_label_for_event(4, "T1"), "imagine_left_fist")
        self.assertEqual(task_label_for_event(4, "T2"), "imagine_right_fist")
        self.assertEqual(task_label_for_event(5, "T1"), "execute_both_fists")
        self.assertEqual(task_label_for_event(5, "T2"), "execute_both_feet")
        self.assertEqual(task_label_for_event(6, "T1"), "imagine_both_fists")
        self.assertEqual(task_label_for_event(6, "T2"), "imagine_both_feet")

    def test_label_family_and_numeric_labels(self):
        self.assertEqual(label_family_for_run(3), "motor_execution")
        self.assertEqual(label_family_for_run(4), "motor_imagery")
        self.assertEqual(numeric_label_for_task("rest"), 0)
        self.assertEqual(numeric_label_for_task("execute_left_fist"), 1)
        self.assertEqual(numeric_label_for_task("imagine_both_feet"), 8)

    def test_invalid_run_and_event_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Unsupported EEGBCI run"):
            run_type_for_run(2)
        with self.assertRaisesRegex(ValueError, "Unsupported EEGBCI event"):
            task_label_for_event(3, "BAD")

    def test_select_eegbci_channels_compat16(self):
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS, select_eegbci_channels

        ch_names = list(EEGBCI_COMPAT_CHANNELS) + ["EXTRA"]
        data = np.arange(len(ch_names) * 100, dtype=float).reshape(len(ch_names), 100)
        selected, selected_names = select_eegbci_channels(data, ch_names, "compat16")
        self.assertEqual(selected.shape, (16, 100))
        self.assertEqual(selected_names, list(EEGBCI_COMPAT_CHANNELS))
        np.testing.assert_allclose(selected[0], data[0])

    def test_select_eegbci_channels_all(self):
        from pyhealth.tasks.eegbci import select_eegbci_channels

        data = np.ones((64, 50))
        ch_names = [f"CH{i}" for i in range(64)]
        selected, selected_names = select_eegbci_channels(data, ch_names, "all")
        self.assertEqual(selected.shape, (64, 50))
        self.assertEqual(selected_names, ch_names)

    def test_select_eegbci_channels_missing_channel_raises(self):
        from pyhealth.tasks.eegbci import select_eegbci_channels

        with self.assertRaisesRegex(ValueError, "Missing EEGBCI channels"):
            select_eegbci_channels(np.ones((2, 20)), ["C3", "C4"], "compat16")

    def test_normalize_signal_95th_percentile(self):
        from pyhealth.tasks.eegbci import normalize_signal

        signal = np.array([[0.0, 1.0, 2.0, 100.0], [0.0, -2.0, 2.0, 4.0]])
        normalized = normalize_signal(signal, "95th_percentile")
        self.assertEqual(normalized.shape, signal.shape)
        self.assertLess(np.max(np.abs(normalized[0])), 2.0)

    def test_compute_band_powers_detects_alpha_sinusoid(self):
        from pyhealth.tasks.eegbci import compute_band_powers

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        alpha = np.sin(2 * np.pi * 10 * times)
        data = np.stack([alpha, alpha])
        features = compute_band_powers(data, sfreq)
        self.assertEqual(features["dominant_band"], "alpha")
        self.assertGreater(features["alpha_relative"], 0.5)
        self.assertGreater(features["alpha_beta_ratio"], 1.0)

    def test_compute_band_powers_detects_beta_sinusoid(self):
        from pyhealth.tasks.eegbci import compute_band_powers

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        beta = np.sin(2 * np.pi * 20 * times)
        data = np.stack([beta, beta])
        features = compute_band_powers(data, sfreq)
        self.assertEqual(features["dominant_band"], "beta")
        self.assertGreater(features["beta_relative"], 0.5)

    def test_interpret_band_profile_returns_cautious_metadata(self):
        from pyhealth.tasks.eegbci import interpret_band_profile

        interpretation = interpret_band_profile(
            {
                "dominant_band": "alpha",
                "alpha_relative": 0.65,
                "beta_relative": 0.10,
                "theta_relative": 0.10,
                "gamma_relative": 0.05,
                "alpha_beta_ratio": 6.5,
                "theta_beta_ratio": 1.0,
            }
        )
        self.assertEqual(interpretation["brain_state_hypothesis"], "relaxed_or_idle")
        self.assertIn(interpretation["confidence"], {"low", "medium", "high"})
        self.assertIn("consistent with", interpretation["interpretation"])


from pyhealth.datasets.eegbci import EEGBCIDataset


class TestEEGBCIDataset(unittest.TestCase):
    def test_prepare_metadata_with_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edf = root / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R03.edf"
            edf.parent.mkdir(parents=True)
            edf.write_bytes(b"")

            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = str(root)
            ds.subjects = [1]
            ds.runs = [3]
            ds.download = False
            ds.prepare_metadata()

            csv_path = root / "eegbci-pyhealth.csv"
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "patient_id"], "S001")
            self.assertEqual(df.loc[0, "record_id"], "R03")
            self.assertEqual(df.loc[0, "subject_id"], 1)
            self.assertEqual(df.loc[0, "run"], 3)
            self.assertEqual(df.loc[0, "run_type"], "motor_execution_left_right")
            self.assertEqual(df.loc[0, "source"], "physionet_eegbci")

    def test_prepare_metadata_download_uses_mne_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_path = root / "S001R04.edf"
            fake_path.write_bytes(b"")
            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = str(root)
            ds.subjects = [1]
            ds.runs = [4]
            ds.download = True

            with patch(
                "pyhealth.datasets.eegbci.mne.datasets.eegbci.load_data",
                return_value=[str(fake_path)],
            ) as load_data:
                ds.prepare_metadata()

            load_data.assert_called_once_with(1, [4], path=str(root), update_path=False)
            df = pd.read_csv(root / "eegbci-pyhealth.csv")
            self.assertEqual(df.loc[0, "record_id"], "R04")
            self.assertEqual(df.loc[0, "run_type"], "motor_imagery_left_right")

    def test_prepare_metadata_missing_local_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = tmp
            ds.subjects = [1]
            ds.runs = [3]
            ds.download = False
            with self.assertRaisesRegex(FileNotFoundError, "download=True"):
                ds.prepare_metadata()

    def test_default_task_returns_pattern_discovery(self):
        from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery

        ds = EEGBCIDataset.__new__(EEGBCIDataset)
        self.assertIsInstance(ds.default_task, EEGBCIPatternDiscovery)


from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI


@dataclass
class _EEGBCIEvent:
    signal_file: str
    record_id: str = "R03"
    subject_id: int = 1
    run: int = 3
    run_type: str = "motor_execution_left_right"
    source: str = "physionet_eegbci"


class _EEGBCIPatient:
    def __init__(self, patient_id: str, events: List[_EEGBCIEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_EEGBCIEvent]:
        if event_type not in (None, "records"):
            return []
        return self._events


class TestEEGBCITasks(unittest.TestCase):
    def test_task_schema_attributes(self):
        task = EEGMotorImageryEEGBCI()
        self.assertEqual(task.task_name, "EEGBCI_motor_imagery")
        self.assertEqual(task.input_schema, {"signal": "tensor", "stft": "tensor"})
        self.assertEqual(task.output_schema, {"label": "multiclass"})

    def test_task_schema_without_stft(self):
        task = EEGMotorImageryEEGBCI(compute_stft=False)
        self.assertEqual(task.input_schema, {"signal": "tensor"})

    def test_pattern_discovery_schema_attributes(self):
        task = EEGBCIPatternDiscovery(compute_stft=False)
        self.assertEqual(task.task_name, "EEGBCI_pattern_discovery")
        self.assertEqual(task.input_schema, {"signal": "tensor"})

    def test_iter_annotation_windows_uses_full_2s_windows(self):
        import mne
        from pyhealth.tasks.eegbci import iter_annotation_windows

        sfreq = 200.0
        raw = mne.io.RawArray(
            np.zeros((2, int(sfreq * 6))),
            mne.create_info(["C3", "C4"], sfreq=sfreq, ch_types=["eeg", "eeg"]),
            verbose="error",
        )
        raw.set_annotations(
            mne.Annotations(onset=[0.5, 2.0], duration=[1.0, 3.0], description=["T0", "T1"])
        )
        windows = iter_annotation_windows(raw, run=3, window_size=2.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["event_code"], "T1")
        self.assertEqual(windows[0]["task_label"], "execute_left_fist")
        self.assertEqual(windows[0]["start_sample"], 400)
        self.assertEqual(windows[0]["end_sample"], 800)

    def test_motor_imagery_task_returns_samples_from_raw(self):
        import mne

        sfreq = 200.0
        raw = mne.io.RawArray(
            np.ones((16, int(sfreq * 5))),
            mne.create_info(
                list(
                    __import__(
                        "pyhealth.tasks.eegbci", fromlist=["EEGBCI_COMPAT_CHANNELS"]
                    ).EEGBCI_COMPAT_CHANNELS
                ),
                sfreq=sfreq,
                ch_types=["eeg"] * 16,
            ),
            verbose="error",
        )
        raw.set_annotations(mne.Annotations(onset=[0.0], duration=[2.0], description=["T1"]))
        patient = _EEGBCIPatient("S001", [_EEGBCIEvent(signal_file="dummy.edf")])
        task = EEGMotorImageryEEGBCI(compute_stft=False, resample_rate=None, bandpass_filter=None)

        with patch("pyhealth.tasks.eegbci.mne.io.read_raw_edf", return_value=raw):
            samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "S001")
        self.assertEqual(sample["record_id"], "R03")
        self.assertEqual(sample["event_code"], "T1")
        self.assertEqual(sample["task_label"], "execute_left_fist")
        self.assertEqual(sample["label"], 1)
        self.assertEqual(tuple(sample["signal"].shape), (16, 400))

    def test_pattern_discovery_adds_bandpower_metadata(self):
        import mne
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        alpha = np.sin(2 * np.pi * 10 * times)
        raw = mne.io.RawArray(
            np.tile(alpha, (16, 1)),
            mne.create_info(list(EEGBCI_COMPAT_CHANNELS), sfreq=sfreq, ch_types=["eeg"] * 16),
            verbose="error",
        )
        raw.set_annotations(mne.Annotations(onset=[0.0], duration=[2.0], description=["T0"]))
        patient = _EEGBCIPatient("S001", [_EEGBCIEvent(signal_file="dummy.edf")])
        task = EEGBCIPatternDiscovery(compute_stft=False, resample_rate=None, bandpass_filter=None)

        with patch("pyhealth.tasks.eegbci.mne.io.read_raw_edf", return_value=raw):
            samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["bandpower"]["dominant_band"], "alpha")
        self.assertEqual(sample["brain_state_hypothesis"], "relaxed_or_idle")
        self.assertIn("interpretation", sample)


@unittest.skipUnless(
    os.environ.get("PYHEALTH_RUN_REAL_EEGBCI") == "1",
    "Set PYHEALTH_RUN_REAL_EEGBCI=1 to download and test real EEGBCI data.",
)
class TestEEGBCIRealDataSmoke(unittest.TestCase):
    def test_real_eegbci_subject_1_run_3_pattern_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = EEGBCIDataset(root=tmp, subjects=[1], runs=[3], download=True)
            sample_dataset = dataset.set_task(
                EEGBCIPatternDiscovery(compute_stft=False, window_size=2.0)
            )
            self.assertGreater(len(sample_dataset), 0)
            sample = sample_dataset[0]
            self.assertIn("signal", sample)
            self.assertEqual(sample["signal"].shape[0], 16)
            self.assertIn(sample["task_label"], set(EEGBCI_LABELS))
            self.assertIn("bandpower", sample)
            self.assertIn("brain_state_hypothesis", sample)
