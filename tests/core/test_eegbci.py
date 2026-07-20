import os
import sys
import unittest
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

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
        self.assertNotIn(
            "This is exploratory signal metadata", interpretation["interpretation"]
        )
        self.assertNotIn("clinical diagnosis", interpretation["interpretation"])


from pyhealth.datasets.eegbci import EEGBCIDataset


class TestEEGBCIDataset(unittest.TestCase):
    def _set_metadata_identity(self, ds):
        ds.selection_key = ds._build_selection_key()
        ds.metadata_file_name = ds._metadata_file_name()

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
            self._set_metadata_identity(ds)
            ds.prepare_metadata()

            csv_path = root / ds.metadata_file_name
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "patient_id"], "S001")
            self.assertEqual(df.loc[0, "record_id"], "R03")
            self.assertEqual(df.loc[0, "subject_id"], 1)
            self.assertEqual(df.loc[0, "run"], 3)
            self.assertEqual(df.loc[0, "run_type"], "motor_execution_left_right")
            self.assertEqual(df.loc[0, "source"], "physionet_eegbci")

            edf.unlink()
            self.assertFalse(ds._metadata_matches_request(csv_path))

    def test_metadata_content_changes_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = tmp
            ds.metadata_file_name = "metadata.csv"
            csv_path = Path(tmp) / ds.metadata_file_name

            csv_path.write_text("signal_file\n/old/path.edf\n", encoding="utf-8")
            first_key = ds._metadata_cache_key()
            csv_path.write_text("signal_file\n/new/path.edf\n", encoding="utf-8")

            self.assertNotEqual(first_key, ds._metadata_cache_key())

    def test_selection_inputs_are_normalized_for_stable_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject, run in [(1, 3), (1, 4), (2, 3), (2, 4)]:
                edf = (
                    root
                    / "files"
                    / "eegmmidb"
                    / "1.0.0"
                    / f"S{subject:03d}"
                    / f"S{subject:03d}R{run:02d}.edf"
                )
                edf.parent.mkdir(parents=True, exist_ok=True)
                edf.write_bytes(b"")

            first = EEGBCIDataset(
                root=str(root),
                subjects=[2, 1, 1],
                runs=[4, 3, 4],
                download=False,
            )
            second = EEGBCIDataset(
                root=str(root),
                subjects=[1, 2],
                runs=[3, 4],
                download=False,
            )

            self.assertEqual(first.subjects, [1, 2])
            self.assertEqual(first.runs, [3, 4])
            self.assertEqual(first.selection_key, second.selection_key)
            self.assertEqual(first.dataset_name, second.dataset_name)

    def test_prepare_metadata_uses_selection_specific_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R03.edf"
            second = root / "files" / "eegmmidb" / "1.0.0" / "S002" / "S002R04.edf"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"")
            second.write_bytes(b"")

            ds_first = EEGBCIDataset(
                root=str(root), subjects=[1], runs=[3], download=False
            )
            ds_second = EEGBCIDataset(
                root=str(root), subjects=[2], runs=[4], download=False
            )

            first_csv = root / ds_first.metadata_file_name
            second_csv = root / ds_second.metadata_file_name
            self.assertNotEqual(first_csv, second_csv)
            self.assertTrue(first_csv.exists())
            self.assertTrue(second_csv.exists())

            first_df = pd.read_csv(first_csv)
            second_df = pd.read_csv(second_csv)
            self.assertEqual(first_df.loc[0, "subject_id"], 1)
            self.assertEqual(first_df.loc[0, "run"], 3)
            self.assertEqual(second_df.loc[0, "subject_id"], 2)
            self.assertEqual(second_df.loc[0, "run"], 4)

    def test_find_local_edf_checks_canonical_mne_path_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R03.edf"
            fallback = root / "other" / "S001R03.edf"
            canonical.parent.mkdir(parents=True)
            fallback.parent.mkdir(parents=True)
            canonical.write_bytes(b"")
            fallback.write_bytes(b"")

            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = str(root)

            self.assertEqual(ds._find_local_edf(1, 3), canonical)

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
            self._set_metadata_identity(ds)

            with patch(
                "pyhealth.datasets.eegbci.mne.datasets.eegbci.load_data",
                return_value=[str(fake_path)],
            ) as load_data:
                ds.prepare_metadata()

            load_data.assert_called_once_with(1, [4], path=str(root), update_path=False)
            df = pd.read_csv(root / ds.metadata_file_name)
            self.assertEqual(df.loc[0, "record_id"], "R04")
            self.assertEqual(df.loc[0, "run_type"], "motor_imagery_left_right")

    def test_prepare_metadata_missing_local_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = tmp
            ds.subjects = [1]
            ds.runs = [3]
            ds.download = False
            self._set_metadata_identity(ds)
            with self.assertRaisesRegex(FileNotFoundError, "download=True"):
                ds.prepare_metadata()

    def test_default_task_returns_motor_imagery(self):
        from pyhealth.tasks.eegbci import EEGMotorImageryEEGBCI

        ds = EEGBCIDataset.__new__(EEGBCIDataset)
        self.assertIs(type(ds.default_task), EEGMotorImageryEEGBCI)

    def test_dataset_set_task_offline_integration(self):
        import mne
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edf = root / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R03.edf"
            edf.parent.mkdir(parents=True)
            edf.write_bytes(b"")
            sfreq = 160.0
            times = np.arange(0, 2, 1 / sfreq)
            signal = np.sin(2 * np.pi * 10 * times)
            raw = mne.io.RawArray(
                np.tile(signal, (16, 1)),
                mne.create_info(
                    list(EEGBCI_COMPAT_CHANNELS), sfreq=sfreq, ch_types=["eeg"] * 16
                ),
                verbose="error",
            )
            raw.set_annotations(
                mne.Annotations(onset=[0.0], duration=[2.0], description=["T1"])
            )
            dataset = EEGBCIDataset(
                root=str(root),
                subjects=[1],
                runs=[3],
                download=False,
                cache_dir=root / "cache",
            )

            with patch("pyhealth.tasks.eegbci.mne.io.read_raw_edf", return_value=raw):
                sample_dataset = dataset.set_task(num_workers=1)

            self.assertEqual(len(sample_dataset), 1)
            self.assertEqual(sample_dataset.task_name, "EEGBCI_motor_imagery")
            sample = sample_dataset[0]
            self.assertEqual(sample["task_label"], "execute_left_fist")
            self.assertEqual(sample["eegbci_label"], 1)
            self.assertEqual(tuple(sample["signal"].shape), (16, 400))
            self.assertIn("stft", sample)


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
        self.assertEqual(task.cache_version, "semantic_labels_v1")

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
        self.assertEqual(sample["label"], "execute_left_fist")
        self.assertEqual(sample["eegbci_label"], 1)
        self.assertEqual(tuple(sample["signal"].shape), (16, 400))

    def test_sparse_task_labels_remain_distinct_for_multiclass_processing(self):
        from pyhealth.processors import MultiClassLabelProcessor

        labels = ["rest", "execute_both_fists", "execute_both_feet"]
        processor = MultiClassLabelProcessor()
        processor.fit([{"label": label} for label in labels], "label")

        self.assertEqual(
            len({processor.process(label).item() for label in labels}), len(labels)
        )

    def test_stft_uses_current_sample_rate(self):
        import mne
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS

        sfreq = 100.0
        raw = mne.io.RawArray(
            np.ones((16, int(sfreq * 2))),
            mne.create_info(
                list(EEGBCI_COMPAT_CHANNELS), sfreq=sfreq, ch_types=["eeg"] * 16
            ),
            verbose="error",
        )
        raw.set_annotations(
            mne.Annotations(onset=[0.0], duration=[2.0], description=["T1"])
        )
        patient = _EEGBCIPatient("S001", [_EEGBCIEvent(signal_file="dummy.edf")])
        task = EEGMotorImageryEEGBCI(resample_rate=None, bandpass_filter=None)

        with patch("pyhealth.tasks.eegbci.mne.io.read_raw_edf", return_value=raw):
            samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(tuple(samples[0]["stft"].shape), (16, 50, 3))

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


class TestEEGBCIMomentReportHelpers(unittest.TestCase):
    def _moment_row(self, **overrides):
        row = {
            "patient_id": "S001",
            "record_id": "R03",
            "subject_id": 1,
            "run": 3,
            "run_type": "motor_execution_left_right",
            "trial_id": "S001_R03_T0_0",
            "event_code": "T0",
            "task_label": "rest",
            "label_family": "rest",
            "label": 0,
            "eegbci_label": 0,
            "model_label": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "dominant_band": "alpha",
            "delta_relative": 0.05,
            "theta_relative": 0.10,
            "alpha_relative": 0.55,
            "beta_relative": 0.20,
            "gamma_relative": 0.10,
            "alpha_beta_ratio": 2.75,
            "theta_beta_ratio": 0.50,
        }
        row.update(overrides)
        return row

    def _sample(self, **overrides):
        sample = {
            "patient_id": "S001",
            "record_id": "R03",
            "subject_id": 1,
            "run": 3,
            "run_type": "motor_execution_left_right",
            "trial_id": "S001_R03_T0_0",
            "event_code": "T0",
            "task_label": "rest",
            "label_family": "rest",
            "label": 0,
            "eegbci_label": 0,
            "start_time": 0.0,
            "end_time": 2.0,
            "brain_state_hypothesis": "relaxed_or_idle",
            "confidence": "medium",
            "quality_flags": "",
            "interpretation": "Alpha-dominant profile.",
            "bandpower": {
                "dominant_band": "alpha",
                "alpha_beta_ratio": 2.75,
                "theta_beta_ratio": 0.50,
                "delta_power": 0.05,
                "theta_power": 0.10,
                "alpha_power": 0.55,
                "beta_power": 0.20,
                "gamma_power": 0.10,
                "delta_relative": 0.05,
                "theta_relative": 0.10,
                "alpha_relative": 0.55,
                "beta_relative": 0.20,
                "gamma_relative": 0.10,
            },
        }
        sample.update(overrides)
        return sample

    def test_analysis_version_constant(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import ANALYSIS_VERSION

        self.assertEqual(ANALYSIS_VERSION, "eegbci_pattern_moment_report_v1")

    def test_parse_int_list_strips_whitespace(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import parse_int_list

        self.assertEqual(parse_int_list("1, 2, 4-6"), [1, 2, 4, 5, 6])

    def test_parse_int_list_rejects_descending_ranges(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import parse_int_list

        with self.assertRaisesRegex(ValueError, "Range start must be <= range end"):
            parse_int_list("5-3")

    def test_build_rest_baselines_uses_rest_rows_only(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import build_rest_baselines

        rows = [
            self._moment_row(task_label="rest", subject_id=1, run=3, alpha_relative=0.50),
            self._moment_row(
                task_label="execute_left_fist", subject_id=1, run=3, alpha_relative=0.90
            ),
            self._moment_row(task_label="rest", subject_id=1, run=4, alpha_relative=0.70),
        ]

        baselines = build_rest_baselines(rows)

        self.assertAlmostEqual(
            baselines["same_subject_run"][(1, 3)]["alpha_relative"], 0.50
        )
        self.assertAlmostEqual(
            baselines["same_subject_all_runs"][1]["alpha_relative"], 0.60
        )
        self.assertAlmostEqual(baselines["global_rest"]["alpha_relative"], 0.60)

    def test_build_rest_baselines_handles_no_rest_rows(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import build_rest_baselines

        rows = [
            self._moment_row(
                task_label="execute_left_fist", label_family="motor_execution"
            )
        ]

        baselines = build_rest_baselines(rows)

        self.assertEqual(baselines["same_subject_run"], {})
        self.assertEqual(baselines["same_subject_all_runs"], {})
        self.assertIsNone(baselines["global_rest"])

    def test_render_summary_reports_rest_baseline_source_rows(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        rows = [
            self._moment_row(task_label="rest"),
            self._moment_row(
                task_label="execute_left_fist", label_family="motor_execution"
            ),
            self._moment_row(task_label="rest", run=4),
        ]

        summary = render_summary(
            rows,
            {
                "subjects": [1],
                "runs": [3, 4],
                "max_windows": None,
                "baseline_row_count": 2,
                "output_was_capped": False,
            },
        )

        self.assertIn("- Baseline source rows: 2", summary)

    def test_annotate_rest_fallback_scopes(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [
            self._moment_row(task_label="rest", subject_id=1, run=3, alpha_relative=0.50),
            self._moment_row(task_label="rest", subject_id=1, run=4, alpha_relative=0.70),
            self._moment_row(
                task_label="execute_left_fist",
                label_family="motor_execution",
                subject_id=1,
                run=3,
                alpha_relative=0.80,
            ),
            self._moment_row(
                task_label="execute_left_fist",
                label_family="motor_execution",
                subject_id=1,
                run=5,
                alpha_relative=0.80,
            ),
            self._moment_row(
                task_label="execute_left_fist",
                label_family="motor_execution",
                subject_id=2,
                run=8,
                alpha_relative=0.80,
            ),
        ]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        self.assertEqual(annotated[2]["rest_reference_scope"], "same_subject_run")
        self.assertEqual(annotated[3]["rest_reference_scope"], "same_subject_all_runs")
        self.assertEqual(annotated[4]["rest_reference_scope"], "global_rest")

    def test_derive_state_hypothesis_detects_profiles(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_state_hypothesis

        cases = [
            (
                self._moment_row(
                    alpha_relative=0.60,
                    beta_relative=0.12,
                    gamma_relative=0.05,
                    alpha_beta_ratio=5.0,
                ),
                "idle_alpha_profile",
            ),
            (
                self._moment_row(
                    alpha_relative=0.12,
                    beta_relative=0.48,
                    gamma_relative=0.16,
                    alpha_beta_ratio=0.25,
                ),
                "sensorimotor_engagement_profile",
            ),
            (
                self._moment_row(
                    delta_relative=0.42,
                    theta_relative=0.36,
                    alpha_relative=0.08,
                    beta_relative=0.08,
                ),
                "slow_wave_dominant_pattern",
            ),
            (
                self._moment_row(
                    gamma_relative=0.48, alpha_relative=0.10, beta_relative=0.12
                ),
                "possible_artifact_profile",
            ),
            (
                self._moment_row(
                    delta_relative=0.18,
                    theta_relative=0.20,
                    alpha_relative=0.22,
                    beta_relative=0.21,
                    gamma_relative=0.19,
                    alpha_beta_ratio=1.05,
                ),
                "mixed_ambiguous_profile",
            ),
        ]

        for row, expected in cases:
            with self.subTest(expected=expected):
                result = derive_state_hypothesis(row)
                self.assertEqual(result["state_hypothesis"], expected)
                self.assertIn(result["state_confidence"], {"low", "medium", "high"})
                self.assertGreaterEqual(result["evidence_score"], 0.0)
                self.assertLessEqual(result["evidence_score"], 1.0)
                self.assertIn("alpha=", result["evidence_summary"])

    def test_state_hypothesis_uses_only_finite_rest_deltas(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_state_hypothesis

        delta_profiles = (
            ({"alpha": 0.10, "beta": -0.05}, "idle_alpha_profile"),
            (
                {"alpha": -0.10, "beta": 0.08, "gamma": 0.02},
                "sensorimotor_engagement_profile",
            ),
            ({"delta": 0.10, "theta": 0.08}, "slow_wave_dominant_pattern"),
        )
        for deltas, expected in delta_profiles:
            row = self._moment_row()
            for band in ("delta", "theta", "alpha", "beta", "gamma"):
                row[f"rest_{band}_relative_delta"] = deltas.get(band, 0.0)
            result = derive_state_hypothesis(row)
            self.assertEqual(result["state_hypothesis"], expected)
            self.assertIn("basis=rest_normalized_delta", result["evidence_summary"])

        for invalid in (float("nan"), float("inf")):
            row = self._moment_row(alpha_relative=0.65, alpha_beta_ratio=6.5)
            for band in ("delta", "theta", "alpha", "beta", "gamma"):
                row[f"rest_{band}_relative_delta"] = invalid
            result = derive_state_hypothesis(row)
            self.assertTrue(np.isfinite(result["evidence_score"]))
            self.assertIn("basis=absolute_band_profile", result["evidence_summary"])

    def test_state_confidence_requires_margin(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            STATE_CONFIDENCE_RANK,
            derive_state_hypothesis,
        )

        clear = derive_state_hypothesis(
            self._moment_row(
                alpha_relative=0.70,
                beta_relative=0.10,
                gamma_relative=0.04,
                alpha_beta_ratio=6.0,
            )
        )
        weaker = derive_state_hypothesis(
            self._moment_row(
                alpha_relative=0.40,
                beta_relative=0.22,
                gamma_relative=0.10,
                alpha_beta_ratio=2.0,
            )
        )

        self.assertEqual(clear["state_hypothesis"], weaker["state_hypothesis"])
        self.assertGreater(
            STATE_CONFIDENCE_RANK[clear["state_confidence"]],
            STATE_CONFIDENCE_RANK[weaker["state_confidence"]],
        )

    def test_task_state_relation_table_is_deterministic(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            derive_task_state_relation,
        )

        cases = [
            ("rest", "rest", "idle_alpha_profile", "supports_label"),
            ("rest", "rest", "mixed_ambiguous_profile", "ambiguous"),
            ("rest", "rest", "possible_artifact_profile", "not_applicable"),
            (
                "execute_left_fist",
                "motor_execution",
                "sensorimotor_engagement_profile",
                "supports_label",
            ),
            (
                "imagine_left_fist",
                "motor_imagery",
                "sensorimotor_engagement_profile",
                "adds_detail",
            ),
            ("execute_left_fist", "motor_execution", "idle_alpha_profile", "disagrees"),
            (
                "imagine_left_fist",
                "motor_imagery",
                "slow_wave_dominant_pattern",
                "adds_detail",
            ),
        ]

        for task_label, label_family, state, expected in cases:
            with self.subTest(state=state, label_family=label_family):
                result = derive_task_state_relation(
                    self._moment_row(
                        task_label=task_label,
                        label_family=label_family,
                        state_hypothesis=state,
                    )
                )
                self.assertEqual(result["task_state_relation"], expected)
                self.assertIn(result["task_state_confidence"], {"low", "medium", "high"})
                self.assertGreater(len(result["task_state_rationale"]), 20)

    def test_quality_booleans_are_parseable(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_quality_columns

        flags = derive_quality_columns(
            self._moment_row(
                state_hypothesis="possible_artifact_profile",
                state_confidence="low",
                quality_flags="low_confidence; high_gamma",
            )
        )

        self.assertTrue(flags["is_low_confidence"])
        self.assertTrue(flags["is_possible_artifact"])
        self.assertFalse(flags["is_mixed_or_ambiguous"])

    def test_quality_booleans_do_not_depend_on_string_parsing_only(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_quality_columns

        flags = derive_quality_columns(
            self._moment_row(
                state_hypothesis="mixed_ambiguous_profile",
                state_confidence="medium",
                quality_flags="",
            )
        )

        self.assertTrue(flags["is_mixed_or_ambiguous"])

    def test_quality_booleans_do_not_conflate_legacy_low_confidence(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import derive_quality_columns

        flags = derive_quality_columns(
            self._moment_row(
                state_hypothesis="idle_alpha_profile",
                state_confidence="medium",
                quality_flags="low_confidence",
            )
        )

        self.assertFalse(flags["is_low_confidence"])

    def test_annotate_moment_rows_adds_required_fields(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            ANALYSIS_VERSION,
            MOMENT_REPORT_COLUMNS,
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [
            self._moment_row(task_label="rest", alpha_relative=0.50, beta_relative=0.20),
            self._moment_row(
                task_label="execute_left_fist",
                label_family="motor_execution",
                alpha_relative=0.20,
                beta_relative=0.45,
            ),
        ]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        for annotated_row in annotated:
            for column in MOMENT_REPORT_COLUMNS:
                self.assertIn(column, annotated_row)
        row = annotated[1]
        self.assertEqual(row["analysis_version"], ANALYSIS_VERSION)
        self.assertIn(
            row["state_hypothesis"],
            {
                "idle_alpha_profile",
                "sensorimotor_engagement_profile",
                "slow_wave_dominant_pattern",
                "possible_artifact_profile",
                "mixed_ambiguous_profile",
            },
        )
        self.assertIn("rest_alpha_relative_delta", row)
        self.assertAlmostEqual(row["rest_alpha_relative_delta"], -0.30)
        self.assertIn("task_state_relation", row)
        self.assertIn("task_state_rationale", row)
        self.assertIn("is_low_confidence", row)

    def test_annotate_moment_rows_marks_unavailable_rest(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [
            self._moment_row(
                task_label="execute_left_fist", label_family="motor_execution"
            )
        ]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        self.assertEqual(annotated[0]["rest_reference_scope"], "unavailable")
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            self.assertEqual(annotated[0][f"rest_{band}_relative_delta"], "")

    def test_rest_delta_values_are_band_specific(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [
            self._moment_row(
                task_label="rest",
                delta_relative=0.10,
                theta_relative=0.20,
                alpha_relative=0.30,
                beta_relative=0.25,
                gamma_relative=0.15,
            ),
            self._moment_row(
                task_label="execute_left_fist",
                label_family="motor_execution",
                delta_relative=0.15,
                theta_relative=0.18,
                alpha_relative=0.25,
                beta_relative=0.35,
                gamma_relative=0.07,
            ),
        ]

        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        self.assertAlmostEqual(annotated[1]["rest_delta_relative_delta"], 0.05)
        self.assertAlmostEqual(annotated[1]["rest_theta_relative_delta"], -0.02)
        self.assertAlmostEqual(annotated[1]["rest_alpha_relative_delta"], -0.05)
        self.assertAlmostEqual(annotated[1]["rest_beta_relative_delta"], 0.10)
        self.assertAlmostEqual(annotated[1]["rest_gamma_relative_delta"], -0.08)

    def test_annotate_moment_rows_adds_report_interpretation(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        row = self._moment_row()

        annotated = annotate_moment_rows([row], build_rest_baselines([row]))

        self.assertIn("consistent with", annotated[0]["interpretation"])
        self.assertIn("task label", annotated[0]["interpretation"])
        self.assertIn(annotated[0]["state_hypothesis"], annotated[0]["interpretation"])

    def test_annotate_moment_rows_does_not_mutate_input_rows(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
        )

        rows = [self._moment_row()]
        original = [dict(row) for row in rows]

        annotate_moment_rows(rows, build_rest_baselines(rows))

        self.assertEqual(rows, original)

    def test_select_representative_windows_is_deterministic(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            select_representative_windows,
        )

        rows = [
            self._moment_row(
                subject_id=2,
                run=4,
                start_time=6.0,
                state_hypothesis="idle_alpha_profile",
                state_confidence="medium",
                evidence_score=0.80,
            ),
            self._moment_row(
                subject_id=1,
                run=3,
                start_time=4.0,
                state_hypothesis="idle_alpha_profile",
                state_confidence="medium",
                evidence_score=0.80,
            ),
            self._moment_row(
                subject_id=1,
                run=3,
                start_time=8.0,
                state_hypothesis="sensorimotor_engagement_profile",
                state_confidence="high",
                evidence_score=0.90,
            ),
            self._moment_row(
                subject_id=1,
                run=3,
                start_time=10.0,
                state_hypothesis="mixed_ambiguous_profile",
                state_confidence="low",
                evidence_score=0.12,
            ),
            self._moment_row(
                subject_id=1,
                run=3,
                start_time=12.0,
                state_hypothesis="idle_alpha_profile",
                task_state_relation="disagrees",
                state_confidence="medium",
                evidence_score=0.70,
            ),
        ]

        selected = select_representative_windows(rows)

        self.assertEqual(selected["cards"]["strongest_idle_like"]["subject_id"], 1)
        self.assertEqual(
            selected["cards"]["strongest_motor_engaged"]["state_hypothesis"],
            "sensorimotor_engagement_profile",
        )
        self.assertEqual(selected["cards"]["most_ambiguous"]["start_time"], 10.0)
        self.assertEqual(
            selected["cards"]["strongest_task_state_disagreement"][
                "task_state_relation"
            ],
            "disagrees",
        )
        self.assertIn("strongest_artifact_like", selected["absent"])

    def test_select_representative_windows_picks_lowest_evidence_ambiguous(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            select_representative_windows,
        )

        rows = [
            self._moment_row(
                subject_id=2,
                run=3,
                start_time=4.0,
                state_hypothesis="mixed_ambiguous_profile",
                state_confidence="low",
                evidence_score=0.20,
            ),
            self._moment_row(
                subject_id=1,
                run=3,
                start_time=8.0,
                state_hypothesis="mixed_ambiguous_profile",
                state_confidence="low",
                evidence_score=0.10,
            ),
        ]

        selected = select_representative_windows(rows)

        self.assertEqual(selected["cards"]["most_ambiguous"]["subject_id"], 1)

    def test_select_representative_windows_picks_strongest_disagreement(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            select_representative_windows,
        )

        rows = [
            self._moment_row(
                subject_id=1,
                run=3,
                start_time=4.0,
                state_hypothesis="idle_alpha_profile",
                task_state_relation="disagrees",
                state_confidence="medium",
                evidence_score=0.50,
            ),
            self._moment_row(
                subject_id=2,
                run=3,
                start_time=6.0,
                state_hypothesis="idle_alpha_profile",
                task_state_relation="disagrees",
                state_confidence="medium",
                evidence_score=0.80,
            ),
        ]

        selected = select_representative_windows(rows)

        self.assertEqual(
            selected["cards"]["strongest_task_state_disagreement"]["subject_id"], 2
        )

    def test_render_summary_contains_required_sections_and_limitations(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            ANALYSIS_VERSION,
            annotate_moment_rows,
            build_rest_baselines,
            render_summary,
        )

        rows = [
            self._moment_row(
                task_label="execute_left_fist", label_family="motor_execution"
            )
        ]
        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))
        summary = render_summary(
            annotated,
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": 1,
                "baseline_row_count": 1,
                "output_was_capped": True,
            },
        )

        self.assertIn(ANALYSIS_VERSION, summary.splitlines()[2])
        for heading in [
            "## Executive Result",
            "## Run Configuration",
            "## Window Coverage",
            "## Moment-State Summary",
            "## Task Label x State Matrix",
            "## Rest-Normalized Bandpower Summary",
            "## Confidence and Quality Audit",
            "## Representative Windows",
            "## Limitations",
            "## Next Checks",
        ]:
            self.assertIn(heading, summary)
        self.assertIn("No rest baseline was available", summary)
        self.assertIn("Output was capped by `--max-windows`", summary)
        self.assertNotIn(
            "Brain-state hypotheses are exploratory signal metadata",
            summary.splitlines()[2],
        )

    def test_render_summary_handles_empty_rows(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        summary = render_summary(
            [],
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": 0,
                "baseline_row_count": 0,
                "output_was_capped": True,
            },
        )

        self.assertIn("No windows were produced", summary)
        self.assertIn("## Limitations", summary)

    def test_render_summary_reports_all_low_confidence_and_same_state(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        rows = [
            self._moment_row(
                state_hypothesis="mixed_ambiguous_profile",
                state_confidence="low",
                evidence_score=0.10,
                task_state_relation="ambiguous",
                task_state_confidence="low",
                rest_reference_scope="unavailable",
                is_low_confidence=True,
                is_possible_artifact=False,
                is_mixed_or_ambiguous=True,
            ),
            self._moment_row(
                start_time=2.0,
                state_hypothesis="mixed_ambiguous_profile",
                state_confidence="low",
                evidence_score=0.12,
                task_state_relation="ambiguous",
                task_state_confidence="low",
                rest_reference_scope="unavailable",
                is_low_confidence=True,
                is_possible_artifact=False,
                is_mixed_or_ambiguous=True,
            ),
        ]

        summary = render_summary(
            rows,
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": None,
                "baseline_row_count": 2,
                "output_was_capped": False,
            },
        )

        self.assertIn("Every window is low confidence", summary)
        self.assertIn("Every window maps to the same state", summary)

    def test_render_summary_reports_task_state_matrix(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        rows = [
            self._moment_row(
                task_label="rest",
                state_hypothesis="idle_alpha_profile",
                state_confidence="medium",
                evidence_score=0.60,
                task_state_relation="supports_label",
                task_state_confidence="medium",
                rest_reference_scope="same_subject_run",
            ),
            self._moment_row(
                task_label="execute_left_fist",
                label_family="motor_execution",
                state_hypothesis="sensorimotor_engagement_profile",
                state_confidence="medium",
                evidence_score=0.70,
                task_state_relation="supports_label",
                task_state_confidence="medium",
                rest_reference_scope="same_subject_run",
            ),
        ]

        summary = render_summary(
            rows,
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": None,
                "baseline_row_count": 2,
                "output_was_capped": False,
            },
        )

        self.assertIn("rest x idle_alpha_profile: 1", summary)
        self.assertIn(
            "execute_left_fist x sensorimotor_engagement_profile: 1", summary
        )

    def test_render_summary_includes_representative_window_details(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        row = self._moment_row(
            state_hypothesis="idle_alpha_profile",
            state_confidence="medium",
            evidence_score=0.75,
            task_state_relation="supports_label",
            task_state_confidence="medium",
            task_state_rationale="The idle-like alpha profile is consistent with rest.",
            rest_reference_scope="same_subject_run",
            rest_delta_relative_delta=0.01,
            rest_theta_relative_delta=0.02,
            rest_alpha_relative_delta=0.03,
            rest_beta_relative_delta=-0.02,
            rest_gamma_relative_delta=-0.01,
            is_low_confidence=False,
            is_possible_artifact=False,
            is_mixed_or_ambiguous=False,
        )

        summary = render_summary(
            [row],
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": None,
                "baseline_row_count": 1,
                "output_was_capped": False,
            },
        )

        for text in [
            "Subject 1 run 3 trial S001_R03_T0_0",
            "Task: rest from 0.0s to 2.0s",
            "State: idle_alpha_profile",
            "Dominant band: alpha",
            "Rest deltas:",
            "Task relation: supports_label",
            "low_confidence=False",
            "Rationale: The idle-like alpha profile",
        ]:
            self.assertIn(text, summary)

    def test_render_summary_moves_nonclinical_warning_to_limitations(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import render_summary

        summary = render_summary(
            [self._moment_row(state_hypothesis="idle_alpha_profile")],
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": None,
                "baseline_row_count": 1,
                "output_was_capped": False,
            },
        )

        opening = "\n".join(summary.splitlines()[:6])
        limitations = summary.split("## Limitations", 1)[1]
        self.assertNotIn("clinical findings", opening)
        self.assertIn("clinical findings", limitations)

    def test_summary_text_does_not_repeat_old_row_level_caveat(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            annotate_moment_rows,
            build_rest_baselines,
            render_summary,
        )

        rows = [
            self._moment_row(
                interpretation="This is exploratory signal metadata, not a diagnosis."
            )
        ]
        annotated = annotate_moment_rows(rows, build_rest_baselines(rows))

        summary = render_summary(
            annotated,
            {
                "subjects": [1],
                "runs": [3],
                "max_windows": None,
                "baseline_row_count": 1,
                "output_was_capped": False,
            },
        )

        self.assertNotIn("This is exploratory signal metadata", summary)

    def test_moment_report_columns_are_declared(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import (
            MOMENT_REPORT_COLUMNS,
            OUTPUT_COLUMNS,
        )

        for column in [
            "patient_id",
            "task_label",
            "alpha_relative",
            "analysis_version",
            "state_hypothesis",
            "state_confidence",
            "evidence_score",
            "evidence_summary",
            "rest_reference_scope",
            "rest_alpha_relative_delta",
            "task_state_relation",
            "task_state_rationale",
            "task_state_confidence",
            "interpretation",
            "is_low_confidence",
            "is_possible_artifact",
            "is_mixed_or_ambiguous",
        ]:
            self.assertIn(column, OUTPUT_COLUMNS)
        self.assertIn("analysis_version", MOMENT_REPORT_COLUMNS)

    def test_output_columns_remove_legacy_task_fields(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import OUTPUT_COLUMNS

        for legacy_column in [
            "brain_state_hypothesis",
            "confidence",
            "quality_flags",
            "legacy_brain_state_hypothesis",
            "legacy_confidence",
            "legacy_quality_flags",
            "legacy_interpretation",
        ]:
            self.assertNotIn(legacy_column, OUTPUT_COLUMNS)

    def test_empty_dataframe_uses_output_columns(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import OUTPUT_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            pd.DataFrame([], columns=OUTPUT_COLUMNS).to_csv(path, index=False)

            df = pd.read_csv(path)

        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), list(OUTPUT_COLUMNS))

    def test_main_max_windows_zero_writes_empty_artifacts(self):
        from examples.eeg.eegbci import eegbci_pattern_discovery as example

        class FakeDataset:
            def __init__(self, *args, **kwargs):
                pass

            def set_task(self, task):
                return [self_sample]

        self_sample = self._sample()
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "eegbci_pattern_discovery.py",
                "--subjects",
                "1",
                "--runs",
                "3",
                "--max-windows",
                "0",
                "--output-dir",
                tmp,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                example, "EEGBCIDataset", FakeDataset
            ):
                example.main()

            csv_path = Path(tmp) / "eegbci_pattern_windows.csv"
            summary_path = Path(tmp) / "eegbci_pattern_summary.md"
            df = pd.read_csv(csv_path)
            summary = summary_path.read_text(encoding="utf-8")

        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), list(example.OUTPUT_COLUMNS))
        self.assertIn("No windows were produced", summary)
        self.assertIn("Output was capped by `--max-windows`", summary)

    def test_main_baseline_uses_uncapped_rows(self):
        from examples.eeg.eegbci import eegbci_pattern_discovery as example

        first = self._sample(
            task_label="execute_left_fist",
            label_family="motor_execution",
            alpha_beta_ratio=0.5,
            bandpower={
                **self._sample()["bandpower"],
                "dominant_band": "beta",
                "alpha_relative": 0.20,
                "beta_relative": 0.45,
                "alpha_beta_ratio": 0.5,
            },
        )
        rest = self._sample(
            task_label="rest",
            start_time=2.0,
            bandpower={
                **self._sample()["bandpower"],
                "alpha_relative": 0.50,
                "beta_relative": 0.20,
            },
        )

        class FakeDataset:
            def __init__(self, *args, **kwargs):
                pass

            def set_task(self, task):
                return [first, rest]

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "eegbci_pattern_discovery.py",
                "--subjects",
                "1",
                "--runs",
                "3",
                "--max-windows",
                "1",
                "--output-dir",
                tmp,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                example, "EEGBCIDataset", FakeDataset
            ):
                example.main()

            df = pd.read_csv(Path(tmp) / "eegbci_pattern_windows.csv")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "rest_reference_scope"], "same_subject_run")
        self.assertAlmostEqual(df.loc[0, "rest_alpha_relative_delta"], -0.30)

    def test_main_writes_analysis_version_to_every_csv_row(self):
        from examples.eeg.eegbci import eegbci_pattern_discovery as example

        samples = [self._sample(), self._sample(start_time=2.0, trial_id="second")]

        class FakeDataset:
            def __init__(self, *args, **kwargs):
                pass

            def set_task(self, task):
                return samples

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "eegbci_pattern_discovery.py",
                "--subjects",
                "1",
                "--runs",
                "3",
                "--output-dir",
                tmp,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                example, "EEGBCIDataset", FakeDataset
            ):
                example.main()

            df = pd.read_csv(Path(tmp) / "eegbci_pattern_windows.csv")

        self.assertEqual(len(df), 2)
        self.assertTrue((df["analysis_version"] == example.ANALYSIS_VERSION).all())

    def test_parse_int_list_rejects_invalid_input_loudly(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import parse_int_list

        with self.assertRaises(ValueError):
            parse_int_list("a")
        with self.assertRaises(ValueError):
            parse_int_list("3-a")

    def test_parse_int_list_accepts_ranges_and_singletons(self):
        from examples.eeg.eegbci.eegbci_pattern_discovery import parse_int_list

        self.assertEqual(parse_int_list("1,3-5"), [1, 3, 4, 5])


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
