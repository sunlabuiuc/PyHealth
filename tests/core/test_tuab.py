import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from unittest.mock import patch


from pyhealth.datasets.tuab import TUABDataset
from pyhealth.tasks.temple_university_EEG_tasks import EEGAbnormalTUAB


@dataclass
class _DummyEvent:
    signal_file: str
    label: str


class _DummyPatient:
    def __init__(self, patient_id: str, events: List[_DummyEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self) -> List[_DummyEvent]:
        return self._events


class TestTUABDataset(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def test_prepare_metadata_creates_expected_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Minimal filesystem layout: train has normal + abnormal, eval too
            train_normal_dir = root / "train" / "normal" / "01_tcp_ar"
            train_abnormal_dir = root / "train" / "abnormal" / "01_tcp_ar"
            eval_normal_dir = root / "eval" / "normal" / "01_tcp_ar"
            eval_abnormal_dir = root / "eval" / "abnormal" / "01_tcp_ar"

            self._touch(train_normal_dir / "aaaaaaav_s004_t000.edf")
            self._touch(train_abnormal_dir / "aaaaaaaq_s004_t000.edf")
            self._touch(eval_normal_dir / "aaaaaqax_s001_t000.edf")
            self._touch(eval_abnormal_dir / "aaaaapys_s002_t001.edf")

            # Call prepare_metadata without invoking BaseDataset init
            ds = TUABDataset.__new__(TUABDataset)
            ds.root = str(root)
            ds.prepare_metadata()

            train_csv = root / "tuab-train-pyhealth.csv"
            eval_csv = root / "tuab-eval-pyhealth.csv"

            self.assertTrue(train_csv.exists())
            self.assertTrue(eval_csv.exists())

            train_df = pd.read_csv(train_csv)
            eval_df = pd.read_csv(eval_csv)

            # Two train files (one normal, one abnormal)
            self.assertEqual(len(train_df), 2)
            # Two eval files (one normal, one abnormal)
            self.assertEqual(len(eval_df), 2)

            # Check train columns
            self.assertIn("patient_id", train_df.columns)
            self.assertIn("record_id", train_df.columns)
            self.assertIn("signal_file", train_df.columns)
            self.assertIn("label", train_df.columns)

            # Check eval columns (same schema)
            self.assertIn("patient_id", eval_df.columns)
            self.assertIn("record_id", eval_df.columns)
            self.assertIn("signal_file", eval_df.columns)
            self.assertIn("label", eval_df.columns)

            # Verify a normal train entry
            normal_row = train_df[train_df["patient_id"] == "aaaaaaav"]
            self.assertEqual(len(normal_row), 1)
            self.assertEqual(normal_row.iloc[0]["record_id"], "s004_t000")
            self.assertEqual(normal_row.iloc[0]["label"], "normal")
            self.assertTrue(
                str(normal_row.iloc[0]["signal_file"]).endswith("aaaaaaav_s004_t000.edf")
            )

            # Verify an abnormal train entry
            abnormal_row = train_df[train_df["patient_id"] == "aaaaaaaq"]
            self.assertEqual(len(abnormal_row), 1)
            self.assertEqual(abnormal_row.iloc[0]["record_id"], "s004_t000")
            self.assertEqual(abnormal_row.iloc[0]["label"], "abnormal")

            # Verify eval entries
            eval_normal = eval_df[eval_df["patient_id"] == "aaaaaqax"]
            self.assertEqual(len(eval_normal), 1)
            self.assertEqual(eval_normal.iloc[0]["label"], "normal")

            eval_abnormal = eval_df[eval_df["patient_id"] == "aaaaapys"]
            self.assertEqual(len(eval_abnormal), 1)
            self.assertEqual(eval_abnormal.iloc[0]["label"], "abnormal")
            self.assertEqual(eval_abnormal.iloc[0]["record_id"], "s002_t001")

            # Idempotency: should not crash when CSVs already exist
            ds.prepare_metadata()

    def test_prepare_metadata_multiple_sessions_same_patient(self):
        """Multiple EDF files for the same subject should produce multiple rows."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_normal_dir = root / "train" / "normal" / "01_tcp_ar"

            self._touch(train_normal_dir / "aaaaaaav_s001_t000.edf")
            self._touch(train_normal_dir / "aaaaaaav_s001_t001.edf")
            self._touch(train_normal_dir / "aaaaaaav_s002_t000.edf")

            ds = TUABDataset.__new__(TUABDataset)
            ds.root = str(root)
            ds.prepare_metadata()

            train_csv = root / "tuab-train-pyhealth.csv"
            self.assertTrue(train_csv.exists())

            train_df = pd.read_csv(train_csv)
            self.assertEqual(len(train_df), 3)
            # All rows should belong to the same patient
            self.assertTrue((train_df["patient_id"] == "aaaaaaav").all())
            # record_ids should be sorted
            record_ids = train_df["record_id"].tolist()
            self.assertEqual(record_ids, sorted(record_ids))

    def test_prepare_metadata_skips_invalid_filenames(self):
        """Filenames that don't match the expected 3-part pattern are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_normal_dir = root / "train" / "normal" / "01_tcp_ar"

            # Valid
            self._touch(train_normal_dir / "aaaaaaav_s001_t000.edf")
            # Invalid (only 2 parts)
            self._touch(train_normal_dir / "badfile_s001.edf")

            ds = TUABDataset.__new__(TUABDataset)
            ds.root = str(root)
            ds.prepare_metadata()

            train_csv = root / "tuab-train-pyhealth.csv"
            train_df = pd.read_csv(train_csv)
            self.assertEqual(len(train_df), 1)
            self.assertEqual(train_df.iloc[0]["patient_id"], "aaaaaaav")

    def test_prepare_metadata_missing_split_dir_no_crash(self):
        """If a split directory is missing, no CSV is written and no crash occurs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Only create train, not eval
            train_normal_dir = root / "train" / "normal" / "01_tcp_ar"
            self._touch(train_normal_dir / "aaaaaaav_s001_t000.edf")

            ds = TUABDataset.__new__(TUABDataset)
            ds.root = str(root)
            ds.prepare_metadata()

            train_csv = root / "tuab-train-pyhealth.csv"
            eval_csv = root / "tuab-eval-pyhealth.csv"

            self.assertTrue(train_csv.exists())
            # eval csv should not exist since there's no eval directory
            self.assertFalse(eval_csv.exists())

    def test_invalid_subset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                TUABDataset(root=tmp, subset="nope")

    def test_default_task_returns_task_instance(self):
        ds = TUABDataset.__new__(TUABDataset)
        task = ds.default_task
        self.assertIsInstance(task, EEGAbnormalTUAB)


class TestEEGAbnormalTUAB(unittest.TestCase):
    def test_convert_to_bipolar_output_shape_and_values(self):
        ch_names = [
            "EEG FP1-REF",
            "EEG F7-REF",
            "EEG T3-REF",
            "EEG T5-REF",
            "EEG O1-REF",
            "EEG FP2-REF",
            "EEG F8-REF",
            "EEG T4-REF",
            "EEG T6-REF",
            "EEG O2-REF",
            "EEG F3-REF",
            "EEG C3-REF",
            "EEG P3-REF",
            "EEG F4-REF",
            "EEG C4-REF",
            "EEG P4-REF",
        ]

        n = 2000
        raw_data = np.arange(len(ch_names) * n, dtype=float).reshape(len(ch_names), n)

        out = EEGAbnormalTUAB.convert_to_bipolar(raw_data, ch_names)
        self.assertEqual(out.shape, (16, n))

        # Verify first channel: FP1 - F7
        fp1 = ch_names.index("EEG FP1-REF")
        f7 = ch_names.index("EEG F7-REF")
        expected0 = raw_data[fp1] - raw_data[f7]
        np.testing.assert_allclose(out[0], expected0)

        # Verify last channel: P4 - O2
        p4 = ch_names.index("EEG P4-REF")
        o2 = ch_names.index("EEG O2-REF")
        expected15 = raw_data[p4] - raw_data[o2]
        np.testing.assert_allclose(out[15], expected15)

    def test_convert_to_bipolar_middle_channels(self):
        """Spot-check a middle channel (channel 9: F3 - C3)."""
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]

        n = 100
        raw_data = np.random.randn(len(ch_names), n)
        out = EEGAbnormalTUAB.convert_to_bipolar(raw_data, ch_names)

        f3 = ch_names.index("EEG F3-REF")
        c3 = ch_names.index("EEG C3-REF")
        np.testing.assert_allclose(out[9], raw_data[f3] - raw_data[c3])

    def test_call_segments_into_10s_windows(self):
        """__call__ should produce the right number of 10-second segments."""
        task = EEGAbnormalTUAB()

        fs = 200
        # 25 seconds of data => 2 full 10-second windows (last 5s discarded)
        n_samples = fs * 25
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]
        raw_data = np.random.randn(len(ch_names), n_samples)

        dummy_patient = _DummyPatient(
            patient_id="patient-0",
            events=[_DummyEvent(signal_file="/dummy/path.edf", label="abnormal")],
        )

        with patch.object(
            EEGAbnormalTUAB,
            "read_and_process_edf",
            return_value=(raw_data, ch_names),
        ):
            samples = task(dummy_patient)

        self.assertEqual(len(samples), 2)  # 25s // 10s = 2 windows
        for s in samples:
            self.assertEqual(s["signal"].shape, (16, 2000))
            self.assertEqual(s["label"], 1)  # abnormal
            self.assertEqual(s["patient_id"], "patient-0")
            self.assertIn("segment_id", s)
            self.assertIn("start_time", s)
            self.assertIn("end_time", s)

    def test_call_label_mapping_normal(self):
        """Normal label should map to 0."""
        task = EEGAbnormalTUAB()

        fs = 200
        n_samples = fs * 10  # exactly one window
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]
        raw_data = np.random.randn(len(ch_names), n_samples)

        dummy_patient = _DummyPatient(
            patient_id="patient-1",
            events=[_DummyEvent(signal_file="/dummy/path.edf", label="normal")],
        )

        with patch.object(
            EEGAbnormalTUAB,
            "read_and_process_edf",
            return_value=(raw_data, ch_names),
        ):
            samples = task(dummy_patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 0)

    def test_call_label_mapping_abnormal(self):
        """Abnormal label should map to 1."""
        task = EEGAbnormalTUAB()

        fs = 200
        n_samples = fs * 10
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]
        raw_data = np.random.randn(len(ch_names), n_samples)

        dummy_patient = _DummyPatient(
            patient_id="patient-2",
            events=[_DummyEvent(signal_file="/dummy/path.edf", label="abnormal")],
        )

        with patch.object(
            EEGAbnormalTUAB,
            "read_and_process_edf",
            return_value=(raw_data, ch_names),
        ):
            samples = task(dummy_patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 1)

    def test_call_too_short_recording_returns_no_samples(self):
        """A recording shorter than 10 seconds should produce zero samples."""
        task = EEGAbnormalTUAB()

        fs = 200
        n_samples = fs * 5  # only 5 seconds
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]
        raw_data = np.random.randn(len(ch_names), n_samples)

        dummy_patient = _DummyPatient(
            patient_id="patient-3",
            events=[_DummyEvent(signal_file="/dummy/path.edf", label="normal")],
        )

        with patch.object(
            EEGAbnormalTUAB,
            "read_and_process_edf",
            return_value=(raw_data, ch_names),
        ):
            samples = task(dummy_patient)

        self.assertEqual(len(samples), 0)

    def test_call_multiple_events_per_patient(self):
        """Multiple EDF files per patient should all be processed."""
        task = EEGAbnormalTUAB()

        fs = 200
        n_samples = fs * 20  # 2 windows each
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]
        raw_data = np.random.randn(len(ch_names), n_samples)

        dummy_patient = _DummyPatient(
            patient_id="patient-4",
            events=[
                _DummyEvent(signal_file="/dummy/file1.edf", label="normal"),
                _DummyEvent(signal_file="/dummy/file2.edf", label="abnormal"),
            ],
        )

        with patch.object(
            EEGAbnormalTUAB,
            "read_and_process_edf",
            return_value=(raw_data, ch_names),
        ):
            samples = task(dummy_patient)

        # 2 events × 2 windows each = 4 samples
        self.assertEqual(len(samples), 4)
        # First two from normal file
        self.assertEqual(samples[0]["label"], 0)
        self.assertEqual(samples[1]["label"], 0)
        # Last two from abnormal file
        self.assertEqual(samples[2]["label"], 1)
        self.assertEqual(samples[3]["label"], 1)

    def test_call_segment_ids_are_sequential(self):
        """segment_id should increment per window within each recording."""
        task = EEGAbnormalTUAB()

        fs = 200
        n_samples = fs * 30  # 3 windows
        ch_names = [
            "EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
            "EEG O1-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF",
            "EEG T6-REF", "EEG O2-REF", "EEG F3-REF", "EEG C3-REF",
            "EEG P3-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
        ]
        raw_data = np.random.randn(len(ch_names), n_samples)

        dummy_patient = _DummyPatient(
            patient_id="patient-5",
            events=[_DummyEvent(signal_file="/dummy/path.edf", label="normal")],
        )

        with patch.object(
            EEGAbnormalTUAB,
            "read_and_process_edf",
            return_value=(raw_data, ch_names),
        ):
            samples = task(dummy_patient)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["segment_id"], "0")
        self.assertEqual(samples[1]["segment_id"], "1")
        self.assertEqual(samples[2]["segment_id"], "2")

        # Verify start/end times
        self.assertEqual(samples[0]["start_time"], 0)
        self.assertEqual(samples[0]["end_time"], 2000)
        self.assertEqual(samples[1]["start_time"], 2000)
        self.assertEqual(samples[1]["end_time"], 4000)
        self.assertEqual(samples[2]["start_time"], 4000)
        self.assertEqual(samples[2]["end_time"], 6000)

    def test_task_schema_attributes(self):
        task = EEGAbnormalTUAB()
        self.assertEqual(task.task_name, "EEG_abnormal")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "binary"})


if __name__ == "__main__":
    unittest.main()