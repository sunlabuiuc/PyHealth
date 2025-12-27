import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from unittest.mock import patch


from pyhealth.datasets.tuev import TUEVDataset
from pyhealth.tasks.temple_university_EEG_tasks import EEGEventsTUEV


@dataclass
class _DummyEvent:
	signal_file: str


class _DummyPatient:
	def __init__(self, patient_id: str, events: List[_DummyEvent]):
		self.patient_id = patient_id
		self._events = events

	def get_events(self) -> List[_DummyEvent]:
		return self._events


class TestTUEVDataset(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        
    def test_prepare_metadata_creates_expected_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Minimal filesystem layout the metadata builder expects
            train_subject = root / "train" / "subject-train"
            eval_subject = root / "eval" / "subject-eval"

            train_edf = train_subject / "00002275_00000001.edf"
            eval_edf = eval_subject / "bckg_032_a_.edf"
            self._touch(train_edf)
            self._touch(eval_edf)

            # Call prepare_metadata without invoking BaseDataset init
            ds = TUEVDataset.__new__(TUEVDataset)
            ds.root = str(root)
            ds.prepare_metadata()

            train_csv = root / "tuev-train-pyhealth.csv"
            eval_csv = root / "tuev-eval-pyhealth.csv"

            self.assertTrue(train_csv.exists())
            self.assertTrue(eval_csv.exists())

            train_df = pd.read_csv(train_csv)
            eval_df = pd.read_csv(eval_csv)

            self.assertEqual(len(train_df), 1)
            self.assertEqual(len(eval_df), 1)

            self.assertIn("patient_id", train_df.columns)
            self.assertIn("record_id", train_df.columns)
            self.assertIn("signal_file", train_df.columns)

            self.assertEqual(train_df.loc[0, "patient_id"], "subject-train")
            self.assertEqual(train_df.loc[0, "record_id"], 1)
            self.assertTrue(str(train_df.loc[0, "signal_file"]).endswith("00002275_00000001.edf"))

            self.assertIn("patient_id", eval_df.columns)
            self.assertIn("label", eval_df.columns)
            self.assertIn("segment_id", eval_df.columns)
            self.assertIn("signal_file", eval_df.columns)

            self.assertEqual(eval_df.loc[0, "patient_id"], "subject-eval")
            self.assertEqual(eval_df.loc[0, "label"], "bckg")
            self.assertEqual(eval_df.loc[0, "segment_id"], "a_")
            self.assertTrue(str(eval_df.loc[0, "signal_file"]).endswith("bckg_032_a_.edf"))

            # Idempotency: should not crash when CSVs already exist
            ds.prepare_metadata()
            
    def test_invalid_subset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                TUEVDataset(root=tmp, subset="nope")

    def test_default_task_returns_task_instance(self):
        ds = TUEVDataset.__new__(TUEVDataset)
        task = ds.default_task
        self.assertIsInstance(task, EEGEventsTUEV)


class TestEEGEventsTUEV(unittest.TestCase):
    def test_convert_signals_output_shape_and_values(self):
        class _Raw:
            def __init__(self, ch_names):
                self.info = {"ch_names": ch_names}

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
        raw = _Raw(ch_names)

        n = 10
        signals = np.arange(len(ch_names) * n, dtype=float).reshape(len(ch_names), n)

        out = EEGEventsTUEV.convert_signals(signals, raw)
        self.assertEqual(out.shape, (16, n))

        fp1 = ch_names.index("EEG FP1-REF")
        f7 = ch_names.index("EEG F7-REF")
        expected0 = signals[fp1] - signals[f7]
        np.testing.assert_allclose(out[0], expected0)

    def test_BuildEvents_single_row_eventdata_and_window_length(self):
        fs = 256
        num_chan = 16
        num_points = 2000
        signals = np.random.randn(num_chan, num_points)
        times = np.arange(num_points) / fs

        # Single-row .rec style data: [chan, start_time, end_time, label]
        # Choose start/end so that (end-start)=1s => 5s window (2s pre + 1s event + 2s post)
        event = np.array([4, 5.0, 6.0, 2])

        feats, offending, labels = EEGEventsTUEV.BuildEvents(signals, times, event)
        self.assertEqual(feats.shape, (1, num_chan, fs * 5))
        self.assertEqual(offending.shape, (1, 1))
        self.assertEqual(labels.shape, (1, 1))
        self.assertEqual(int(offending.squeeze()), 4)
        self.assertEqual(int(labels.squeeze()), 2)

    def test_call_returns_one_sample_per_event_and_adjusts_label(self):
        task = EEGEventsTUEV()

        dummy_patient = _DummyPatient(
            patient_id="patient-0",
            events=[_DummyEvent(signal_file=os.path.join("C:\\", "dummy.edf"))],
        )

        feats = np.zeros((2, 16, 256 * 5), dtype=float)
        offending = np.array([[3], [7]])
        labels = np.array([[1], [6]])  # will become 0 and 5 in output

        with patch.object(EEGEventsTUEV, "readEDF", return_value=(None, None, None, None)):
            with patch.object(EEGEventsTUEV, "convert_signals", return_value=None):
                with patch.object(
                    EEGEventsTUEV,
                    "BuildEvents",
                    return_value=(feats, offending, labels),
                ):
                    samples = task(dummy_patient)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["patient_id"], "patient-0")
        self.assertIn("signal", samples[0])
        self.assertEqual(samples[0]["signal"].shape, (16, 256 * 5))
        self.assertEqual(samples[0]["offending_channel"], 3)
        self.assertEqual(samples[0]["label"], 0)
        self.assertEqual(samples[1]["offending_channel"], 7)
        self.assertEqual(samples[1]["label"], 5)


if __name__ == "__main__":
    unittest.main()
