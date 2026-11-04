"""SleepStagingEarEEG on synthetic arrays. EEGLAB reading is mocked."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import torch

from pyhealth.datasets import EESM23Dataset, SubjectSplit
from pyhealth.tasks.sleep_staging_ear_eeg import (
    EAR_EEG_CHANNELS,
    SleepStagingEarEEG,
    normalize_subject,
)

SFREQ = 250.0


@dataclass
class _DummyEvent:
    signal_file: str
    scoring_file: str
    session_id: str
    has_scoring: str = "1"


class _DummyPatient:
    def __init__(self, patient_id: str, events: list[_DummyEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> list[_DummyEvent]:
        if event_type not in (None, "recordings"):
            return []
        return self._events


def _signal(seconds: float, value: float = 1.0, channels: int = 4) -> np.ndarray:
    n = int(SFREQ * seconds)
    return np.full((channels, n), value, dtype=float)


class TestSleepStagingEarEEG(unittest.TestCase):
    def _write_scoring(self, path: Path, rows: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "onset\tduration\tscoring_idx\tscoring\n" + rows
        path.write_text(text)
        return str(path)

    def _run(
        self,
        signal: np.ndarray,
        scoring_rows: str,
        task: SleepStagingEarEEG | None = None,
        patient_id: str = "sub-001",
        ch_names: list[str] | None = None,
    ):
        task = task or SleepStagingEarEEG(
            compute_stft=False,
            bandpass_filter=None,
            notch_filter=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            scoring_path = Path(tmp) / "scoring.tsv"
            self._write_scoring(scoring_path, scoring_rows)
            patient = _DummyPatient(
                patient_id,
                [
                    _DummyEvent(
                        signal_file="/tmp/night.set",
                        scoring_file=str(scoring_path),
                        session_id="ses-001",
                    )
                ],
            )
            names = ch_names or list(EAR_EEG_CHANNELS)
            loaded = (signal, SFREQ, names)
            with patch.object(
                SleepStagingEarEEG, "read_recording", return_value=loaded
            ):
                return task(patient)

    def test_labels_follow_scoring_idx_minus_one(self):
        rows = "0\t30\t1\tWake\n30\t30\t2\tREM\n"
        samples = self._run(_signal(70), rows)
        self.assertEqual([s["label"] for s in samples], [0, 1])
        self.assertEqual(samples[0]["signal"].shape, (4, 6000))

    def test_nan_epochs_are_skipped(self):
        signal = _signal(70)
        signal[:, int(30 * SFREQ) : int(30 * SFREQ) + 10] = np.nan
        rows = "0\t30\t1\tWake\n30\t30\t2\tREM\n"
        samples = self._run(signal, rows)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 0)

    def test_missing_ear_channel_skips_recording(self):
        task = SleepStagingEarEEG(
            compute_stft=False, bandpass_filter=None, notch_filter=None
        )
        patient = _DummyPatient(
            "sub-001",
            [
                _DummyEvent(
                    "/tmp/night.set",
                    "/tmp/scoring.tsv",
                    "ses-001",
                )
            ],
        )
        with patch.object(SleepStagingEarEEG, "read_recording", return_value=None):
            self.assertEqual(task(patient), [])

    def test_include_artefact_false_drops_artefact_epochs(self):
        rows = "0\t30\t1\tWake\n30\t30\t6\tArtefact\n60\t30\t4\tN2\n"
        task = SleepStagingEarEEG(
            compute_stft=False,
            bandpass_filter=None,
            notch_filter=None,
            include_artefact=False,
        )
        samples = self._run(_signal(100), rows, task=task)
        self.assertEqual([s["label"] for s in samples], [0, 3])

    def test_subjects_are_stored_as_a_sorted_tuple(self):
        task = SleepStagingEarEEG(subjects=["sub-3", "1", "SUB2", "sub-001"])
        self.assertEqual(task.subjects, ("sub-001", "sub-002", "sub-003"))

    def test_normalize_subject(self):
        self.assertEqual(normalize_subject("10"), "sub-010")
        self.assertEqual(normalize_subject("SUB-2"), "sub-002")

    def test_normalize_subject_shares_the_generic_helper(self):
        self.assertEqual(normalize_subject("sub7"), "sub-007")
        self.assertEqual(normalize_subject("Custom"), "custom")

    def test_compute_stft_adds_stft_field(self):
        rows = "0\t30\t1\tWake\n"
        task = SleepStagingEarEEG(
            bandpass_filter=None,
            notch_filter=None,
            compute_stft=True,
            resample_rate=200,
        )
        samples = self._run(_signal(35), rows, task=task)
        self.assertIn("stft", samples[0])
        self.assertIsInstance(samples[0]["stft"], torch.Tensor)

    def test_unscored_event_yields_no_samples(self):
        task = SleepStagingEarEEG(compute_stft=False)
        patient = _DummyPatient(
            "sub-001",
            [
                _DummyEvent(
                    "/tmp/night.set",
                    "",
                    "ses-003",
                    has_scoring="0",
                )
            ],
        )
        self.assertEqual(task(patient), [])

    def test_set_task_subject_split_returns_three_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("sub-001", "sub-002", "sub-003"):
                eeg_dir = root / sub / "ses-001" / "eeg"
                eeg_dir.mkdir(parents=True)
                set_path = eeg_dir / f"{sub}_ses-001_task-sleep_acq-earEEG_eeg.set"
                set_path.write_bytes(b"")
                scoring = eeg_dir / f"{sub}_ses-001_task-sleep_acq-scoring_events.tsv"
                scoring.write_text(
                    "onset\tduration\tscoring_idx\tscoring\n0\t30\t1\tWake\n"
                )

            def fake_read(_self, file_name: str):
                return _signal(35), SFREQ, list(EAR_EEG_CHANNELS)

            task = SleepStagingEarEEG(
                compute_stft=False,
                bandpass_filter=None,
                notch_filter=None,
                subjects=["sub-001", "sub-002", "sub-003"],
            )
            split = SubjectSplit(
                (("sub-001",), ("sub-002",), ("sub-003",)),
            )
            with patch.object(SleepStagingEarEEG, "read_recording", fake_read):
                dataset = EESM23Dataset(root=str(root), cache_dir=str(root / "cache"))
                train, val, test = dataset.set_task(task, split=split)
            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(test), 1)
            self.assertEqual(train[0]["patient_id"], "sub-001")


def _raw_array(ch_names: list[str], seconds: float = 2.0) -> mne.io.RawArray:
    """Build an in-memory recording; channel ``i`` is a constant ``(i+1)`` uV."""
    n = int(SFREQ * seconds)
    data = np.stack([np.full(n, (i + 1) * 1e-6) for i in range(len(ch_names))])
    info = mne.create_info(ch_names, SFREQ, ch_types="eeg", verbose="error")
    return mne.io.RawArray(data, info, verbose="error")


class TestReadRecording(unittest.TestCase):
    """``read_recording`` against a real ``mne.io.RawArray``."""

    def test_selects_ear_channels_in_order_in_microvolts(self):
        # File order differs from the task order, and ELE is present.
        raw = _raw_array(["ELE", "LT", "RB", "LB", "RT"])
        with patch("mne.io.read_raw_eeglab", return_value=raw) as reader:
            loaded = SleepStagingEarEEG().read_recording("night.set")
        reader.assert_called_once()
        data, sfreq, ch_names = loaded
        self.assertEqual(ch_names, list(EAR_EEG_CHANNELS))
        self.assertEqual(sfreq, SFREQ)
        self.assertEqual(data.shape, (4, int(SFREQ * 2)))
        # RB is the 3rd file channel (3 uV), RT the 5th, LB the 4th, LT the 2nd.
        np.testing.assert_allclose(data[:, 0], [3.0, 5.0, 4.0, 2.0])

    def test_missing_ear_channel_returns_none(self):
        raw = _raw_array(["RB", "RT", "LB", "ELE"])
        with patch("mne.io.read_raw_eeglab", return_value=raw):
            self.assertIsNone(SleepStagingEarEEG().read_recording("night.set"))

    def test_read_failure_returns_none(self):
        for exc in (OSError("corrupt"), ValueError("bad header"), RuntimeError("x")):
            with patch("mne.io.read_raw_eeglab", side_effect=exc):
                self.assertIsNone(SleepStagingEarEEG().read_recording("night.set"))


class TestSkippedInputs(unittest.TestCase):
    """Guards that return no samples instead of raising."""

    def _call(self, task, scoring_text=None, scoring_file=None, seconds=40):
        with tempfile.TemporaryDirectory() as tmp:
            if scoring_file is None:
                path = Path(tmp) / "scoring.tsv"
                path.write_text(scoring_text)
                scoring_file = str(path)
            patient = _DummyPatient(
                "sub-001", [_DummyEvent("night.set", scoring_file, "ses-001")]
            )
            loaded = (_signal(seconds), SFREQ, list(EAR_EEG_CHANNELS))
            with patch.object(
                SleepStagingEarEEG, "read_recording", return_value=loaded
            ):
                return task(patient)

    def _task(self, **kwargs):
        return SleepStagingEarEEG(
            compute_stft=False, bandpass_filter=None, notch_filter=None, **kwargs
        )

    def test_subject_outside_filter_is_skipped(self):
        rows = "onset\tduration\tscoring_idx\tscoring\n0\t30\t1\tWake\n"
        task = self._task(subjects=["sub-002"])
        self.assertEqual(self._call(task, rows), [])

    def test_unreadable_scoring_file_is_skipped(self):
        self.assertEqual(
            self._call(self._task(), scoring_file="/nonexistent/scoring.tsv"), []
        )

    def test_epoch_past_end_of_recording_is_skipped(self):
        rows = "onset\tduration\tscoring_idx\tscoring\n0\t30\t1\tWake\n30\t30\t2\tREM\n"
        samples = self._call(self._task(), rows, seconds=45)
        self.assertEqual([s["label"] for s in samples], [0])

    def test_empty_scoring_path_is_skipped(self):
        patient = _DummyPatient("sub-001", [_DummyEvent("night.set", "", "ses-001")])
        self.assertEqual(self._task()(patient), [])


class TestDefaultFiltering(unittest.TestCase):
    """The default per-epoch filters run on real arrays."""

    def test_filters_remove_dc_and_notch_line_noise(self):
        t = np.arange(int(30 * SFREQ)) / SFREQ
        noise = 100.0 * np.sin(2 * np.pi * 50.0 * t)
        epoch = np.tile(50.0 + noise, (4, 1))
        task = SleepStagingEarEEG(compute_stft=False, normalization=None)
        out = task._filter_epoch(epoch, SFREQ)
        self.assertEqual(out.shape, epoch.shape)
        # 0.1 Hz high-pass drops the offset; the notch drops the 50 Hz tone.
        # Skip the edges, where the filters have transients.
        inner = out[:, 1000:-1000]
        self.assertLess(abs(inner.mean()), 5.0)
        self.assertLess(np.abs(inner).max(), 50.0)  # input tone amplitude is 100

    def test_epochs_flow_through_default_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            scoring = Path(tmp) / "scoring.tsv"
            scoring.write_text("onset\tduration\tscoring_idx\tscoring\n0\t30\t3\tN1\n")
            event = _DummyEvent("night.set", str(scoring), "ses-001")
            rng = np.random.default_rng(0)
            signal = rng.standard_normal((4, int(40 * SFREQ)))
            task = SleepStagingEarEEG(compute_stft=False)
            with patch.object(
                SleepStagingEarEEG,
                "read_recording",
                return_value=(signal, SFREQ, list(EAR_EEG_CHANNELS)),
            ):
                samples = task._samples_from_event("sub-001", event)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 2)
        self.assertEqual(samples[0]["signal"].shape, (4, 6000))


if __name__ == "__main__":
    unittest.main()
