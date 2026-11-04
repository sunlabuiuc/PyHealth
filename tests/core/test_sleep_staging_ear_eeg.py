"""SleepStagingEarEEG on synthetic arrays. EEGLAB reading is mocked."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
