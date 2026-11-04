"""SeizureDetectionCHBMIT on synthetic arrays. EDF reading is mocked."""

import unittest
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import torch

from pyhealth.tasks.seizure_detection_chbmit import (
    BIPOLAR_CHANNELS,
    SeizureDetectionCHBMIT,
    select_bipolar_channels,
)

SFREQ = 256.0
_USE_SIGNAL = object()


@dataclass
class _DummyEvent:
    signal_file: str
    record_id: str
    seizure_starts: str
    seizure_ends: str


class _DummyPatient:
    def __init__(self, patient_id: str, events: List[_DummyEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_DummyEvent]:
        if event_type not in (None, "recordings"):
            return []
        return self._events


def _signal(seconds: float, value: float = 1.0) -> np.ndarray:
    n = int(SFREQ * seconds)
    return np.full((len(BIPOLAR_CHANNELS), n), value, dtype=float)


class TestSelectBipolarChannels(unittest.TestCase):
    def test_mne_duplicate_suffix_and_zero_fill(self):
        ch_names = ["FP1-F7", "T8-P8-0", "T8-P8-1", "FZ-CZ"]
        raw = np.arange(len(ch_names) * 4, dtype=float).reshape(len(ch_names), 4)
        out = select_bipolar_channels(raw, ch_names)
        self.assertEqual(out.shape, (16, 4))
        np.testing.assert_allclose(out[0], raw[0])  # FP1-F7
        np.testing.assert_allclose(out[6], raw[1])  # T8-P8, the -0 copy
        np.testing.assert_allclose(out[1], 0.0)  # F7-T7 was missing

    def test_no_bipolar_channel_returns_none(self):
        raw = np.ones((2, 8))
        self.assertIsNone(select_bipolar_channels(raw, ["FZ-CZ", "CZ-PZ"]))


class TestSeizureDetectionCHBMIT(unittest.TestCase):
    def _run(
        self,
        signal: np.ndarray,
        starts: str,
        ends: str,
        task: Optional[SeizureDetectionCHBMIT] = None,
        patient_id: str = "chb01",
        read_result=_USE_SIGNAL,
    ):
        task = task or SeizureDetectionCHBMIT(
            compute_stft=False, oversample_seizures=False
        )
        patient = _DummyPatient(
            patient_id,
            [_DummyEvent("/tmp/chb01_03.edf", "chb01_03", starts, ends)],
        )
        loaded = (signal, SFREQ) if read_result is _USE_SIGNAL else read_result
        with patch.object(SeizureDetectionCHBMIT, "read_edf", return_value=loaded):
            return task(patient)

    def test_overlap_and_cbramod_labels(self):
        # 25 s -> two 10 s windows. Seizure at 12-15 s hits only the second.
        signal = _signal(25)
        overlap = self._run(
            signal,
            "12",
            "15",
            SeizureDetectionCHBMIT(
                compute_stft=False, oversample_seizures=False, label_mode="overlap"
            ),
        )
        cbramod = self._run(
            signal,
            "12",
            "15",
            SeizureDetectionCHBMIT(
                compute_stft=False, oversample_seizures=False, label_mode="cbramod"
            ),
        )
        self.assertEqual([s["label"] for s in overlap], [0, 1])
        self.assertEqual([s["label"] for s in cbramod], [0, 1])
        self.assertEqual(overlap[0]["signal"].shape, (16, 2000))
        self.assertEqual(overlap[0]["segment_id"], "0")
        self.assertEqual(overlap[1]["segment_id"], "2560")
        self.assertTrue(all(s["is_augmented"] == 0 for s in overlap))

    def test_short_recording_is_dropped(self):
        samples = self._run(_signal(5), "1", "2")
        self.assertEqual(samples, [])

    def test_oversampled_windows_are_tagged(self):
        task = SeizureDetectionCHBMIT(
            compute_stft=False,
            oversample_seizures=True,
            label_mode="overlap",
        )
        samples = self._run(_signal(25), "12", "15", task)
        augmented = [s for s in samples if s["is_augmented"] == 1]
        regular = [s for s in samples if s["is_augmented"] == 0]
        self.assertEqual(len(regular), 2)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(augmented[0]["label"], 1)
        self.assertTrue(augmented[0]["segment_id"].startswith("s-0-add-"))

    def test_unreadable_file_is_skipped(self):
        samples = self._run(_signal(25), "", "", read_result=None)
        self.assertEqual(samples, [])

    def test_subjects_filter(self):
        task = SeizureDetectionCHBMIT(
            compute_stft=False, oversample_seizures=False, subjects=["1"]
        )
        kept = self._run(_signal(20), "", "", task, patient_id="CHB1")
        dropped = self._run(_signal(20), "", "", task, patient_id="chb02")
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_normalizations(self):
        # Native rate equals the target, so resampling does not change values.
        signal = _signal(10, value=100.0)
        by_100 = self._run(
            signal,
            "",
            "",
            SeizureDetectionCHBMIT(
                compute_stft=False,
                oversample_seizures=False,
                resample_rate=SFREQ,
                normalization="div_by_100",
            ),
        )
        self.assertTrue(torch.allclose(by_100[0]["signal"], torch.ones(16, 2560)))

        percentile = self._run(
            signal,
            "",
            "",
            SeizureDetectionCHBMIT(
                compute_stft=False,
                oversample_seizures=False,
                resample_rate=SFREQ,
                normalization="95th_percentile",
            ),
        )
        scale = 100.0 + 1e-8
        self.assertTrue(
            torch.allclose(
                percentile[0]["signal"], torch.full((16, 2560), 100.0 / scale)
            )
        )

    def test_schema_and_bad_label_mode(self):
        self.assertEqual(
            SeizureDetectionCHBMIT().input_schema,
            {"signal": "tensor", "stft": "tensor"},
        )
        self.assertEqual(
            SeizureDetectionCHBMIT(compute_stft=False).input_schema,
            {"signal": "tensor"},
        )
        self.assertEqual(
            SeizureDetectionCHBMIT().output_schema, {"label": "multiclass"}
        )
        with self.assertRaises(ValueError):
            SeizureDetectionCHBMIT(label_mode="nope")
        with self.assertRaises(ValueError):
            SeizureDetectionCHBMIT(normalization="zscore")

    def test_stft_is_attached(self):
        task = SeizureDetectionCHBMIT(compute_stft=True, oversample_seizures=False)
        samples = self._run(_signal(10), "", "", task)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["stft"].shape[0], 16)
        self.assertEqual(samples[0]["stft"].ndim, 3)


if __name__ == "__main__":
    unittest.main()
