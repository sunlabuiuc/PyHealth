"""Unit tests for SleepStagingIBI task."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pyhealth.tasks import SleepStagingIBI
from pyhealth.tasks.sleep_staging_ibi import _MAX_EPOCHS, _SAMPLES_PER_EPOCH


def _make_event(npz_path: str, ahi: float = 5.0):
    event = MagicMock()
    event.npz_path = npz_path
    event.ahi = ahi
    return event


def _make_patient(events, pid: str = "S001"):
    patient = MagicMock()
    patient.patient_id = pid
    patient.get_events.return_value = events
    return patient


def _make_npz(
    directory: str,
    name: str = "S001",
    n_samples: int = 1500,
    stages=None,
    fs: int = 25,
    ahi: float = 3.0,
) -> str:
    if stages is None:
        stages = np.zeros(n_samples, dtype=np.int32)
        for i in range(n_samples):
            stages[i] = i % 5
    data = np.random.default_rng(0).random(n_samples).astype(np.float32)
    path = str(Path(directory) / f"{name}.npz")
    np.savez(
        path,
        data=data,
        stages=stages.astype(np.int32),
        fs=np.int64(fs),
        ahi=np.float32(ahi),
    )
    return path


class TestSleepStagingIBI(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_task_name(self):
        self.assertEqual(SleepStagingIBI.task_name, "SleepStagingIBI")

    def test_input_output_schema(self):
        self.assertEqual(SleepStagingIBI.input_schema["signal"], "tensor")
        self.assertEqual(SleepStagingIBI.output_schema["label"], "multiclass")

    def test_3class_mapping(self):
        data_arr = np.zeros(5 * _SAMPLES_PER_EPOCH, dtype=np.float32)
        stages_arr = np.zeros(5 * _SAMPLES_PER_EPOCH, dtype=np.int32)
        for i, s in enumerate([0, 1, 2, 3, 4]):
            stages_arr[i * _SAMPLES_PER_EPOCH:(i + 1) * _SAMPLES_PER_EPOCH] = s
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path, data=data_arr, stages=stages_arr, fs=np.int64(25), ahi=np.float32(0.0)
        )

        task = SleepStagingIBI(num_classes=3)
        patient = _make_patient([_make_event(path)])
        labels = [s["label"] for s in task(patient)]
        self.assertEqual(labels, [0, 1, 1, 1, 2])

    def test_5class_mapping(self):
        data_arr = np.zeros(5 * _SAMPLES_PER_EPOCH, dtype=np.float32)
        stages_arr = np.zeros(5 * _SAMPLES_PER_EPOCH, dtype=np.int32)
        for i, s in enumerate([0, 1, 2, 3, 4]):
            stages_arr[i * _SAMPLES_PER_EPOCH:(i + 1) * _SAMPLES_PER_EPOCH] = s
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path, data=data_arr, stages=stages_arr, fs=np.int64(25), ahi=np.float32(0.0)
        )

        task = SleepStagingIBI(num_classes=5)
        patient = _make_patient([_make_event(path)])
        self.assertEqual([s["label"] for s in task(patient)], [0, 1, 2, 3, 4])

    def test_invalid_label_skipped(self):
        stages_arr = np.full(_SAMPLES_PER_EPOCH, -1, dtype=np.int32)
        data_arr = np.zeros(_SAMPLES_PER_EPOCH, dtype=np.float32)
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path, data=data_arr, stages=stages_arr, fs=np.int64(25), ahi=np.float32(0.0)
        )

        task = SleepStagingIBI(num_classes=3)
        patient = _make_patient([_make_event(path)])
        self.assertEqual(task(patient), [])

    def test_max_epochs_cap(self):
        n = (_MAX_EPOCHS + 50) * _SAMPLES_PER_EPOCH
        data_arr = np.zeros(n, dtype=np.float32)
        stages_arr = np.zeros(n, dtype=np.int32)
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path, data=data_arr, stages=stages_arr, fs=np.int64(25), ahi=np.float32(0.0)
        )

        task = SleepStagingIBI(num_classes=3)
        patient = _make_patient([_make_event(path)])
        self.assertEqual(len(task(patient)), _MAX_EPOCHS)

    def test_empty_on_short_record(self):
        data_arr = np.zeros(100, dtype=np.float32)
        stages_arr = np.zeros(100, dtype=np.int32)
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path, data=data_arr, stages=stages_arr, fs=np.int64(25), ahi=np.float32(0.0)
        )

        task = SleepStagingIBI()
        patient = _make_patient([_make_event(path)])
        self.assertEqual(task(patient), [])

    def test_signal_shape(self):
        path = _make_npz(str(self.tmp_path), n_samples=3 * _SAMPLES_PER_EPOCH)
        task = SleepStagingIBI()
        patient = _make_patient([_make_event(path)])
        samples = task(patient)
        for s in samples:
            self.assertEqual(s["signal"].shape, (_SAMPLES_PER_EPOCH,))

    def test_ahi_nan_passthrough(self):
        data_arr = np.zeros(_SAMPLES_PER_EPOCH, dtype=np.float32)
        stages_arr = np.zeros(_SAMPLES_PER_EPOCH, dtype=np.int32)
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path,
            data=data_arr,
            stages=stages_arr,
            fs=np.int64(25),
            ahi=np.float32(float("nan")),
        )

        task = SleepStagingIBI()
        event = _make_event(path, ahi=float("nan"))
        patient = _make_patient([event])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertTrue(math.isnan(samples[0]["ahi"]))

    def test_wrong_fs_raises(self):
        data_arr = np.zeros(_SAMPLES_PER_EPOCH, dtype=np.float32)
        stages_arr = np.zeros(_SAMPLES_PER_EPOCH, dtype=np.int32)
        path = str(self.tmp_path / "s.npz")
        np.savez(
            path, data=data_arr, stages=stages_arr, fs=np.int64(50), ahi=np.float32(0.0)
        )

        task = SleepStagingIBI()
        patient = _make_patient([_make_event(path)])
        with self.assertRaisesRegex(ValueError, "fs=50"):
            task(patient)


if __name__ == "__main__":
    unittest.main()
