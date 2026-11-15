"""SleepStagingSHHS: epoch slicing, channel selection and label filtering."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pyhealth.tasks import sleep_staging_shhs as task_module
from pyhealth.tasks.sleep_staging_shhs import SleepStagingSHHS

SHHS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
    <EpochLength>30</EpochLength>
    <SleepStages>
        <SleepStage>0</SleepStage>
        <SleepStage>4</SleepStage>
        <SleepStage>2</SleepStage>
        <SleepStage>1</SleepStage>
    </SleepStages>
</PSGAnnotation>
"""

SHHS_CHANNELS = [
    "EEG",
    "EEG(sec)",
    "EMG",
    "EOG(L)",
    "EOG(R)",
    "ECG",
    "HR",
    "RESP",
    "THOR RES",
    "ABDO RES",
    "POSITION",
    "LIGHT",
    "SaO2",
    "OX STAT",
]


class _FakeRawEDF:
    """Minimal stand-in for the ``mne.io.Raw`` returned by read_raw_edf."""

    def __init__(self, data, ch_names, sfreq=125.0):
        self._data = data
        self.ch_names = list(ch_names)
        self.info = {"sfreq": sfreq}
        self.closed = False

    def get_data(self, picks=None, units=None):
        data = self._data
        if picks is not None:
            data = data[list(picks), :]
        return data

    def close(self):
        self.closed = True


class TestSleepStagingSHHS(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.annotation_file = self.root / "shhs1-200001-profusion.xml"
        self.annotation_file.write_text(SHHS_XML)

    def _run(self, data, ch_names=None, label_file=None, **options):
        """Run the task on fake EDF data without touching the filesystem."""
        task = SleepStagingSHHS(**options)
        channels = SHHS_CHANNELS[: data.shape[0]] if ch_names is None else ch_names
        raw = _FakeRawEDF(data, channels)
        event = SimpleNamespace(
            visit="shhs1",
            record_id="shhs1-200001",
            signal_file="shhs1-200001.edf",
            annotation_file=str(label_file or self.annotation_file),
        )
        patient = SimpleNamespace(patient_id="200001", get_events=lambda: [event])
        with mock.patch.object(task_module.mne.io, "read_raw_edf", return_value=raw):
            return task(patient)

    def _data(self, num_epochs, num_channels=14):
        """Samples where the value encodes the channel index."""
        return np.repeat(
            np.arange(num_channels, dtype=float)[:, None], num_epochs * 125, axis=1
        )

    def test_epochs_are_labelled_from_the_xml(self):
        samples = self._run(self._data(4), epoch_sec=1)

        self.assertEqual(len(samples), 4)
        self.assertEqual([s["label"] for s in samples], [0, 4, 2, 1])
        self.assertEqual([s["epoch_index"] for s in samples], [0, 1, 2, 3])
        self.assertEqual(
            [s["record_id"] for s in samples],
            [f"shhs1-200001-{i}" for i in range(4)],
        )
        self.assertTrue(all(s["patient_id"] == "200001" for s in samples))
        self.assertTrue(all(s["visit"] == "shhs1" for s in samples))

    def test_default_channels_are_the_two_eeg_channels(self):
        samples = self._run(self._data(4), epoch_sec=1)
        signal = samples[0]["signal"]
        self.assertEqual(tuple(signal.shape), (2, 125))
        np.testing.assert_array_equal(signal[:, 0], [0, 1])

    def test_named_channels_are_selected_by_name(self):
        samples = self._run(self._data(4), epoch_sec=1, channels=["EEG(sec)"])
        signal = samples[0]["signal"]
        self.assertEqual(tuple(signal.shape), (1, 125))
        np.testing.assert_array_equal(signal[:, 0], [1])

    def test_channel_lookup_ignores_case_and_spacing(self):
        samples = self._run(self._data(4), epoch_sec=1, channels=["eeg (sec)"])
        np.testing.assert_array_equal(samples[0]["signal"][:, 0], [1])

    def test_missing_channel_drops_the_recording(self):
        with self.assertLogs(task_module.logger, level="WARNING") as logs:
            samples = self._run(self._data(4), epoch_sec=1, channels=["iEEG"])
        self.assertEqual(samples, [])
        self.assertTrue(any("iEEG" in message for message in logs.output))

    def test_unnamed_channels_fall_back_to_legacy_positions(self):
        channels = ["ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7"]
        with self.assertLogs(task_module.logger, level="WARNING") as logs:
            samples = self._run(self._data(4), ch_names=channels, epoch_sec=1)
        self.assertTrue(any("falling back" in message for message in logs.output))
        np.testing.assert_array_equal(samples[0]["signal"][:, 0], [2, 7])

    def test_skips_signal_epochs_beyond_available_labels(self):
        samples = self._run(self._data(6), epoch_sec=1)
        self.assertEqual([s["label"] for s in samples], [0, 4, 2, 1])

    def test_ignores_labels_without_a_complete_signal_epoch(self):
        samples = self._run(self._data(2), epoch_sec=1)
        self.assertEqual([s["label"] for s in samples], [0, 4])

    def test_movement_epochs_are_dropped(self):
        label_file = self.root / "with-movement.xml"
        label_file.write_text(
            "<PSGAnnotation><SleepStages>"
            "<SleepStage>0</SleepStage><SleepStage>9</SleepStage>"
            "<SleepStage>5</SleepStage>"
            "</SleepStages></PSGAnnotation>"
        )
        samples = self._run(self._data(3), epoch_sec=1, label_file=label_file)
        self.assertEqual([s["label"] for s in samples], [0, 5])
        self.assertEqual([s["epoch_index"] for s in samples], [0, 2])

    def test_unreadable_annotation_drops_the_recording(self):
        missing = self.root / "absent.xml"
        with self.assertLogs(task_module.logger, level="WARNING"):
            samples = self._run(self._data(4), epoch_sec=1, label_file=missing)
        self.assertEqual(samples, [])

    def test_subjects_filter_excludes_other_participants(self):
        samples = self._run(self._data(4), epoch_sec=1, subjects=["200999"])
        self.assertEqual(samples, [])

    def test_normalization_is_applied(self):
        # Non-constant data: constant rows survive 95th-percentile scaling.
        rng = np.random.default_rng(0)
        data = rng.normal(50.0, 10.0, size=(14, 4 * 125))
        plain = self._run(data, epoch_sec=1)[0]["signal"]
        scaled = self._run(data, epoch_sec=1, normalization="95th_percentile")[0][
            "signal"
        ]
        self.assertFalse(np.allclose(plain.numpy(), scaled.numpy()))

    def test_epoch_sec_must_keep_annotations_aligned(self):
        for epoch_sec in (7, 20, 45, 0):
            with self.subTest(epoch_sec=epoch_sec), self.assertRaises(ValueError):
                SleepStagingSHHS(epoch_sec=epoch_sec)

    def test_task_schemas(self):
        task = SleepStagingSHHS()
        self.assertEqual(task.task_name, "SleepStagingSHHS")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "multiclass"})


if __name__ == "__main__":
    unittest.main()
