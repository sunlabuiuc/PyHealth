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

# Samples in one scored 30 s epoch at the released 125 Hz.
_SAMPLES_PER_SCORE = 30 * 125


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

    def _run(self, data, ch_names=None, label_file=None, sfreq=125.0, **options):
        """Run the task on fake EDF data without touching the filesystem."""
        task = SleepStagingSHHS(**options)
        channels = SHHS_CHANNELS[: data.shape[0]] if ch_names is None else ch_names
        raw = _FakeRawEDF(data, channels, sfreq=sfreq)
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
        """Samples where the value encodes the channel index.

        ``num_epochs`` counts scored 30 s epochs, so the signal is exactly as
        long as the annotations it gets paired with.
        """
        return np.repeat(
            np.arange(num_channels, dtype=float)[:, None],
            num_epochs * _SAMPLES_PER_SCORE,
            axis=1,
        )

    def _ramp(self, num_epochs, num_channels=14):
        """Samples whose value encodes the absolute sample index."""
        return np.tile(
            np.arange(num_epochs * _SAMPLES_PER_SCORE, dtype=float),
            (num_channels, 1),
        )

    def test_epochs_are_labelled_from_the_xml(self):
        samples = self._run(self._data(4))

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
        samples = self._run(self._data(4))
        signal = samples[0]["signal"]
        self.assertEqual(tuple(signal.shape), (2, _SAMPLES_PER_SCORE))
        np.testing.assert_array_equal(signal[:, 0], [0, 1])

    def test_epochs_cover_consecutive_thirty_second_windows(self):
        samples = self._run(self._ramp(3))

        starts = [int(s["signal"][0, 0]) for s in samples]
        self.assertEqual(starts, [0, _SAMPLES_PER_SCORE, 2 * _SAMPLES_PER_SCORE])
        for sample in samples:
            self.assertEqual(tuple(sample["signal"].shape), (2, _SAMPLES_PER_SCORE))

    def test_named_channels_are_selected_by_name(self):
        samples = self._run(self._data(4), channels=["EEG(sec)"])
        signal = samples[0]["signal"]
        self.assertEqual(tuple(signal.shape), (1, _SAMPLES_PER_SCORE))
        np.testing.assert_array_equal(signal[:, 0], [1])

    def test_channel_lookup_ignores_case_and_spacing(self):
        samples = self._run(self._data(4), channels=["eeg (sec)"])
        np.testing.assert_array_equal(samples[0]["signal"][:, 0], [1])

    def test_missing_channel_drops_the_recording(self):
        with self.assertLogs(task_module.logger, level="WARNING") as logs:
            samples = self._run(self._data(4), channels=["iEEG"])
        self.assertEqual(samples, [])
        self.assertTrue(any("iEEG" in message for message in logs.output))

    def test_unnamed_channels_fall_back_to_legacy_positions(self):
        channels = ["ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7"]
        with self.assertLogs(task_module.logger, level="WARNING") as logs:
            samples = self._run(self._data(4), ch_names=channels)
        self.assertTrue(any("falling back" in message for message in logs.output))
        np.testing.assert_array_equal(samples[0]["signal"][:, 0], [2, 7])

    def test_skips_signal_epochs_beyond_available_labels(self):
        samples = self._run(self._data(6))
        self.assertEqual([s["label"] for s in samples], [0, 4, 2, 1])

    def test_ignores_labels_without_a_complete_signal_epoch(self):
        samples = self._run(self._data(2))
        self.assertEqual([s["label"] for s in samples], [0, 4])

    def test_ignores_labels_without_a_complete_signal_epoch_warns(self):
        with self.assertLogs(task_module.logger, level="WARNING") as logs:
            self._run(self._data(2))
        self.assertTrue(any("scored epochs" in m for m in logs.output))

    def test_movement_epochs_are_dropped(self):
        label_file = self.root / "with-movement.xml"
        label_file.write_text(
            "<PSGAnnotation><SleepStages>"
            "<SleepStage>0</SleepStage><SleepStage>9</SleepStage>"
            "<SleepStage>5</SleepStage>"
            "</SleepStages></PSGAnnotation>"
        )
        samples = self._run(self._data(3), label_file=label_file)
        self.assertEqual([s["label"] for s in samples], [0, 5])
        self.assertEqual([s["epoch_index"] for s in samples], [0, 2])

    def test_unreadable_annotation_drops_the_recording(self):
        missing = self.root / "absent.xml"
        with self.assertLogs(task_module.logger, level="WARNING"):
            samples = self._run(self._data(4), label_file=missing)
        self.assertEqual(samples, [])

    def test_corrupt_annotation_drops_the_recording(self):
        # A truncated download is not well-formed XML, and ET.ParseError is a
        # SyntaxError, so it has to be caught explicitly or one bad night
        # aborts the whole run.
        truncated = self.root / "truncated.xml"
        truncated.write_text("<PSGAnnotation><SleepStages><SleepStage>0")
        empty = self.root / "empty.xml"
        empty.write_text("")
        for label_file in (truncated, empty):
            with self.subTest(label_file=label_file.name):
                with self.assertLogs(task_module.logger, level="WARNING") as logs:
                    samples = self._run(self._data(4), label_file=label_file)
                self.assertEqual(samples, [])
                self.assertTrue(any("Failed to read" in m for m in logs.output))

    def test_subjects_filter_excludes_other_participants(self):
        samples = self._run(self._data(4), subjects=["200999"])
        self.assertEqual(samples, [])

    def test_subjects_filter_keeps_listed_participants(self):
        samples = self._run(self._data(4), subjects=["200001"])
        self.assertEqual(len(samples), 4)

    def test_normalization_is_applied(self):
        # Non-constant data: constant rows survive 95th-percentile scaling.
        rng = np.random.default_rng(0)
        data = rng.normal(50.0, 10.0, size=(14, 4 * _SAMPLES_PER_SCORE))
        plain = self._run(data)[0]["signal"]
        scaled = self._run(data, normalization="95th_percentile")[0]["signal"]
        self.assertFalse(np.allclose(plain.numpy(), scaled.numpy()))

    def test_short_epochs_share_the_score_of_their_window(self):
        # 10 s windows: each 30 s score labels three consecutive samples.
        samples = self._run(self._ramp(4), epoch_seconds=10)

        self.assertEqual(len(samples), 12)
        self.assertEqual(
            [s["label"] for s in samples],
            [0, 0, 0, 4, 4, 4, 2, 2, 2, 1, 1, 1],
        )
        self.assertEqual([s["epoch_index"] for s in samples], list(range(12)))
        self.assertEqual(
            [int(s["signal"][0, 0]) for s in samples],
            [i * 1250 for i in range(12)],
        )
        for sample in samples:
            self.assertEqual(tuple(sample["signal"].shape), (2, 1250))

    def test_movement_epochs_are_dropped_across_all_their_sub_epochs(self):
        label_file = self.root / "movement.xml"
        label_file.write_text(
            "<PSGAnnotation><SleepStages>"
            "<SleepStage>0</SleepStage><SleepStage>9</SleepStage>"
            "<SleepStage>5</SleepStage>"
            "</SleepStages></PSGAnnotation>"
        )
        samples = self._run(self._ramp(3), label_file=label_file, epoch_seconds=10)
        self.assertEqual([s["label"] for s in samples], [0, 0, 0, 5, 5, 5])
        self.assertEqual([s["epoch_index"] for s in samples], [0, 1, 2, 6, 7, 8])

    def test_epoch_seconds_must_divide_thirty(self):
        for epoch_seconds in (7, 20, 45, 60, 90, 120, 300, 0, -10, 12.5, "abc"):
            with (
                self.subTest(epoch_seconds=epoch_seconds),
                self.assertRaises(ValueError),
            ):
                SleepStagingSHHS(epoch_seconds=epoch_seconds)

    def test_epoch_seconds_accepts_divisors_of_thirty(self):
        for epoch_seconds in (1, 2, 3, 5, 6, 10, 15, 30):
            with self.subTest(epoch_seconds=epoch_seconds):
                self.assertEqual(
                    SleepStagingSHHS(epoch_seconds=epoch_seconds).epoch_seconds,
                    epoch_seconds,
                )

    def test_empty_channel_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            SleepStagingSHHS(channels=[])

    def test_recording_is_not_preloaded(self):
        task = SleepStagingSHHS()
        raw = _FakeRawEDF(self._data(4), SHHS_CHANNELS)
        event = SimpleNamespace(
            visit="shhs1",
            record_id="shhs1-200001",
            signal_file="shhs1-200001.edf",
            annotation_file=str(self.annotation_file),
        )
        patient = SimpleNamespace(patient_id="200001", get_events=lambda: [event])
        with mock.patch.object(
            task_module.mne.io, "read_raw_edf", return_value=raw
        ) as reader:
            task(patient)
        self.assertFalse(reader.call_args.kwargs["preload"])

    def test_unexpected_sampling_rate_warns_and_still_slices(self):
        with self.assertLogs(task_module.logger, level="WARNING") as logs:
            samples = self._run(self._data(4), sfreq=100.0)
        self.assertTrue(any("100.0 Hz" in m for m in logs.output))
        # Boundaries follow the file, so a 30 s epoch is 3000 samples here.
        self.assertEqual(len(samples), 4)
        self.assertEqual(tuple(samples[0]["signal"].shape), (2, 3000))

    def test_task_schemas(self):
        task = SleepStagingSHHS()
        self.assertEqual(task.task_name, "SleepStagingSHHS")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "multiclass"})


if __name__ == "__main__":
    unittest.main()
