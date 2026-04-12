import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from pyhealth.tasks.shhs_sleep_staging import (
    SleepStagingSHHS,
    _STAGE_MAP,
    _downsample,
    _ecg_to_ibi,
    _parse_profusion_stages,
    _pick_ecg_channel,
)

SHHS1_CHANNELS = [
    "SaO2", "H.R.", "EEG(sec)", "ECG", "EMG",
    "EOG(L)", "EOG(R)", "EEG", "THOR RES", "ABDO RES",
    "POSITION", "LIGHT", "NEW AIR", "OX stat",
]

SHHS2_CHANNELS = [
    "SaO2", "PR", "EEG(sec)", "ECG", "EMG",
    "EOG(L)", "EOG(R)", "EEG", "AIRFLOW", "THOR RES",
    "ABDO RES", "POSITION", "LIGHT", "OX STAT",
]


def _write_profusion_xml(path: str, stages: list[int]) -> None:
    """Write a minimal Profusion-format XML annotation file."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<PSGAnnotation>",
        "  <SleepStages>",
    ]
    for s in stages:
        lines.append(f"    <SleepStage>{s}</SleepStage>")
    lines.extend(["  </SleepStages>", "</PSGAnnotation>"])
    with open(path, "w") as f:
        f.write("\n".join(lines))


class TestPickEcgChannel(unittest.TestCase):

    def test_shhs1_channels(self):
        self.assertEqual(_pick_ecg_channel(SHHS1_CHANNELS), 3)

    def test_shhs2_channels(self):
        self.assertEqual(_pick_ecg_channel(SHHS2_CHANNELS), 3)

    def test_case_insensitive(self):
        channels = ["SaO2", "eeg", "ecg", "emg"]
        self.assertEqual(_pick_ecg_channel(channels), 2)

    def test_ecg_substring(self):
        channels = ["EEG", "ECG1", "EMG"]
        self.assertEqual(_pick_ecg_channel(channels), 1)

    def test_missing_raises(self):
        channels = ["SaO2", "H.R.", "EEG", "EMG", "EOG(L)"]
        with self.assertRaises(ValueError):
            _pick_ecg_channel(channels)


class TestParseProfusionStages(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parses_stages(self):
        stages = [0, 0, 1, 2, 2, 3, 5, 0]
        xml_path = f"{self.tmpdir.name}/test.xml"
        _write_profusion_xml(xml_path, stages)
        self.assertEqual(_parse_profusion_stages(xml_path), stages)

    def test_single_stage(self):
        xml_path = f"{self.tmpdir.name}/single.xml"
        _write_profusion_xml(xml_path, [2])
        self.assertEqual(_parse_profusion_stages(xml_path), [2])

    def test_missing_sleep_stages_raises(self):
        xml_path = f"{self.tmpdir.name}/bad.xml"
        with open(xml_path, "w") as f:
            f.write("<PSGAnnotation><Events/></PSGAnnotation>")
        with self.assertRaises(ValueError):
            _parse_profusion_stages(xml_path)


class TestEcgToIbi(unittest.TestCase):

    @patch("pyhealth.tasks.shhs_sleep_staging.nk.ecg_process")
    def test_regular_heartbeat(self, mock_ecg):
        fs = 125
        n_samples = 1000
        rpeaks = np.arange(125, n_samples, 125)
        mock_ecg.return_value = (
            None, {"ECG_R_Peaks": rpeaks}
        )

        signal = np.zeros(n_samples, dtype=np.float32)
        ibi = _ecg_to_ibi(signal, fs)

        self.assertEqual(len(ibi), n_samples)
        self.assertAlmostEqual(ibi[200], 1.0, places=5)
        self.assertEqual(ibi[0], 0.0)

    @patch("pyhealth.tasks.shhs_sleep_staging.nk.ecg_process")
    def test_outlier_removal(self, mock_ecg):
        fs = 125
        n_samples = 1000
        # Peaks at 100, 200, 700 -> IBIs: 0.8s, 4.0s
        rpeaks = np.array([100, 200, 700])
        mock_ecg.return_value = (
            None, {"ECG_R_Peaks": rpeaks}
        )

        signal = np.zeros(n_samples, dtype=np.float32)
        ibi = _ecg_to_ibi(signal, fs)

        self.assertAlmostEqual(ibi[150], 0.8, places=5)
        self.assertEqual(ibi[400], 0.0)
        self.assertEqual(ibi[800], 0.0)

    @patch("pyhealth.tasks.shhs_sleep_staging.nk.ecg_process")
    def test_fewer_than_two_peaks(self, mock_ecg):
        mock_ecg.return_value = (
            None, {"ECG_R_Peaks": np.array([100])}
        )

        signal = np.zeros(500, dtype=np.float32)
        ibi = _ecg_to_ibi(signal, fs=125)

        self.assertEqual(len(ibi), 500)
        self.assertTrue(np.all(ibi == 0.0))

    def test_with_neurokit(self):
        import neurokit2 as nk
        ecg = nk.ecg_simulate(duration=10, sampling_rate=125)
        ibi = _ecg_to_ibi(
            np.array(ecg, dtype=np.float32), fs=125
        )

        nonzero = ibi[ibi > 0]
        self.assertGreater(len(nonzero), 0)
        self.assertTrue(np.all(nonzero < 2.0))
        self.assertTrue(np.all(nonzero > 0.3))


class TestDownsample(unittest.TestCase):

    def test_integer_ratio(self):
        signal = np.arange(1250, dtype=np.float32)
        result = _downsample(signal, source_hz=125, target_hz=25)
        self.assertEqual(len(result), 250)

    def test_non_integer_ratio(self):
        signal = np.arange(2560, dtype=np.float32)
        result = _downsample(
            signal, source_hz=256, target_hz=25
        )
        expected_len = int(round(2560 * 25 / 256))
        self.assertAlmostEqual(len(result), expected_len, delta=2)

    def test_same_rate(self):
        signal = np.arange(100, dtype=np.float32)
        result = _downsample(signal, source_hz=25, target_hz=25)
        np.testing.assert_array_equal(result, signal)


class TestSleepStagingSHHSTask(unittest.TestCase):

    def test_task_schema(self):
        self.assertEqual(
            SleepStagingSHHS.task_name, "SleepStagingSHHS"
        )
        self.assertEqual(
            SleepStagingSHHS.input_schema, {"signal": "tensor"}
        )
        self.assertEqual(
            SleepStagingSHHS.output_schema, {"label": "multiclass"}
        )

    def test_defaults(self):
        task = SleepStagingSHHS()
        self.assertEqual(task.epoch_seconds, 30)
        self.assertEqual(task.seq_len, 20)
        self.assertEqual(task.target_hz, 25)
        self.assertEqual(task.max_epochs, 1100)

    def test_custom_params(self):
        task = SleepStagingSHHS(
            epoch_seconds=10,
            seq_len=5,
            target_hz=50,
            max_epochs=500,
        )
        self.assertEqual(task.epoch_seconds, 10)
        self.assertEqual(task.seq_len, 5)
        self.assertEqual(task.target_hz, 50)
        self.assertEqual(task.max_epochs, 500)

    def test_stage_map(self):
        self.assertEqual(_STAGE_MAP[0], 0)
        self.assertEqual(_STAGE_MAP[1], 1)
        self.assertEqual(_STAGE_MAP[2], 1)
        self.assertEqual(_STAGE_MAP[3], 1)
        self.assertEqual(_STAGE_MAP[4], 1)
        self.assertEqual(_STAGE_MAP[5], 2)


class TestSleepStagingProcessing(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.seq_len = 5
        self.target_hz = 25
        self.epoch_seconds = 30
        self.source_hz = 125
        self.samples_per_epoch = self.target_hz * self.epoch_seconds

        self.task = SleepStagingSHHS(
            epoch_seconds=self.epoch_seconds,
            seq_len=self.seq_len,
            target_hz=self.target_hz,
        )

        self.n_epochs = 10
        self.stages = [0, 1, 2, 2, 3, 5, 0, 1, 2, 3]
        self.xml_path = (
            f"{self.tmpdir.name}/shhs1-200001-profusion.xml"
        )
        _write_profusion_xml(self.xml_path, self.stages)

        n_source_samples = (
            self.n_epochs * self.epoch_seconds * self.source_hz
        )
        self.ecg_signal = np.sin(
            2 * np.pi * np.arange(n_source_samples) / self.source_hz
        ).astype(np.float32)

        self.rpeaks = np.arange(125, n_source_samples, 125)

        self.event = SimpleNamespace(
            signal_file=f"{self.tmpdir.name}/shhs1-200001.edf",
            annotation_file=self.xml_path,
            ecg_sample_rate="125",
            visitnumber=1,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_process_event(self):
        mock_raw = MagicMock()
        mock_raw.ch_names = SHHS1_CHANNELS
        mock_raw.get_data.return_value = (
            self.ecg_signal.reshape(1, -1)
        )

        with (
            patch(
                "mne.io.read_raw_edf", return_value=mock_raw
            ),
            patch(
                "pyhealth.tasks.shhs_sleep_staging.nk.ecg_process",
                return_value=(
                    None, {"ECG_R_Peaks": self.rpeaks}
                ),
            ),
        ):
            return self.task._process_event("200001", self.event)

    def test_returns_samples(self):
        samples = self._run_process_event()
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)

    def test_sample_keys(self):
        samples = self._run_process_event()
        for sample in samples:
            self.assertIn("patient_id", sample)
            self.assertIn("record_id", sample)
            self.assertIn("signal", sample)
            self.assertIn("label", sample)

    def test_patient_id(self):
        samples = self._run_process_event()
        for sample in samples:
            self.assertEqual(sample["patient_id"], "200001")

    def test_signal_shape(self):
        samples = self._run_process_event()
        expected = (self.seq_len, self.samples_per_epoch)
        for sample in samples:
            self.assertEqual(sample["signal"].shape, expected)

    def test_signal_dtype(self):
        samples = self._run_process_event()
        self.assertEqual(samples[0]["signal"].dtype, np.float32)

    def test_label_range(self):
        samples = self._run_process_event()
        for sample in samples:
            self.assertIn(sample["label"], {0, 1, 2})

    def test_record_id_format(self):
        samples = self._run_process_event()
        pattern = re.compile(r"^200001-v1-\d+$")
        for sample in samples:
            self.assertRegex(sample["record_id"], pattern)

    def test_sliding_window_count(self):
        samples = self._run_process_event()
        expected = self.n_epochs - self.seq_len + 1
        self.assertEqual(len(samples), expected)

    def test_too_few_epochs_returns_empty(self):
        short_xml = f"{self.tmpdir.name}/short.xml"
        _write_profusion_xml(short_xml, [0, 1, 2])
        event = SimpleNamespace(
            signal_file=self.event.signal_file,
            annotation_file=short_xml,
            ecg_sample_rate="125",
            visitnumber=1,
        )

        mock_raw = MagicMock()
        mock_raw.ch_names = SHHS1_CHANNELS
        n_samples = 3 * self.epoch_seconds * self.source_hz
        mock_raw.get_data.return_value = np.zeros(
            (1, n_samples), dtype=np.float32
        )
        rpeaks = np.arange(125, n_samples, 125)

        with (
            patch(
                "mne.io.read_raw_edf", return_value=mock_raw
            ),
            patch(
                "pyhealth.tasks.shhs_sleep_staging.nk.ecg_process",
                return_value=(
                    None, {"ECG_R_Peaks": rpeaks}
                ),
            ),
        ):
            samples = self.task._process_event("200001", event)

        self.assertEqual(samples, [])

    def test_invalid_stages_skipped(self):
        # Stage 6 not in _STAGE_MAP -> mapped to -1, windows skipped
        stages_with_invalid = [0, 1, 2, 2, 3, 6, 0, 1, 2, 3]
        xml_path = f"{self.tmpdir.name}/invalid.xml"
        _write_profusion_xml(xml_path, stages_with_invalid)
        event = SimpleNamespace(
            signal_file=self.event.signal_file,
            annotation_file=xml_path,
            ecg_sample_rate="125",
            visitnumber=1,
        )

        mock_raw = MagicMock()
        mock_raw.ch_names = SHHS1_CHANNELS
        mock_raw.get_data.return_value = (
            self.ecg_signal.reshape(1, -1)
        )

        with (
            patch(
                "mne.io.read_raw_edf", return_value=mock_raw
            ),
            patch(
                "pyhealth.tasks.shhs_sleep_staging.nk.ecg_process",
                return_value=(
                    None, {"ECG_R_Peaks": self.rpeaks}
                ),
            ),
        ):
            samples = self.task._process_event("200001", event)

        all_valid_count = self.n_epochs - self.seq_len + 1
        self.assertLess(len(samples), all_valid_count)
        for sample in samples:
            self.assertIn(sample["label"], {0, 1, 2})

    def test_call_handles_processing_errors(self):
        mock_patient = MagicMock()
        mock_patient.patient_id = "200001"
        bad_event = SimpleNamespace(
            signal_file="/nonexistent/file.edf",
            annotation_file="/nonexistent/annotation.xml",
            ecg_sample_rate="125",
            visitnumber=1,
        )
        mock_patient.get_events.return_value = [bad_event]

        samples = self.task(mock_patient)
        self.assertEqual(samples, [])

    def test_max_epochs_limits_output(self):
        task = SleepStagingSHHS(
            epoch_seconds=self.epoch_seconds,
            seq_len=self.seq_len,
            target_hz=self.target_hz,
            max_epochs=7,
        )
        mock_raw = MagicMock()
        mock_raw.ch_names = SHHS1_CHANNELS
        mock_raw.get_data.return_value = (
            self.ecg_signal.reshape(1, -1)
        )

        with (
            patch(
                "mne.io.read_raw_edf", return_value=mock_raw
            ),
            patch(
                "pyhealth.tasks.shhs_sleep_staging.nk.ecg_process",
                return_value=(
                    None, {"ECG_R_Peaks": self.rpeaks}
                ),
            ),
        ):
            samples = task._process_event("200001", self.event)

        self.assertEqual(len(samples), 7 - self.seq_len + 1)


if __name__ == "__main__":
    unittest.main()
