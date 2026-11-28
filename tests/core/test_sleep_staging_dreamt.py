"""Tests for DREAMT sleep staging using synthetic BVP recordings."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks.sleep_staging_dreamt import SleepStagingDREAMT
from pyhealth.tasks.sleep_staging_dreamt_sequence import SleepStagingDREAMTSequence


@dataclass
class _DummyEvent:
    file_64hz: str
    age: float = 60.0
    gender: str = "M"
    ahi: float = 5.0


class _DummyPatient:
    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None):
        return self._events


class TestSleepStagingDREAMT(unittest.TestCase):
    def _run_task(self, stages, invalid_epoch=None, task=None):
        """Create a temporary 64 Hz CSV and run the DREAMT task."""
        samples_per_epoch = 64 * 30
        bvp = np.sin(np.arange(len(stages) * samples_per_epoch) * 2 * np.pi / 64)

        stage_values = []
        for stage in stages:
            stage_values.extend([stage] * samples_per_epoch)

        if invalid_epoch is not None:
            start = invalid_epoch * samples_per_epoch
            bvp[start : start + 10] = np.nan

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "synthetic_dreamt.csv"
            pd.DataFrame({"BVP": bvp, "Sleep_Stage": stage_values}).to_csv(
                csv_path, index=False
            )

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            task = task or SleepStagingDREAMT()

            # Use predictable IBI values so these tests focus on task behavior.
            with patch.object(
                task,
                "_extract_ibi",
                side_effect=lambda epoch: np.full(len(epoch), 0.8, dtype=np.float32),
            ):
                return task(patient)

    def test_stage_mapping(self):
        task = SleepStagingDREAMT()
        self.assertEqual(task._map_stage("W"), 0)
        self.assertEqual(task._map_stage("N1"), 1)
        self.assertEqual(task._map_stage("N2"), 1)
        self.assertEqual(task._map_stage("N3"), 1)
        self.assertEqual(task._map_stage("R"), 2)
        self.assertIsNone(task._map_stage("Missing"))

    def test_three_class_labels_and_epoch_shape(self):
        samples = self._run_task(["W", "N2", "R"])

        self.assertEqual([s["label"] for s in samples], [0, 1, 2])
        self.assertEqual(samples[0]["signal"].shape, (1, 1920))
        self.assertEqual(samples[0]["patient_id"], "S_TEST")
        self.assertEqual(samples[0]["signal"].dtype, np.float32)

    def test_sequence_first_output_shape(self):
        task = SleepStagingDREAMT(sequence_first=True)
        samples = self._run_task(["W"], task=task)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["signal"].shape, (1920, 1))
        self.assertEqual(samples[0]["signal"].dtype, np.float32)

    def test_default_output_shape_is_preserved(self):
        task = SleepStagingDREAMT()
        samples = self._run_task(["W"], task=task)

        self.assertEqual(samples[0]["signal"].shape, (1, 1920))

    def test_sequence_first_must_be_boolean(self):
        with self.assertRaises(ValueError):
            SleepStagingDREAMT(sequence_first="yes")

    def test_mixed_stage_epoch_is_skipped(self):
        samples_per_epoch = 64 * 30
        bvp = np.ones(samples_per_epoch, dtype=np.float64)
        stages = np.array(["N2"] * samples_per_epoch, dtype=object)

        # Make the final 10 seconds have a different sleep stage.
        stages[-640:] = "R"

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed_stage.csv"
            pd.DataFrame({"BVP": bvp, "Sleep_Stage": stages}).to_csv(
                csv_path, index=False
            )

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            task = SleepStagingDREAMT()
            with patch.object(
                task,
                "_extract_ibi",
                return_value=np.full(1920, 0.8, dtype=np.float32),
            ):
                samples = task(patient)

        self.assertEqual(samples, [])

    def test_nan_epoch_is_skipped(self):
        samples = self._run_task(["W", "N2", "R"], invalid_epoch=1)
        self.assertEqual([s["label"] for s in samples], [0, 2])

    def test_missing_file_is_skipped(self):
        patient = _DummyPatient(
            "S_TEST",
            [_DummyEvent(file_64hz="/nonexistent/dreamt.csv")],
        )
        self.assertEqual(SleepStagingDREAMT()(patient), [])

    def test_empty_patient_has_no_samples(self):
        patient = _DummyPatient("S_TEST", [])
        self.assertEqual(SleepStagingDREAMT()(patient), [])

    def test_flat_bvp_returns_zero_ibi(self):
        """Test that flat BVP produces a finite zero IBI sequence."""
        task = SleepStagingDREAMT()
        bvp = np.zeros(1920, dtype=np.float64)

        ibi = task._extract_ibi(bvp)

        self.assertEqual(ibi.shape, (1920,))
        self.assertEqual(ibi.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(ibi)))
        self.assertEqual(np.count_nonzero(ibi), 0)

    def test_short_recording_returns_no_samples(self):
        """Test that a recording shorter than one epoch is skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "short_recording.csv"
            pd.DataFrame(
                {
                    "BVP": np.ones(1000),
                    "Sleep_Stage": ["W"] * 1000,
                }
            ).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            samples = SleepStagingDREAMT()(patient)

        self.assertEqual(samples, [])

    def test_missing_and_unknown_stage_epochs_are_skipped(self):
        """Test that missing and unsupported stage labels are skipped."""
        samples_per_epoch = 64 * 30
        bvp = np.ones(samples_per_epoch * 2)
        stages = [None] * samples_per_epoch + ["Unknown"] * samples_per_epoch

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "invalid_stages.csv"
            pd.DataFrame(
                {
                    "BVP": bvp,
                    "Sleep_Stage": stages,
                }
            ).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            samples = SleepStagingDREAMT()(patient)

        self.assertEqual(samples, [])

    def test_invalid_bvp_epoch_shape_raises_value_error(self):
        """Test that empty and non-1D BVP epochs are rejected."""
        task = SleepStagingDREAMT()

        invalid_signals = [
            np.array([]),
            np.zeros((2, 1920)),
        ]

        for signal in invalid_signals:
            with self.subTest(shape=signal.shape):
                with self.assertRaises(ValueError):
                    task._extract_ibi(signal)

    def test_actual_bvp_to_ibi_extraction(self):
        """Test IBI extraction on a synthetic periodic BVP signal."""
        try:
            import neurokit2  # noqa: F401
        except ImportError:
            self.skipTest("NeuroKit2 is an optional DREAMT dependency.")

        task = SleepStagingDREAMT()
        time = np.arange(1920) / 64
        bvp = np.sin(2 * np.pi * time)

        ibi = task._extract_ibi(bvp)

        self.assertEqual(ibi.shape, (1920,))
        self.assertEqual(ibi.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(ibi)))
        self.assertGreater(np.count_nonzero(ibi), 0)

    def test_invalid_parameters_raise_value_error(self):
        """Test that invalid task configuration is rejected."""
        invalid_configs = [
            {"sampling_rate": 0},
            {"sampling_rate": -64},
            {"sampling_rate": 64.5},
            {"chunk_duration": 0},
            {"chunk_duration": -30},
            {"chunk_duration": 30.5},
            {"max_ibi_seconds": 0},
            {"max_ibi_seconds": -2.0},
            {"max_ibi_seconds": float("nan")},
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    SleepStagingDREAMT(**config)

    def test_empty_csv_is_skipped(self):
        """Test that an empty CSV file is skipped with a warning."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "empty.csv"
            csv_path.write_text("")

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            with self.assertLogs(
                "pyhealth.tasks.sleep_staging_dreamt",
                level="WARNING",
            ):
                samples = SleepStagingDREAMT()(patient)

        self.assertEqual(samples, [])

    def test_missing_required_column_is_skipped(self):
        """Test that a CSV missing a required column is skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "missing_column.csv"
            pd.DataFrame({"BVP": np.ones(1920)}).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            with self.assertLogs(
                "pyhealth.tasks.sleep_staging_dreamt",
                level="WARNING",
            ):
                samples = SleepStagingDREAMT()(patient)

        self.assertEqual(samples, [])

    def test_custom_column_names(self):
        """Test processing with custom BVP and sleep-stage columns."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "custom_columns.csv"
            pd.DataFrame(
                {
                    "Pulse": np.ones(1920),
                    "Stage": ["W"] * 1920,
                }
            ).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )

            task = SleepStagingDREAMT(
                bvp_column="Pulse",
                stage_column="Stage",
            )

            with patch.object(
                task,
                "_extract_ibi",
                return_value=np.full(1920, 0.8, dtype=np.float32),
            ):
                samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 0)
        self.assertEqual(samples[0]["signal"].shape, (1, 1920))

    def test_invalid_column_names_raise_value_error(self):
        """Test that empty or invalid column names are rejected."""
        invalid_configs = [
            {"bvp_column": ""},
            {"bvp_column": "   "},
            {"bvp_column": None},
            {"stage_column": ""},
            {"stage_column": "   "},
            {"stage_column": None},
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    SleepStagingDREAMT(**config)

    def test_bvp_is_the_default_signal_type(self):
        """Test that BVP remains the default input signal."""
        task = SleepStagingDREAMT()
        self.assertEqual(task.signal_type, "BVP")
        self.assertEqual(task.ibi_column, "IBI")

    def test_invalid_signal_type_raises_value_error(self):
        """Test that unsupported signal types are rejected."""
        for signal_type in ["ECG", "", None]:
            with self.subTest(signal_type=signal_type):
                with self.assertRaises(ValueError):
                    SleepStagingDREAMT(signal_type=signal_type)

    def test_precomputed_ibi_input(self):
        """Test that precomputed IBI values are used directly."""
        samples_per_epoch = 64 * 30
        ibi = np.full(samples_per_epoch, 0.9, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "synthetic_ibi.csv"
            pd.DataFrame(
                {
                    "BVP": np.ones(samples_per_epoch),
                    "IBI": ibi,
                    "Sleep_Stage": ["N2"] * samples_per_epoch,
                }
            ).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )
            task = SleepStagingDREAMT(signal_type="IBI")

            with patch.object(
                task,
                "_extract_ibi",
                side_effect=AssertionError(
                    "BVP extraction should not run in IBI mode."
                ),
            ):
                samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 1)
        self.assertEqual(samples[0]["signal"].shape, (1, samples_per_epoch))
        np.testing.assert_allclose(samples[0]["signal"], 0.9)

    def test_precomputed_ibi_invalid_values_become_zero(self):
        """Test that missing, negative, and long IBI values become zero."""
        samples_per_epoch = 64 * 30
        ibi = np.full(samples_per_epoch, 0.9, dtype=np.float64)
        ibi[0] = np.nan
        ibi[1] = -0.5
        ibi[2] = 0.0
        ibi[3] = 2.0
        ibi[4] = 2.5
        ibi[5] = np.inf

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "invalid_ibi.csv"
            pd.DataFrame(
                {
                    "BVP": np.ones(samples_per_epoch),
                    "IBI": ibi,
                    "Sleep_Stage": ["W"] * samples_per_epoch,
                }
            ).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )
            samples = SleepStagingDREAMT(signal_type="IBI")(patient)

        signal = samples[0]["signal"][0]
        np.testing.assert_array_equal(signal[:6], np.zeros(6))
        self.assertTrue(np.all(signal[6:] == np.float32(0.9)))

    def test_custom_ibi_column_name(self):
        """Test that a custom precomputed IBI column can be selected."""
        samples_per_epoch = 64 * 30

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "custom_ibi.csv"
            pd.DataFrame(
                {
                    "PulseInterval": np.full(samples_per_epoch, 0.75),
                    "Stage": ["R"] * samples_per_epoch,
                }
            ).to_csv(csv_path, index=False)

            patient = _DummyPatient(
                "S_TEST",
                [_DummyEvent(file_64hz=str(csv_path))],
            )
            task = SleepStagingDREAMT(
                signal_type="IBI",
                ibi_column="PulseInterval",
                stage_column="Stage",
            )
            samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 2)
        np.testing.assert_allclose(samples[0]["signal"], 0.75)


class TestDREAMTDatasetSleepStagingIntegration(unittest.TestCase):
    """Test loading synthetic DREAMT files through the dataset and task."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        data_dir = self.root / "data_64Hz"
        data_dir.mkdir()

        patient_file = data_dir / "S_TEST_whole_df.csv"
        pd.DataFrame(
            {
                "IBI": np.full(1920, 0.8, dtype=np.float32),
                "Sleep_Stage": ["W"] * 1920,
            }
        ).to_csv(patient_file, index=False)

        pd.DataFrame(
            [
                {
                    "patient_id": "S_TEST",
                    "age": 60,
                    "gender": "M",
                    "bmi": 25.0,
                    "oahi": 5.0,
                    "ahi": 5.0,
                    "mean_sao2": 95.0,
                    "arousal_index": 10.0,
                    "medical_history": "Synthetic test record",
                    "file_64hz": str(patient_file),
                    "file_100hz": "",
                }
            ]
        ).to_csv(self.root / "dreamt-metadata.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_to_sleep_staging_pipeline(self):
        """Load a fake DREAMT patient and generate a sleep-staging sample."""
        dataset = DREAMTDataset(
            root=str(self.root),
            cache_dir=str(self.root / "cache"),
        )

        self.assertEqual(dataset.unique_patient_ids, ["S_TEST"])

        samples = dataset.set_task(SleepStagingDREAMT(signal_type="IBI"))

        self.assertGreater(len(samples), 0)

        sample = samples[0]
        self.assertEqual(sample["patient_id"], "S_TEST")
        self.assertEqual(sample["signal"].shape, (1, 1920))
        self.assertEqual(sample["label"], 0)
        self.assertTrue(np.allclose(sample["signal"], 0.8))


class TestSleepStagingDREAMTSequence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.task = SleepStagingDREAMTSequence()
        self.task._extract_ibi = lambda epoch: np.full(
            len(epoch), 0.8, dtype=np.float32
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_sequence(self, stages):
        """Create a synthetic 100 Hz recording and process it."""
        bvp = np.sin(np.arange(len(stages)) / 10)
        path = self.root / "synthetic_100hz.csv"
        pd.DataFrame({"BVP": bvp, "Sleep_Stage": stages}).to_csv(path, index=False)

        event = type(
            "Event",
            (),
            {
                "file_100hz": str(path),
                "age": 60,
                "gender": "M",
                "ahi": 5.0,
            },
        )()
        patient = _DummyPatient("S_SEQUENCE", [event])
        return self.task(patient)

    def test_valid_wake_nrem_rem_sequence(self):
        """A valid recording produces three correctly shaped epochs."""
        stages = np.repeat(["W", "N1", "R"], 3000)
        samples = self._run_sequence(stages)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["signal"].shape, (3, 750))
        self.assertEqual(sample["label"]["values"], [0, 1, 2])
        self.assertEqual(sample["label"]["time"], [0.0, 30.0, 60.0])
        self.assertTrue(np.allclose(sample["signal"], 0.8))

    def test_inconsistent_epoch_is_skipped_and_time_is_preserved(self):
        """Skipping an invalid epoch must not collapse the timeline."""
        stages = np.concatenate(
            [
                np.repeat("W", 3000),
                np.concatenate([np.repeat("W", 1500), np.repeat("N1", 1500)]),
                np.repeat("R", 3000),
            ]
        )
        samples = self._run_sequence(stages)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["signal"].shape, (2, 750))
        self.assertEqual(sample["label"]["values"], [0, 2])
        self.assertEqual(sample["label"]["time"], [0.0, 60.0])

    def test_recording_without_valid_epochs_returns_empty_list(self):
        """A recording containing only unmapped stages produces no sample."""
        stages = np.repeat("Missing", 3000)
        samples = self._run_sequence(stages)

        self.assertEqual(samples, [])

    def test_extract_ibi_from_synthetic_ppg(self):
        """Real NeuroKit2 extraction returns a valid sample-level IBI signal."""
        try:
            import neurokit2 as nk
        except ImportError:
            self.skipTest("NeuroKit2 is an optional DREAMT dependency.")

        task = SleepStagingDREAMTSequence()
        ppg = nk.ppg_simulate(
            duration=30,
            sampling_rate=task.sampling_rate,
            heart_rate=60,
            random_state=42,
        )

        ibi = task._extract_ibi(ppg)

        self.assertEqual(ibi.shape, (3000,))
        self.assertTrue(np.isfinite(ibi).all())
        self.assertGreater(np.count_nonzero(ibi), 0)
        self.assertLessEqual(float(ibi.max()), task.max_ibi_seconds)

    def test_extract_ibi_from_constant_signal_returns_zeros(self):
        """A constant BVP signal cannot contain detectable pulse intervals."""
        task = SleepStagingDREAMTSequence()
        bvp = np.ones(3000, dtype=np.float64)

        ibi = task._extract_ibi(bvp)

        self.assertEqual(ibi.shape, (3000,))
        self.assertTrue(np.all(ibi == 0))

    def test_realigns_epoch_after_non_aligned_stage_boundary(self):
        """A new stage run can begin at a non-30-second boundary."""
        stages = np.concatenate(
            [
                np.repeat("W", 3500),  # 35 seconds
                np.repeat("N1", 3000),  # Next complete epoch
            ]
        )
        samples = self._run_sequence(stages)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["signal"].shape, (2, 750))
        self.assertEqual(sample["label"]["values"], [0, 1])
        self.assertEqual(sample["label"]["time"], [0.0, 35.0])


class TestSleepStagingDREAMTSequenceContext(unittest.TestCase):
    """Regression tests for optional context-aware IBI extraction."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.task = SleepStagingDREAMTSequence(context_seconds=5)
        self.bvp = np.sin(np.arange(9000) / 10)
        self.timestamps = np.arange(9000, dtype=np.float64) / 100
        self.baseline = np.full(750, 0.8, dtype=np.float32)

    @staticmethod
    def _mock_extract(segment):
        value = 0.8 if len(segment) == 3000 else 0.9
        return np.full(len(segment), value, dtype=np.float32)

    def _apply(self, timestamps, bvp=None):
        return self.task._apply_ibi_context(
            self.bvp if bvp is None else bvp,
            timestamps,
            3000,
            6000,
            self.baseline,
        )

    def _make_patient(self, include_timestamps=True):
        data = {
            "BVP": self.bvp,
            "Sleep_Stage": np.repeat(["W", "N2", "R"], 3000),
        }
        if include_timestamps:
            data["TIMESTAMP"] = self.timestamps

        path = self.root / "context_recording.csv"
        pd.DataFrame(data).to_csv(path, index=False)

        event = type(
            "Event",
            (),
            {
                "file_100hz": str(path),
                "age": 60,
                "gender": "M",
                "ahi": 5.0,
            },
        )()

        return _DummyPatient("S_CONTEXT", [event])

    def test_default_context_is_zero(self):
        task = SleepStagingDREAMTSequence()
        self.assertEqual(task.context_seconds, 0)

    def test_invalid_context_parameters(self):
        for value in (-1, 1.5, True, False, "5", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "context_seconds"):
                    SleepStagingDREAMTSequence(context_seconds=value)

    def test_zero_context_preserves_default_output(self):
        patient = self._make_patient()
        default = SleepStagingDREAMTSequence()
        explicit = SleepStagingDREAMTSequence(context_seconds=0)

        with patch.object(default, "_extract_ibi",
                          side_effect=self._mock_extract):
            first = default(patient)[0]

        with patch.object(explicit, "_extract_ibi",
                          side_effect=self._mock_extract):
            second = explicit(patient)[0]

        np.testing.assert_array_equal(first["signal"], second["signal"])
        self.assertEqual(first["label"], second["label"])

    def test_context_preserves_labels_times_and_shape(self):
        patient = self._make_patient()
        default = SleepStagingDREAMTSequence()

        with patch.object(default, "_extract_ibi",
                          side_effect=self._mock_extract):
            baseline = default(patient)[0]

        with patch.object(self.task, "_extract_ibi",
                          side_effect=self._mock_extract):
            context = self.task(patient)[0]

        self.assertEqual(context["signal"].shape, (3, 750))
        self.assertEqual(context["signal"].dtype, np.float32)
        self.assertEqual(context["label"], baseline["label"])
        self.assertEqual(context["label"]["values"], [0, 1, 2])
        self.assertEqual(context["label"]["time"], [0.0, 30.0, 60.0])
        np.testing.assert_array_equal(
            context["signal"],
            np.full((3, 750), 0.9, dtype=np.float32),
        )

    def test_context_crops_and_downsamples_correctly(self):
        expanded = np.linspace(0.1, 1.5, 4000, dtype=np.float32)

        with patch.object(
            self.task, "_extract_ibi", return_value=expanded
        ) as extract:
            result = self._apply(self.timestamps)

        extract.assert_called_once()
        np.testing.assert_array_equal(
            extract.call_args.args[0], self.bvp[2500:6500]
        )
        np.testing.assert_array_equal(
            result, expanded[500:3500:4]
        )

    def test_missing_timestamp_column_falls_back(self):
        patient = self._make_patient(include_timestamps=False)

        with patch.object(self.task, "_extract_ibi",
                          side_effect=self._mock_extract) as extract:
            sample = self.task(patient)[0]

        self.assertEqual(extract.call_count, 3)
        np.testing.assert_array_equal(
            sample["signal"],
            np.full((3, 750), 0.8, dtype=np.float32),
        )

    def test_timestamp_discontinuity_falls_back(self):
        timestamps = self.timestamps.copy()
        timestamps[2800:] += 1.0

        with patch.object(self.task, "_extract_ibi") as extract:
            result = self._apply(timestamps)

        extract.assert_not_called()
        np.testing.assert_array_equal(result, self.baseline)

    def test_nonfinite_timestamp_falls_back(self):
        timestamps = self.timestamps.copy()
        timestamps[2800] = np.nan

        with patch.object(self.task, "_extract_ibi") as extract:
            result = self._apply(timestamps)

        extract.assert_not_called()
        np.testing.assert_array_equal(result, self.baseline)

    def test_nonfinite_context_bvp_falls_back(self):
        bvp = self.bvp.copy()
        bvp[2800] = np.nan

        with patch.object(self.task, "_extract_ibi") as extract:
            result = self._apply(self.timestamps, bvp=bvp)

        extract.assert_not_called()
        np.testing.assert_array_equal(result, self.baseline)

    def test_context_extraction_failure_falls_back(self):
        with patch.object(
            self.task, "_extract_ibi",
            side_effect=RuntimeError("Synthetic extraction failure"),
        ):
            result = self._apply(self.timestamps)

        np.testing.assert_array_equal(result, self.baseline)

    def test_invalid_context_candidate_falls_back(self):
        candidates = [
            np.zeros(100, dtype=np.float32),
            np.full(4000, np.nan, dtype=np.float32),
            np.full(4000, -0.1, dtype=np.float32),
            np.full(4000, 2.0, dtype=np.float32),
        ]

        for candidate in candidates:
            with self.subTest(candidate_shape=candidate.shape):
                with patch.object(
                    self.task, "_extract_ibi", return_value=candidate
                ):
                    result = self._apply(self.timestamps)

                np.testing.assert_array_equal(result, self.baseline)

    def test_context_is_clipped_at_recording_boundaries(self):
        with patch.object(self.task, "_extract_ibi",
                          side_effect=self._mock_extract) as extract:
            first = self.task._apply_ibi_context(
                self.bvp, self.timestamps, 0, 3000, self.baseline
            )
            last = self.task._apply_ibi_context(
                self.bvp, self.timestamps, 6000, 9000, self.baseline
            )

        self.assertEqual(extract.call_count, 2)
        np.testing.assert_array_equal(
            extract.call_args_list[0].args[0], self.bvp[:3500]
        )
        np.testing.assert_array_equal(
            extract.call_args_list[1].args[0], self.bvp[5500:]
        )
        self.assertEqual(first.shape, (750,))
        self.assertEqual(last.shape, (750,))



if __name__ == "__main__":
    unittest.main()



class TestDREAMTSequenceDefensiveValidation(unittest.TestCase):
    """Verify sequence preprocessing failure paths."""

    def test_invalid_sequence_parameters(self):
        cases = [
            {"sampling_rate": 0},
            {"target_sampling_rate": 0},
            {"chunk_duration": 0},
            {"sampling_rate": 101},
            {"max_ibi_seconds": 0},
            {"bvp_column": ""},
            {"stage_column": ""},
        ]

        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    SleepStagingDREAMTSequence(**config)

    def test_invalid_sequence_signals_and_missing_stage(self):
        task = SleepStagingDREAMTSequence()

        for signal in (
            np.zeros((2, 10)),
            np.array([0.0, np.nan, 1.0]),
            np.array([0.0, np.inf, 1.0]),
        ):
            with self.subTest(shape=signal.shape):
                with self.assertRaisesRegex(
                    ValueError, "finite and one-dimensional"
                ):
                    task._extract_ibi(signal)

        self.assertIsNone(task._map_stage(np.nan))
        self.assertIsNone(task._map_stage(None))

    def test_missing_and_malformed_sequence_recordings(self):
        from types import SimpleNamespace

        task = SleepStagingDREAMTSequence()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "missing_columns.csv"

            pd.DataFrame(
                {"unrelated_column": [1, 2, 3]}
            ).to_csv(malformed, index=False)

            for path in (
                None,
                str(root / "missing.csv"),
                str(malformed),
            ):
                with self.subTest(path=path):
                    patient = _DummyPatient(
                        "synthetic",
                        [SimpleNamespace(file_100hz=path)],
                    )
                    self.assertEqual(task(patient), [])

    def test_sequence_extraction_failure_handling(self):
        from types import SimpleNamespace

        task = SleepStagingDREAMTSequence()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"

            pd.DataFrame({
                "BVP": np.sin(np.arange(3000) / 10),
                "Sleep_Stage": ["W"] * 3000,
            }).to_csv(path, index=False)

            patient = _DummyPatient(
                "synthetic",
                [SimpleNamespace(
                    file_100hz=str(path),
                    age=60,
                    gender="M",
                    ahi=5.0,
                )],
            )

            with patch.object(
                task, "_extract_ibi",
                side_effect=RuntimeError("synthetic failure"),
            ):
                self.assertEqual(task(patient), [])

            with patch.object(
                task, "_extract_ibi",
                return_value=np.zeros(12),
            ):
                self.assertEqual(task(patient), [])

            with patch.object(
                task, "_extract_ibi",
                side_effect=ImportError("synthetic dependency"),
            ):
                with self.assertRaises(ImportError):
                    task(patient)
