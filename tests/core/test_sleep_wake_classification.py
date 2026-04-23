"""Unit tests for the SleepWakeClassification task.

Author:
    Diego Farias Castro (diegof4@illinois.edu)
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks.sleep_wake_classification import SleepWakeClassification


class TestSleepWakeClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        (cls.root / "data_64Hz").mkdir()

        cls.valid_record_dataframe = cls._build_valid_record()
        cls.valid_record_path = cls.root / "data_64Hz" / "S001_whole_df.csv"
        cls.valid_record_dataframe.to_csv(
            cls.valid_record_path,
            index=False,
        )
        cls.valid_patient = cls._build_patient("S001", cls.valid_record_path)
        cls.missing_file_patient = cls._build_patient(
            "S002",
            cls.root / "data_64Hz" / "S002_whole_df.csv",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @staticmethod
    def _build_valid_record(num_rows: int = 4) -> pd.DataFrame:
        sleep_stages = ["W", "W", "N2", "N2", "REM", "REM", "W", "W"][:num_rows]
        return pd.DataFrame(
            {
                "TIMESTAMP": list(range(num_rows)),
                "BVP": np.linspace(0.1, 0.8, num_rows),
                "EDA": np.linspace(0.01, 0.08, num_rows),
                "TEMP": np.linspace(36.1, 36.5, num_rows),
                "ACC_X": np.linspace(1.0, 2.0, num_rows),
                "ACC_Y": np.linspace(0.5, 1.5, num_rows),
                "ACC_Z": np.linspace(0.2, 1.2, num_rows),
                "HR": np.linspace(60, 67, num_rows),
                "Sleep_Stage": sleep_stages,
            }
        )

    @staticmethod
    def _build_patient(patient_id: str, file_64hz: Path) -> Patient:
        data_source = pl.DataFrame(
            {
                "patient_id": [patient_id],
                "timestamp": [pd.Timestamp("2026-01-01").to_pydatetime()],
                "event_type": ["dreamt_sleep"],
                "dreamt_sleep/file_64hz": [str(file_64hz)],
            }
        )
        return Patient(patient_id=patient_id, data_source=data_source)

    def test_convert_sleep_stage_to_binary_label(self):
        task = SleepWakeClassification()

        self.assertEqual(task._convert_sleep_stage_to_binary_label("WAKE"), 1)
        self.assertEqual(task._convert_sleep_stage_to_binary_label("W"), 1)
        self.assertEqual(task._convert_sleep_stage_to_binary_label("N2"), 0)
        self.assertEqual(task._convert_sleep_stage_to_binary_label("REM"), 0)
        self.assertIsNone(task._convert_sleep_stage_to_binary_label(None))
        self.assertIsNone(task._convert_sleep_stage_to_binary_label("UNKNOWN"))

    def test_split_signal_into_epochs(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
        signal = np.array([0, 1, 2, 3, 4])

        epochs = task._split_signal_into_epochs(signal, sampling_rate_hz=1)

        self.assertEqual(len(epochs), 2)
        self.assertTrue(np.array_equal(epochs[0], np.array([0, 1])))
        self.assertTrue(np.array_equal(epochs[1], np.array([2, 3])))

    def test_extract_binary_label_for_epoch(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
        record_dataframe = pd.DataFrame({"Sleep_Stage": ["W", "W", "N2", "N2"]})

        self.assertEqual(
            task._extract_binary_label_for_epoch(record_dataframe, 0, 2), 1
        )
        self.assertEqual(
            task._extract_binary_label_for_epoch(record_dataframe, 1, 2), 0
        )

    def test_base_feature_name_contract_matches_feature_vector(self):
        task = SleepWakeClassification()
        feature_sets = {
            "accelerometer_x": [{"trimmed_mean": 1.0, "max": 2.0, "iqr": 3.0}],
            "accelerometer_y": [{"trimmed_mean": 4.0, "max": 5.0, "iqr": 6.0}],
            "accelerometer_z": [{"trimmed_mean": 7.0, "max": 8.0, "iqr": 9.0}],
            "accelerometer_magnitude_deviation": [{"mad": 10.0}],
            "temperature": [{"mean": 11.0, "min": 12.0, "max": 13.0, "std": 14.0}],
            "blood_volume_pulse": [{"rmssd": 15.0, "sdnn": 16.0, "pnn50": 17.0}],
            "electrodermal_activity": [
                {
                    "scr_amp_mean": 18.0,
                    "scr_amp_max": 19.0,
                    "scr_rise_mean": 20.0,
                    "scr_recovery_mean": 21.0,
                }
            ],
        }

        features = task._build_epoch_feature_vector(feature_sets, epoch_index=0)

        self.assertEqual(len(task.base_feature_names), 21)
        self.assertEqual(len(features), len(task.base_feature_names))
        self.assertEqual(features, [float(i) for i in range(1, 22)])

    def test_build_record_epoch_feature_matrix_returns_empty_when_columns_missing(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
        record_dataframe = pd.DataFrame(
            {
                "ACC_X": [1, 2, 3, 4],
                "ACC_Y": [1, 2, 3, 4],
                "ACC_Z": [1, 2, 3, 4],
                "TEMP": [36, 36, 36, 36],
                "Sleep_Stage": ["W", "W", "N2", "N2"],
            }
        )

        self.assertEqual(task._build_record_epoch_feature_matrix(record_dataframe), [])

    def test_load_wearable_record_dataframe_returns_none_for_missing_file(self):
        task = SleepWakeClassification()
        missing_event = self.missing_file_patient.get_events(event_type="dreamt_sleep")[
            0
        ]

        self.assertIsNone(task._load_wearable_record_dataframe(missing_event))

    def test_task_returns_empty_when_patient_has_no_sleep_events(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
        patient = Patient(
            patient_id="S999",
            data_source=pl.DataFrame(
                {
                    "timestamp": [],
                    "event_type": [],
                },
                schema={
                    "timestamp": pl.Datetime,
                    "event_type": pl.Utf8,
                },
            ),
        )

        self.assertEqual(task(patient), [])

    def test_task_returns_empty_when_sleep_stage_column_is_missing(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
        record_dataframe = self._build_valid_record().drop(columns=["Sleep_Stage"])

        with patch.object(
            task,
            "_load_wearable_record_dataframe",
            return_value=record_dataframe,
        ):
            self.assertEqual(task(self.valid_patient), [])

    def test_task_skips_epochs_with_unsupported_labels(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)
        record_dataframe = self._build_valid_record(num_rows=4)
        record_dataframe["Sleep_Stage"] = ["X", "X", "N2", "N2"]

        with (
            patch.object(
                task,
                "_load_wearable_record_dataframe",
                return_value=record_dataframe,
            ),
            patch.object(
                task,
                "_build_record_epoch_feature_matrix",
                return_value=[[1.0, 2.0], [3.0, 4.0]],
            ),
        ):
            samples = task(self.valid_patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["epoch_index"], 1)
        self.assertEqual(samples[0]["label"], 0)

    def test_task_runs_full_flow_with_lightweight_feature_stub(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)

        with patch.object(
            task,
            "_build_record_epoch_feature_matrix",
            return_value=[
                [1.0, 10.0],
                [2.0, 20.0],
            ],
        ):
            samples = task(self.valid_patient)

        self.assertEqual(len(samples), 2)
        self.assertTrue(all("features" in sample for sample in samples))
        self.assertTrue(all("label" in sample for sample in samples))
        self.assertTrue(all("record_id" in sample for sample in samples))
        self.assertEqual(samples[0]["record_id"], "S001-event0-epoch0")
        self.assertEqual(samples[0]["label"], 1)
        self.assertEqual(samples[1]["label"], 0)

    def test_task_uses_minimum_epoch_count_between_labels_and_features(self):
        task = SleepWakeClassification(epoch_seconds=2, sampling_rate=1)

        with patch.object(
            task,
            "_build_record_epoch_feature_matrix",
            return_value=[[1.0], [2.0]],
        ):
            samples = task(self.valid_patient)

        self.assertEqual(len(samples), 2)
        self.assertEqual([sample["epoch_index"] for sample in samples], [0, 1])


if __name__ == "__main__":
    unittest.main()
