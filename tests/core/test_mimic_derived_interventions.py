import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from pyhealth.data import Patient
from pyhealth.tasks.mimic_derived_interventions import (
    VasopressorDurationTask,
    VentilationDurationTask,
)


class TestVasopressorDurationTask(unittest.TestCase):
    """Unit tests for vasopressor duration task logic."""

    def test_uses_duration_column_when_present(self):
        ts1 = datetime(2024, 1, 1, 0, 0)
        ts2 = datetime(2024, 1, 1, 4, 0)
        patient_df = pl.DataFrame(
            {
                "patient_id": ["stay_duration"] * 2,
                "timestamp": [ts1, ts2],
                "event_type": ["vasopressin", "vasopressin"],
                "vasopressin/duration_hours": [1.0, None],
            }
        )
        patient = Patient(patient_id="stay_duration", data_source=patient_df)

        task = VasopressorDurationTask()
        samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "stay_duration")
        timestamps, values = sample["vasopressor"]
        np.testing.assert_array_equal(timestamps, [ts1, ts2])
        np.testing.assert_allclose(values.flatten(), [1.0, 0.0])
        self.assertEqual(sample["vasopressor_hours"], 1.0)

    def test_computes_duration_from_endtime_when_missing_column(self):
        ts1 = datetime(2024, 2, 1, 3, 0)
        ts2 = datetime(2024, 2, 1, 6, 30)
        end1 = ts1 + timedelta(hours=2)
        end2 = ts2 + timedelta(hours=1.5)
        patient_df = pl.DataFrame(
            {
                "patient_id": ["stay_endtime"] * 2,
                "timestamp": [ts1, ts2],
                "event_type": ["vasopressor_durations", "vasopressor_durations"],
                "vasopressor_durations/endtime": [end1, end2],
            }
        )
        patient = Patient(patient_id="stay_endtime", data_source=patient_df)

        task = VasopressorDurationTask(event_type="vasopressor_durations")
        samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        timestamps, values = sample["vasopressor"]
        np.testing.assert_array_equal(timestamps, [ts1, ts2])
        np.testing.assert_allclose(values.flatten(), [2.0, 1.5])
        self.assertAlmostEqual(sample["vasopressor_hours"], 3.5)


class TestVentilationDurationTask(unittest.TestCase):
    """Unit tests for ventilation duration task logic."""

    def test_status_encoding_and_duration_sum(self):
        ts1 = datetime(2024, 3, 1, 0, 0)
        ts2 = datetime(2024, 3, 1, 5, 0)
        ts3 = datetime(2024, 3, 1, 9, 0)
        patient_df = pl.DataFrame(
            {
                "patient_id": ["stay_status"] * 3,
                "timestamp": [ts1, ts2, ts3],
                "event_type": ["ventilation", "ventilation", "ventilation"],
                "ventilation/duration_hours": [4.0, 2.0, 1.0],
                "ventilation/ventilation_status": ["Invasive", "NIV", "Invasive"],
            }
        )
        patient = Patient(patient_id="stay_status", data_source=patient_df)

        task = VentilationDurationTask()
        samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        timestamps, values = sample["ventilation"]
        np.testing.assert_array_equal(timestamps, [ts1, ts2, ts3])
        np.testing.assert_allclose(values[:, 0], [4.0, 2.0, 1.0])
        np.testing.assert_array_equal(values[:, 1], [0.0, 1.0, 0.0])
        self.assertEqual(sample["ventilation_hours"], 7.0)
        self.assertEqual(sample["ventilation_statuses"], ["Invasive", "NIV"])

    def test_defaults_when_status_and_duration_absent(self):
        ts1 = datetime(2024, 4, 1, 1, 0)
        ts2 = datetime(2024, 4, 1, 6, 0)
        end1 = ts1 + timedelta(hours=1)
        end2 = ts2 + timedelta(hours=3)
        patient_df = pl.DataFrame(
            {
                "patient_id": ["stay_unknown"] * 2,
                "timestamp": [ts1, ts2],
                "event_type": ["ventilation_durations", "ventilation_durations"],
                "ventilation_durations/endtime": [end1, end2],
            }
        )
        patient = Patient(patient_id="stay_unknown", data_source=patient_df)

        task = VentilationDurationTask(event_type="ventilation_durations")
        samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        timestamps, values = sample["ventilation"]
        np.testing.assert_array_equal(timestamps, [ts1, ts2])
        np.testing.assert_allclose(values[:, 0], [1.0, 3.0])
        np.testing.assert_array_equal(values[:, 1], [0.0, 0.0])
        self.assertEqual(sample["ventilation_hours"], 4.0)
        self.assertEqual(sample["ventilation_statuses"], ["Unknown"])


class TestDerivedConfigs(unittest.TestCase):
    """Ensure derived config files expose expected table metadata."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.config_dir = root / "pyhealth" / "datasets" / "configs"

    def test_mimic3_derived_config(self):
        config_path = self.config_dir / "mimic3_derived.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual(config["version"], "3.1")
        tables = config["tables"]
        self.assertIn("vasopressor_durations", tables)
        self.assertIn("ventilation_durations", tables)

        vasopressor = tables["vasopressor_durations"]
        self.assertEqual(vasopressor["patient_id"], "icustay_id")
        self.assertEqual(vasopressor["timestamp"], "starttime")
        self.assertIn("endtime", vasopressor["attributes"])
        self.assertIn("duration_hours", vasopressor["attributes"])

        ventilation = tables["ventilation_durations"]
        self.assertEqual(ventilation["patient_id"], "stay_id")
        self.assertEqual(ventilation["timestamp"], "starttime")
        self.assertIn("endtime", ventilation["attributes"])
        self.assertIn("duration_hours", ventilation["attributes"])

    def test_mimic4_derived_config(self):
        config_path = self.config_dir / "mimic4_derived.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual(config["version"], "3.1")
        tables = config["tables"]
        self.assertIn("vasopressin", tables)
        self.assertIn("ventilation", tables)

        vasopressor = tables["vasopressin"]
        self.assertEqual(vasopressor["patient_id"], "stay_id")
        self.assertEqual(vasopressor["timestamp"], "starttime")
        self.assertIn("endtime", vasopressor["attributes"])

        ventilation = tables["ventilation"]
        self.assertEqual(ventilation["patient_id"], "stay_id")
        self.assertEqual(ventilation["timestamp"], "starttime")
        self.assertIn("endtime", ventilation["attributes"])
        self.assertIn("ventilation_status", ventilation["attributes"])


if __name__ == "__main__":
    unittest.main()
