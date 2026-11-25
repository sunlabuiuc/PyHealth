"""Tests for NHANES mortality prediction tasks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pyhealth.tasks import BaseTask
from pyhealth.tasks.nhanes_mortality import (
    NHANESMortalityPrediction,
    build_window_inputs,
    get_mortality_label,
)


class TestNHANESMortalityPrediction(unittest.TestCase):
    """Tests mortality labels and activity windows."""

    def setUp(self) -> None:
        self.epochs = pd.DataFrame(
            {
                "epoch": np.arange(1, 2017),
                "intensity": np.full(2016, 100.0),
                "wear_minutes": np.full(2016, 5),
                "wear_fraction": np.ones(2016),
            }
        )

    def test_task_inheritance(self) -> None:
        self.assertTrue(issubclass(NHANESMortalityPrediction, BaseTask))

    def test_mortality_labels(self) -> None:
        cases = [
            (1, 1, 30, 1),
            (1, 1, 60, 1),
            (1, 1, 61, 0),
            (1, 0, 100, 0),
            (1, 0, 30, None),
            (1, 0, 60, None),
            (0, 1, 30, None),
            (2, 1, 30, None),
            (1, np.nan, 30, None),
            (1, 1, np.nan, None),
            (1, 1, -1, None),
        ]

        for elig, mort, months, expected in cases:
            with self.subTest(elig=elig, mort=mort, months=months):
                self.assertEqual(
                    get_mortality_label(elig, mort, months),
                    expected,
                )

    def test_prediction_horizon(self) -> None:
        self.assertEqual(get_mortality_label(1, 1, 48, 60), 1)
        self.assertEqual(get_mortality_label(1, 1, 48, 36), 0)

    def test_default_windows(self) -> None:
        X, info = build_window_inputs(self.epochs)

        self.assertEqual(X.shape, (2, 1008, 2))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(len(info), 2)

        self.assertEqual(info[0]["window_id"], 1)
        self.assertEqual(info[1]["window_id"], 2)

    def test_invalid_window(self) -> None:
        epochs = self.epochs.copy()

        epochs.loc[:1007, "wear_minutes"] = 0
        epochs.loc[:1007, "wear_fraction"] = 0

        X, info = build_window_inputs(epochs)

        self.assertEqual(X.shape, (1, 1008, 2))
        self.assertEqual(info[0]["window_id"], 2)

    def test_window_threshold_boundary(self) -> None:
        for valid_count, expected in [(302, 0), (303, 1)]:
            with self.subTest(valid_count=valid_count):
                epochs = self.epochs.iloc[:1008].copy()
                epochs["wear_minutes"] = 0
                epochs["wear_fraction"] = 0

                epochs.loc[: valid_count - 1, "wear_minutes"] = 5
                epochs.loc[: valid_count - 1, "wear_fraction"] = 1

                X, _ = build_window_inputs(epochs)

                self.assertEqual(len(X), expected)

    def test_all_windows_invalid(self) -> None:
        epochs = self.epochs.copy()
        epochs["wear_minutes"] = 0
        epochs["wear_fraction"] = 0

        X, info = build_window_inputs(epochs)

        self.assertEqual(X.shape, (0, 1008, 2))
        self.assertEqual(info, [])

    def test_intensity_only_ablation(self) -> None:
        baseline, baseline_info = build_window_inputs(self.epochs)

        ablation, ablation_info = build_window_inputs(
            self.epochs,
            include_wear_fraction=False,
        )

        self.assertEqual(ablation.shape, (2, 1008, 1))
        self.assertEqual(baseline_info, ablation_info)

        np.testing.assert_array_equal(
            baseline[:, :, 0],
            ablation[:, :, 0],
        )

    def test_full_week_window(self) -> None:
        X, info = build_window_inputs(
            self.epochs,
            window_size=2016,
        )

        self.assertEqual(X.shape, (1, 2016, 2))
        self.assertEqual(len(info), 1)

    def test_invalid_epoch_index(self) -> None:
        epochs = self.epochs.copy()
        epochs.loc[1, "epoch"] = 1

        with self.assertRaises(ValueError):
            build_window_inputs(epochs)

    def test_invalid_parameters(self) -> None:
        cases = [
            {"prediction_months": 0},
            {"window_size": 0},
            {"window_size": 2017},
            {"min_wear_minutes": 0},
            {"min_wear_minutes": 6},
            {"min_valid_fraction": -0.1},
            {"min_valid_fraction": 1.1},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NHANESMortalityPrediction(root=".", **kwargs)

    def test_patient_to_samples(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "activity.parquet"
            self.epochs.to_parquet(path, index=False)

            event = SimpleNamespace(
                eligstat=1,
                mortstat=1,
                permth_int=30,
                activity_file="activity.parquet",
            )

            patient = Mock()
            patient.patient_id = "21005"
            patient.get_events.return_value = [event]

            task = NHANESMortalityPrediction(root=root)
            samples = task(patient)

            self.assertEqual(len(samples), 2)

            for sample in samples:
                self.assertEqual(sample["patient_id"], "21005")
                self.assertEqual(sample["label"], 1)
                self.assertEqual(sample["activity"].shape, (1008, 2))

    def test_ineligible_patient(self) -> None:
        patient = Mock()
        patient.patient_id = "21005"

        patient.get_events.return_value = [
            SimpleNamespace(
                eligstat=0,
                mortstat=1,
                permth_int=30,
            )
        ]

        task = NHANESMortalityPrediction(root=".")

        self.assertEqual(task(patient), [])

    def test_missing_activity_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            patient = Mock()
            patient.patient_id = "21005"

            patient.get_events.return_value = [
                SimpleNamespace(
                    eligstat=1,
                    mortstat=1,
                    permth_int=30,
                    activity_file="missing.parquet",
                )
            ]

            task = NHANESMortalityPrediction(root=root)

            with self.assertRaises(FileNotFoundError):
                task(patient)


if __name__ == "__main__":
    unittest.main()
