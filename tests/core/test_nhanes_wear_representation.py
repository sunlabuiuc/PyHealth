"""Tests for NHANES wear-coverage representations."""

import unittest

import numpy as np
import pandas as pd

from pyhealth.tasks.nhanes_mortality import (
    NHANESMortalityPrediction,
    build_window_inputs,
)


class TestNHANESWearRepresentation(unittest.TestCase):
    """Tests binary and continuous wear features."""

    def setUp(self) -> None:
        self.epochs = pd.DataFrame(
            {
                "epoch": np.arange(1, 7),
                "intensity": [0, 10, 20, 30, 40, 50],
                "wear_minutes": [0, 1, 2, 3, 4, 5],
                "wear_fraction": [0, 0.2, 0.4, 0.6, 0.8, 1],
            }
        )

    def test_binary_representation(self) -> None:
        """Binary coverage should follow the wear threshold."""
        activity, _ = build_window_inputs(
            self.epochs,
            window_size=6,
            wear_representation="binary",
        )

        np.testing.assert_array_equal(
            activity[0, :, 1],
            [0, 1, 1, 1, 1, 1],
        )

    def test_fraction_representation(self) -> None:
        """Continuous coverage should preserve wear fractions."""
        activity, _ = build_window_inputs(
            self.epochs,
            window_size=6,
            wear_representation="fraction",
        )

        np.testing.assert_allclose(
            activity[0, :, 1],
            [0, 0.2, 0.4, 0.6, 0.8, 1],
        )

    def test_same_intensity_and_windows(self) -> None:
        """Coverage representation must not alter intensity."""
        binary, binary_info = build_window_inputs(
            self.epochs,
            window_size=6,
            wear_representation="binary",
        )

        fraction, fraction_info = build_window_inputs(
            self.epochs,
            window_size=6,
            wear_representation="fraction",
        )

        self.assertEqual(binary.shape, fraction.shape)
        self.assertEqual(binary_info, fraction_info)

        np.testing.assert_array_equal(
            binary[:, :, 0],
            fraction[:, :, 0],
        )

    def test_custom_wear_threshold(self) -> None:
        """Binary coverage must respect custom wear thresholds."""
        activity, _ = build_window_inputs(
            self.epochs,
            window_size=6,
            min_wear_minutes=3,
            wear_representation="binary",
        )

        np.testing.assert_array_equal(
            activity[0, :, 1],
            [0, 0, 0, 1, 1, 1],
        )

        np.testing.assert_array_equal(
            activity[0, :, 0],
            [0, 0, 0, 30, 40, 50],
        )

    def test_intensity_only_compatibility(self) -> None:
        """Disabling wear features should preserve intensity."""
        binary, binary_info = build_window_inputs(
            self.epochs,
            window_size=6,
            include_wear_fraction=False,
            wear_representation="binary",
        )

        fraction, fraction_info = build_window_inputs(
            self.epochs,
            window_size=6,
            include_wear_fraction=False,
            wear_representation="fraction",
        )

        self.assertEqual(binary.shape, (1, 6, 1))
        self.assertEqual(binary_info, fraction_info)

        np.testing.assert_array_equal(binary, fraction)

    def test_invalid_representation(self) -> None:
        """Unsupported representations should raise errors."""
        with self.assertRaises(ValueError):
            build_window_inputs(
                self.epochs,
                window_size=6,
                wear_representation="invalid",
            )

    def test_invalid_task_representation(self) -> None:
        """The task constructor should validate representation."""
        with self.assertRaises(ValueError):
            NHANESMortalityPrediction(
                root=".",
                wear_representation="invalid",
            )


if __name__ == "__main__":
    unittest.main()
