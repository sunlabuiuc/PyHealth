"""Test cases for the ICUMortalityDuETTMIMIC4 task.

Author: Shubham Srivastava (ss253@illinois.edu)

Description:
    Unit tests for the DuETT mortality prediction task. Tests cover
    task instantiation, schema validation, and binning logic.
"""

import unittest

from pyhealth.tasks.icu_mortality_duett_mimic4 import (
    ICUMortalityDuETTMIMIC4,
)


class TestICUMortalityDuETTMIMIC4(unittest.TestCase):
    """Test cases for the ICUMortalityDuETTMIMIC4 task."""

    def test_task_instantiation_defaults(self):
        """Test task initializes with default parameters."""
        task = ICUMortalityDuETTMIMIC4()

        self.assertEqual(task.task_name, "ICUMortalityDuETTMIMIC4")
        self.assertEqual(task.n_time_bins, 24)
        self.assertEqual(task.input_window_hours, 48)

    def test_task_instantiation_custom(self):
        """Test task initializes with custom parameters."""
        task = ICUMortalityDuETTMIMIC4(
            n_time_bins=12, input_window_hours=24
        )

        self.assertEqual(task.n_time_bins, 12)
        self.assertEqual(task.input_window_hours, 24)

    def test_input_schema(self):
        """Test that input schema has the correct keys and types."""
        task = ICUMortalityDuETTMIMIC4()

        self.assertIn("ts_values", task.input_schema)
        self.assertIn("ts_counts", task.input_schema)
        self.assertIn("static", task.input_schema)
        self.assertIn("times", task.input_schema)

        for key in task.input_schema:
            self.assertEqual(task.input_schema[key], "tensor")

    def test_output_schema(self):
        """Test that output schema has the correct key and type."""
        task = ICUMortalityDuETTMIMIC4()

        self.assertIn("mortality", task.output_schema)
        self.assertEqual(task.output_schema["mortality"], "binary")

    def test_lab_categories(self):
        """Test that lab categories are properly defined."""
        task = ICUMortalityDuETTMIMIC4()

        self.assertEqual(len(task.LAB_CATEGORY_NAMES), 10)
        self.assertEqual(task.D_VARS, 10)
        self.assertIn("Sodium", task.LAB_CATEGORY_NAMES)
        self.assertIn("Glucose", task.LAB_CATEGORY_NAMES)

        # All itemids should map to a category index
        for itemid, idx in task._ITEMID_TO_CAT_IDX.items():
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 10)

    def test_binning_output_shape(self):
        """Test that _bin_observations produces correct shapes."""
        import polars as pl
        from datetime import datetime

        task = ICUMortalityDuETTMIMIC4(
            n_time_bins=4, input_window_hours=8
        )

        admission_time = datetime(2023, 1, 1, 0, 0, 0)

        # Create a minimal labevents DataFrame
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 1, 0, 0),
                    datetime(2023, 1, 1, 3, 0, 0),
                    datetime(2023, 1, 1, 5, 0, 0),
                ],
                "labevents/itemid": ["50983", "50971", "50902"],
                "labevents/valuenum": [140.0, 4.2, 102.0],
                "labevents/storetime": [
                    "2023-01-01 01:00:00",
                    "2023-01-01 03:00:00",
                    "2023-01-01 05:00:00",
                ],
            }
        )

        ts_values, ts_counts = task._bin_observations(
            df, admission_time
        )

        # Should be (n_time_bins, D_VARS) = (4, 10)
        self.assertEqual(len(ts_values), 4)
        self.assertEqual(len(ts_values[0]), 10)
        self.assertEqual(len(ts_counts), 4)
        self.assertEqual(len(ts_counts[0]), 10)

    def test_binning_values_correct(self):
        """Test that binning produces correct values."""
        import polars as pl
        from datetime import datetime

        task = ICUMortalityDuETTMIMIC4(
            n_time_bins=2, input_window_hours=4
        )

        admission_time = datetime(2023, 1, 1, 0, 0, 0)

        # Sodium (idx 0): two events in first bin
        # 50983 is Sodium itemid
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 0, 30, 0),
                    datetime(2023, 1, 1, 1, 30, 0),
                ],
                "labevents/itemid": ["50983", "50983"],
                "labevents/valuenum": [140.0, 142.0],
                "labevents/storetime": [
                    "2023-01-01 00:30:00",
                    "2023-01-01 01:30:00",
                ],
            }
        )

        ts_values, ts_counts = task._bin_observations(
            df, admission_time
        )

        # Sodium is category index 0
        # Both events are in bin 0 (first half of 4-hour window)
        self.assertAlmostEqual(ts_values[0][0], 141.0)  # mean
        self.assertEqual(ts_counts[0][0], 2.0)

        # Bin 1 should be zero-imputed
        self.assertEqual(ts_values[1][0], 0.0)
        self.assertEqual(ts_counts[1][0], 0.0)

    def test_empty_patient_returns_empty(self):
        """Test that a patient with no demographics returns empty."""
        task = ICUMortalityDuETTMIMIC4()

        class MockPatient:
            patient_id = "test"

            def get_events(self, **kwargs):
                return []

        result = task(MockPatient())
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
