import os
import unittest
import polars as pl

from pyhealth.datasets.wesad import WESADDataset  # Adjust this import based on your actual path

class TestWESADDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up paths and load the dataset once for all tests."""
        cls.test_root = "test_data/wesad"  # Replace with actual test data dir
        cls.dataset = WESADDataset(root=cls.test_root, config_path="pyhealth/datasets/configs/wesad.yaml")

    def test_initialization(self):
        """Test that the dataset initializes with correct number of .pkl files."""
        self.assertIsInstance(self.dataset.pkl_files, list)
        self.assertGreater(len(self.dataset.pkl_files), 0, "No valid .pkl files found.")

    def test_load_data_returns_lazyframe(self):
        """Test that load_data returns a Polars LazyFrame."""
        df_lazy = self.dataset.load_data()
        self.assertIsInstance(df_lazy, pl.LazyFrame)

    def test_load_data_content(self):
        """Check the presence of expected columns in the DataFrame."""
        df_lazy = self.dataset.load_data()
        df = df_lazy.collect()
        required_columns = {"patient_id", "timestamp", "event_type", "label"}
        self.assertTrue(required_columns.issubset(set(df.columns)))

    def test_label_values(self):
        """Test that label column contains expected value types (integers)."""
        df_lazy = self.dataset.load_data()
        df = df_lazy.select("label").collect()
        self.assertTrue(df["label"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64])

    def test_sample_row_validity(self):
        """Test that at least one row has non-NaN sensor values."""
        df_lazy = self.dataset.load_data()
        df = df_lazy.collect()
        sensor_cols = [col for col in df.columns if "chest_" in col or "wrist_" in col]
        any_non_nan = df.select(sensor_cols).drop_nulls().height > 0
        self.assertTrue(any_non_nan, "All sensor values are NaN.")

if __name__ == "__main__":
    unittest.main()