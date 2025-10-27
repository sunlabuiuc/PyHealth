import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import polars as pl

from tests.base import BaseTestCase
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.tasks.base_task import BaseTask
from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.data import Patient


class MockTask(BaseTask):
    """Mock task for testing purposes."""

    def __init__(self, task_name="test_task"):
        self.task_name = task_name
        self.input_schema = {"test_attribute": "raw"}
        self.output_schema = {"test_label": "binary"}

    def __call__(self, patient):
        """Return mock samples based on patient data."""
        # Extract patient's test data from the patient's data source
        patient_data = patient.data_source

        samples = []
        for row in patient_data.iter_rows(named=True):
            sample = {
                "test_attribute": row["test/test_attribute"],
                "test_label": row["test/test_label"],
                "patient_id": row["patient_id"],
            }
            samples.append(sample)

        return samples


class MockDataset(BaseDataset):
    """Mock dataset for testing purposes."""

    def __init__(self):
        # Initialize without calling parent __init__ to avoid file dependencies
        self.dataset_name = "TestDataset"
        self.dev = False

        # Create realistic test data with patient_id, test_attribute, and test_label
        self._collected_global_event_df = pl.DataFrame(
            {
                "patient_id": ["1", "2", "1", "2"],
                "event_type": ["test", "test", "test", "test"],
                "timestamp": [None, None, None, None],
                "test/test_attribute": [
                    "pat_1_attr_1",
                    "pat_2_attr_1",
                    "pat_1_attr_2",
                    "pat_2_attr_2",
                ],
                "test/test_label": [0, 1, 1, 0],
            }
        )
        self._unique_patient_ids = ["1", "2"]

    @property
    def collected_global_event_df(self):
        return self._collected_global_event_df

    @property
    def unique_patient_ids(self):
        return self._unique_patient_ids

    def iter_patients(self, df=None):
        """Mock patient iterator that returns real Patient objects."""
        if df is None:
            df = self.collected_global_event_df

        grouped = df.group_by("patient_id")
        for patient_id, patient_df in grouped:
            patient_id = patient_id[0]
            yield Patient(patient_id=patient_id, data_source=patient_df)


class TestCachingFunctionality(BaseTestCase):
    """Test cases for caching functionality in BaseDataset.set_task()."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset = MockDataset()
        self.task = MockTask()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_task_signature(self):
        """Test that set_task has the correct method signature."""
        import inspect

        sig = inspect.signature(BaseDataset.set_task)
        params = list(sig.parameters.keys())

        expected_params = ["self", "task", "num_workers", "cache_dir", "cache_format", "input_processors", "output_processors"]
        self.assertEqual(params, expected_params)

        # Check default values
        self.assertEqual(sig.parameters["task"].default, None)
        self.assertEqual(sig.parameters["num_workers"].default, 1)
        self.assertEqual(sig.parameters["cache_dir"].default, None)
        self.assertEqual(sig.parameters["cache_format"].default, "parquet")
        self.assertEqual(sig.parameters["input_processors"].default, None)
        self.assertEqual(sig.parameters["output_processors"].default, None)

    def test_set_task_no_caching(self):
        """Test set_task without caching (cache_dir=None)."""
        sample_dataset = self.dataset.set_task(self.task)

        self.assertIsInstance(sample_dataset, SampleDataset)
        self.assertEqual(len(sample_dataset), 4)  # Two patients, two samples each
        self.assertEqual(sample_dataset.dataset_name, "TestDataset")

        # Check that samples have the correct structure
        sample = sample_dataset[0]
        self.assertIn("test_attribute", sample)
        self.assertIn("test_label", sample)
        self.assertIn("patient_id", sample)

    def test_full_parquet_caching_cycle(self):
        """Test complete save and load cycle with parquet caching."""
        cache_path = Path(self.temp_dir) / f"{self.task.task_name}.parquet"

        # Step 1: First call - should generate samples and save to cache
        self.assertFalse(cache_path.exists(), "Cache file should not exist initially")

        sample_dataset_1 = self.dataset.set_task(
            self.task, cache_dir=self.temp_dir, cache_format="parquet"
        )

        # Verify cache file was created
        self.assertTrue(
            cache_path.exists(), "Cache file should be created after first call"
        )

        # Verify the sample dataset is correct
        self.assertIsInstance(sample_dataset_1, SampleDataset)
        self.assertEqual(
            len(sample_dataset_1), 4
        )  # Should have 4 samples from our mock data

        # Step 2: Second call - should load from cache (not regenerate)
        sample_dataset_2 = self.dataset.set_task(
            self.task, cache_dir=self.temp_dir, cache_format="parquet"
        )

        # Verify the loaded dataset matches the original
        self.assertIsInstance(sample_dataset_2, SampleDataset)
        self.assertEqual(len(sample_dataset_2), 4)

        # Step 3: Verify the actual cached data is correct
        # Load the parquet file directly to check its contents
        cached_df = pl.read_parquet(cache_path)
        cached_samples = cached_df.to_dicts()

        self.assertEqual(len(cached_samples), 4)

        # Verify sample content matches expected structure
        for sample in cached_samples:
            self.assertIn("test_attribute", sample)
            self.assertIn("test_label", sample)
            self.assertIn("patient_id", sample)
            self.assertIn(sample["patient_id"], ["1", "2"])
            self.assertIn(sample["test_label"], [0, 1])

    def test_full_pickle_caching_cycle(self):
        """Test complete save and load cycle with pickle caching."""
        cache_path = Path(self.temp_dir) / f"{self.task.task_name}.pickle"

        # Step 1: First call - should generate samples and save to cache
        self.assertFalse(cache_path.exists(), "Cache file should not exist initially")

        sample_dataset_1 = self.dataset.set_task(
            self.task, cache_dir=self.temp_dir, cache_format="pickle"
        )

        # Verify cache file was created
        self.assertTrue(
            cache_path.exists(), "Cache file should be created after first call"
        )

        # Verify the sample dataset is correct
        self.assertIsInstance(sample_dataset_1, SampleDataset)
        self.assertEqual(
            len(sample_dataset_1), 4
        )  # Should have 4 samples from our mock data

        # Step 2: Second call - should load from cache (not regenerate)
        sample_dataset_2 = self.dataset.set_task(
            self.task, cache_dir=self.temp_dir, cache_format="pickle"
        )

        # Verify the loaded dataset matches the original
        self.assertIsInstance(sample_dataset_2, SampleDataset)
        self.assertEqual(len(sample_dataset_2), 4)

        # Step 3: Verify the actual cached data is correct
        # Load the pickle file directly to check its contents
        import pickle

        with open(cache_path, "rb") as f:
            cached_samples = pickle.load(f)

        self.assertEqual(len(cached_samples), 4)

        # Verify sample content matches expected structure
        for sample in cached_samples:
            self.assertIn("test_attribute", sample)
            self.assertIn("test_label", sample)
            self.assertIn("patient_id", sample)
            self.assertIn(sample["patient_id"], ["1", "2"])
            self.assertIn(sample["test_label"], [0, 1])

    def test_set_task_invalid_cache_format(self):
        """Test set_task with invalid cache format."""
        # This should not raise an error during set_task call,
        # but should log a warning when trying to save
        sample_dataset = self.dataset.set_task(
            self.task, cache_dir=self.temp_dir, cache_format="invalid_format"
        )

        self.assertIsInstance(sample_dataset, SampleDataset)
        self.assertEqual(len(sample_dataset), 4)  # Generated samples

    @patch("polars.read_parquet")
    def test_set_task_cache_load_failure_fallback(self, mock_read_parquet):
        """Test fallback to generation when cache loading fails."""
        # Make read_parquet raise an exception
        mock_read_parquet.side_effect = Exception("Failed to read cache")

        # Create a dummy cache file
        cache_path = Path(self.temp_dir) / f"{self.task.task_name}.parquet"
        cache_path.touch()

        sample_dataset = self.dataset.set_task(
            self.task, cache_dir=self.temp_dir, cache_format="parquet"
        )

        # Should still work by falling back to generation
        self.assertIsInstance(sample_dataset, SampleDataset)
        self.assertEqual(len(sample_dataset), 4)  # Generated samples

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        nested_cache_dir = os.path.join(self.temp_dir, "nested", "cache", "dir")

        # Ensure the nested directory doesn't exist
        self.assertFalse(os.path.exists(nested_cache_dir))

        with patch("polars.DataFrame.write_parquet"):
            sample_dataset = self.dataset.set_task(
                self.task, cache_dir=nested_cache_dir, cache_format="parquet"
            )

        # Directory should be created
        self.assertTrue(os.path.exists(nested_cache_dir))
        self.assertIsInstance(sample_dataset, SampleDataset)


if __name__ == "__main__":
    unittest.main()
