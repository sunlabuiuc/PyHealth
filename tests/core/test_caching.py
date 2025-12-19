import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import polars as pl
import dask.dataframe as dd
import torch

from tests.base import BaseTestCase
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.tasks.base_task import BaseTask
from pyhealth.datasets.sample_dataset import SampleDataset


class MockTask(BaseTask):
    """Mock task for testing purposes."""

    def __init__(self, task_name="test_task"):
        self.task_name = task_name
        self.input_schema = {"test_attribute": "raw"}
        self.output_schema = {"test_label": "binary"}
        self.call_count = 0

    def __call__(self, patient):
        """Return mock samples based on patient data."""
        # Extract patient's test data from the patient's data source
        patient_data = patient.data_source
        self.call_count += 1

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

    def __init__(self, cache_dir: str | Path | None = None):
        super().__init__(
            root="",
            tables=[],
            dataset_name="TestDataset",
            cache_dir=cache_dir,
            dev=False,
        )

    def load_data(self) -> dd.DataFrame:
        import pandas as pd

        return dd.from_pandas(
            pd.DataFrame(
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
            ),
            npartitions=1,
        )


class TestCachingFunctionality(BaseTestCase):
    """Test cases for caching functionality in BaseDataset.set_task()."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dataset = MockDataset(cache_dir=self.temp_dir)
        self.task = MockTask()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _task_cache_dir(self) -> Path:
        cache_dir = self.temp_dir / "task_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def test_set_task_signature(self):
        """Test that set_task has the correct method signature."""
        import inspect

        sig = inspect.signature(BaseDataset.set_task)
        params = list(sig.parameters.keys())

        expected_params = [
            "self",
            "task",
            "num_workers",
            "cache_dir",
            "cache_format",
            "input_processors",
            "output_processors",
        ]
        self.assertEqual(params, expected_params)

        # Check default values
        self.assertEqual(sig.parameters["task"].default, None)
        self.assertEqual(sig.parameters["num_workers"].default, None)
        self.assertEqual(sig.parameters["cache_dir"].default, None)
        self.assertEqual(sig.parameters["cache_format"].default, "parquet")
        self.assertEqual(sig.parameters["input_processors"].default, None)
        self.assertEqual(sig.parameters["output_processors"].default, None)

    def test_set_task_writes_cache_and_metadata(self):
        """Ensure set_task materializes cache files and schema metadata."""
        cache_dir = self._task_cache_dir()
        sample_dataset = self.dataset.set_task(
            self.task, cache_dir=cache_dir, cache_format="parquet"
        )

        self.assertIsInstance(sample_dataset, SampleDataset)
        self.assertEqual(sample_dataset.dataset_name, "TestDataset")
        self.assertEqual(sample_dataset.task_name, self.task.task_name)
        self.assertEqual(len(sample_dataset), 4)
        self.assertEqual(self.task.call_count, 2)

        # Cache artifacts should be present for StreamingDataset
        self.assertTrue((cache_dir / "index.json").exists())
        self.assertTrue((cache_dir / "schema.pkl").exists())

        # Check processed sample structure and metadata persisted
        sample = sample_dataset[0]
        self.assertIn("test_attribute", sample)
        self.assertIn("test_label", sample)
        self.assertIn("patient_id", sample)
        self.assertIsInstance(sample["test_label"], torch.Tensor)
        self.assertIn("test_attribute", sample_dataset.input_processors)
        self.assertIn("test_label", sample_dataset.output_processors)
        self.assertEqual(set(sample_dataset.patient_to_index), {"1", "2"})
        self.assertTrue(
            all(len(indexes) == 2 for indexes in sample_dataset.patient_to_index.values())
        )
        self.assertEqual(sample_dataset.record_to_index, {})

    def test_default_cache_dir_is_used(self):
        """When cache_dir is omitted, default cache dir should be used."""
        task_cache = self.dataset.cache_dir / "tasks" / self.task.task_name
        sample_dataset = self.dataset.set_task(self.task)

        self.assertTrue(task_cache.exists())
        self.assertTrue((task_cache / "index.json").exists())
        self.assertTrue((self.dataset.cache_dir / "global_event_df.parquet").exists())
        self.assertEqual(len(sample_dataset), 4)

    def test_reuses_existing_cache_without_regeneration(self):
        """Second call should reuse cached samples instead of recomputing."""
        cache_dir = self._task_cache_dir()
        _ = self.dataset.set_task(self.task, cache_dir=cache_dir)
        self.assertEqual(self.task.call_count, 2)

        with patch.object(
            self.task, "__call__", side_effect=AssertionError("Task should not rerun")
        ):
            cached_dataset = self.dataset.set_task(
                self.task, cache_dir=cache_dir, cache_format="parquet"
            )

        self.assertEqual(len(cached_dataset), 4)
        self.assertEqual(self.task.call_count, 2)


if __name__ == "__main__":
    unittest.main()
