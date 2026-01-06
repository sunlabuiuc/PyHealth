import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import dask.dataframe as dd
import torch
import json
import uuid

from tests.base import BaseTestCase
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.tasks.base_task import BaseTask
from pyhealth.datasets.sample_dataset import SampleDataset


class MockTask(BaseTask):
    """Mock task for testing purposes."""
    task_name = "test_task"
    input_schema = {"test_attribute": "raw"}
    output_schema = {"test_label": "binary"}

    def __init__(self, param=None):
        self.call_count = 0
        if param:
            self.param = param

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

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dataset = MockDataset(cache_dir=cls.temp_dir.name)

    def setUp(self):
        self.task = MockTask()
        self.cache_dir = Path(self.temp_dir.name) / "task_cache"
        self.cache_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.cache_dir)

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
        with self.dataset.set_task(
            self.task, cache_dir=self.cache_dir, cache_format="parquet"
        ) as sample_dataset:
            self.assertIsInstance(sample_dataset, SampleDataset)
            self.assertEqual(sample_dataset.dataset_name, "TestDataset")
            self.assertEqual(sample_dataset.task_name, self.task.task_name)
            self.assertEqual(len(sample_dataset), 4)
            self.assertEqual(self.task.call_count, 2)

            # Ensure intermediate cache files are created
            self.assertTrue((self.cache_dir / "task_df.ld" / "index.json").exists())

            # Cache artifacts should be present for StreamingDataset
            assert sample_dataset.input_dir.path is not None
            sample_dir = Path(sample_dataset.input_dir.path)
            self.assertTrue((sample_dir / "index.json").exists())
            self.assertTrue((sample_dir / "schema.pkl").exists())

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
        # Ensure directory is cleaned up after context exit
        self.assertFalse((sample_dir / "index.json").exists())
        self.assertFalse((sample_dir / "schema.pkl").exists())
        # Ensure intermediate cache files are still present
        self.assertTrue((self.cache_dir / "task_df.ld" / "index.json").exists())


    def test_default_cache_dir_is_used(self):
        """When cache_dir is omitted, default cache dir should be used."""
        task_params = json.dumps(
            {"call_count": 0},
            sort_keys=True,
            default=str
        )

        task_cache = self.dataset.cache_dir / "tasks" / f"{self.task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params)}"
        sample_dataset = self.dataset.set_task(self.task)

        self.assertTrue(task_cache.exists())
        self.assertTrue((task_cache / "task_df.ld" / "index.json").exists())
        self.assertTrue((self.dataset.cache_dir / "global_event_df.parquet").exists())
        self.assertEqual(len(sample_dataset), 4)

        sample_dataset.close()

    def test_reuses_existing_cache_without_regeneration(self):
        """Second call should reuse cached samples instead of recomputing."""
        sample_dataset = self.dataset.set_task(self.task, cache_dir=self.cache_dir)
        self.assertEqual(self.task.call_count, 2)

        with patch.object(
            self.task, "__call__", side_effect=AssertionError("Task should not rerun")
        ):
            cached_dataset = self.dataset.set_task(
                self.task, cache_dir=self.cache_dir, cache_format="parquet"
            )

        self.assertEqual(len(cached_dataset), 4)
        self.assertEqual(self.task.call_count, 2)

        sample_dataset.close()
        cached_dataset.close()

    def test_tasks_with_diff_param_values_get_diff_caches(self):
        sample_dataset1 = self.dataset.set_task(MockTask(param=1))
        sample_dataset2 = self.dataset.set_task(MockTask(param=2))

        task_params1 = json.dumps(
            {"call_count": 0, "param": 2},
            sort_keys=True,
            default=str
        )

        task_params2 = json.dumps(
            {"call_count": 0, "param": 2},
            sort_keys=True,
            default=str
        )

        task_cache1 = self.dataset.cache_dir / "tasks" / f"{self.task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params1)}"
        task_cache2 = self.dataset.cache_dir / "tasks" / f"{self.task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params2)}"

        self.assertTrue(task_cache1.exists())
        self.assertTrue(task_cache2.exists())
        self.assertTrue((task_cache1 / "task_df.ld" / "index.json").exists())
        self.assertTrue((task_cache2 / "task_df.ld" / "index.json").exists())
        self.assertTrue((self.dataset.cache_dir / "global_event_df.parquet").exists())
        self.assertEqual(len(sample_dataset1), 4)
        self.assertEqual(len(sample_dataset2), 4)

        sample_dataset1.close()
        sample_dataset2.close()

    def test_clear_cache_removes_all_caches(self):
        """Test that clear_cache removes entire dataset cache."""
        # Create cache by accessing global_event_df
        _ = self.dataset.global_event_df
        cache_path = self.dataset.cache_dir

        # Verify cache exists
        self.assertTrue(cache_path.exists())
        self.assertTrue((cache_path / "global_event_df.parquet").exists())

        # Create a task cache
        sample_dataset = self.dataset.set_task(self.task)
        task_cache_dir = self.dataset._get_task_cache_dir(self.task)
        self.assertTrue(task_cache_dir.exists())

        # Clear entire cache
        self.dataset.clear_cache()

        # Verify everything is removed
        self.assertFalse(cache_path.exists())

        # Verify cached attributes are reset
        self.assertIsNone(self.dataset._cache_dir)
        self.assertIsNone(self.dataset._global_event_df)
        self.assertIsNone(self.dataset._unique_patient_ids)

        sample_dataset.close()

    def test_clear_cache_handles_nonexistent_cache(self):
        """Test that clear_cache handles the case when no cache exists."""
        # Create a fresh dataset without any cache
        fresh_dataset = MockDataset(cache_dir=self.cache_dir / "fresh")

        # This should not raise an error
        fresh_dataset.clear_cache()

    def test_clear_task_cache_removes_only_specified_task(self):
        """Test that clear_task_cache removes only the specified task cache."""
        # Create two different tasks
        task1 = MockTask(param=1)
        task2 = MockTask(param=2)

        # Set both tasks to create their caches
        sample_dataset1 = self.dataset.set_task(task1)
        sample_dataset2 = self.dataset.set_task(task2)

        # Get cache directories
        task1_cache_dir = self.dataset._get_task_cache_dir(task1)
        task2_cache_dir = self.dataset._get_task_cache_dir(task2)
        global_cache = self.dataset.cache_dir / "global_event_df.parquet"

        # Verify all caches exist
        self.assertTrue(task1_cache_dir.exists())
        self.assertTrue(task2_cache_dir.exists())
        self.assertTrue(global_cache.exists())

        # Clear only task1 cache
        self.dataset.clear_task_cache(task1)

        # Verify task1 cache is removed but others remain
        self.assertFalse(task1_cache_dir.exists())
        self.assertTrue(task2_cache_dir.exists())
        self.assertTrue(global_cache.exists())

        sample_dataset1.close()
        sample_dataset2.close()

    def test_clear_task_cache_handles_nonexistent_cache(self):
        """Test that clear_task_cache handles the case when task cache doesn't exist."""
        task = MockTask(param=999)

        # This should not raise an error even though cache doesn't exist
        self.dataset.clear_task_cache(task)

    def test_get_task_cache_dir_consistency(self):
        """Test that _get_task_cache_dir produces consistent paths."""
        task = MockTask(param=42)

        # Get path multiple times
        path1 = self.dataset._get_task_cache_dir(task)
        path2 = self.dataset._get_task_cache_dir(task)

        # Should be identical
        self.assertEqual(path1, path2)

        # Should match the pattern used in set_task
        task_params = json.dumps(
            vars(task),
            sort_keys=True,
            default=str
        )
        expected_path = self.dataset.cache_dir / "tasks" / f"{task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params)}"
        self.assertEqual(path1, expected_path)


if __name__ == "__main__":
    unittest.main()
