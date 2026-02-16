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


class MockTask2(BaseTask):
    """Second mock task with a different output schema than the first"""
    task_name = "test_task"
    input_schema = {"test_attribute": "raw"}
    output_schema = {"test_label": "multiclass"}

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

    def __init__(self, root: str = "", tables = [], dataset_name = "TestDataset", cache_dir: str | Path | None = None, dev = False):
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            dev=dev,
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
            "input_processors",
            "output_processors",
        ]
        self.assertEqual(params, expected_params)

        # Check default values
        self.assertEqual(sig.parameters["task"].default, None)
        self.assertEqual(sig.parameters["num_workers"].default, None)
        self.assertEqual(sig.parameters["input_processors"].default, None)
        self.assertEqual(sig.parameters["output_processors"].default, None)

    def test_set_task_writes_cache_and_metadata(self):
        """Ensure set_task materializes cache files and schema metadata."""
        with self.dataset.set_task(self.task) as sample_dataset:
            self.assertIsInstance(sample_dataset, SampleDataset)
            self.assertEqual(sample_dataset.dataset_name, "TestDataset")
            self.assertEqual(sample_dataset.task_name, self.task.task_name)
            self.assertEqual(len(sample_dataset), 4)
            self.assertEqual(self.task.call_count, 2)

            # Ensure intermediate cache files are created in default location
            task_params = json.dumps(
                {
                    **vars(self.task),
                    "input_schema": self.task.input_schema,
                    "output_schema": self.task.output_schema,
                },
                sort_keys=True,
                default=str
            )
            task_cache_dir = self.dataset.cache_dir / "tasks" / f"{self.task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params)}"
            self.assertTrue((task_cache_dir / "task_df.ld" / "index.json").exists())

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
        self.assertTrue((task_cache_dir / "task_df.ld" / "index.json").exists())


    def test_default_cache_dir_is_used(self):
        """When cache_dir is omitted, default cache dir should be used."""
        task_params = json.dumps(
            {"input_schema": {"test_attribute": "raw"}, "output_schema": {"test_label": "binary"}, "call_count": 0},
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
        sample_dataset = self.dataset.set_task(self.task)
        self.assertEqual(self.task.call_count, 2)

        with patch.object(
            self.task, "__call__", side_effect=AssertionError("Task should not rerun")
        ):
            cached_dataset = self.dataset.set_task(self.task)

        self.assertEqual(len(cached_dataset), 4)
        self.assertEqual(self.task.call_count, 2)

        sample_dataset.close()
        cached_dataset.close()

    def test_tasks_with_diff_param_values_get_diff_caches(self):
        sample_dataset1 = self.dataset.set_task(MockTask(param=1))
        sample_dataset2 = self.dataset.set_task(MockTask(param=2))

        self.assertNotEqual(sample_dataset1.path, sample_dataset2.path)

        task_params1 = json.dumps(
            {"input_schema": {"test_attribute": "raw"}, "output_schema": {"test_label": "binary"}, "call_count": 0, "param": 1},
            sort_keys=True,
            default=str
        )

        task_params2 = json.dumps(
            {"input_schema": {"test_attribute": "raw"}, "output_schema": {"test_label": "binary"}, "call_count": 0, "param": 2},
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

    def test_tasks_with_diff_output_schemas_get_diff_caches(self):
        sample_dataset1 = self.dataset.set_task(MockTask())
        sample_dataset2 = self.dataset.set_task(MockTask2())

        self.assertNotEqual(sample_dataset1.path, sample_dataset2.path)

        task_params1 = json.dumps(
            {"input_schema": {"test_attribute": "raw"}, "output_schema": {"test_label": "binary"}, "call_count": 0},
            sort_keys=True,
            default=str
        )

        task_params2 = json.dumps(
            {"input_schema": {"test_attribute": "raw"}, "output_schema": {"test_label": "multiclass"}, "call_count": 0},
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

    def test_datasets_with_diff_roots_get_diff_caches(self):
        dataset1 = MockDataset(root=tempfile.TemporaryDirectory().name, cache_dir=self.temp_dir.name)
        dataset2 = MockDataset(root=tempfile.TemporaryDirectory().name, cache_dir=self.temp_dir.name)

        self.assertNotEqual(dataset1.cache_dir, dataset2.cache_dir)

    def test_datasets_with_diff_tables_get_diff_caches(self):
        dataset1 = MockDataset(tables=["one", "two", ], cache_dir=self.temp_dir.name)
        dataset2 = MockDataset(tables=["one", "two", "three"], cache_dir=self.temp_dir.name)
        dataset3 = MockDataset(tables=["one", "three"], cache_dir=self.temp_dir.name)
        dataset4 = MockDataset(tables=[], cache_dir=self.temp_dir.name)

        caches = [dataset1.cache_dir, dataset2.cache_dir, dataset3.cache_dir, dataset4.cache_dir]

        self.assertEqual(len(caches), len(set(caches)))

    def test_datasets_with_diff_names_get_diff_caches(self):
        dataset1 = MockDataset(dataset_name="one", cache_dir=self.temp_dir.name)
        dataset2 = MockDataset(dataset_name="two", cache_dir=self.temp_dir.name)

        self.assertNotEqual(dataset1.cache_dir, dataset2.cache_dir)

    def test_datasets_with_diff_dev_values_get_diff_caches(self):
        dataset1 = MockDataset(dev=True, cache_dir=self.temp_dir.name)
        dataset2 = MockDataset(dev=False, cache_dir=self.temp_dir.name)

        self.assertNotEqual(dataset1.cache_dir, dataset2.cache_dir)

if __name__ == "__main__":
    unittest.main()
