import tempfile
import unittest
from unittest.mock import patch

import polars as pl
import pandas as pd
import dask.dataframe as dd

from pyhealth.datasets.base_dataset import BaseDataset


class MockDataset(BaseDataset):
    """Dataset that bypasses file loading for tests."""

    def __init__(self, data: dd.DataFrame, **kwargs):
        self._data = data
        super().__init__(**kwargs)

    def load_data(self) -> dd.DataFrame:
        return self._data


class TestBaseDataset(unittest.TestCase):
    def _single_row_data(self) -> dd.DataFrame:
        return dd.from_pandas(
            pd.DataFrame(
                {
                    "patient_id": ["1"],
                    "event_type": ["test"],
                    "timestamp": [None],
                    "test/value": [0],
                }
            ),
            npartitions=1,
        )

    def test_cache_dir_varies_with_core_identifiers(self):
        base_kwargs = dict(
            tables=["table_a"],
            dataset_name="CacheDataset",
            dev=False,
        )

        with tempfile.TemporaryDirectory() as cache_root, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ):
            datasets = [
                MockDataset(
                    data=self._single_row_data(),
                    root="/data/root_a",
                    **base_kwargs,
                ),
                MockDataset(
                    data=self._single_row_data(),
                    root="/data/root_b",  # different root
                    **base_kwargs,
                ),
                MockDataset(
                    data=self._single_row_data(),
                    root="/data/root_a",
                    tables=["table_b"],  # different tables
                    dataset_name="CacheDataset",
                    dev=False,
                ),
                MockDataset(
                    data=self._single_row_data(),
                    root="/data/root_a",
                    tables=["table_a"],
                    dataset_name="OtherDataset",  # different dataset name
                    dev=False,
                ),
                MockDataset(
                    data=self._single_row_data(),
                    root="/data/root_a",
                    tables=["table_a"],
                    dataset_name="CacheDataset",
                    dev=True,  # different dev flag
                ),
            ]

            cache_dirs = [ds.cache_dir for ds in datasets]
            self.assertEqual(
                len(cache_dirs),
                len(set(cache_dirs)),
                "cache_dir should change when root/tables/dataset_name/dev change",
            )

    def test_event_df_cache_is_physically_sorted(self):
        unsorted_data = dd.from_pandas(
            pd.DataFrame(
                {
                    "patient_id": ["3", "1", "2", "1"],
                    "event_type": ["test"] * 4,
                    "timestamp": [None] * 4,
                    "test/value": [10, 20, 30, 40],
                }
            ),
            npartitions=1,
        )
        original_order = unsorted_data["patient_id"].compute().tolist()

        with tempfile.TemporaryDirectory() as cache_root, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ):
            dataset = MockDataset(
                data=unsorted_data,
                root="/data/root_sort",
                tables=["table_a"],
                dataset_name="SortingDataset",
                dev=False,
            )

            # Trigger caching of global_event_df.parquet
            _ = dataset.global_event_df
            cache_path = dataset.cache_dir / "global_event_df.parquet"
            self.assertTrue(cache_path.exists(), "global_event_df cache should be created")

            cached_df = pl.read_parquet(cache_path)
            cached_order = cached_df["patient_id"].to_list()

            self.assertNotEqual(
                cached_order, original_order, "cache should not keep the unsorted order"
            )
            self.assertEqual(
                cached_order,
                sorted(cached_order),
                "cached global_event_df parquet must be sorted by patient_id",
            )


if __name__ == "__main__":
    unittest.main()
