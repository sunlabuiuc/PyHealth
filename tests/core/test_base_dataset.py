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

    def test_empty_string_handling(self):
        import os
        from dataclasses import dataclass
        from typing import List

        # Create a temporary directory and a CSV file
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "data.csv")
            # Create CSV with empty strings
            # pid, time, val
            # p1, 2020-01-01, v1
            # p2, "", v2  -> missing time
            # "", 2020-01-02, v3 -> missing pid
            # p3, 2020-01-03, "" -> missing val
            with open(csv_path, "w") as f:
                f.write("pid,time,val\n")
                f.write("p1,2020-01-01,v1\n")
                f.write("p2,,v2\n")
                f.write(",2020-01-02,v3\n")
                f.write("p3,2020-01-03,\n")

            @dataclass
            class TableConfig:
                file_path: str
                patient_id: str
                timestamp: str
                timestamp_format: str
                attributes: List[str]
                join: List = None

                def __post_init__(self):
                    if self.join is None:
                        self.join = []

            @dataclass
            class Config:
                tables: dict

            config = Config(
                tables={
                    "table1": TableConfig(
                        file_path="data.csv",
                        patient_id="pid",
                        timestamp="time",
                        timestamp_format="%Y-%m-%d",
                        attributes=["val"]
                    )
                }
            )
            
            class ConcreteDataset(BaseDataset):
                pass

            dataset = ConcreteDataset(
                root=tmp_dir,
                tables=["table1"],
                dataset_name="TestDataset",
                cache_dir=tmp_dir
            )
            dataset.config = config

            # Load data
            # load_table returns a dask dataframe
            df = dataset.load_table("table1")
            # Compute to get pandas dataframe
            pdf = df.compute()

            # Verify
            # Row 0: p1, 2020-01-01, v1
            self.assertEqual(pdf.iloc[0]["patient_id"], "p1")
            self.assertEqual(pdf.iloc[0]["timestamp"], pd.Timestamp("2020-01-01"))
            self.assertEqual(pdf.iloc[0]["table1/val"], "v1")

            # Row 1: p2, NaT, v2
            self.assertEqual(pdf.iloc[1]["patient_id"], "p2")
            self.assertTrue(pd.isna(pdf.iloc[1]["timestamp"]))
            self.assertEqual(pdf.iloc[1]["table1/val"], "v2")

            # Row 2: <NA>, 2020-01-02, v3
            self.assertTrue(pd.isna(pdf.iloc[2]["patient_id"]))
            self.assertEqual(pdf.iloc[2]["timestamp"], pd.Timestamp("2020-01-02"))
            self.assertEqual(pdf.iloc[2]["table1/val"], "v3")

            # Row 3: p3, 2020-01-03, <NA>
            self.assertEqual(pdf.iloc[3]["patient_id"], "p3")
            self.assertEqual(pdf.iloc[3]["timestamp"], pd.Timestamp("2020-01-03"))
            self.assertTrue(pd.isna(pdf.iloc[3]["table1/val"]))

    def test_empty_string_handling_composite_timestamp(self):
        import os
        from dataclasses import dataclass
        from typing import List

        # Create a temporary directory and a CSV file
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "data_composite.csv")
            # Create CSV with empty strings in composite timestamp fields
            # pid, year, month, day, val
            # p1, 2020, 01, 01, v1 -> 2020-01-01
            # p2, 2020, , 02, v2   -> missing month -> NaT
            # p3, , 01, 03, v3     -> missing year -> NaT
            with open(csv_path, "w") as f:
                f.write("pid,year,month,day,val\n")
                f.write("p1,2020,01,01,v1\n")
                f.write("p2,2020,,02,v2\n")
                f.write("p3,,01,03,v3\n")

            @dataclass
            class TableConfig:
                file_path: str
                patient_id: str
                timestamp: List[str]
                timestamp_format: str
                attributes: List[str]
                join: List = None

                def __post_init__(self):
                    if self.join is None:
                        self.join = []

            @dataclass
            class Config:
                tables: dict

            config = Config(
                tables={
                    "table1": TableConfig(
                        file_path="data_composite.csv",
                        patient_id="pid",
                        timestamp=["year", "month", "day"],
                        timestamp_format="%Y%m%d",
                        attributes=["val"]
                    )
                }
            )
            
            class ConcreteDataset(BaseDataset):
                pass

            dataset = ConcreteDataset(
                root=tmp_dir,
                tables=["table1"],
                dataset_name="TestDatasetComposite",
                cache_dir=tmp_dir
            )
            dataset.config = config

            # Load data
            df = dataset.load_table("table1")
            pdf = df.compute()

            # Verify
            # Row 0: p1, 2020-01-01
            self.assertEqual(pdf.iloc[0]["patient_id"], "p1")
            self.assertEqual(pdf.iloc[0]["timestamp"], pd.Timestamp("2020-01-01"))
            self.assertEqual(pdf.iloc[0]["table1/val"], "v1")

            # Row 1: p2, NaT
            self.assertEqual(pdf.iloc[1]["patient_id"], "p2")
            self.assertTrue(pd.isna(pdf.iloc[1]["timestamp"]))
            self.assertEqual(pdf.iloc[1]["table1/val"], "v2")

            # Row 2: p3, NaT
            self.assertEqual(pdf.iloc[2]["patient_id"], "p3")
            self.assertTrue(pd.isna(pdf.iloc[2]["timestamp"]))
            self.assertEqual(pdf.iloc[2]["table1/val"], "v3")


if __name__ == "__main__":
    unittest.main()
