import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import polars as pl
import psutil
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

    def test_dev_mode_keeps_1000_patients_across_partitions(self):
        # The patients are spread over many partitions. Dev mode must keep
        # 1000 of them, not only those in the first partition.
        data = dd.from_pandas(
            pd.DataFrame(
                {
                    "patient_id": [str(i) for i in range(1500)],
                    "event_type": ["test"] * 1500,
                    "timestamp": [None] * 1500,
                    "test/value": list(range(1500)),
                }
            ),
            npartitions=10,
        )
        with tempfile.TemporaryDirectory() as cache_root, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ):
            dataset = MockDataset(
                data=data,
                root="/data/root_dev",
                tables=["table_a"],
                dataset_name="DevDataset",
                dev=True,
            )
            patients = dataset.global_event_df.select("patient_id").collect()
            self.assertEqual(patients["patient_id"].n_unique(), 1000)

    @unittest.skipIf(os.name == "nt", "open file descriptor counts are POSIX-only")
    def test_event_df_build_does_not_leak_file_descriptors(self):
        # Each build starts and closes a dask LocalCluster. A leak here exhausts the
        # macOS default `ulimit -n` (256) partway through the test suite.
        proc = psutil.Process()

        def build(i: int) -> None:
            with tempfile.TemporaryDirectory() as cache_root, patch(
                "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
                return_value=cache_root,
            ):
                dataset = MockDataset(
                    data=self._single_row_data(),
                    root=f"/data/root_fd_{i}",
                    tables=["table_a"],
                    dataset_name="FdLeakDataset",
                    dev=False,
                )
                _ = dataset.global_event_df

        build(0)  # warm up one-time imports and handles
        before = proc.num_fds()
        for i in range(1, 5):
            build(i)
        leaked = proc.num_fds() - before
        self.assertLess(
            leaked, 4, f"{leaked} file descriptors leaked across 4 dataset builds"
        )

    def test_interrupted_event_df_build_is_not_reused(self):
        # A build interrupted (Ctrl-C) after writing part of the table must not
        # leave that part at the cache path, where the next run would reuse it.
        data = dd.from_pandas(
            pd.DataFrame(
                {
                    "patient_id": ["1", "2", "3"],
                    "event_type": ["test"] * 3,
                    "timestamp": [None] * 3,
                    "test/value": [1, 2, 3],
                }
            ),
            npartitions=1,
        )

        with tempfile.TemporaryDirectory() as cache_root, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ):
            def dataset():
                return MockDataset(
                    data=data,
                    root="/data/root_interrupt",
                    tables=["table_a"],
                    dataset_name="InterruptDataset",
                    dev=False,
                )

            first = dataset()
            cache_path = first.cache_dir / "global_event_df.parquet"

            def interrupted(output_dir):
                # one part file is on disk when Ctrl-C arrives
                output_dir.mkdir(parents=True)
                data.head(1).to_parquet(output_dir / "part.0.parquet", index=False)
                self.assertFalse(cache_path.exists())  # nothing published yet
                raise KeyboardInterrupt

            with patch.object(first, "_event_transform", interrupted), \
                    self.assertRaises(KeyboardInterrupt):
                _ = first.global_event_df
            self.assertFalse(cache_path.exists())
            self.assertEqual(list(first.cache_dir.glob("*.building")), [])

            second = dataset()
            patients = second.global_event_df.collect()["patient_id"].to_list()
            self.assertEqual(sorted(patients), ["1", "2", "3"])
            self.assertEqual(list(second.cache_dir.glob("*.building")), [])

    def test_concurrent_event_df_builds_run_once(self):
        # Two datasets on the same cache: while the first builds (holding the
        # lock), the second waits, then finds the published table and does not
        # build it again.
        data = dd.from_pandas(
            pd.DataFrame(
                {
                    "patient_id": ["1", "2", "3"],
                    "event_type": ["test"] * 3,
                    "timestamp": [None] * 3,
                    "test/value": [1, 2, 3],
                }
            ),
            npartitions=1,
        )

        with tempfile.TemporaryDirectory() as cache_root, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ):
            def dataset():
                return MockDataset(
                    data=data,
                    root="/data/root_concurrent",
                    tables=["table_a"],
                    dataset_name="ConcurrentDataset",
                    dev=False,
                )

            first, second = dataset(), dataset()
            self.assertEqual(first.cache_dir, second.cache_dir)
            building, release = threading.Event(), threading.Event()
            builds, errors = [], []

            def transform(name):
                def build(output_dir):
                    builds.append(name)
                    building.set()
                    self.assertTrue(release.wait(30))
                    output_dir.mkdir(parents=True)
                    data.compute().to_parquet(
                        output_dir / "part.0.parquet", index=False
                    )

                return build

            def read(ds):
                try:
                    _ = ds.global_event_df
                except BaseException as e:  # surfaced by the main thread
                    errors.append(e)

            with patch.object(first, "_event_transform", transform("first")), \
                    patch.object(second, "_event_transform", transform("second")):
                a = threading.Thread(target=read, args=(first,))
                a.start()
                self.assertTrue(building.wait(30))
                b = threading.Thread(target=read, args=(second,))
                b.start()
                b.join(1.0)
                self.assertTrue(b.is_alive())  # waiting for the lock
                release.set()
                a.join(30)
                b.join(30)

            self.assertEqual(errors, [])
            self.assertEqual(builds, ["first"])
            patients = second.global_event_df.collect()["patient_id"].to_list()
            self.assertEqual(sorted(patients), ["1", "2", "3"])
            self.assertEqual(list(second.cache_dir.glob("*.building")), [])

    def test_event_df_builds_in_notebook_without_ipywidgets(self):
        # In a Jupyter kernel dask's progress() draws an ipywidgets bar, but
        # ipywidgets is not a PyHealth dependency. Simulate a kernel without it;
        # in_notebook() also switches dask to threaded workers, as in Jupyter.
        # distributed.scheduler is where a progress feed cancelled by closing
        # the cluster too early logs its CancelledError traceback, so this also
        # fails without the wait after the build.
        with tempfile.TemporaryDirectory() as cache_root, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ), patch(
            "pyhealth.datasets.base_dataset.in_notebook", return_value=True
        ), patch(
            "distributed.diagnostics.progressbar.is_kernel", return_value=True
        ), patch.dict(sys.modules, {"ipywidgets": None}), self.assertNoLogs(
            "distributed.scheduler", level="ERROR"
        ):
            dataset = MockDataset(
                data=self._single_row_data(),
                root="/data/root_no_ipywidgets",
                tables=["table_a"],
                dataset_name="NoIpywidgetsDataset",
                dev=False,
            )
            _ = dataset.global_event_df
            self.assertTrue(
                (dataset.cache_dir / "global_event_df.parquet").exists()
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

    def test_null_patient_id_unique_across_partitions(self):
        """Regression test for the Dask per-partition reset_index bug.

        A table config with ``patient_id: null`` must give every row a
        globally unique ``patient_id``, even when the frame spans multiple
        Dask partitions. The old code used ``df.reset_index(drop=True)`` +
        ``df.index``, which Dask resets per partition, so rows from different
        partitions collided onto the same id and were silently merged into
        one "patient".
        """
        from pyhealth.datasets.configs.config import DatasetConfig

        class _NullPatientDataset(BaseDataset):
            def __init__(self, config, root="/tmp/x"):
                self.config = config
                self.root = root

        ds = _NullPatientDataset(
            DatasetConfig(
                version="1.0",
                tables={"t": {"file_path": "t.csv", "attributes": ["value"]}},
            )
        )
        # 6 rows across 3 partitions; the buggy code produced ids [0,1,0,1,0,1].
        frame = dd.from_pandas(pd.DataFrame({"value": list(range(6))}), npartitions=3)
        with patch.object(ds, "_scan_table", return_value=frame):
            out = ds.load_table("t").compute()
        self.assertEqual(len(out), 6)
        self.assertEqual(out["patient_id"].nunique(), 6)
        self.assertEqual(sorted(out["patient_id"].tolist()), [str(i) for i in range(6)])


if __name__ == "__main__":
    unittest.main()
