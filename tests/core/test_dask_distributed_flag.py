"""Opt out of the Dask distributed cluster for event caching.

``BaseDataset(use_dask_distributed=False)`` makes ``_event_transform`` write
the cache with the local Dask scheduler instead of starting a distributed
cluster, for sandboxes where the cluster cannot start. The default ``True``
keeps the distributed path (with ``dashboard_address=None``) unchanged, and
the flag does not affect the cache key or the written contents.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
      python -m pytest tests/core/test_dask_distributed_flag.py -q
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dask.dataframe as dd
import pandas as pd


class TestDaskDistributedFlag(unittest.TestCase):
    def _dataset(self, cache_root: str, use_dask_distributed: bool = True, dev=False):
        from pyhealth.datasets.base_dataset import BaseDataset

        class MockDataset(BaseDataset):
            def __init__(self, data: dd.DataFrame, **kwargs):
                self._data = data
                super().__init__(**kwargs)

            def load_data(self) -> dd.DataFrame:
                return self._data

        data = dd.from_pandas(
            pd.DataFrame(
                {
                    # Unsorted on purpose: both paths sort by patient_id.
                    "patient_id": ["2", "1", "3", "2"],
                    "event_type": ["test", "test", "test", "test"],
                    "timestamp": [None, None, None, None],
                    "test/value": [2, 0, 1, 3],
                }
            ),
            npartitions=2,
        )
        with patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=cache_root,
        ):
            dataset = MockDataset(
                data=data,
                root="/data/root_a",
                tables=["table_a"],
                dataset_name="FlagDataset",
                dev=dev,
                num_workers=2,
                use_dask_distributed=use_dask_distributed,
            )
        return dataset

    def _cached_frame(self, out):
        frame = dd.read_parquet(out).compute()
        return (
            frame.sort_values(list(frame.columns))
            .reset_index(drop=True)
            .astype({"test/value": "int64"})
        )

    def test_flag_defaults_to_distributed(self):
        import inspect

        from pyhealth.datasets.base_dataset import BaseDataset

        sig = inspect.signature(BaseDataset.__init__)
        self.assertIn("use_dask_distributed", sig.parameters)
        self.assertTrue(sig.parameters["use_dask_distributed"].default)

    def test_false_bypasses_cluster(self):
        with tempfile.TemporaryDirectory() as cache_root:
            dataset = self._dataset(cache_root, use_dask_distributed=False)
            self.assertFalse(dataset.use_dask_distributed)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "events.parquet"
                with patch(
                    "pyhealth.datasets.base_dataset.DaskCluster",
                    side_effect=AssertionError("cluster must not start"),
                ):
                    dataset._event_transform(out)
                self.assertTrue(out.exists())

    def test_true_uses_cluster(self):
        with tempfile.TemporaryDirectory() as cache_root:
            dataset = self._dataset(cache_root, use_dask_distributed=True)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "events.parquet"
                with patch(
                    "pyhealth.datasets.base_dataset.DaskCluster",
                    side_effect=RuntimeError("cluster entry reached"),
                ):
                    with self.assertRaises(RuntimeError):
                        dataset._event_transform(out)

    def test_flag_does_not_change_cache_key(self):
        with tempfile.TemporaryDirectory() as cache_root:
            on = self._dataset(cache_root, use_dask_distributed=True)
            off = self._dataset(cache_root, use_dask_distributed=False)
            self.assertEqual(on.cache_dir, off.cache_dir)

    def test_flag_path_matches_distributed_contents(self):
        for dev in (False, True):
            with tempfile.TemporaryDirectory() as cache_root:
                local = self._dataset(
                    cache_root, use_dask_distributed=False, dev=dev
                )
                dist = self._dataset(
                    cache_root, use_dask_distributed=True, dev=dev
                )
                # Same key so the same directory would be reused; write the
                # two arms to separate output dirs to compare contents.
                with tempfile.TemporaryDirectory() as tmp:
                    flag_out = Path(tmp) / "flag.parquet"
                    dist_out = Path(tmp) / "dist.parquet"
                    local._event_transform(flag_out)
                    dist._event_transform(dist_out)
                    pd.testing.assert_frame_equal(
                        self._cached_frame(flag_out),
                        self._cached_frame(dist_out),
                    )
