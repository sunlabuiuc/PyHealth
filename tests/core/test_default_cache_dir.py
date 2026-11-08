"""Default cache location: warn loudly, and allow requiring an explicit cache_dir.

Without ``cache_dir``, a dataset writes processed copies of its source data
(the event table, task samples) to the user cache folder. For identified
clinical data that silently copies row-level data outside the project's
controlled storage, so it must be visible, and preventable.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dask.dataframe as dd
import pandas as pd

from pyhealth.datasets.base_dataset import REQUIRE_CACHE_DIR_ENV, BaseDataset

LOGGER = "pyhealth.datasets.base_dataset"


class MockDataset(BaseDataset):
    """Dataset that bypasses file loading for tests."""

    def load_data(self) -> dd.DataFrame:
        return dd.from_pandas(
            pd.DataFrame(
                {"patient_id": ["1"], "event_type": ["t"], "timestamp": [None]}
            ),
            npartitions=1,
        )


def _make(root: str, **kwargs) -> MockDataset:
    return MockDataset(
        root=root, tables=["t"], dataset_name="CacheDirDataset", dev=False, **kwargs
    )


class TestDefaultCacheDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=self.tmp.name,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(REQUIRE_CACHE_DIR_ENV, None)

    def test_default_cache_dir_logs_a_warning_with_the_path(self):
        with self.assertLogs(LOGGER, "WARNING") as logs:
            dataset = _make("/data/root_warn")
        self.assertTrue(str(dataset.cache_dir).startswith(self.tmp.name))
        self.assertIn(str(dataset.cache_dir), "\n".join(logs.output))
        self.assertIn(REQUIRE_CACHE_DIR_ENV, "\n".join(logs.output))

    def test_warns_once_per_path(self):
        with self.assertLogs(LOGGER, "WARNING") as logs:
            _make("/data/root_once")
            _make("/data/root_once")
        warnings = [r for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)

    def test_explicit_cache_dir_does_not_warn(self):
        with tempfile.TemporaryDirectory() as own:
            with self.assertNoLogs(LOGGER, "WARNING"):
                dataset = _make("/data/root_explicit", cache_dir=own)
            self.assertTrue(str(dataset.cache_dir).startswith(own))

    def test_required_cache_dir_raises_without_one(self):
        for value in ("1", "true", "YES"):
            with self.subTest(value=value):
                os.environ[REQUIRE_CACHE_DIR_ENV] = value
                with self.assertRaisesRegex(ValueError, "cache_dir"):
                    _make(f"/data/root_required_{value}")
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_required_cache_dir_allows_explicit_dir(self):
        os.environ[REQUIRE_CACHE_DIR_ENV] = "1"
        with tempfile.TemporaryDirectory() as own:
            dataset = _make("/data/root_required_ok", cache_dir=own)
            self.assertTrue(str(dataset.cache_dir).startswith(own))


if __name__ == "__main__":
    unittest.main()
