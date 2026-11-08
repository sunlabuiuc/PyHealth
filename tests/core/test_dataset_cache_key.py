"""The dataset cache key must change when the source data or config changes.

Before, the key covered only {root, tables, dataset_name, dev}: rewriting a
source file at the same path, or editing the YAML config, silently reused the
stale cached event table.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

from pyhealth.datasets import BaseDataset

CONFIG = """version: "1.0"
tables:
  events:
    file_path: "events.csv"
    patient_id: "patient_id"
    timestamp: "time"
    attributes:
      - "code"
"""


def _write_csv(path: Path, n_patients: int) -> None:
    rows = ["patient_id,time,code"] + [
        f"p{i},2020-01-0{1 + i % 9} 00:00:00,c{i}" for i in range(n_patients)
    ]
    path.write_text("\n".join(rows) + "\n")


class TestDatasetCacheKey(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "data"
        self.cache = Path(tmp.name) / "cache"
        self.root.mkdir()
        self.config = self.root / "config.yaml"
        self.config.write_text(CONFIG)
        self.csv = self.root / "events.csv"
        _write_csv(self.csv, n_patients=3)

    def _dataset(self) -> BaseDataset:
        return BaseDataset(
            root=str(self.root),
            tables=["events"],
            dataset_name="CacheKeyDataset",
            config_path=str(self.config),
            cache_dir=str(self.cache),
        )

    def _n_patients(self, dataset: BaseDataset) -> int:
        return dataset.global_event_df.select("patient_id").unique().collect().height

    def test_rewritten_source_file_is_reloaded(self):
        first = self._dataset()
        self.assertEqual(self._n_patients(first), 3)

        time.sleep(0.01)  # make sure the modification time moves
        _write_csv(self.csv, n_patients=5)  # same path, new content

        second = self._dataset()
        self.assertNotEqual(first.cache_dir, second.cache_dir)
        self.assertEqual(self._n_patients(second), 5)

    def test_unchanged_source_reuses_the_cache(self):
        first = self._dataset()
        _ = first.global_event_df
        second = self._dataset()
        self.assertEqual(first.cache_dir, second.cache_dir)

    def test_edited_config_changes_the_cache(self):
        first = self._dataset()
        self.config.write_text(CONFIG.replace('      - "code"\n', '      - "code"\n      - "time"\n'))
        second = self._dataset()
        self.assertNotEqual(first.cache_dir, second.cache_dir)

    def test_touching_mtime_only_changes_the_cache(self):
        # Size + modification time is the signal (like make); content is not hashed.
        first = self._dataset()
        stat = self.csv.stat()
        os.utime(self.csv, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        second = self._dataset()
        self.assertNotEqual(first.cache_dir, second.cache_dir)


if __name__ == "__main__":
    unittest.main()
