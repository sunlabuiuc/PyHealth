"""Proof that BaseDataset still scans parquet files.

Will's a0f1422 deleted _scan_table/_scan_parquet while MEDS still calls
_scan_parquet. These tests load a real parquet file, not just inspect source.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestP0ParquetScan(unittest.TestCase):
    def test_scan_parquet_reads_a_real_file(self):
        from pyhealth.datasets.base_dataset import BaseDataset

        self.assertTrue(callable(getattr(BaseDataset, "_scan_table")))
        self.assertTrue(callable(getattr(BaseDataset, "_scan_parquet")))

        class _Host:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.parquet"
            pd.DataFrame({"subject_id": ["a", "b"], "n": [1, 2]}).to_parquet(path)
            df = BaseDataset._scan_parquet(_Host(), str(path))
            out = df.compute()
            self.assertEqual(len(out), 2)
            self.assertIn("subject_id", out.columns)

    def test_scan_table_routes_a_parquet_file_to_the_parquet_scanner(self):
        from pyhealth.datasets.base_dataset import BaseDataset

        class _Host:
            def _scan_parquet(self, source_path):
                return f"parquet:{source_path}"

            def _scan_csv_tsv_gz(self, source_path):
                return f"csv:{source_path}"

        host = _Host()
        pq = BaseDataset._scan_table(host, "/tmp/events.parquet")
        csv = BaseDataset._scan_table(host, "/tmp/events.csv.gz")
        self.assertTrue(pq.startswith("parquet:"))
        self.assertTrue(csv.startswith("csv:"))


class TestP0AbsoluteTablePath(unittest.TestCase):
    def test_resolve_table_path_keeps_absolute(self):
        from pyhealth.datasets.base_dataset import resolve_table_path

        abs_csv = "/tmp/generated/mimic-cxr-2.0.0-metadata-pyhealth-sunlab.csv"
        resolved = resolve_table_path("/data/root", abs_csv)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.endswith("mimic-cxr-2.0.0-metadata-pyhealth-sunlab.csv"))
        self.assertEqual(resolved, str(Path(abs_csv).expanduser().resolve()))
        rel = resolve_table_path("/data/root", "hosp/patients.csv.gz")
        self.assertTrue(rel.endswith("hosp/patients.csv.gz"))
