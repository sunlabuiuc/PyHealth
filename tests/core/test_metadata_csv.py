"""Shared metadata-CSV and subject-id helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets.metadata_csv import (
    metadata_cache_dir,
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)
from pyhealth.tasks.subject_ids import normalize_prefixed_subject


class TestMetadataCsv(unittest.TestCase):
    def test_cache_dir_is_keyed_by_dataset_and_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir()
            b.mkdir()
            with patch("pathlib.Path.home", return_value=home):
                self.assertNotEqual(
                    metadata_cache_dir(a, "demo"), metadata_cache_dir(b, "demo")
                )
                self.assertNotEqual(
                    metadata_cache_dir(a, "demo"), metadata_cache_dir(a, "other")
                )
                self.assertEqual(metadata_cache_dir(a, "demo").parent.name, "demo")

    def test_write_prefers_root_and_exists_sees_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = pd.DataFrame({"patient_id": ["a"]})
            with patch("pathlib.Path.home", return_value=root / "home"):
                self.assertFalse(metadata_csv_exists(root, "demo", "demo.csv"))
                path = write_metadata_csv(df, root, "demo", "demo.csv")
                self.assertEqual(path, root / "demo.csv")
                self.assertTrue(metadata_csv_exists(root, "demo", "demo.csv"))
                self.assertEqual(
                    resolve_metadata_root(root, "demo", "demo.csv"), str(root)
                )

    def test_write_falls_back_to_cache_when_root_is_not_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            root.mkdir()
            home = Path(tmp) / "home"
            df = pd.DataFrame({"patient_id": ["a"]})
            real_to_csv = pd.DataFrame.to_csv

            def fail_in_root(self_df, target, *args, **kwargs):
                if Path(target).parent == root:
                    raise PermissionError("read-only")
                return real_to_csv(self_df, target, *args, **kwargs)

            with (
                patch("pathlib.Path.home", return_value=home),
                patch.object(pd.DataFrame, "to_csv", fail_in_root),
            ):
                path = write_metadata_csv(df, root, "demo", "demo.csv")
                self.assertEqual(path.parent, metadata_cache_dir(root, "demo"))
                self.assertTrue(path.is_file())
                self.assertFalse((root / "demo.csv").exists())
                # BaseDataset must now read from the cache directory.
                self.assertEqual(
                    resolve_metadata_root(root, "demo", "demo.csv"),
                    str(metadata_cache_dir(root, "demo")),
                )


class TestNormalizePrefixedSubject(unittest.TestCase):
    def test_separator_and_width(self):
        for raw in ("7", "sub7", "SUB-7", "sub-007"):
            self.assertEqual(normalize_prefixed_subject(raw, "sub", 3, "-"), "sub-007")

    def test_no_separator(self):
        for raw in ("2", "CHB2", "chb02"):
            self.assertEqual(normalize_prefixed_subject(raw, "chb", 2), "chb02")

    def test_non_numeric_ids_pass_through_lowercased(self):
        self.assertEqual(
            normalize_prefixed_subject("Cohort-A", "sub", 3, "-"), "cohort-a"
        )
        self.assertEqual(normalize_prefixed_subject("chb-1", "chb", 2), "chb-1")


if __name__ == "__main__":
    unittest.main()
