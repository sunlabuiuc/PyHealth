"""Tests for the CMMD mammography dataset loader."""

import builtins
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

try:
    import openpyxl
except ImportError:  # pragma: no cover - optional dependency
    openpyxl = None

from pyhealth.datasets import CMMDDataset
from tests.core.cmmd_test_utils import CLINICAL, build_cmmd_root, pydicom


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCMMDDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "cmmd"
        self.root.mkdir()
        build_cmmd_root(self.root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_labels_follow_laterality(self):
        dataset = CMMDDataset(root=str(self.root), cache_dir=str(self.temp_dir / "c"))
        self.assertFalse((self.root / "cmmd-metadata-pyhealth.csv").exists())
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(metadata), 8)
        by_key = {
            (r.patient_id, r.laterality, r.view): (r.has_label, r.label)
            for r in metadata.itertuples()
        }
        self.assertEqual(by_key[("D1-0002", "L", "CC")], (1, 1.0))
        self.assertEqual(by_key[("D1-0002", "R", "CC")], (1, 0.0))
        # D2-0001's right breast has no row: unlabelled, not malignant.
        self.assertEqual(by_key[("D2-0001", "R", "CC")][0], 0)
        self.assertTrue(pd.isna(by_key[("D2-0001", "R", "CC")][1]))
        self.assertEqual(by_key[("D2-0001", "L", "MLO")], (1, 1.0))
        self.assertEqual(set(metadata["cohort"]), {"D1", "D2"})
        self.assertEqual(
            sorted(dataset.unique_patient_ids), ["D1-0001", "D1-0002", "D2-0001"]
        )

    def test_classification_mapping_is_strict(self):
        clinical = CLINICAL.copy()
        clinical["classification"] = [" benign ", "MALIGNANT", "Unknown", "malignant"]
        build_cmmd_root(self.root, clinical)
        with self.assertLogs("pyhealth.datasets.cmmd", level="WARNING") as logs:
            dataset = CMMDDataset(
                root=str(self.root), cache_dir=str(self.temp_dir / "c")
            )
        self.assertIn("Unknown", "\n".join(logs.output))
        metadata = pd.read_csv(dataset.metadata_path)
        by_key = {
            (r.patient_id, r.laterality, r.view): (r.has_label, r.label)
            for r in metadata.itertuples()
        }
        self.assertEqual(by_key[("D1-0001", "R", "CC")], (1, 0.0))
        self.assertEqual(by_key[("D1-0002", "L", "CC")], (1, 1.0))
        # "Unknown" is neither benign nor malignant: unlabelled, not benign.
        self.assertEqual(by_key[("D1-0002", "R", "CC")][0], 0)
        self.assertTrue(pd.isna(by_key[("D1-0002", "R", "CC")][1]))
        self.assertEqual(by_key[("D2-0001", "L", "CC")], (1, 1.0))

    def test_metadata_refreshes_when_sheet_edited_in_place(self):
        cache = str(self.temp_dir / "c")
        dataset = CMMDDataset(root=str(self.root), cache_dir=cache)
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(int(metadata["has_label"].sum()), 6)
        clinical = CLINICAL.copy()
        clinical.loc[clinical["ID1"] == "D1-0001", "classification"] = "Malignant"
        sheet = self.root / "CMMD_clinicaldata_revision.csv"
        clinical.to_csv(sheet, index=False)
        # Same path; force a different mtime in case the edit lands in
        # the same timestamp tick.
        stat = sheet.stat()
        os.utime(sheet, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        dataset = CMMDDataset(root=str(self.root), cache_dir=cache)
        metadata = pd.read_csv(dataset.metadata_path)
        row = metadata[
            (metadata["patient_id"] == "D1-0001") & (metadata["view"] == "CC")
        ].iloc[0]
        self.assertEqual(row["label"], 1.0)

    def test_xlsx_without_openpyxl_raises_clear_error(self):
        (self.root / "CMMD_clinicaldata_revision.xlsx").write_bytes(b"PK")
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("No module named 'openpyxl'")
            return real_import(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=fake_import),
            self.assertRaisesRegex(ImportError, "openpyxl"),
        ):
            CMMDDataset(root=str(self.root), cache_dir=str(self.temp_dir / "c"))

    @unittest.skipUnless(openpyxl is not None, "openpyxl is not installed")
    def test_xlsx_clinical_sheet(self):
        (self.root / "CMMD_clinicaldata_revision.csv").unlink()
        CLINICAL.to_excel(
            self.root / "CMMD_clinicaldata_revision.xlsx", index=False
        )
        dataset = CMMDDataset(root=str(self.root), cache_dir=str(self.temp_dir / "c"))
        self.assertEqual(dataset.clinical_path.suffix, ".xlsx")
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(metadata), 8)
        self.assertEqual(int(metadata["has_label"].sum()), 6)

    def test_missing_clinical_file(self):
        with self.assertRaises(FileNotFoundError):
            CMMDDataset(
                root=str(self.root),
                clinical_file="missing.xlsx",
                cache_dir=str(self.temp_dir / "c"),
            )

    def test_default_task(self):
        dataset = CMMDDataset(root=str(self.root), cache_dir=str(self.temp_dir / "c"))
        self.assertEqual(dataset.default_task.task_name, "CMMDMalignancyClassification")


if __name__ == "__main__":
    unittest.main()
