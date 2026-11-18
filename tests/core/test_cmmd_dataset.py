"""Tests for the CMMD mammography dataset loader."""

import builtins
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import pandas as pd

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from pyhealth.datasets import CMMDDataset

_VIEW_MEANING = {"CC": "cranio-caudal", "MLO": "medio-lateral oblique"}

# Clinical sheet: D1-0001 right benign; D1-0002 bilateral with different
# labels; D2-0001 left malignant only (its right breast has no row).
CLINICAL = pd.DataFrame(
    {
        "ID1": ["D1-0001", "D1-0002", "D1-0002", "D2-0001"],
        "LeftRight": ["R", "L", "R", "L"],
        "Age": [44, 50, 50, 64],
        "number": [2, 2, 2, 2],
        "abnormality": ["calcification", "mass", "mass", "both"],
        "classification": ["Benign", "Malignant", "Benign", "Malignant"],
        "subtype": [None, None, None, "Luminal B"],
    }
)

# (patient folder, file, laterality, view)
IMAGES = [
    ("D1-0001", "1-1.dcm", "R", "CC"),
    ("D1-0001", "1-2.dcm", "R", "MLO"),
    ("D1-0002", "1-1.dcm", "L", "CC"),
    ("D1-0002", "1-2.dcm", "R", "CC"),
    ("D2-0001", "1-1.dcm", "R", "CC"),
    ("D2-0001", "1-2.dcm", "R", "MLO"),
    ("D2-0001", "1-3.dcm", "L", "CC"),
    ("D2-0001", "1-4.dcm", "L", "MLO"),
]


def write_dicom(path: Path, patient_id: str, laterality: str, view: str) -> None:
    """Write a tiny 8x8 mammogram DICOM with CMMD-style header fields."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    suffix = zlib.crc32(str(path).encode("utf-8"))
    file_meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.3680043.8.498.{suffix}"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = patient_id
    ds.ImageLaterality = laterality
    code = Dataset()
    code.CodeMeaning = _VIEW_MEANING[view]
    ds.ViewCodeSequence = Sequence([code])
    ds.Modality = "MG"
    ds.Rows = 8
    ds.Columns = 8
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = bytes(range(64))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def build_cmmd_root(root: Path) -> None:
    """Write a CSV clinical sheet and NBIA-style DICOM folders under ``root``."""
    CLINICAL.to_csv(root / "CMMD_clinicaldata_revision.csv", index=False)
    for patient, name, side, view in IMAGES:
        write_dicom(root / "CMMD" / patient / "study" / name, patient, side, view)


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
