"""Synthetic CMMD root (clinical CSV and tiny DICOMs) shared by the CMMD tests."""

import zlib
from pathlib import Path

import pandas as pd

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

VIEW_MEANING = {"CC": "cranio-caudal", "MLO": "medio-lateral oblique"}

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
    code.CodeMeaning = VIEW_MEANING[view]
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


def build_cmmd_root(root: Path, clinical: pd.DataFrame | None = None) -> None:
    """Write a CSV clinical sheet and NBIA-style DICOM folders under ``root``."""
    (CLINICAL if clinical is None else clinical).to_csv(root / "CMMD_clinicaldata_revision.csv", index=False)
    for patient, name, side, view in IMAGES:
        write_dicom(root / "CMMD" / patient / "study" / name, patient, side, view)
