"""Synthetic CBIS-DDSM root (CSVs and tiny DICOMs) shared by the CBIS-DDSM tests."""

import zlib
from pathlib import Path

import pandas as pd

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

STEMS = {
    ("mass", "train"): ("mass_case_description_train_set", "Mass-Training"),
    ("mass", "test"): ("mass_case_description_test_set", "Mass-Test"),
    ("calc", "train"): ("calc_case_description_train_set", "Calc-Training"),
    ("calc", "test"): ("calc_case_description_test_set", "Calc-Test"),
}

# (kind, split, patient, side, view, abnormality id, assessment, pathology,
#  series uid, on disk)
ROWS = [
    ("mass", "train", "P_00001", "LEFT", "CC", 1, 4, "MALIGNANT", "1.1", True),
    ("mass", "train", "P_00001", "LEFT", "CC", 2, 3, "BENIGN", "1.1", True),
    ("calc", "train", "P_00002", "RIGHT", "MLO", 1, 0, "BENIGN", "2.1", True),
    ("calc", "train", "P_00002", "RIGHT", "MLO", 2, 2, "BENIGN_WITHOUT_CALLBACK",
     "2.1", True),
    ("mass", "test", "P_00003", "LEFT", "MLO", 1, 5, "MALIGNANT", "3.1", True),
    # P_00004 is in both official splits (same image, different series).
    ("calc", "train", "P_00004", "LEFT", "CC", 1, 4, "BENIGN", "4.1", True),
    ("mass", "test", "P_00004", "LEFT", "CC", 1, 4, "MALIGNANT", "4.2", True),
    # No DICOM on disk: skipped.
    ("mass", "test", "P_00005", "RIGHT", "CC", 1, 4, "MALIGNANT", "5.1", False),
]


def write_dicom(path: Path) -> None:
    """Write a tiny 8x8 single-frame DICOM."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    suffix = zlib.crc32(str(path).encode("utf-8"))
    file_meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.3680043.8.498.{suffix}"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MG"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = bytes(range(64))
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), enforce_file_format=True)


def build_cbis_root(root: Path, rows=ROWS, density=None) -> None:
    """Write the four case-description CSVs and the DICOMs under ``root``.

    ``density`` optionally maps a patient id to its breast density (default 2).
    """
    tables = {key: [] for key in STEMS}
    for kind, split, pid, side, view, abn, birads, path, uid, on_disk in rows:
        prefix = STEMS[(kind, split)][1]
        tables[(kind, split)].append(
            {
                "patient_id": pid,
                # mass CSVs say breast_density, calc CSVs breast density.
                ("breast_density" if kind == "mass" else "breast density"): (
                    (density or {}).get(pid, 2)
                ),
                "left or right breast": side,
                "image view": view,
                "abnormality id": abn,
                "abnormality type": "mass" if kind == "mass" else "calcification",
                "assessment": birads,
                "pathology": path,
                "subtlety": 3,
                "image file path": f"{prefix}_{pid}_{side}_{view}/9.9/{uid}/000000.dcm",
            }
        )
        if on_disk:
            write_dicom(root / "dicom" / uid / "00000001.dcm")
    columns = [
        "patient_id", "breast density", "left or right breast", "image view",
        "abnormality id", "abnormality type", "assessment", "pathology",
        "subtlety", "image file path",
    ]
    for key, (stem, _) in STEMS.items():
        table = pd.DataFrame(tables[key], columns=None if tables[key] else columns)
        table.to_csv(root / f"{stem}.csv", index=False)
