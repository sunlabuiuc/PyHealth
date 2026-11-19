"""Tests for the CBIS-DDSM mammography dataset loader."""

import shutil
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import pandas as pd

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from pyhealth.datasets import CBISDDSMDataset, SubjectSplit

_STEMS = {
    ("mass", "train"): ("mass_case_description_train_set", "Mass-Training"),
    ("mass", "test"): ("mass_case_description_test_set", "Mass-Test"),
    ("calc", "train"): ("calc_case_description_train_set", "Calc-Training"),
    ("calc", "test"): ("calc_case_description_test_set", "Calc-Test"),
}

# (kind, split, patient, side, view, abnormality id, assessment, pathology,
#  series uid, on disk)
_ROWS = [
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


def build_cbis_root(root: Path, rows=_ROWS) -> None:
    """Write the four case-description CSVs and the DICOMs under ``root``."""
    tables = {key: [] for key in _STEMS}
    for kind, split, pid, side, view, abn, birads, path, uid, on_disk in rows:
        prefix = _STEMS[(kind, split)][1]
        tables[(kind, split)].append(
            {
                "patient_id": pid,
                # mass CSVs say breast_density, calc CSVs breast density.
                ("breast_density" if kind == "mass" else "breast density"): 2,
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
    for key, (stem, _) in _STEMS.items():
        table = pd.DataFrame(tables[key], columns=None if tables[key] else columns)
        table.to_csv(root / f"{stem}.csv", index=False)


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCBISDDSMDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "cbis"
        self.root.mkdir()
        build_cbis_root(self.root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dataset(self, **kwargs):
        return CBISDDSMDataset(
            root=str(self.root), cache_dir=str(self.temp_dir / "cache"), **kwargs
        )

    def test_metadata_rows(self):
        dataset = self._dataset()
        self.assertFalse((self.root / "cbis-ddsm-metadata-pyhealth.csv").exists())
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(metadata), 7)  # P_00005 has no DICOM
        self.assertEqual(
            sorted(dataset.unique_patient_ids),
            ["P_00001", "P_00002", "P_00003", "P_00004"],
        )
        event = dataset.get_patient("P_00003").get_events("lesions")[0]
        self.assertEqual(event.image_id, "P_00003_LEFT_MLO")
        self.assertEqual(event.split, "test")
        self.assertEqual(event.series_uid, "3.1")
        self.assertTrue(event.image_path.endswith("3.1/00000001.dcm"))
        self.assertEqual(set(metadata["breast_density"]), {2})

    def test_drop_overlap_patients_and_official_split(self):
        with self.assertRaises(ValueError):
            self._dataset().official_split()
        dataset = self._dataset(drop_overlap_patients=True)
        self.assertNotIn("P_00004", dataset.unique_patient_ids)
        split = dataset.official_split()
        self.assertIsInstance(split, SubjectSplit)
        self.assertEqual(
            split.to_dict()["subjects"], [["P_00001", "P_00002"], ["P_00003"]]
        )

    def test_data_retriever_layout(self):
        retriever = self.temp_dir / "retriever"
        retriever.mkdir()
        for stem, _ in _STEMS.values():
            shutil.copy(self.root / f"{stem}.csv", retriever / f"{stem}.csv")
        folder = retriever / "CBIS-DDSM" / "Mass-Test_P_00003_LEFT_MLO" / "full"
        write_dicom(folder / "1-1.dcm")
        pd.DataFrame(
            {"Series UID": ["3.1"], "File Location": ["./CBIS-DDSM/Mass-Test_P_00003_LEFT_MLO/full"]}
        ).to_csv(retriever / "metadata.csv", index=False)
        dataset = CBISDDSMDataset(
            root=str(retriever),
            cache_dir=str(self.temp_dir / "cache_retriever"),
            metadata_dir=str(self.temp_dir / "meta_retriever"),
        )
        self.assertEqual(dataset.unique_patient_ids, ["P_00003"])

    def test_missing_csv_raises(self):
        (self.root / "calc_case_description_test_set.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_download_skips_csvs_on_disk(self):
        with mock.patch("pyhealth.datasets.cbis_ddsm.download_file") as fetch:
            self._dataset(download=True)
        fetch.assert_not_called()

    def test_download_fetches_missing_csv(self):
        stem = "calc_case_description_test_set"
        saved = (self.root / f"{stem}.csv").read_bytes()
        (self.root / f"{stem}.csv").unlink()

        def fake_download(url, path):
            Path(path).write_bytes(saved)

        with mock.patch(
            "pyhealth.datasets.cbis_ddsm.download_file", side_effect=fake_download
        ) as fetch:
            dataset = self._dataset(download=True)
        fetch.assert_called_once()
        url, path = fetch.call_args.args
        self.assertEqual(
            url, f"https://www.cancerimagingarchive.net/wp-content/uploads/{stem}.csv"
        )
        self.assertEqual(Path(path), self.root.resolve() / f"{stem}.csv")
        self.assertEqual(len(dataset.unique_patient_ids), 4)

    def test_download_series_extracts_and_removes_archive(self):
        dataset = self._dataset()
        payload = self.temp_dir / "payload.dcm"
        write_dicom(payload)

        def fake_download(url, path):
            with zipfile.ZipFile(path, "w") as handle:
                handle.write(payload, "1-1.dcm")

        with mock.patch(
            "pyhealth.datasets.cbis_ddsm.download_file", side_effect=fake_download
        ) as fetch:
            dataset.download_series(["5.1", "1.1"])  # 1.1 is already on disk
        fetch.assert_called_once()
        self.assertTrue(fetch.call_args.args[0].endswith("SeriesInstanceUID=5.1"))
        dicom_dir = self.root / "dicom"
        self.assertTrue((dicom_dir / "5.1" / "1-1.dcm").is_file())
        self.assertFalse((dicom_dir / "5.1.zip").exists())

        refreshed = self._dataset(refresh_metadata=True)
        self.assertIn("P_00005", refreshed.unique_patient_ids)

    def test_download_series_rejects_unsafe_uid(self):
        dataset = self._dataset()
        with mock.patch("pyhealth.datasets.cbis_ddsm.download_file") as fetch:
            for uid in ["../evil", "1.2/3", "", "1..2"]:
                with self.assertRaises(ValueError):
                    dataset.download_series([uid])
        fetch.assert_not_called()

    def test_default_task(self):
        self.assertEqual(
            self._dataset().default_task.task_name,
            "CBISDDSMPathologyClassification",
        )


if __name__ == "__main__":
    unittest.main()
