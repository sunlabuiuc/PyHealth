"""Synthetic tests for the RSNA Pneumonia dataset loader."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from pyhealth.datasets import RSNAPneumoniaDataset

_IMAGE_DIR = "mdai_rsna_project_x9N20BZa_images_2018-07-20-153330"
_MAPPING = "pneumonia-challenge-dataset-mappings_2018.json"


def _write_dicom(path: Path, patient_id: str, view_position: str = "AP") -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = (
        "1.2.826.0.1.3680043.8.498." + str(abs(hash(str(path))))
    )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientID = patient_id
    dataset.Modality = "CR"
    dataset.ViewPosition = view_position
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = bytes(8 * 8)
    dataset.save_as(str(path), enforce_file_format=True)


class TestRSNAPneumoniaDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir) / "rsna"
        self.image_dir = self.root / _IMAGE_DIR
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.mapping = [
            {
                "img_id": "image_1.png",
                "subset_group": 0,
                "subset_init_label": 1,
                "orig_labels": ["Infiltration"],
                "StudyInstanceUID": "study_1",
                "SeriesInstanceUID": "series_1",
                "SOPInstanceUID": "sop_1",
            },
            {
                "img_id": "image_2.png",
                "subset_group": 1,
                "subset_init_label": 0,
                "orig_labels": ["No Finding"],
                "StudyInstanceUID": "study_2",
                "SeriesInstanceUID": "series_2",
                "SOPInstanceUID": "sop_2",
            },
            {
                "img_id": "image_3.png",
                "subset_group": 2,
                "subset_init_label": 2,
                "orig_labels": ["No Lung Opacity / Not Normal"],
                "StudyInstanceUID": "study_3",
                "SeriesInstanceUID": "series_3",
                "SOPInstanceUID": "sop_3",
            },
        ]
        with open(self.root / _MAPPING, "w") as handle:
            json.dump(self.mapping, handle)
        for entry in self.mapping:
            path = (
                self.image_dir
                / entry["StudyInstanceUID"]
                / entry["SeriesInstanceUID"]
                / f"{entry['SOPInstanceUID']}.dcm"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mapping_metadata(self):
        metadata_dir = Path(self.temp_dir) / "metadata"
        dataset = RSNAPneumoniaDataset(
            root=str(self.root),
            cache_dir=str(self.cache_dir),
            metadata_dir=str(metadata_dir),
        )
        self.assertEqual(
            set(dataset.unique_patient_ids),
            {"study_1", "study_2", "study_3"},
        )
        positive = dataset.get_patient("study_1").get_events()[0]
        self.assertEqual(str(positive.label), "1")
        self.assertEqual(str(positive.subset_init_label), "1")
        self.assertEqual(positive.orig_labels, "Infiltration")
        self.assertEqual(positive.record_id, "sop_1")
        self.assertTrue(positive.image_path.endswith("sop_1.dcm"))
        self.assertEqual(str(positive.fold), "0")
        negative = dataset.get_patient("study_2").get_events()[0]
        self.assertEqual(str(negative.label), "0")
        self.assertTrue(
            (metadata_dir / "rsna-pneumonia-metadata-pyhealth.csv").exists()
        )
        self.assertFalse(
            (self.root / "rsna-pneumonia-metadata-pyhealth.csv").exists()
        )

    @unittest.skipUnless(pydicom is not None, "pydicom is not installed")
    def test_read_dicom_patient_id(self):
        path = self.image_dir / "study_1" / "series_1" / "sop_1.dcm"
        _write_dicom(path, patient_id="patient_1", view_position="PA")
        metadata_dir = Path(self.temp_dir) / "metadata_dicom"
        dataset = RSNAPneumoniaDataset(
            root=str(self.root),
            cache_dir=str(self.cache_dir),
            metadata_dir=str(metadata_dir),
            read_dicom_patient_id=True,
        )
        self.assertIn("patient_1", dataset.unique_patient_ids)
        event = dataset.get_patient("patient_1").get_events()[0]
        self.assertEqual(event.view_position, "PA")
        self.assertEqual(int(event.rows), 8)


if __name__ == "__main__":
    unittest.main()
