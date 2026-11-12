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

    def test_annotation_labels_override_mapping(self):
        annotations = {
            "labelGroups": [
                {
                    "labels": [
                        {"id": "L_pos", "name": "Lung Opacity"},
                        {
                            "id": "L_neg",
                            "name": "No Lung Opacity / Not Normal",
                        },
                    ]
                }
            ],
            "datasets": [
                {
                    "annotations": [
                        {
                            "SOPInstanceUID": "sop_1",
                            "labelId": "L_neg",
                            "data": None,
                        },
                        {
                            "SOPInstanceUID": "sop_2",
                            "labelId": "L_pos",
                            "data": {"x": 1, "y": 1, "width": 2, "height": 2},
                        },
                    ]
                }
            ],
        }
        annotations_path = self.root / "annotations.json"
        with open(annotations_path, "w") as handle:
            json.dump(annotations, handle)
        dataset = RSNAPneumoniaDataset(
            root=str(self.root),
            cache_dir=str(self.cache_dir),
            metadata_dir=str(Path(self.temp_dir) / "metadata_annotations"),
            annotations_file=str(annotations_path),
            label_source="annotations",
        )
        self.assertEqual(dataset.label_source, "annotations")
        labels = {
            patient_id: int(
                dataset.get_patient(patient_id).get_events()[0].label
            )
            for patient_id in dataset.unique_patient_ids
        }
        # sop_1 is overridden to negative, sop_2 to positive, sop_3 falls
        # back to the mapping label.
        self.assertEqual(labels, {"study_1": 0, "study_2": 1, "study_3": 0})

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

    def test_error_branches_and_cached_marker_reuse(self):
        with self.assertRaises(FileNotFoundError):
            RSNAPneumoniaDataset(
                root=str(self.root),
                image_dir=str(self.root / "missing_images"),
                cache_dir=str(self.cache_dir),
            )

        with self.assertRaises(FileNotFoundError):
            RSNAPneumoniaDataset(
                root=str(self.root),
                mapping_file=str(self.root / "missing_mapping.json"),
                cache_dir=str(self.cache_dir),
            )

        with self.assertRaises(ValueError):
            RSNAPneumoniaDataset(
                root=str(self.root),
                label_source="invalid",
                cache_dir=str(self.cache_dir),
            )

        with self.assertRaises(FileNotFoundError):
            RSNAPneumoniaDataset(
                root=str(self.root),
                annotations_file=str(self.root / "missing_annotations.json"),
                label_source="annotations",
                cache_dir=str(self.cache_dir),
            )

        # Default metadata_dir=None and subsequent reuse of existing marker
        dataset1 = RSNAPneumoniaDataset(
            root=str(self.root),
            cache_dir=str(self.cache_dir),
        )
        dataset2 = RSNAPneumoniaDataset(
            root=str(self.root),
            cache_dir=str(self.cache_dir),
        )
        self.assertEqual(
            set(dataset1.unique_patient_ids),
            set(dataset2.unique_patient_ids),
        )


if __name__ == "__main__":
    unittest.main()
