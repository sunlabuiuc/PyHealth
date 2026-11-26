"""Tests for the RSNA pneumonia classification task."""

import json
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

import torch

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from pyhealth.datasets import RSNAPneumoniaDataset, get_dataloader

_IMAGE_DIR = "mdai_rsna_project_x9N20BZa_images_2018-07-20-153330"
_MAPPING = "pneumonia-challenge-dataset-mappings_2018.json"


def _write_dicom(path: Path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    uid_suffix = zlib.crc32(str(path).encode("utf-8"))
    file_meta.MediaStorageSOPInstanceUID = (
        f"1.2.826.0.1.3680043.8.498.{uid_suffix}"
    )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CR"
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


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestRSNAPneumoniaTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir) / "rsna"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images = self.root / _IMAGE_DIR
        self.mapping = [
            {
                "subset_group": 0,
                "subset_init_label": 1,
                "orig_labels": ["Infiltration"],
                "StudyInstanceUID": "study_1",
                "SeriesInstanceUID": "series_1",
                "SOPInstanceUID": "sop_1",
            },
            {
                "subset_group": 1,
                "subset_init_label": 0,
                "orig_labels": ["No Finding"],
                "StudyInstanceUID": "study_2",
                "SeriesInstanceUID": "series_2",
                "SOPInstanceUID": "sop_2",
            },
        ]
        with open(self.root / _MAPPING, "w") as handle:
            json.dump(self.mapping, handle)
        for entry in self.mapping:
            path = (
                self.images
                / entry["StudyInstanceUID"]
                / entry["SeriesInstanceUID"]
                / f"{entry['SOPInstanceUID']}.dcm"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_dicom(path)
        self.dataset = RSNAPneumoniaDataset(
            root=str(self.root),
            cache_dir=str(Path(self.temp_dir) / "cache"),
            metadata_dir=str(Path(self.temp_dir) / "metadata"),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_task(self):
        self.assertEqual(
            self.dataset.default_task.task_name,
            "RSNAPneumoniaClassification",
        )

    def test_set_task(self):
        samples = self.dataset.set_task(num_workers=1)
        self.assertEqual(len(samples), 2)
        labels = {
            sample["patient_id"]: int(sample["pneumonia"])
            for sample in samples
        }
        self.assertEqual(labels, {"study_1": 1, "study_2": 0})
        sample = samples[0]
        image = samples.input_processors["image"].materialize(sample["image"])
        self.assertEqual(tuple(image.shape), (1, 224, 224))
        self.assertIsInstance(sample["pneumonia"], torch.Tensor)
        self.assertEqual(set(samples.input_schema), {"image"})
        self.assertEqual(set(samples.output_schema), {"pneumonia"})

        loader = get_dataloader(samples, batch_size=2)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 224, 224))
        self.assertEqual(tuple(batch["pneumonia"].shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
