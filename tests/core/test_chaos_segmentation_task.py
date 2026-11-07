"""Tests for the CHAOS segmentation task."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from pyhealth.datasets import CHAOSDataset, get_dataloader
from pyhealth.processors.image_io.types import ImageRef


def _write_slice(
    path: Path, z: float, study_uid: str, series_uid: str
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = (
        "1.2.826.0.1.3680043.8.498." + str(abs(hash(str(path))))
    )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientID = "ct_1"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 2.0
    dataset.InstanceNumber = int(z) + 1
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0.0, 0.0, z]
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    array = np.full((8, 8), 1024, dtype=np.int16)
    if z == 0:
        array[0, 0] = 2000
    dataset.PixelData = array.tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCHAOSTasks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir) / "CHAOS"
        series_dir = self.root / "Train_Sets/CT/1/DICOM_anon"
        mask_dir = self.root / "Train_Sets/CT/1/Ground"
        series_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        study_uid = generate_uid()
        series_uid = generate_uid()
        _write_slice(series_dir / "slice_0.dcm", 4.0, study_uid, series_uid)
        _write_slice(series_dir / "slice_1.dcm", 0.0, study_uid, series_uid)
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
            mask_dir / "mask_0.png"
        )
        marker = np.zeros((8, 8), dtype=np.uint8)
        marker[0, 0] = 1
        Image.fromarray(marker).save(mask_dir / "mask_1.png")
        self.dataset = CHAOSDataset(
            root=str(self.root),
            cache_dir=str(Path(self.temp_dir) / "cache"),
            metadata_dir=str(Path(self.temp_dir) / "metadata"),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_task(self):
        self.assertEqual(
            self.dataset.default_task.task_name,
            "CHAOSAbdominalSegmentationCTMRI",
        )

    def test_official_task_classes(self):
        from pyhealth.tasks import (
            CHAOSAbdominalSegmentationCTMRI,
            CHAOSAbdominalSegmentationMRIonly,
            CHAOSLiverSegmentationCTMRI,
            CHAOSLiverSegmentationCTonly,
            CHAOSLiverSegmentationMRIonly,
        )

        expected = {
            CHAOSLiverSegmentationCTMRI: (1, ("CT", "MR"), 2),
            CHAOSLiverSegmentationCTonly: (2, ("CT",), 2),
            CHAOSLiverSegmentationMRIonly: (3, ("MR",), 2),
            CHAOSAbdominalSegmentationCTMRI: (4, ("CT", "MR"), 5),
            CHAOSAbdominalSegmentationMRIonly: (5, ("MR",), 5),
        }
        for task_class, (official, modalities, classes) in expected.items():
            task = task_class()
            self.assertEqual(task.official_task, official)
            self.assertEqual(task.modalities, modalities)
            self.assertEqual(task.num_classes, classes)
            self.assertEqual(task.task_name, task_class.__name__)

        task = CHAOSLiverSegmentationCTonly()
        samples = self.dataset.set_task(task, num_workers=1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples.output_schema["mask"][1]["num_classes"], 2)

        patient = self.dataset.get_patient(
            self.dataset.unique_patient_ids[0]
        )
        self.assertEqual(CHAOSLiverSegmentationMRIonly()(patient), [])

    def test_set_task_with_target_shape(self):
        from pyhealth.tasks import CHAOSAbdominalSegmentationCTMRI

        task = CHAOSAbdominalSegmentationCTMRI(target_shape=(1, 4, 4))
        samples = self.dataset.set_task(task, num_workers=1)
        sample = samples[0]
        image = samples.input_processors["image"].materialize(sample["image"])
        mask = samples.output_processors["mask"].materialize(sample["mask"])
        self.assertEqual(tuple(image.shape), (1, 1, 4, 4))
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))

    def test_set_task(self):
        samples = self.dataset.set_task(num_workers=1)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        image = samples.input_processors["image"].materialize(sample["image"])
        mask = samples.output_processors["mask"].materialize(sample["mask"])
        self.assertEqual(tuple(image.shape), (1, 2, 8, 8))
        self.assertEqual(tuple(mask.shape), (1, 2, 8, 8))
        self.assertEqual(str(mask.dtype), "torch.int64")
        self.assertEqual(set(mask.unique().tolist()), {0, 1})
        # The default task is CT+MRI multi-organ; CT supervises background and
        # liver only, not the unannotated spleen/kidney classes.
        self.assertEqual(sample["class_mask"], [1, 1, 0, 0, 0])
        # DICOM position order reverses filename order; mask order must follow
        # the image order, so mask_1 (ones) comes before mask_0 (zeros).
        self.assertEqual(int(mask[0, 0].sum()), 1)
        self.assertEqual(int(mask[0, 1].sum()), 0)
        image_marker = np.unravel_index(
            int(np.argmax(image[0, 0].numpy())),
            image[0, 0].shape,
        )
        mask_marker = tuple(
            np.argwhere(mask[0, 0].numpy() > 0)[0].tolist()
        )
        self.assertEqual(image_marker, mask_marker)
        self.assertEqual(set(samples.input_schema), {"image"})
        self.assertEqual(set(samples.output_schema), {"mask"})
        self.assertEqual(
            samples.output_schema["mask"],
            (
                "image_mask_dir",
                {
                    "lazy": True,
                    "num_classes": 5,
                    "label_map": {
                        0: 0,
                        1: 1,
                        63: 1,
                        126: 2,
                        189: 3,
                        252: 4,
                    },
                    "strict_mask_names": False,
                },
            ),
        )

        # The processed cache must store references, not the full 3D masks.
        samples._resolve_refs = False
        raw = samples[0]
        self.assertIsInstance(raw["image"], ImageRef)
        self.assertIsInstance(raw["mask"], ImageRef)
        self.assertEqual(raw["mask"].reader, "image_mask")
        samples._resolve_refs = True

        loader = get_dataloader(samples, batch_size=1)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["image"].shape), (1, 1, 2, 8, 8))
        self.assertEqual(tuple(batch["mask"].shape), (1, 1, 2, 8, 8))


if __name__ == "__main__":
    unittest.main()
