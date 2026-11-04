"""Tests for the DICOM readers and processors using synthetic slices."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from pyhealth.processors import DicomProcessor, DicomSeriesProcessor
from pyhealth.processors.image_io.readers.dicom import DicomReader, DicomSeriesReader
from pyhealth.processors.image_io.types import ImageRef


def _write_slice(
    path: Path,
    stored_value: int,
    z: float,
    instance: int,
    series_uid: str,
    study_uid: str,
    *,
    include_position: bool = True,
) -> None:
    """Write one small CT DICOM slice."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.PatientID = "p1"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.Rows = 6
    dataset.Columns = 8
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelSpacing = [1.0, 2.0]
    dataset.SliceThickness = 2.0
    dataset.InstanceNumber = instance
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    if include_position:
        dataset.ImagePositionPatient = [0.0, 0.0, z]
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    array = np.full((6, 8), stored_value, dtype=np.int16)
    dataset.PixelData = array.tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestDicomReaderAndProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.series_uid = generate_uid()
        self.study_uid = generate_uid()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_single(self, value: int = 1024) -> Path:
        path = self.root / "one.dcm"
        _write_slice(
            path,
            value,
            0.0,
            1,
            self.series_uid,
            self.study_uid,
        )
        return path

    def test_single_metadata_and_load(self):
        path = self._write_single(1024 + 47)
        metadata = DicomReader().read_metadata(str(path))
        self.assertEqual(metadata.shape, (1, 6, 8))
        self.assertEqual(metadata.spacing, (1.0, 2.0))
        self.assertEqual(metadata.modality, "CT")
        array = DicomReader().load(str(path), normalize=None)
        self.assertEqual(array.shape, (6, 8))
        self.assertEqual(float(array.min()), 47.0)
        self.assertEqual(float(array.max()), 47.0)

    def test_single_processor(self):
        path = self._write_single(1024 + 10)
        processor = DicomProcessor(image_size=4, mode="L", lazy=False)
        tensor = processor.process(str(path))
        self.assertEqual(tuple(tensor.shape), (1, 4, 4))
        self.assertTrue(processor.stores_tensor)

        lazy_processor = DicomProcessor(image_size=4, mode="L", lazy=True)
        ref = lazy_processor.process(str(path))
        self.assertIsInstance(ref, ImageRef)
        self.assertFalse(lazy_processor.stores_tensor)
        lazy_tensor = lazy_processor.materialize(ref)
        self.assertEqual(tuple(lazy_tensor.shape), (1, 4, 4))

    def _write_series(self, *, include_position: bool = True) -> None:
        values = {0: 1024, 1: 1025, 2: 1026}
        positions = {0: 4.0, 1: 0.0, 2: 2.0}
        for index in range(3):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                values[index],
                positions[index],
                index + 1,
                self.series_uid,
                self.study_uid,
                include_position=include_position,
            )

    def test_series_sorts_by_position(self):
        self._write_series()
        metadata = DicomSeriesReader().read_metadata(self.root)
        self.assertEqual(metadata.shape, (1, 3, 6, 8))
        self.assertEqual(metadata.orientation, "RAS")
        self.assertTrue(metadata.source[0].endswith("IMG-1.dcm"))
        volume = DicomSeriesReader().load(self.root, metadata=metadata)
        self.assertEqual(volume.shape, (1, 3, 6, 8))
        self.assertEqual(set(np.unique(volume).tolist()), {0.0, 1.0, 2.0})

    def test_series_falls_back_to_instance_number(self):
        self._write_series(include_position=False)
        metadata = DicomSeriesReader().read_metadata(self.root)
        self.assertTrue(metadata.source[0].endswith("IMG-0.dcm"))

    def test_multiple_series_require_uid(self):
        other_series = generate_uid()
        _write_slice(
            self.root / "a.dcm", 1024, 0.0, 1, self.series_uid, self.study_uid
        )
        _write_slice(
            self.root / "b.dcm", 1024, 0.0, 1, other_series, self.study_uid
        )
        with self.assertRaises(ValueError):
            DicomSeriesReader().read_metadata(self.root)

    def test_auto_normalize_requires_modality(self):
        path = self._write_single()
        dataset = pydicom.dcmread(str(path))
        del dataset.Modality
        dataset.save_as(str(path), enforce_file_format=True)
        with self.assertRaises(ValueError):
            DicomSeriesReader().load(str(path), normalize="auto")

    def test_series_processor(self):
        self._write_series()
        processor = DicomSeriesProcessor()
        ref = processor.process(self.root)
        self.assertIsInstance(ref, ImageRef)
        self.assertFalse(processor.stores_tensor)
        tensor = processor.materialize(ref)
        self.assertEqual(tuple(tensor.shape), (1, 3, 6, 8))

        eager = DicomSeriesProcessor(lazy=False)
        eager_tensor = eager.process(self.root)
        self.assertTrue(eager.stores_tensor)
        self.assertEqual(tuple(eager_tensor.shape), (1, 3, 6, 8))


if __name__ == "__main__":
    unittest.main()
