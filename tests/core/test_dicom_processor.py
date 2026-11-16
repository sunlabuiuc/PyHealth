"""Tests for the DICOM readers and processors using synthetic slices."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
except ImportError:  # pragma: no cover - test env installs the extra
    pydicom = None

from PIL import Image

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CNN
from pyhealth.processors import (
    DicomProcessor,
    DicomSeriesProcessor,
    ImageMaskDirectoryProcessor,
    read_image_metadata,
    transform_batch,
    transform_processor,
)
from pyhealth.processors.image_io.readers.dicom import DicomReader, DicomSeriesReader


def _write_slice(
    path: Path,
    stored_value: int,
    z: float,
    instance: int,
    series_uid: str,
    study_uid: str,
    *,
    include_position: bool = True,
    iop: list[float] | None = None,
    position: list[float] | None = None,
    photometric: str = "MONOCHROME2",
    window_center: float | None = None,
    window_width: float | None = None,
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
    dataset.PhotometricInterpretation = photometric
    dataset.PixelSpacing = [1.0, 2.0]
    dataset.SliceThickness = 2.0
    dataset.InstanceNumber = instance
    dataset.ImageOrientationPatient = iop or [1, 0, 0, 0, 1, 0]
    if include_position:
        dataset.ImagePositionPatient = position or [0.0, 0.0, z]
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    if window_center is not None:
        dataset.WindowCenter = window_center
    if window_width is not None:
        dataset.WindowWidth = window_width
    array = np.full((6, 8), stored_value, dtype=np.int16)
    dataset.PixelData = array.tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def _write_multiframe(path: Path, series_uid: str, study_uid: str) -> None:
    """Write a small multi-frame DICOM file."""
    _write_slice(path, 1024, 0.0, 1, series_uid, study_uid)
    dataset = pydicom.dcmread(str(path))
    dataset.NumberOfFrames = 2
    dataset.PixelData = np.zeros((2, 6, 8), dtype=np.int16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def _write_sized_slice(
    path: Path, rows: int, columns: int, study_uid: str
) -> None:
    """Write one CT slice of ``rows`` x ``columns`` pixels."""
    _write_slice(path, 1024, 0.0, 1, generate_uid(), study_uid)
    dataset = pydicom.dcmread(str(path))
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.PixelData = np.zeros((rows, columns), dtype=np.int16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def _write_rgb_slice(path: Path) -> None:
    """Write one small RGB DICOM slice."""
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
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Rows = 6
    dataset.Columns = 8
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 3
    dataset.PhotometricInterpretation = "RGB"
    dataset.PlanarConfiguration = 0
    dataset.PixelData = np.zeros((6, 8, 3), dtype=np.uint8).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def _write_masks(
    mask_dir: Path, count: int, prefix: str = "mask", separator: str = "_"
) -> None:
    """Write small single-pixel marker masks."""
    mask_dir.mkdir()
    for index in range(count):
        array = np.zeros((6, 8), dtype=np.uint8)
        array[0, 0] = index + 1
        Image.fromarray(array).save(
            mask_dir / f"{prefix}{separator}{index}.png"
        )


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestDicomReaderAndProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.series_uid = generate_uid()
        self.study_uid = generate_uid()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_single(
        self, value: int = 1024, photometric: str = "MONOCHROME2"
    ) -> Path:
        path = self.root / "one.dcm"
        _write_slice(
            path,
            value,
            0.0,
            1,
            self.series_uid,
            self.study_uid,
            photometric=photometric,
        )
        return path

    def test_single_metadata_and_load(self):
        path = self._write_single(1024 + 47)
        metadata = DicomReader().read_metadata(str(path))
        self.assertEqual(metadata["shape"], (1, 6, 8))
        self.assertEqual(metadata["spacing"], (1.0, 2.0))
        self.assertEqual(metadata["modality"], "CT")
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
        value = lazy_processor.process(str(path))
        self.assertIsInstance(value, str)
        self.assertFalse(lazy_processor.stores_tensor)
        lazy_tensor = lazy_processor.materialize(value)
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
        self.assertEqual(metadata["shape"], (1, 3, 6, 8))
        self.assertEqual(metadata["orientation"], "RAS")
        self.assertTrue(metadata["source"][0].endswith("IMG-1.dcm"))
        volume = DicomSeriesReader().load(self.root, metadata=metadata)
        self.assertEqual(volume.shape, (1, 3, 6, 8))
        self.assertEqual(set(np.unique(volume).tolist()), {0.0, 1.0, 2.0})

    def test_non_axial_series_metadata_matches_loaded(self):
        iop = [0, 1, 0, 0, 0, 1]
        for index in range(3):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                iop=iop,
                position=[float(index) * 3.0, 0.0, 0.0],
            )
        metadata = DicomSeriesReader().read_metadata(self.root)
        volume = DicomSeriesReader().load(self.root, metadata=metadata)
        self.assertEqual(metadata["shape"], volume.shape)
        # The position gap is 3 mm, so it wins over SliceThickness=2 mm.
        self.assertEqual(metadata["spacing"], (1.0, 2.0, 3.0))
        self.assertEqual(metadata["orientation"], "RAS")
        import nibabel as nib

        self.assertEqual(
            "".join(nib.aff2axcodes(np.asarray(metadata["affine"]))),
            "RAS",
        )

    def test_mask_names_matching_dicom_order(self):
        self._write_series()
        mask_dir = self.root / "masks"
        # IMG-N.png matches IMG-N.dcm, so this uses the name-matching path.
        _write_masks(mask_dir, 3, "IMG", "-")
        processor = ImageMaskDirectoryProcessor(lazy=False, canonical=False)
        tensor = processor.process((mask_dir, self.root))
        # DICOM positions are z=(4, 0, 2), so the mask order is IMG-1, IMG-2,
        # IMG-0 and the marker values are (2, 3, 1).
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [2, 3, 1])

    def test_processor_mask_keeps_integer_labels(self):
        path = self._write_single(1024 + 1)
        processor = DicomProcessor(
            image_size=4, mode="L", is_mask=True, lazy=False
        )
        tensor = processor.process(str(path))
        self.assertEqual(str(tensor.dtype), "torch.int64")
        self.assertEqual(set(tensor.unique().tolist()), {1025})

    def test_default_processor_is_eager(self):
        path = self._write_single(1024 + 1)
        processor = DicomProcessor(image_size=4, mode="L")
        tensor = processor.process(str(path))
        self.assertTrue(processor.stores_tensor)
        self.assertEqual(tuple(tensor.shape), (1, 4, 4))

    def test_default_processor_works_with_cnn(self):
        paths = []
        for index in range(2):
            path = self.root / f"cnn_{index}.dcm"
            _write_slice(
                path,
                1024 + index,
                0.0,
                1,
                generate_uid(),
                self.study_uid,
            )
            paths.append(path)
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "image": str(paths[0]), "label": 0},
                {"patient_id": "p1", "image": str(paths[1]), "label": 1},
            ],
            {"image": ("dicom", {"image_size": 8})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            model = CNN(dataset=dataset)
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            output = model(**batch)
            self.assertIn("loss", output)
        finally:
            dataset.close()

    def test_fit_records_the_output_shape_from_dicom_headers(self):
        tall = self._write_single()  # 6 x 8
        wide = self.root / "wide.dcm"
        _write_sized_slice(wide, 4, 10, self.study_uid)
        samples = [{"image": str(tall)}, {"image": None}, {"image": str(wide)}]
        missing = [{"image": str(self.root / "missing.dcm")}]
        for mode, channels in ((None, 1), ("L", 1), ("RGB", 3), ("RGBA", 4)):
            with self.subTest(mode=mode):
                processor = DicomProcessor(image_size=None, mode=mode)
                processor.fit(samples, "image")
                self.assertEqual(processor._shape, (channels, 6, 10))
                # With an image_size, fit() reads no file.
                processor = DicomProcessor(image_size=5, mode=mode)
                processor.fit(missing, "image")
                self.assertEqual(processor._shape, (channels, 5, 5))
                self.assertEqual(
                    tuple(processor.process(str(wide)).shape), (channels, 5, 5)
                )
        processor = DicomProcessor(image_size=None)
        processor.fit([{"image": None}, {}], "image")
        self.assertIsNone(processor._shape)

    def test_unified_bridge_uses_the_fitted_shape(self):
        tall = self._write_single()  # 6 x 8
        wide = self.root / "wide.dcm"
        _write_sized_slice(wide, 4, 10, self.study_uid)
        processor = DicomProcessor(image_size=None)
        processor.fit([{"image": str(tall)}, {"image": str(wide)}], "image")
        processors = transform_processor({"image": processor})
        self.assertEqual(processors["image"].value_dim(), {0: (1, 6, 10)})
        batch = {"image": processor.collate([processor.process(str(tall))])}
        values, offsets = transform_batch(processors, batch)["image"]
        self.assertEqual(tuple(values.shape), (1, 1, 6, 10))
        self.assertEqual(offsets.tolist(), [0, 1])

    def test_multiframe_dicom_is_rejected_while_reading_metadata(self):
        path = self.root / "multi.dcm"
        _write_multiframe(path, self.series_uid, self.study_uid)
        with self.assertRaisesRegex(NotImplementedError, "Multi-frame"):
            DicomReader().read_metadata(str(path))
        with self.assertRaisesRegex(NotImplementedError, "Multi-frame"):
            DicomSeriesReader().read_metadata(str(path))

    def test_pil_output_collates_without_lazy(self):
        paths = []
        for index in range(2):
            path = self.root / f"pil_{index}.dcm"
            _write_slice(
                path,
                1024 + index,
                0.0,
                1,
                generate_uid(),
                self.study_uid,
            )
            paths.append(path)
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "image": str(paths[0]), "label": 0},
                {"patient_id": "p1", "image": str(paths[1]), "label": 1},
            ],
            {
                "image": (
                    "dicom",
                    {
                        "lazy": False,
                        "to_tensor": False,
                        "image_size": 4,
                        "mode": "L",
                    },
                )
            },
            {"label": "binary"},
            in_memory=False,
        )
        try:
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            self.assertEqual(len(batch["image"]), 2)
            self.assertIsInstance(batch["image"][0], Image.Image)
        finally:
            dataset.close()

    def test_lazy_dicom_dataloader_materializes(self):
        paths = []
        for index in range(2):
            path = self.root / f"single_{index}.dcm"
            _write_slice(
                path,
                1024 + index,
                0.0,
                1,
                generate_uid(),
                self.study_uid,
            )
            paths.append(path)
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "image": str(paths[0]), "label": 0},
                {"patient_id": "p1", "image": str(paths[1]), "label": 1},
            ],
            {"image": ("dicom", {"lazy": True, "image_size": 4, "mode": "L"})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            self.assertIsInstance(dataset[0]["image"], str)
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 4, 4))
        finally:
            dataset.close()

    def test_mask_requires_tensor_output(self):
        with self.assertRaisesRegex(ValueError, "to_tensor=True"):
            DicomProcessor(is_mask=True, to_tensor=False)
        with self.assertRaisesRegex(ValueError, "normalize=True"):
            DicomProcessor(
                is_mask=True,
                normalize=True,
                mean=[0.0],
                std=[1.0],
            )

    def test_rgba_from_grayscale(self):
        path = self._write_single(1024 + 1)
        tensor = DicomProcessor(
            image_size=4, mode="RGBA", lazy=False
        ).process(str(path))
        self.assertEqual(tuple(tensor.shape), (4, 4, 4))

    def test_mask_skips_rescale_and_window(self):
        path = self._write_single(1024 + 7)
        array = DicomReader().load(
            str(path),
            is_mask=True,
            window=True,
            window_center=0.0,
            window_width=10.0,
        )
        self.assertEqual(int(array.min()), 1031)
        self.assertEqual(int(array.max()), 1031)

    def test_series_mask_skips_rescale(self):
        self._write_series()
        volume = DicomSeriesReader().load(
            self.root, is_mask=True, canonical=False
        )
        values = sorted(int(value) for value in np.unique(volume))
        self.assertEqual(values, [1024, 1025, 1026])

    def test_monochrome1_window_is_applied_before_inversion(self):
        path = self._write_single(1024 + 25, photometric="MONOCHROME1")
        array = DicomReader().load(
            str(path),
            normalize=None,
            window=True,
            window_center=0.0,
            window_width=100.0,
        )
        # Rescaled 25 -> windowed 0.75 -> inverted 0.25.
        self.assertAlmostEqual(float(array[0, 0]), 0.25)

    def test_unreadable_dicom_is_not_skipped(self):
        self._write_single()
        (self.root / "broken.dcm").write_bytes(b"not a dicom")
        original = pydicom.dcmread

        def fake_dcmread(path, *args, **kwargs):
            if Path(path).name == "broken.dcm":
                raise OSError("truncated")
            return original(path, *args, **kwargs)

        with (
            mock.patch("pydicom.dcmread", side_effect=fake_dcmread),
            self.assertRaisesRegex(ValueError, "Failed to read DICOM"),
        ):
            DicomSeriesReader().read_metadata(self.root)

    def test_read_image_metadata_mask_tuple(self):
        self._write_series()
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        metadata = read_image_metadata(
            (mask_dir, self.root),
            reader="image_mask",
            target_shape=(2, 3, 4),
        )
        self.assertEqual(metadata["shape"], (1, 2, 3, 4))
        self.assertEqual(len(metadata["spacing"]), 3)

    def test_read_image_metadata_mask_tuple_non_axial(self):
        iop = [0, 1, 0, 0, 0, 1]
        for index in range(3):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                iop=iop,
                position=[float(index) * 3.0, 0.0, 0.0],
            )
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        metadata = read_image_metadata(
            (mask_dir, self.root), reader="image_mask"
        )
        tensor = ImageMaskDirectoryProcessor(
            lazy=False, canonical=True
        ).process((mask_dir, self.root))
        self.assertEqual(metadata["shape"], tuple(tensor.shape))

    def test_read_image_metadata_target_requires_3d(self):
        path = self._write_single()
        with self.assertRaisesRegex(ValueError, "3D volume"):
            read_image_metadata(str(path), target_shape=(2, 3, 4))
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        with self.assertRaisesRegex(ValueError, "3D volume"):
            read_image_metadata(
                mask_dir,
                reader="image_mask",
                target_shape=(2, 3, 4),
            )

    def test_colour_dicom_is_rejected(self):
        path = self.root / "colour.dcm"
        _write_rgb_slice(path)
        with self.assertRaisesRegex(NotImplementedError, "Colour DICOM"):
            DicomReader().read_metadata(str(path))
        with self.assertRaisesRegex(NotImplementedError, "Colour DICOM"):
            DicomReader().load(str(path))

    def test_mask_names_match_non_dcm_extension(self):
        series_dir = self.root / "ima_series"
        series_dir.mkdir()
        positions = {0: 4.0, 1: 0.0, 2: 2.0}
        for index in range(3):
            _write_slice(
                series_dir / f"IMG-{index}.ima",
                1024,
                positions[index],
                index + 1,
                self.series_uid,
                self.study_uid,
            )
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        processor = ImageMaskDirectoryProcessor(lazy=False, canonical=False)
        tensor = processor.process((mask_dir, series_dir))
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [2, 3, 1])

    def test_mask_rejects_in_plane_size_mismatch(self):
        self._write_series()
        mask_dir = self.root / "masks"
        mask_dir.mkdir()
        for index in range(3):
            array = np.zeros((5, 7), dtype=np.uint8)
            array[0, 0] = index + 1
            Image.fromarray(array).save(mask_dir / f"IMG-{index}.png")
        for lazy in (False, True):
            for kwargs in ({}, {"target_shape": (2, 3, 4)}):
                processor = ImageMaskDirectoryProcessor(
                    lazy=lazy, canonical=False, **kwargs
                )
                with self.assertRaisesRegex(
                    ValueError, "does not match the reference"
                ):
                    processor.process((mask_dir, self.root))

    def test_mask_alignment_requires_readable_series(self):
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3)
        processor = ImageMaskDirectoryProcessor(lazy=False)
        with self.assertRaisesRegex(ValueError, "Could not read"):
            processor.process((mask_dir, self.root / "missing.dcm"))

    def test_mask_alignment_rejects_slice_count_mismatch(self):
        self._write_series()
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 2)
        processor = ImageMaskDirectoryProcessor(lazy=False)
        with self.assertRaisesRegex(ValueError, "2 slices"):
            processor.process((mask_dir, self.root))

    def test_mask_alignment_falls_back_to_positional_order(self):
        self._write_series()
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3)
        processor = ImageMaskDirectoryProcessor(
            lazy=False, canonical=False, strict_mask_names=False
        )
        with self.assertLogs(
            "pyhealth.processors.image_io.readers.image_mask", level="WARNING"
        ):
            tensor = processor.process((mask_dir, self.root))
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [2, 3, 1])

    def test_mask_alignment_rejects_mismatched_names_by_default(self):
        self._write_series()
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3)
        processor = ImageMaskDirectoryProcessor(lazy=False)
        with self.assertRaisesRegex(ValueError, "filenames do not match"):
            processor.process((mask_dir, self.root))

    def test_mask_metadata_uses_processor_canonical_flag(self):
        iop = [0, 1, 0, 0, 0, 1]
        for index in range(3):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                iop=iop,
                position=[float(index) * 3.0, 0.0, 0.0],
            )
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        processor = ImageMaskDirectoryProcessor(lazy=True, canonical=False)
        value = processor.process((mask_dir, self.root))
        tensor = processor.materialize(value)
        self.assertEqual(tuple(tensor.shape), (1, 3, 6, 8))

    def test_series_rejects_duplicate_positions(self):
        for index, z in enumerate((0.0, 0.0, 2.0)):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                z,
                index + 1,
                self.series_uid,
                self.study_uid,
            )
        with self.assertRaisesRegex(ValueError, "duplicate or overlapping"):
            DicomSeriesReader().read_metadata(self.root)

    def test_series_warns_on_non_uniform_spacing(self):
        for index, z in enumerate((0.0, 1.0, 3.0)):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                z,
                index + 1,
                self.series_uid,
                self.study_uid,
            )
        with self.assertLogs(
            "pyhealth.processors.image_io.readers.dicom", level="WARNING"
        ):
            metadata = DicomSeriesReader().read_metadata(self.root)
        self.assertEqual(metadata["spacing"][0], 1.5)

    def test_series_prefers_measured_slice_spacing(self):
        for index in range(3):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                float(index) * 3.0,
                index + 1,
                self.series_uid,
                self.study_uid,
            )
        metadata = DicomSeriesReader().read_metadata(self.root)
        self.assertEqual(metadata["spacing"][0], 3.0)

    def test_series_monochrome1_uses_one_inversion_offset(self):
        for index, value in enumerate((10, 20, 30)):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                value,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                photometric="MONOCHROME1",
            )
        volume = DicomSeriesReader().load(
            self.root, canonical=False, normalize=None
        )
        values = sorted(int(value) for value in np.unique(volume))
        self.assertEqual(values, [0, 10, 20])
        mask_volume = DicomSeriesReader().load(
            self.root, canonical=False, is_mask=True
        )
        self.assertEqual(
            sorted(int(value) for value in np.unique(mask_volume)),
            [10, 20, 30],
        )

    def test_series_rejects_mixed_window_tags(self):
        for index in range(3):
            kwargs = (
                {"window_center": -1000.0, "window_width": 100.0}
                if index == 0
                else {}
            )
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "only some slices"):
            DicomSeriesReader().load(self.root, window=True)

    def test_series_rejects_different_window_values(self):
        for index in range(3):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                1024,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                window_center=-1000.0 + index * 100.0,
                window_width=100.0,
            )
        with self.assertRaisesRegex(ValueError, "different window"):
            DicomSeriesReader().load(self.root, window=True)

    def test_single_dicom_rejects_half_window(self):
        path = self.root / "half_window.dcm"
        _write_slice(
            path,
            1024,
            0.0,
            1,
            self.series_uid,
            self.study_uid,
            window_center=-1000.0,
        )
        with self.assertRaisesRegex(ValueError, "both WindowCenter"):
            DicomReader().load(str(path), window=True)

    def test_series_monochrome1_window_is_applied_before_inversion(self):
        for index, value in enumerate((10, 20, 30)):
            _write_slice(
                self.root / f"IMG-{index}.dcm",
                value,
                float(index),
                index + 1,
                self.series_uid,
                self.study_uid,
                photometric="MONOCHROME1",
                window_center=-1000.0,
                window_width=100.0,
            )
        volume = DicomSeriesReader().load(
            self.root,
            canonical=False,
            normalize=None,
            window=True,
        )
        values = sorted(float(value) for value in np.unique(volume))
        np.testing.assert_allclose(values, [0.44, 0.54, 0.64], atol=1e-6)

    def test_lazy_series_metadata_reflects_target_shape(self):
        self._write_series()
        processor = DicomSeriesProcessor(lazy=True, target_shape=(2, 3, 4))
        value = processor.process(self.root)
        self.assertIsInstance(value, list)
        self.assertEqual(tuple(processor.materialize(value).shape), (1, 2, 3, 4))
        metadata = read_image_metadata(
            self.root, reader="dicom_series", target_shape=(2, 3, 4)
        )
        self.assertEqual(metadata["shape"], (1, 2, 3, 4))

    def test_lazy_mask_metadata_reflects_target_shape(self):
        self._write_series()
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        processor = ImageMaskDirectoryProcessor(
            lazy=True, canonical=False, target_shape=(2, 3, 4)
        )
        value = processor.process((mask_dir, self.root))
        self.assertIsInstance(value, dict)
        self.assertEqual(tuple(processor.materialize(value).shape), (1, 2, 3, 4))

    def test_dataloader_pads_variable_dicom_series(self):
        series_paths = []
        for name, count in (("a", 3), ("b", 2)):
            series_dir = self.root / name
            series_dir.mkdir()
            uid = generate_uid()
            for index in range(count):
                _write_slice(
                    series_dir / f"IMG-{index}.dcm",
                    1024,
                    float(index),
                    index + 1,
                    uid,
                    self.study_uid,
                )
            series_paths.append(series_dir)
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "image": str(series_paths[0]), "label": 0},
                {"patient_id": "p1", "image": str(series_paths[1]), "label": 1},
            ],
            {"image": ("dicom_series", {"lazy": False, "canonical": False})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            shapes = [tuple(dataset[index]["image"].shape) for index in range(2)]
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            expected = (2,) + tuple(
                max(shape[axis] for shape in shapes) for axis in range(4)
            )
            self.assertEqual(tuple(batch["image"].shape), expected)
        finally:
            dataset.close()

    def test_series_falls_back_to_instance_number(self):
        self._write_series(include_position=False)
        metadata = DicomSeriesReader().read_metadata(self.root)
        self.assertTrue(metadata["source"][0].endswith("IMG-0.dcm"))

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

    def test_lazy_mask_dataloader_with_reference_series(self):
        self._write_series()
        mask_dir = self.root / "masks"
        _write_masks(mask_dir, 3, "IMG", "-")
        dataset = create_sample_dataset(
            [
                {
                    "patient_id": "p0",
                    "mask": (str(mask_dir), str(self.root)),
                    "label": 0,
                },
                {
                    "patient_id": "p1",
                    "mask": (str(mask_dir), str(self.root)),
                    "label": 1,
                },
            ],
            {
                "mask": (
                    "image_mask_dir",
                    {"lazy": True, "canonical": False},
                )
            },
            {"label": "binary"},
            in_memory=False,
        )
        try:
            self.assertIsInstance(dataset[0]["mask"], dict)
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch["mask"].shape), (2, 1, 3, 6, 8))
        finally:
            dataset.close()

    def test_series_uid_selection(self):
        other_uid = generate_uid()
        _write_slice(
            self.root / "a.dcm",
            1024,
            0.0,
            1,
            self.series_uid,
            self.study_uid,
        )
        _write_slice(
            self.root / "b.dcm",
            1024,
            0.0,
            1,
            other_uid,
            self.study_uid,
        )
        metadata = DicomSeriesReader().read_metadata(
            self.root, series_uid=other_uid
        )
        self.assertEqual(len(metadata["source"]), 1)
        self.assertTrue(metadata["source"][0].endswith("b.dcm"))

    def test_series_processor(self):
        self._write_series()
        processor = DicomSeriesProcessor(lazy=True)
        value = processor.process(self.root)
        self.assertIsInstance(value, list)
        self.assertFalse(processor.stores_tensor)
        tensor = processor.materialize(value)
        self.assertEqual(tuple(tensor.shape), (1, 3, 6, 8))

        eager = DicomSeriesProcessor(lazy=False)
        eager_tensor = eager.process(self.root)
        self.assertTrue(eager.stores_tensor)
        self.assertEqual(tuple(eager_tensor.shape), (1, 3, 6, 8))


if __name__ == "__main__":
    unittest.main()
