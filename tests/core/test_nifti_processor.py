"""Tests for the NIfTI reader and processor using synthetic volumes."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.processors import NiftiProcessor, read_image_metadata
from pyhealth.processors.image_io.readers.nifti import NiftiReader


@unittest.skipUnless(nib is not None, "nibabel is not installed")
class TestNiftiReaderAndProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.path = self.root / "scan.nii.gz"
        self.data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        self.affine = np.diag([2.0, 3.0, 4.0, 1.0])
        nib.save(nib.Nifti1Image(self.data, self.affine), str(self.path))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_metadata(self):
        metadata = NiftiReader().read_metadata(str(self.path))
        self.assertEqual(metadata["shape"], (1, 6, 5, 4))
        self.assertEqual(metadata["spacing"], (4.0, 3.0, 2.0))
        self.assertEqual(metadata["orientation"], "RAS")
        self.assertEqual(metadata["dtype"], "int16")
        self.assertEqual(metadata["source"], (str(self.path),))

    def test_load_returns_dhw(self):
        array = NiftiReader().load(str(self.path))
        self.assertEqual(array.shape, (1, 6, 5, 4))
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(float(array.min()), 0.0)
        self.assertEqual(float(array.max()), 119.0)

    def test_canonical_orientation(self):
        flipped = self.root / "flipped.nii.gz"
        nib.save(
            nib.Nifti1Image(self.data, np.diag([-2.0, 3.0, 4.0, 1.0])),
            str(flipped),
        )
        metadata = NiftiReader().read_metadata(str(flipped))
        self.assertEqual(metadata["orientation"], "RAS")
        self.assertEqual(metadata["shape"], (1, 6, 5, 4))
        array = NiftiReader().load(str(flipped))
        self.assertEqual(array.shape, (1, 6, 5, 4))

    def test_4d_channels(self):
        path = self.root / "multi.nii.gz"
        data = np.zeros((4, 5, 6, 2), dtype=np.float32)
        data[..., 0] = 1.0
        data[..., 1] = 2.0
        nib.save(nib.Nifti1Image(data, self.affine), str(path))
        array = NiftiReader().load(str(path), channels=2)
        self.assertEqual(array.shape, (2, 6, 5, 4))
        self.assertAlmostEqual(float(array[0].mean()), 1.0)
        self.assertAlmostEqual(float(array[1].mean()), 2.0)

    def test_channel_first_4d_is_rejected(self):
        path = self.root / "channel_first.nii.gz"
        data = np.zeros((2, 4, 5, 6), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, self.affine), str(path))
        with self.assertRaisesRegex(ValueError, "last axis"):
            NiftiReader().read_metadata(str(path), channel_axis=0)
        with self.assertRaisesRegex(ValueError, "last axis"):
            NiftiReader().load(str(path), channel_axis=0)
        with self.assertRaisesRegex(ValueError, "last axis"):
            NiftiProcessor(channel_axis=0).process(str(path))

    def test_auto_normalize_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalize='auto'"):
            NiftiProcessor(normalize="auto")

    def test_2d_target_is_rejected(self):
        path = self.root / "two_d.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((4, 5), dtype=np.float32), self.affine),
            str(path),
        )
        with self.assertRaisesRegex(ValueError, "3D NIfTI"):
            NiftiReader().load(str(path), target_shape=(2, 3, 4))

    def test_5d_nifti_is_rejected(self):
        path = self.root / "five_d.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 2, 2, 2), dtype=np.float32), self.affine),
            str(path),
        )
        with self.assertRaisesRegex(ValueError, "2, 3 or 4"):
            NiftiReader().read_metadata(str(path))

    def test_mask_keeps_labels(self):
        path = self.root / "mask.nii.gz"
        data = (self.data % 3).astype(np.uint8)
        nib.save(nib.Nifti1Image(data, self.affine), str(path))
        array = NiftiReader().load(str(path), is_mask=True)
        self.assertEqual(array.dtype, np.int64)
        self.assertEqual(set(np.unique(array).tolist()), {0, 1, 2})

    def test_resample_target_shape(self):
        array = NiftiReader().load(str(self.path), target_shape=(3, 4, 5))
        self.assertEqual(array.shape, (1, 3, 4, 5))

    def test_mask_processor_keeps_integer_dtype(self):
        path = self.root / "mask.nii.gz"
        data = (self.data % 3).astype(np.uint8)
        nib.save(nib.Nifti1Image(data, self.affine), str(path))
        tensor = NiftiProcessor(lazy=False, is_mask=True).process(str(path))
        self.assertEqual(str(tensor.dtype), "torch.int64")

    def test_lazy_path_is_absolute(self):
        relative = os.path.relpath(self.path, Path.cwd())
        value = NiftiProcessor(lazy=True).process(relative)
        self.assertTrue(Path(value).is_absolute())
        self.assertEqual(Path(value), self.path.resolve())

    def test_default_is_lazy(self):
        value = NiftiProcessor().process(str(self.path))
        self.assertIsInstance(value, str)
        self.assertFalse(NiftiProcessor().stores_tensor)

    def test_lazy_processor(self):
        processor = NiftiProcessor(lazy=True)
        value = processor.process(str(self.path))
        self.assertIsInstance(value, str)
        self.assertFalse(processor.stores_tensor)
        tensor = processor.materialize(value)
        self.assertEqual(tuple(tensor.shape), (1, 6, 5, 4))
        self.assertEqual(str(tensor.dtype), "torch.float32")

    def test_eager_processor(self):
        processor = NiftiProcessor(lazy=False)
        tensor = processor.process(str(self.path))
        self.assertTrue(processor.stores_tensor)
        self.assertEqual(tuple(tensor.shape), (1, 6, 5, 4))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            NiftiProcessor().process(str(self.root / "missing.nii.gz"))

    def test_read_image_metadata_reflects_target_shape(self):
        metadata = read_image_metadata(str(self.path), target_shape=(3, 4, 5))
        self.assertEqual(metadata["shape"], (1, 3, 4, 5))
        self.assertEqual(tuple(NiftiReader().load(str(self.path), target_shape=(3, 4, 5)).shape), (1, 3, 4, 5))

    def test_read_image_metadata_reflects_target_spacing(self):
        metadata = read_image_metadata(str(self.path), target_spacing=(1, 1, 1))
        tensor = NiftiReader().load(str(self.path), target_spacing=(1, 1, 1))
        self.assertEqual(metadata["shape"], tuple(tensor.shape))
        self.assertEqual(tuple(metadata["spacing"]), (1.0, 1.0, 1.0))

    def test_dataloader_pads_variable_volumes(self):
        from pyhealth.datasets import create_sample_dataset, get_dataloader

        second = self.root / "small.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), self.affine),
            str(second),
        )
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "image": str(self.path), "label": 0},
                {"patient_id": "p1", "image": str(second), "label": 1},
            ],
            {"image": ("nifti", {"lazy": False, "canonical": False})},
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


if __name__ == "__main__":
    unittest.main()
