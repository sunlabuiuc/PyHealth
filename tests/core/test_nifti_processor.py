"""Tests for the NIfTI reader and processor using synthetic volumes."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - test env installs the extra
    nib = None

from pyhealth.processors import NiftiProcessor
from pyhealth.processors.image_io.readers.nifti import NiftiReader
from pyhealth.processors.image_io.types import ImageRef


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
        self.assertEqual(metadata.shape, (1, 6, 5, 4))
        self.assertEqual(metadata.spacing, (4.0, 3.0, 2.0))
        self.assertEqual(metadata.orientation, "RAS")
        self.assertEqual(metadata.dtype, "int16")
        self.assertEqual(metadata.source, (str(self.path),))

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
        self.assertEqual(metadata.orientation, "RAS")
        self.assertEqual(metadata.shape, (1, 6, 5, 4))
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

    def test_lazy_processor(self):
        processor = NiftiProcessor()
        ref = processor.process(str(self.path))
        self.assertIsInstance(ref, ImageRef)
        self.assertFalse(processor.stores_tensor)
        tensor = processor.materialize(ref)
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


if __name__ == "__main__":
    unittest.main()
