"""Tests for the image mask-directory processor."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pyhealth.processors import ImageMaskDirectoryProcessor
from pyhealth.processors.image_io.readers.image_mask import ImageMaskReader
from pyhealth.processors.image_io.types import ImageMetadata, ImageRef


class TestImageMaskDirectoryProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mask_dir = Path(self.temp_dir) / "masks"
        self.mask_dir.mkdir()
        for index in range(3):
            array = np.zeros((4, 5), dtype=np.uint8)
            array[index, index] = 1
            Image.fromarray(array).save(self.mask_dir / f"mask_{index}.png")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lazy_process(self):
        processor = ImageMaskDirectoryProcessor()
        self.assertFalse(processor.stores_tensor)
        reference = processor.process(self.mask_dir)
        self.assertIsInstance(reference, ImageRef)
        self.assertEqual(reference.reader, "image_mask")
        self.assertEqual(reference.metadata.shape, (1, 3, 4, 5))

        tensor = processor.materialize(reference)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 5))
        self.assertEqual(str(tensor.dtype), "torch.int64")
        self.assertEqual(int(tensor[0, 0, 0, 0]), 1)
        self.assertEqual(int(tensor[0, 1, 1, 1]), 1)

        raw = reference.load()
        self.assertEqual(raw.shape, (1, 3, 4, 5))
        self.assertEqual(str(raw.dtype), "int64")

    def test_eager_process_matches_lazy(self):
        lazy_tensor = ImageMaskDirectoryProcessor().process(self.mask_dir)
        lazy_tensor = ImageMaskDirectoryProcessor().materialize(lazy_tensor)
        eager_processor = ImageMaskDirectoryProcessor(lazy=False)
        self.assertTrue(eager_processor.stores_tensor)
        eager_tensor = eager_processor.process(self.mask_dir)
        self.assertTrue(torch.equal(lazy_tensor, eager_tensor))

    def test_target_shape_resamples_mask(self):
        files = tuple(sorted(self.mask_dir.glob("*.png")))
        metadata = ImageMetadata(
            format="image_mask",
            shape=(1, 3, 4, 5),
            affine=tuple(tuple(row) for row in np.eye(4)),
            source=tuple(str(path) for path in files),
        )
        volume = ImageMaskReader().load(
            files,
            metadata=metadata,
            canonical=False,
            target_shape=(2, 3, 4),
        )
        self.assertEqual(tuple(volume.shape), (1, 2, 3, 4))

    def test_colour_mask_is_rejected(self):
        mask_dir = Path(self.temp_dir) / "colour"
        mask_dir.mkdir()
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(
            mask_dir / "mask.png"
        )
        with self.assertRaisesRegex(ValueError, "single-channel"):
            ImageMaskDirectoryProcessor(lazy=False).process(mask_dir)

    def test_natural_filename_order(self):
        mask_dir = Path(self.temp_dir) / "natural_order"
        mask_dir.mkdir()
        for index in (1, 2, 10):
            array = np.zeros((2, 2), dtype=np.uint8)
            array[0, 0] = index
            Image.fromarray(array).save(mask_dir / f"mask_{index}.png")
        tensor = ImageMaskDirectoryProcessor(lazy=False).process(mask_dir)
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [1, 2, 10])


if __name__ == "__main__":
    unittest.main()
