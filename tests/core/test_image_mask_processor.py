"""Tests for the image mask-directory processor."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.processors import (
    ImageMaskDirectoryProcessor,
    ImageProcessor,
    read_image_metadata,
)
from pyhealth.processors.image_io.readers.image_mask import ImageMaskReader


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

    def test_is_not_a_2d_image_processor(self):
        self.assertNotIsInstance(
            ImageMaskDirectoryProcessor(), ImageProcessor
        )

    def test_lazy_process(self):
        processor = ImageMaskDirectoryProcessor(lazy=True)
        self.assertFalse(processor.stores_tensor)
        value = processor.process(self.mask_dir)
        self.assertIsInstance(value, dict)
        self.assertEqual(Path(value["mask_dir"]), self.mask_dir.resolve())
        self.assertIsNone(value["series_path"])

        tensor = processor.materialize(value)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 5))
        self.assertEqual(str(tensor.dtype), "torch.int64")
        self.assertEqual(int(tensor[0, 0, 0, 0]), 1)
        self.assertEqual(int(tensor[0, 1, 1, 1]), 1)

    def test_read_image_metadata(self):
        metadata = read_image_metadata(self.mask_dir, reader="image_mask")
        self.assertEqual(metadata["shape"], (1, 3, 4, 5))
        self.assertEqual(len(metadata["source"]), 3)

    def test_eager_process_matches_lazy(self):
        lazy_tensor = ImageMaskDirectoryProcessor(lazy=True).process(self.mask_dir)
        lazy_tensor = ImageMaskDirectoryProcessor(lazy=True).materialize(lazy_tensor)
        eager_processor = ImageMaskDirectoryProcessor(lazy=False)
        self.assertTrue(eager_processor.stores_tensor)
        eager_tensor = eager_processor.process(self.mask_dir)
        self.assertTrue(torch.equal(lazy_tensor, eager_tensor))

    def test_target_shape_resamples_mask(self):
        files = tuple(sorted(self.mask_dir.glob("*.png")))
        metadata = {
            "format": "image_mask",
            "shape": (1, 3, 4, 5),
            "affine": tuple(tuple(row) for row in np.eye(4)),
            "source": tuple(str(path) for path in files),
        }
        volume = ImageMaskReader().load(
            files,
            metadata=metadata,
            canonical=False,
            target_shape=(2, 3, 4),
        )
        self.assertEqual(tuple(volume.shape), (1, 2, 3, 4))

    def test_mask_file_list_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "not as a list"):
            ImageMaskDirectoryProcessor(lazy=False).process(
                [self.mask_dir / "mask_0.png"]
            )

    def test_colour_mask_is_rejected(self):
        mask_dir = Path(self.temp_dir) / "colour"
        mask_dir.mkdir()
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(
            mask_dir / "mask.png"
        )
        with self.assertRaisesRegex(ValueError, "single-channel"):
            ImageMaskDirectoryProcessor(lazy=False).process(mask_dir)

    def test_dataloader_pads_variable_mask_volumes(self):
        second = Path(self.temp_dir) / "small"
        second.mkdir()
        for index in range(2):
            array = np.zeros((2, 3), dtype=np.uint8)
            array[0, 0] = index + 1
            Image.fromarray(array).save(second / f"mask_{index}.png")
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "mask": str(self.mask_dir), "label": 0},
                {"patient_id": "p1", "mask": str(second), "label": 1},
            ],
            {"mask": ("image_mask_dir", {"lazy": False})},
            {"label": "binary"},
            in_memory=False,
        )
        try:
            shapes = [tuple(dataset[index]["mask"].shape) for index in range(2)]
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            expected = (2,) + tuple(
                max(shape[axis] for shape in shapes) for axis in range(4)
            )
            self.assertEqual(tuple(batch["mask"].shape), expected)
        finally:
            dataset.close()

    def test_target_without_reference_series_fails_fast(self):
        processor = ImageMaskDirectoryProcessor(target_shape=(2, 3, 4))
        with self.assertRaisesRegex(ValueError, "reference affine"):
            processor.process(self.mask_dir)

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
