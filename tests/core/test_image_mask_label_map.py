"""Tests for mask label remapping in ImageMaskDirectoryProcessor."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pyhealth.processors import ImageMaskDirectoryProcessor


class TestImageMaskLabelMap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mask_dir = Path(self.temp_dir) / "masks"
        self.mask_dir.mkdir()
        for index, value in enumerate([0, 63, 126]):
            array = np.full((4, 5), value, dtype=np.uint8)
            Image.fromarray(array, mode="L").save(
                self.mask_dir / f"slice_{index}.png"
            )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_eager_label_mapping(self):
        processor = ImageMaskDirectoryProcessor(
            canonical=False,
            lazy=False,
            label_map={0: 0, 63: 1, 126: 2},
            num_classes=3,
        )
        tensor = processor.process(self.mask_dir)
        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 5))
        self.assertEqual(sorted(tensor.unique().tolist()), [0, 1, 2])
        self.assertEqual(processor.num_classes, 3)

    def test_lazy_label_mapping(self):
        processor = ImageMaskDirectoryProcessor(
            canonical=False,
            lazy=True,
            label_map={0: 0, 63: 1, 126: 2},
        )
        ref = processor.process(self.mask_dir)
        tensor = processor.materialize(ref)
        self.assertEqual(sorted(tensor.unique().tolist()), [0, 1, 2])

    def test_unknown_label_raises(self):
        processor = ImageMaskDirectoryProcessor(
            canonical=False,
            lazy=False,
            label_map={0: 0},
        )
        with self.assertRaises(ValueError):
            processor.process(self.mask_dir)


if __name__ == "__main__":
    unittest.main()
