"""Tests for generic image IO helper functions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets.imaging.io import _resolve_metadata_dir
from pyhealth.processors.image_io.readers.factory import (
    get_reader,
    get_reader_by_name,
)
from pyhealth.processors.image_io.tensor_utils import (
    _channels_first,
    _normalize_tensor,
    _resize_tensor,
    _tensor_to_pil,
    _to_torch,
)


class TestImageReaderFactory(unittest.TestCase):
    def test_known_reader_names(self):
        for name in ("nifti", "dicom", "dicom_series", "image_mask"):
            self.assertEqual(get_reader_by_name(name).name, name)

    def test_unknown_reader_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown imaging reader"):
            get_reader_by_name("unknown")

    def test_reader_inference(self):
        self.assertEqual(get_reader("scan.nii.gz").name, "nifti")
        self.assertEqual(get_reader("slice.dcm").name, "dicom")
        self.assertEqual(get_reader(("a.dcm", "b.dcm")).name, "dicom_series")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(get_reader(directory).name, "dicom_series")


class TestTensorUtils(unittest.TestCase):
    def test_to_torch_and_channels_first(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = _to_torch(array, "float32")
        self.assertEqual(tuple(tensor.shape), (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(_channels_first(array).shape), (1, 3, 4))
        rgb = np.zeros((3, 4, 2), dtype=np.float32)
        self.assertEqual(tuple(_channels_first(rgb).shape), (2, 3, 4))
        with self.assertRaisesRegex(ValueError, "2D or 3D"):
            _channels_first(np.zeros((2, 2, 2, 2), dtype=np.float32))

    def test_resize_and_normalize(self):
        tensor = torch.zeros(1, 4, 4)
        self.assertEqual(
            tuple(_resize_tensor(tensor, 8, mode="nearest").shape), (1, 8, 8)
        )
        self.assertEqual(
            tuple(_resize_tensor(tensor, None).shape), (1, 4, 4)
        )
        normalized = _normalize_tensor(torch.ones(2, 2, 2), [1.0, 2.0], [1.0, 2.0])
        self.assertEqual(tuple(normalized.shape), (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "length"):
            _normalize_tensor(torch.ones(2, 2, 2), [1.0], [1.0])

    def test_tensor_to_pil(self):
        self.assertEqual(_tensor_to_pil(torch.zeros(1, 2, 2)).mode, "L")
        self.assertEqual(_tensor_to_pil(torch.zeros(3, 2, 2)).mode, "RGB")
        self.assertEqual(_tensor_to_pil(torch.zeros(4, 2, 2)).mode, "RGBA")
        with self.assertRaisesRegex(ValueError, "channels"):
            _tensor_to_pil(torch.zeros(2, 2, 2))


class TestMetadataDirectory(unittest.TestCase):
    def test_resolve_metadata_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _resolve_metadata_dir(directory, "demo")
            self.assertEqual(root, Path(directory) / "metadata" / "demo")
