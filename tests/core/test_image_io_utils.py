"""Tests for generic image IO helper functions."""

import tempfile
import unittest

import numpy as np
import torch

from pyhealth.processors.image_io.readers.factory import get_reader, get_reader_by_name
from pyhealth.processors.image_io.tensor_utils import (
    _channels_first,
    _normalize_tensor,
    _resize_tensor,
    _tensor_to_pil,
    _to_torch,
)


class TestImageIOHelpers(unittest.TestCase):
    def test_reader_factory(self):
        for name in ("nifti", "dicom", "dicom_series", "image_mask"):
            self.assertEqual(get_reader_by_name(name).name, name)
        with self.assertRaisesRegex(ValueError, "Unknown imaging reader"):
            get_reader_by_name("unknown")
        self.assertEqual(get_reader("scan.nii.gz").name, "nifti")
        self.assertEqual(get_reader("slice.dcm").name, "dicom")
        self.assertEqual(get_reader(("a.dcm", "b.dcm")).name, "dicom_series")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(get_reader(directory).name, "dicom_series")

    def test_tensor_utils(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = _to_torch(array)
        self.assertEqual(tuple(tensor.shape), (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(_channels_first(array).shape), (1, 3, 4))
        self.assertEqual(
            tuple(_channels_first(np.zeros((3, 4, 2))).shape), (2, 3, 4)
        )
        with self.assertRaisesRegex(ValueError, "2D or 3D"):
            _channels_first(np.zeros((2, 2, 2, 2)))
        self.assertEqual(
            tuple(_resize_tensor(torch.zeros(1, 4, 4), 8, mode="nearest").shape),
            (1, 8, 8),
        )
        self.assertEqual(tuple(_resize_tensor(torch.zeros(1, 4, 4), None).shape), (1, 4, 4))
        self.assertEqual(
            tuple(_normalize_tensor(torch.ones(2, 2, 2), [1, 2], [1, 2]).shape),
            (2, 2, 2),
        )
        with self.assertRaisesRegex(ValueError, "length"):
            _normalize_tensor(torch.ones(2, 2, 2), [1], [1])
        for channels, mode in ((1, "L"), (3, "RGB"), (4, "RGBA")):
            self.assertEqual(_tensor_to_pil(torch.zeros(channels, 2, 2)).mode, mode)
        with self.assertRaisesRegex(ValueError, "channels"):
            _tensor_to_pil(torch.zeros(2, 2, 2))
