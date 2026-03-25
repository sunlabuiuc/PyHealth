"""
Unit tests for DataLoaderToNumpy.

Tests cover:
- Tensor/list/ndarray conversion to numpy
- Feature flattening behavior (1D, 2D, >2D)
- Batch padding and truncation
- Label processing
- Transform integration across batches
- Edge cases (inconsistent shapes, no padding, etc.)
"""

import unittest

import numpy as np
import torch

from pyhealth.models.utils import DataLoaderToNumpy


class MockDataLoader:
    """Synthetic PyHealth DataLoader for testing."""

    def __init__(self, batches):
        """Initialize with batches."""
        self.batches = batches

    def __iter__(self):
        """Return iterator over batches."""
        return iter(self.batches)


class TestDataLoaderToNumpyToNumpy(unittest.TestCase):
    """Tests _to_numpy static method."""

    def test_tensor_conversion(self):
        """Test conversion from torch.Tensor to numpy."""
        arr = DataLoaderToNumpy._to_numpy(torch.tensor([1, 2, 3]))

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)

    def test_list_conversion(self):
        """Test conversion from Python list to numpy."""
        arr = DataLoaderToNumpy._to_numpy([1, 2, 3])

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)

    def test_numpy_passthrough(self):
        """Test numpy input is preserved with correct dtype."""
        arr = DataLoaderToNumpy._to_numpy(np.array([1, 2, 3]))

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)


class TestDataLoaderToNumpyFlatten(unittest.TestCase):
    """Tests for feature flattening behavior."""

    def setUp(self):
        """Initialize converter for flattening tests."""
        self.converter = DataLoaderToNumpy(["a"], "y")

    def test_1d_to_2d(self):
        """Test that 1D arrays are reshaped to 2D."""
        arr = np.array([1, 2, 3])

        result = self.converter._flatten_feature(arr, "a")
        self.assertEqual(result.shape, (3, 1))

    def test_2d_unchanged(self):
        """Test that 2D arrays persist."""
        arr = np.array([[1, 2], [3, 4]])

        result = self.converter._flatten_feature(arr, "a")
        self.assertEqual(result.shape, (2, 2))

    def test_3d_flatten(self):
        """Test that > 2D arrays are flattened correctly."""
        arr = np.ones((2, 3, 4))

        result = self.converter._flatten_feature(arr, "a")
        self.assertEqual(result.shape, (2, 12))


class TestDataLoaderToNumpyPadding(unittest.TestCase):
    """Tests for padding and truncation across batches."""

    def test_padding_enabled(self):
        """Test padding when batch width is smaller than expected."""
        converter = DataLoaderToNumpy(["a"], "y", pad_batches=True)

        batch1 = {"a": np.array([[1, 2, 3]]), "y": [0]}
        batch2 = {"a": np.array([[4, 5]]), "y": [1]}

        converter._process_features(batch1)
        converter._fitted = True

        result = converter._process_features(batch2)

        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result[0, 2], 0.0)

    def test_truncation_enabled(self):
        """Test truncation when batch width exceeds expected."""
        converter = DataLoaderToNumpy(["a"], "y", pad_batches=True)

        batch1 = {"a": np.array([[1, 2]]), "y": [0]}
        batch2 = {"a": np.array([[3, 4, 5]]), "y": [1]}

        converter._process_features(batch1)
        converter._fitted = True

        result = converter._process_features(batch2)

        self.assertEqual(result.shape[1], 2)
        self.assertTrue(np.array_equal(result[0], [3, 4]))

    def test_padding_disabled_error(self):
        """Test error when inconsistent widths and padding disabled."""
        converter = DataLoaderToNumpy(["a"], "y", pad_batches=False)

        batch1 = {"a": np.array([[1, 2]]),    "y": [0]}
        batch2 = {"a": np.array([[3, 4, 5]]), "y": [1]}

        converter._process_features(batch1)
        converter._fitted = True

        with self.assertRaises(ValueError):
            converter._process_features(batch2)


class TestDataLoaderToNumpyFeatures(unittest.TestCase):
    """Tests feature concatenation."""

    def test_multiple_features_concat(self):
        """Test concatenation of multiple feature arrays."""
        converter = DataLoaderToNumpy(["a", "b"], "y")

        batch = {
            "a": np.array([[1, 2]]),
            "b": np.array([[3, 4, 5]]),
            "y": [0],
        }

        result = converter._process_features(batch)

        self.assertEqual(result.shape, (1, 5))
        self.assertTrue(np.array_equal(result[0], [1, 2, 3, 4, 5]))


class TestDataLoaderToNumpyLabels(unittest.TestCase):
    """Tests label processing."""

    def test_labels_flatten(self):
        """Test that labels are flattened to 1D."""
        converter = DataLoaderToNumpy(["a"], "y")

        batch = {"a": [[1]], "y": [[1], [2], [3]]}

        result = converter._process_labels(batch)

        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.array_equal(result, [1, 2, 3]))


class TestDataLoaderToNumpyTransform(unittest.TestCase):
    """Tests for transform method."""

    def test_basic_transform(self):
        """Test end-to-end transform with consistent batches."""
        converter = DataLoaderToNumpy(["a", "b"], "y")

        dataloader = MockDataLoader([
            {"a": [[1, 2]], "b": [[3, 4]], "y": [0]},
            {"a": [[5, 6]], "b": [[7, 8]], "y": [1]},
        ])

        x, y = converter.transform(dataloader)

        self.assertEqual(x.shape, (2, 4))
        self.assertEqual(y.shape, (2,))
        self.assertTrue(np.array_equal(y, [0, 1]))

    def test_padding_across_batches(self):
        """Test padding across multiple batches."""
        converter = DataLoaderToNumpy(["a"], "y", pad_batches=True)

        dataloader = MockDataLoader([
            {"a": [[1, 2, 3]], "y": [0]},
            {"a": [[4, 5]], "y": [1]},
        ])

        x, y = converter.transform(dataloader)

        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x[1, 2], 0.0)

    def test_multiple_batches_concat(self):
        """Test concatenation of batches."""
        converter = DataLoaderToNumpy(["a"], "y")

        dataloader = MockDataLoader([
            {"a": [[1], [2]], "y": [0, 1]},
            {"a": [[3], [4]], "y": [2, 3]},
        ])

        x, y = converter.transform(dataloader)

        self.assertEqual(x.shape, (4, 1))
        self.assertEqual(y.shape, (4,))
        self.assertTrue(np.array_equal(y, [0, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
