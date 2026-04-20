import unittest

import torch

from pyhealth.processors.label_processor import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
)


class TestBinaryLabelProcessor(unittest.TestCase):
    """Test cases for the BinaryLabelProcessor."""

    def test_fit_with_int_labels(self):
        """Test fitting with integer 0/1 labels."""
        processor = BinaryLabelProcessor()
        samples = [{"label": 0}, {"label": 1}, {"label": 0}]
        processor.fit(samples, "label")
        self.assertEqual(processor.label_vocab, {0: 0, 1: 1})

    def test_fit_with_bool_labels(self):
        """Test fitting with boolean labels."""
        processor = BinaryLabelProcessor()
        samples = [{"label": True}, {"label": False}]
        processor.fit(samples, "label")
        self.assertEqual(processor.label_vocab, {False: 0, True: 1})

    def test_fit_with_string_labels(self):
        """Test fitting with string labels."""
        processor = BinaryLabelProcessor()
        samples = [{"label": "yes"}, {"label": "no"}, {"label": "yes"}]
        processor.fit(samples, "label")
        self.assertEqual(len(processor.label_vocab), 2)

    def test_fit_non_binary_raises(self):
        """Test that fitting with 3+ classes raises ValueError."""
        processor = BinaryLabelProcessor()
        samples = [{"label": 0}, {"label": 1}, {"label": 2}]
        with self.assertRaises(ValueError):
            processor.fit(samples, "label")

    def test_process_returns_tensor(self):
        """Test that process returns a float tensor."""
        processor = BinaryLabelProcessor()
        samples = [{"label": 0}, {"label": 1}]
        processor.fit(samples, "label")
        result = processor.process(1)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (1,))

    def test_process_correct_values(self):
        """Test that process maps labels correctly."""
        processor = BinaryLabelProcessor()
        samples = [{"label": 0}, {"label": 1}]
        processor.fit(samples, "label")
        self.assertEqual(processor.process(0).item(), 0.0)
        self.assertEqual(processor.process(1).item(), 1.0)

    def test_size(self):
        """Test that size returns 1 for binary labels."""
        processor = BinaryLabelProcessor()
        samples = [{"label": 0}, {"label": 1}]
        processor.fit(samples, "label")
        self.assertEqual(processor.size(), 1)

    def test_schema(self):
        """Test that schema returns ('value',)."""
        processor = BinaryLabelProcessor()
        self.assertEqual(processor.schema(), ("value",))

    def test_is_not_token(self):
        """Test that binary labels are not token-based."""
        processor = BinaryLabelProcessor()
        self.assertFalse(processor.is_token())


class TestMultiClassLabelProcessor(unittest.TestCase):
    """Test cases for the MultiClassLabelProcessor."""

    def test_fit_with_int_labels(self):
        """Test fitting with sequential integer labels."""
        processor = MultiClassLabelProcessor()
        samples = [{"label": 0}, {"label": 1}, {"label": 2}]
        processor.fit(samples, "label")
        self.assertEqual(processor.label_vocab, {0: 0, 1: 1, 2: 2})

    def test_fit_with_string_labels(self):
        """Test fitting with string labels."""
        processor = MultiClassLabelProcessor()
        samples = [
            {"label": "cat"}, {"label": "dog"}, {"label": "bird"},
        ]
        processor.fit(samples, "label")
        self.assertEqual(len(processor.label_vocab), 3)

    def test_process_returns_tensor(self):
        """Test that process returns a long tensor."""
        processor = MultiClassLabelProcessor()
        samples = [{"label": 0}, {"label": 1}, {"label": 2}]
        processor.fit(samples, "label")
        result = processor.process(1)
        self.assertIsInstance(result, torch.Tensor)

    def test_process_correct_mapping(self):
        """Test that labels map to sequential indices."""
        processor = MultiClassLabelProcessor()
        samples = [
            {"label": "cat"}, {"label": "dog"}, {"label": "bird"},
        ]
        processor.fit(samples, "label")
        indices = set()
        for label in ["cat", "dog", "bird"]:
            idx = processor.process(label).item()
            indices.add(idx)
        self.assertEqual(indices, {0, 1, 2})

    def test_size(self):
        """Test that size returns number of classes."""
        processor = MultiClassLabelProcessor()
        samples = [{"label": 0}, {"label": 1}, {"label": 2}]
        processor.fit(samples, "label")
        self.assertEqual(processor.size(), 3)

    def test_schema(self):
        """Test that schema returns ('value',)."""
        processor = MultiClassLabelProcessor()
        self.assertEqual(processor.schema(), ("value",))


if __name__ == "__main__":
    unittest.main()
