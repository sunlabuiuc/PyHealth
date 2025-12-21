"""
Unit tests for StageNet processors.

Tests cover:
- Unknown token handling (must be len(vocab) - 1, not -1)
- Vocabulary building for flat and nested codes
- Time processing
- Padding for nested sequences
- Forward-fill for numeric values
- Edge cases (empty sequences, None values, etc.)
"""

import unittest
import torch
import numpy as np

from pyhealth.processors import StageNetProcessor, StageNetTensorProcessor


class TestStageNetProcessor(unittest.TestCase):
    """Tests for StageNetProcessor (categorical codes)."""

    def test_unknown_token_index(self):
        """Test that <unk> token is len(vocab) - 1, not -1."""
        processor = StageNetProcessor()
        samples = [
            {"data": ([0.0, 1.0], [["A", "B"], ["C", "D", "E"]])},
            {"data": ([0.0], [["F"]])},
        ]
        processor.fit(samples, "data")

        # <unk> should be len(vocab) - 1 (last index)
        expected_unk_idx = len(processor.code_vocab) - 1
        self.assertEqual(processor.code_vocab["<unk>"], expected_unk_idx)

        # <unk> must be >= 0 for nn.Embedding compatibility
        self.assertGreaterEqual(processor.code_vocab["<unk>"], 0)

        # <pad> should be 0
        self.assertEqual(processor.code_vocab["<pad>"], 0)

        # Verify vocab size includes both special tokens
        # Vocab: <unk>, <pad>, A, B, C, D, E, F = 8 tokens
        self.assertEqual(len(processor.code_vocab), 8)
        self.assertEqual(processor.code_vocab["<unk>"], 7)

    def test_unknown_token_embedding_compatibility(self):
        """Test that <unk> index works with nn.Embedding."""
        processor = StageNetProcessor()
        samples = [{"data": ([0.0], [["A", "B"]])}]
        processor.fit(samples, "data")

        # Create an embedding layer with vocab_size
        vocab_size = processor.size()
        embedding = torch.nn.Embedding(vocab_size, 64)

        # Process data with unknown codes
        time, values = processor.process(([0.0], [["A", "UNKNOWN"]]))

        # Should not raise IndexError
        try:
            embedded = embedding(values)
            self.assertEqual(embedded.shape, (1, 2, 64))
        except IndexError:
            self.fail("nn.Embedding raised IndexError with <unk> token")

    def test_flat_codes(self):
        """Test processing flat code sequences."""
        processor = StageNetProcessor()
        samples = [
            {"data": ([0.0, 1.5, 2.3], ["code1", "code2", "code3"])},
        ]
        processor.fit(samples, "data")

        # Check structure detection
        self.assertFalse(processor._is_nested)

        # Process data
        time, values = processor.process(([0.0, 1.5], ["code1", "code2"]))

        # Check shapes
        self.assertEqual(time.shape, (2,))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(values.dtype, torch.long)

        # Check values are encoded correctly
        self.assertEqual(values[0].item(), processor.code_vocab["code1"])
        self.assertEqual(values[1].item(), processor.code_vocab["code2"])

    def test_nested_codes(self):
        """Test processing nested code sequences."""
        processor = StageNetProcessor(padding=0)
        samples = [
            {"data": ([0.0, 1.5], [["A", "B"], ["C", "D", "E"]])},
            {"data": ([0.0], [["F"]])},
        ]
        processor.fit(samples, "data")

        # Check structure detection
        self.assertTrue(processor._is_nested)

        # Max inner length should be 3 (from ["C", "D", "E"])
        self.assertEqual(processor._max_nested_len, 3)

        # Process data
        time, values = processor.process(([0.0, 1.5], [["A", "B"], ["C"]]))

        # Check shapes
        self.assertEqual(time.shape, (2,))
        self.assertEqual(values.shape, (2, 3))  # 2 visits, padded to 3
        self.assertEqual(values.dtype, torch.long)

        # Check padding is applied
        self.assertEqual(values[1, 1].item(), processor.code_vocab["<pad>"])
        self.assertEqual(values[1, 2].item(), processor.code_vocab["<pad>"])

    def test_nested_codes_with_padding(self):
        """Test nested codes with custom padding parameter."""
        processor = StageNetProcessor(padding=20)
        samples = [
            {"data": ([0.0, 1.5], [["A", "B"], ["C", "D", "E"]])},
        ]
        processor.fit(samples, "data")

        # Max inner length should be 3 + 20 = 23
        self.assertEqual(processor._max_nested_len, 23)
        self.assertEqual(processor._padding, 20)

        # Process data
        time, values = processor.process(([0.0], [["A", "B"]]))

        # Check shape includes padding
        self.assertEqual(values.shape, (1, 23))

    def test_unknown_codes_flat(self):
        """Test handling of unknown codes in flat sequences."""
        processor = StageNetProcessor()
        samples = [{"data": ([0.0], ["A", "B"])}]
        processor.fit(samples, "data")

        # Process with unknown code
        time, values = processor.process(([0.0, 1.0], ["A", "UNKNOWN"]))

        self.assertEqual(values[0].item(), processor.code_vocab["A"])
        self.assertEqual(values[1].item(), processor.code_vocab["<unk>"])

    def test_unknown_codes_nested(self):
        """Test handling of unknown codes in nested sequences."""
        processor = StageNetProcessor(padding=0)
        samples = [{"data": ([0.0], [["A", "B"]])}]
        processor.fit(samples, "data")

        # Process with unknown code
        time, values = processor.process(([0.0], [["A", "UNKNOWN"]]))

        self.assertEqual(values[0, 0].item(), processor.code_vocab["A"])
        self.assertEqual(values[0, 1].item(), processor.code_vocab["<unk>"])

    def test_none_codes(self):
        """Test handling of None codes."""
        processor = StageNetProcessor(padding=0)
        samples = [{"data": ([0.0], [["A", "B"]])}]
        processor.fit(samples, "data")

        # Process with None code
        time, values = processor.process(([0.0], [["A", None]]))

        self.assertEqual(values[0, 0].item(), processor.code_vocab["A"])
        self.assertEqual(values[0, 1].item(), processor.code_vocab["<unk>"])

    def test_time_processing(self):
        """Test time data processing."""
        processor = StageNetProcessor()
        samples = [{"data": ([0.0, 1.5, 2.3], ["A", "B", "C"])}]
        processor.fit(samples, "data")

        # Test with time data
        time, values = processor.process(([0.0, 1.5], ["A", "B"]))
        self.assertIsNotNone(time)
        self.assertEqual(time.dtype, torch.float)
        self.assertEqual(time[0].item(), 0.0)
        self.assertEqual(time[1].item(), 1.5)

    def test_no_time_data(self):
        """Test processing without time data."""
        processor = StageNetProcessor()
        samples = [{"data": ([0.0], ["A", "B"])}]
        processor.fit(samples, "data")

        # Process without time
        time, values = processor.process((None, ["A", "B"]))

        self.assertIsNone(time)
        self.assertEqual(values.shape, (2,))

    def test_empty_codes_flat(self):
        """Test processing empty code list (flat)."""
        processor = StageNetProcessor()
        samples = [{"data": ([0.0], ["A", "B"])}]
        processor.fit(samples, "data")

        time, values = processor.process((None, []))

        # Should return single padding token
        self.assertEqual(values.shape, (1,))
        self.assertEqual(values[0].item(), processor.code_vocab["<pad>"])

    def test_empty_codes_nested(self):
        """Test processing empty nested codes."""
        processor = StageNetProcessor(padding=0)
        samples = [{"data": ([0.0], [["A", "B"]])}]
        processor.fit(samples, "data")

        time, values = processor.process((None, []))

        # Should return single row of padding tokens
        self.assertEqual(values.shape, (1, 2))
        self.assertEqual(values[0, 0].item(), processor.code_vocab["<pad>"])
        self.assertEqual(values[0, 1].item(), processor.code_vocab["<pad>"])

    def test_vocab_size_method(self):
        """Test vocab_size() returns correct size."""
        processor = StageNetProcessor()
        samples = [
            {"data": ([0.0], [["A", "B", "C"]])},
        ]
        processor.fit(samples, "data")

        # Vocab: <unk>, <pad>, A, B, C = 5
        self.assertEqual(processor.size(), 5)
        self.assertEqual(len(processor.code_vocab), 5)

    def test_repr(self):
        """Test string representation includes key info."""
        processor = StageNetProcessor(padding=10)
        samples = [{"data": ([0.0], [["A", "B"]])}]
        processor.fit(samples, "data")

        repr_str = repr(processor)
        self.assertIn("StageNetProcessor", repr_str)
        self.assertIn("is_nested=True", repr_str)
        self.assertIn("vocab_size=4", repr_str)
        self.assertIn("max_nested_len=12", repr_str)  # 2 + 10
        self.assertIn("padding=10", repr_str)


class TestStageNetTensorProcessor(unittest.TestCase):
    """Tests for StageNetTensorProcessor (numeric values)."""

    def test_flat_numerics(self):
        """Test processing flat numeric sequences."""
        processor = StageNetTensorProcessor()
        samples = [
            {"data": ([0.0, 1.5, 2.3], [1.0, 2.0, 3.0])},
        ]
        processor.fit(samples, "data")

        # Check structure detection
        self.assertFalse(processor._is_nested)
        self.assertEqual(processor.size(), 1)

        # Process data
        time, values = processor.process(([0.0, 1.5], [1.5, 2.5]))

        # Check shapes
        self.assertEqual(time.shape, (2,))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(values.dtype, torch.float)

        # Check values
        self.assertAlmostEqual(values[0].item(), 1.5, places=5)
        self.assertAlmostEqual(values[1].item(), 2.5, places=5)

    def test_nested_numerics(self):
        """Test processing nested numeric sequences (feature vectors)."""
        processor = StageNetTensorProcessor()
        samples = [
            {"data": ([0.0, 1.5], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        ]
        processor.fit(samples, "data")

        # Check structure detection
        self.assertTrue(processor._is_nested)
        self.assertEqual(processor.size(), 3)  # 3 features

        # Process data
        time, values = processor.process(
            ([0.0, 1.5], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

        # Check shapes
        self.assertEqual(time.shape, (2,))
        self.assertEqual(values.shape, (2, 3))  # 2 timesteps, 3 features
        self.assertEqual(values.dtype, torch.float)

    def test_forward_fill_flat(self):
        """Test forward-fill imputation for flat numerics."""
        processor = StageNetTensorProcessor()
        samples = [{"data": ([0.0], [1.0, 2.0, 3.0])}]
        processor.fit(samples, "data")

        # Process with None/NaN
        time, values = processor.process(([0.0, 1.0, 2.0], [1.0, None, 3.0]))

        # None should be forward-filled to 1.0
        self.assertAlmostEqual(values[0].item(), 1.0, places=5)
        self.assertAlmostEqual(values[1].item(), 1.0, places=5)  # Forward filled
        self.assertAlmostEqual(values[2].item(), 3.0, places=5)

    def test_forward_fill_nested(self):
        """Test forward-fill imputation for nested numerics.

        Forward-fill works per feature dimension across timesteps:
        - For each feature column, None is filled with previous timestep's value
        - If no previous value exists, it becomes 0.0
        """
        processor = StageNetTensorProcessor()
        samples = [{"data": ([0.0], [[1.0, 2.0, 3.0]])}]
        processor.fit(samples, "data")

        # Process with None values
        time, values = processor.process(
            ([0.0, 1.0], [[1.0, None, 3.0], [None, 5.0, 6.0]])
        )

        # First timestep: None at position 1 becomes 0.0 (no prior value for feature 1)
        self.assertAlmostEqual(values[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(values[0, 1].item(), 0.0, places=5)  # No prior
        self.assertAlmostEqual(values[0, 2].item(), 3.0, places=5)

        # Second timestep: None at position 0 is forward-filled from first timestep
        self.assertAlmostEqual(values[1, 0].item(), 1.0, places=5)  # Forward filled
        self.assertAlmostEqual(values[1, 1].item(), 5.0, places=5)
        self.assertAlmostEqual(values[1, 2].item(), 6.0, places=5)

    def test_forward_fill_first_value_none(self):
        """Test forward-fill when first value is None (should be 0.0)."""
        processor = StageNetTensorProcessor()
        samples = [{"data": ([0.0], [1.0, 2.0])}]
        processor.fit(samples, "data")

        # Process with None as first value
        time, values = processor.process(([0.0, 1.0], [None, 2.0]))

        # First None should become 0.0 (no prior value)
        self.assertEqual(values[0].item(), 0.0)
        self.assertAlmostEqual(values[1].item(), 2.0, places=5)

    def test_time_processing_tensor(self):
        """Test time data processing for tensor processor."""
        processor = StageNetTensorProcessor()
        samples = [{"data": ([0.0, 1.5], [[1.0, 2.0]])}]
        processor.fit(samples, "data")

        # Test with time data
        time, values = processor.process(([0.0, 1.5], [[1.0, 2.0], [3.0, 4.0]]))

        self.assertIsNotNone(time)
        self.assertEqual(time.dtype, torch.float)
        self.assertEqual(time[0].item(), 0.0)
        self.assertEqual(time[1].item(), 1.5)

    def test_no_time_tensor(self):
        """Test processing without time data."""
        processor = StageNetTensorProcessor()
        samples = [{"data": ([0.0], [[1.0, 2.0]])}]
        processor.fit(samples, "data")

        # Process without time
        time, values = processor.process((None, [[1.0, 2.0], [3.0, 4.0]]))

        self.assertIsNone(time)
        self.assertEqual(values.shape, (2, 2))

    def test_repr_tensor(self):
        """Test string representation for tensor processor."""
        processor = StageNetTensorProcessor()
        samples = [{"data": ([0.0], [[1.0, 2.0, 3.0]])}]
        processor.fit(samples, "data")

        repr_str = repr(processor)
        self.assertIn("StageNetTensorProcessor", repr_str)
        self.assertIn("is_nested=True", repr_str)
        self.assertIn("feature_dim=3", repr_str)


class TestStageNetProcessorIntegration(unittest.TestCase):
    """Integration tests for realistic scenarios."""

    def test_mortality_prediction_scenario(self):
        """Test realistic mortality prediction with ICD codes and labs."""
        icd_processor = StageNetProcessor(padding=20)
        lab_processor = StageNetTensorProcessor()

        # Simulate patient data
        icd_samples = [
            {
                "icd_codes": (
                    [0.0, 24.0, 48.0],
                    [["D1", "D2"], ["D3", "D4", "D5"], ["D6"]],
                )
            },
        ]
        lab_samples = [
            {
                "labs": (
                    [0.0, 12.0, 24.0],
                    [[98.6, 120.0, 80.0], [99.1, 125.0, 85.0], [98.0, 115.0, 75.0]],
                )
            },
        ]

        icd_processor.fit(icd_samples, "icd_codes")
        lab_processor.fit(lab_samples, "labs")

        # Process new patient with unseen codes
        icd_time, icd_values = icd_processor.process(
            ([0.0, 24.0], [["D1", "NEWCODE"], ["D3"]])
        )

        # Check unknown code handling
        self.assertEqual(icd_values[0, 1].item(), icd_processor.code_vocab["<unk>"])

        # Check padding (max is 3 + 20 = 23)
        self.assertEqual(icd_values.shape[1], 23)

        # Process labs with None values
        # Forward-fill works per feature column across timesteps
        lab_time, lab_values = lab_processor.process(
            ([0.0, 12.0], [[98.6, None, 80.0], [None, 125.0, 85.0]])
        )

        # Check forward-fill for labs
        # Feature 1: None at first timestep becomes 0.0 (no prior)
        self.assertAlmostEqual(lab_values[0, 1].item(), 0.0, places=5)
        # Feature 0: None at second timestep filled from first (98.6)
        self.assertAlmostEqual(lab_values[1, 0].item(), 98.6, places=5)

    def test_vocab_size_for_embedding_layer(self):
        """Test that vocab_size() returns correct size for nn.Embedding."""
        processor = StageNetProcessor(padding=0)
        samples = [
            {"data": ([0.0], [["A", "B", "C", "D", "E"]])},
        ]
        processor.fit(samples, "data")

        # Create embedding layer
        vocab_size = processor.size()
        embedding = torch.nn.Embedding(vocab_size, 128)

        # Process data with all codes including unknown
        time, values = processor.process(([0.0], [["A", "B", "UNKNOWN"]]))

        # Should work without IndexError
        # Shape is (1, max_nested_len, 128) where max_nested_len=5
        embedded = embedding(values)
        self.assertEqual(embedded.shape[0], 1)  # 1 visit
        self.assertEqual(embedded.shape[1], 5)  # Padded to max_nested_len
        self.assertEqual(embedded.shape[2], 128)  # Embedding dim

        # Verify max index is within bounds
        max_idx = values.max().item()
        self.assertLess(max_idx, vocab_size)


if __name__ == "__main__":
    unittest.main()
