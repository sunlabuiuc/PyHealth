"""
Unit tests for NestedSequenceProcessor and NestedFloatsProcessor.

Tests cover:
- Basic encoding and padding
- Forward-fill for missing/empty visits
- Vocabulary building
- Edge cases (empty sequences, single visits, etc.)
"""

import unittest
import torch
from pyhealth.processors import (
    NestedSequenceProcessor,
    NestedFloatsProcessor,
)


class TestNestedSequenceProcessor(unittest.TestCase):
    """Tests for NestedSequenceProcessor (categorical codes with vocab)."""

    def test_basic_encoding(self):
        """Test basic vocabulary building and encoding."""
        processor = NestedSequenceProcessor()
        samples = [
            {"codes": [["A", "B"], ["C", "D", "E"]]},
            {"codes": [["F"]]},
        ]
        processor.fit(samples, "codes")

        # Check vocabulary size (A, B, C, D, E, F + <unk>, <pad>)
        self.assertEqual(processor.vocab_size(), 8)

        # Check max inner length (max is 3 from ["C", "D", "E"], with default padding=20, total=23)
        self.assertEqual(processor._max_inner_len, 23)

        # Test processing
        result = processor.process([["A", "B"], ["C"]])
        self.assertEqual(result.shape, (2, 23))  # 2 visits, padded to length 23
        self.assertEqual(result.dtype, torch.long)

        # Check padding is applied (second visit should be padded)
        self.assertEqual(result[1, 1].item(), processor.code_vocab["<pad>"])
        self.assertEqual(result[1, 2].item(), processor.code_vocab["<pad>"])

    def test_unknown_codes(self):
        """Test handling of unknown codes."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A", "B"]]}]
        processor.fit(samples, "codes")

        # Process with unknown code
        result = processor.process([["A", "X"]])  # X is unknown
        self.assertEqual(result[0, 1].item(), processor.code_vocab["<unk>"])

    def test_padding_empty_visits(self):
        """Test padding for empty visits."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A", "B"], ["C"], ["D", "E"]]}]
        processor.fit(samples, "codes")

        # Process with empty visit in the middle
        result = processor.process([["A", "B"], [], ["D"]])

        # Second visit (empty) should be all padding tokens
        self.assertEqual(result[1, 0].item(), processor.code_vocab["<pad>"])
        self.assertEqual(result[1, 1].item(), processor.code_vocab["<pad>"])

    def test_padding_none_visits(self):
        """Test padding for None visits."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A", "B"], ["C"]]}]
        processor.fit(samples, "codes")

        # Process with None visit
        result = processor.process([["A", "B"], None, ["C"]])

        # Second visit (None) should be all padding tokens
        self.assertEqual(result[1, 0].item(), processor.code_vocab["<pad>"])
        self.assertEqual(result[1, 1].item(), processor.code_vocab["<pad>"])

    def test_empty_first_visit(self):
        """Test when first visit is empty (no prior history)."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A", "B"]]}]
        processor.fit(samples, "codes")

        # Process with empty first visit
        result = processor.process([[], ["A"]])

        # First visit should be all padding
        self.assertEqual(result[0, 0].item(), processor.code_vocab["<pad>"])
        self.assertEqual(result[0, 1].item(), processor.code_vocab["<pad>"])

    def test_empty_sequence(self):
        """Test processing completely empty sequence."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A"]]}]
        processor.fit(samples, "codes")

        result = processor.process([])
        self.assertEqual(result.shape, (1, 1))  # Returns single padded row
        self.assertEqual(result[0, 0].item(), processor.code_vocab["<pad>"])

    def test_single_visit(self):
        """Test processing single visit."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A", "B", "C"]]}]
        processor.fit(samples, "codes")

        result = processor.process([["A", "B"]])
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result[0, 0].item(), processor.code_vocab["A"])
        self.assertEqual(result[0, 1].item(), processor.code_vocab["B"])
        self.assertEqual(result[0, 2].item(), processor.code_vocab["<pad>"])

    def test_all_empty_visits(self):
        """Test when ALL visits are empty (drug recommendation first sample)."""
        processor = NestedSequenceProcessor(padding=0)
        samples = [{"codes": [["A", "B"], ["C"]]}]
        processor.fit(samples, "codes")

        # Process with all visits being empty (like drugs_hist first sample)
        result = processor.process([[], [], []])

        # All visits should be padded
        self.assertEqual(result.shape, (3, 2))  # 3 visits, max_len=2
        # All positions should be <pad>
        for i in range(3):
            for j in range(2):
                self.assertEqual(
                    result[i, j].item(),
                    processor.code_vocab["<pad>"],
                    f"Position [{i}, {j}] should be <pad>",
                )


class TestNestedSequenceFloatsProcessor(unittest.TestCase):
    """Tests for NestedFloatsProcessor (numerical values)."""

    def test_basic_processing(self):
        """Test basic numerical processing and padding."""
        processor = NestedFloatsProcessor()
        samples = [
            {"values": [[1.0, 2.0], [3.0, 4.0, 5.0]]},
            {"values": [[6.0]]},
        ]
        processor.fit(samples, "values")

        # Check max inner length (max is 3 from [3.0, 4.0, 5.0], with default padding=20, total=23)
        self.assertEqual(processor._max_inner_len, 23)

        # Test processing
        result = processor.process([[1.0, 2.0], [3.0]])
        self.assertEqual(result.shape, (2, 23))  # 2 visits, padded to length 23
        self.assertEqual(result.dtype, torch.float)

        # Check values
        self.assertEqual(result[0, 0].item(), 1.0)
        self.assertEqual(result[0, 1].item(), 2.0)
        # Check padding is NaN initially, but forward-filled to 2.0
        self.assertEqual(result[0, 2].item(), 2.0)

    def test_forward_fill_within_features(self):
        """Test forward fill for NaN values within feature dimensions."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0, 3.0]]}]
        processor.fit(samples, "values")

        # Process with NaN values
        result = processor.process([[1.0, None, 3.0], [None, 5.0, 6.0]])

        # First visit: None at position 1 should be forward-filled to 1.0
        self.assertEqual(result[0, 0].item(), 1.0)
        self.assertEqual(result[0, 1].item(), 1.0)  # Forward filled
        self.assertEqual(result[0, 2].item(), 3.0)

        # Second visit: None at position 0 should be forward-filled to 1.0
        self.assertEqual(result[1, 0].item(), 1.0)  # Forward filled
        self.assertEqual(result[1, 1].item(), 5.0)
        self.assertEqual(result[1, 2].item(), 6.0)

    def test_forward_fill_empty_visits(self):
        """Test forward fill for empty visits."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0], [3.0]]}]
        processor.fit(samples, "values")

        # Process with empty visit in the middle
        result = processor.process([[1.0, 2.0], [], [5.0]])

        # Second visit (empty) should be forward-filled from first visit
        self.assertEqual(result[1, 0].item(), 1.0)
        self.assertEqual(result[1, 1].item(), 2.0)

    def test_forward_fill_none_visits(self):
        """Test forward fill for None visits."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0]]}]
        processor.fit(samples, "values")

        # Process with None visit
        result = processor.process([[1.0, 2.0], None, [5.0]])

        # Second visit (None) should be forward-filled from first visit
        self.assertEqual(result[1, 0].item(), 1.0)
        self.assertEqual(result[1, 1].item(), 2.0)

    def test_empty_first_visit(self):
        """Test when first visit is empty (no prior history)."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0]]}]
        processor.fit(samples, "values")

        # Process with empty first visit
        result = processor.process([[], [3.0]])

        # First visit should be zeros (no prior history)
        self.assertEqual(result[0, 0].item(), 0.0)
        self.assertEqual(result[0, 1].item(), 0.0)

    def test_none_first_visit(self):
        """Test when first visit is None (no prior history)."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0]]}]
        processor.fit(samples, "values")

        # Process with None first visit
        result = processor.process([None, [3.0]])

        # First visit should be zeros (no prior history)
        self.assertEqual(result[0, 0].item(), 0.0)
        self.assertEqual(result[0, 1].item(), 0.0)

    def test_empty_sequence(self):
        """Test processing completely empty sequence."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0]]}]
        processor.fit(samples, "values")

        result = processor.process([])
        self.assertEqual(result.shape, (1, 1))  # Returns single row
        # Should be NaN for empty sequence
        self.assertTrue(torch.isnan(result[0, 0]))

    def test_single_visit(self):
        """Test processing single visit."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0, 3.0]]}]
        processor.fit(samples, "values")

        result = processor.process([[1.0, 2.0]])
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result[0, 0].item(), 1.0)
        self.assertEqual(result[0, 1].item(), 2.0)
        # Position 2 is padded NaN, then forward filled to 2.0
        self.assertEqual(result[0, 2].item(), 2.0)

    def test_all_none_values(self):
        """Test handling when all values in a visit are None."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0]]}]
        processor.fit(samples, "values")

        # All None in first visit
        result = processor.process([[None, None], [3.0, 4.0]])

        # First visit: Nones should be forward-filled to 0.0
        self.assertEqual(result[0, 0].item(), 0.0)
        self.assertEqual(result[0, 1].item(), 0.0)

        # Second visit: Normal values
        self.assertEqual(result[1, 0].item(), 3.0)
        self.assertEqual(result[1, 1].item(), 4.0)

    def test_mixed_valid_and_invalid_types(self):
        """Test handling of mixed valid/invalid value types."""
        processor = NestedFloatsProcessor(padding=0)
        samples = [{"values": [[1.0, 2.0]]}]
        processor.fit(samples, "values")

        # Mix of valid floats, None, and strings (should convert to NaN)
        result = processor.process([[1.0, "invalid"], [None, 3.0]])

        # First visit
        self.assertEqual(result[0, 0].item(), 1.0)
        # "invalid" should be NaN, then forward filled to 1.0
        self.assertEqual(result[0, 1].item(), 1.0)

        # Second visit
        # None should be forward filled to 1.0 from previous visit
        self.assertEqual(result[1, 0].item(), 1.0)
        self.assertEqual(result[1, 1].item(), 3.0)


class TestNestedSequencePaddingFeature(unittest.TestCase):
    """Tests for the padding parameter in NestedSequenceProcessor."""

    def test_nested_sequence_padding_default(self):
        """Test NestedSequenceProcessor with default padding (20)."""
        processor = NestedSequenceProcessor()
        self.assertEqual(processor._padding, 20)

        # Fit on nested codes with max length 3
        samples = [
            {"codes": [["A", "B", "C"], ["D", "E"]]},
            {"codes": [["F"]]},
        ]
        processor.fit(samples, "codes")

        # Check that max_inner_len = observed_max (3) + padding (20) = 23
        self.assertEqual(processor._max_inner_len, 23)

        # Process a sample
        result = processor.process([["A", "B"], ["C"]])
        self.assertEqual(result.shape[1], 23)  # Padded to 23

    def test_nested_sequence_padding_custom(self):
        """Test NestedSequenceProcessor with custom padding."""
        processor = NestedSequenceProcessor(padding=50)
        self.assertEqual(processor._padding, 50)

        # Fit on nested codes with max length 2
        samples = [
            {"codes": [["A", "B"], ["C"]]},
            {"codes": [["D", "E"]]},
        ]
        processor.fit(samples, "codes")

        # Check that max_inner_len = observed_max (2) + padding (50) = 52
        self.assertEqual(processor._max_inner_len, 52)

        # Process a sample
        result = processor.process([["A"], ["B", "C"]])
        self.assertEqual(result.shape[1], 52)  # Padded to 52

    def test_nested_sequence_padding_zero(self):
        """Test NestedSequenceProcessor with zero padding."""
        processor = NestedSequenceProcessor(padding=0)
        self.assertEqual(processor._padding, 0)

        # Fit on nested codes with max length 4
        samples = [
            {"codes": [["A", "B", "C", "D"], ["E"]]},
            {"codes": [["F", "G"]]},
        ]
        processor.fit(samples, "codes")

        # Check that max_inner_len = observed_max (4) + padding (0) = 4
        self.assertEqual(processor._max_inner_len, 4)

        # Process a sample
        result = processor.process([["A", "B"], ["C"]])
        self.assertEqual(result.shape[1], 4)  # Padded to exactly 4

    def test_nested_sequence_padding_additive_behavior(self):
        """Test that padding is truly additive (observed_max + padding)."""
        processor = NestedSequenceProcessor(padding=15)

        # Fit on nested codes with max length 5
        samples = [
            {"codes": [["A", "B", "C", "D", "E"], ["F"]]},
            {"codes": [["G", "H", "I"]]},
        ]
        processor.fit(samples, "codes")

        # Verify: max_inner_len should be 5 + 15 = 20
        self.assertEqual(processor._max_inner_len, 20)

        # Process samples and verify they can handle sequences up to length 20
        result = processor.process([["A", "B", "C", "D", "E"], ["F", "G", "H"]])
        self.assertEqual(result.shape, (2, 20))  # 2 visits, padded to 20

    def test_nested_sequence_padding_repr(self):
        """Test that processor __repr__ includes padding information."""
        processor = NestedSequenceProcessor(padding=10)
        samples = [{"codes": [["A", "B"], ["C"]]}]
        processor.fit(samples, "codes")

        repr_str = repr(processor)
        self.assertIn("padding=10", repr_str)
        self.assertIn("max_inner_len=12", repr_str)  # 2 + 10


class TestNestedFloatsPaddingFeature(unittest.TestCase):
    """Tests for the padding parameter in NestedFloatsProcessor."""

    def test_nested_floats_padding_default(self):
        """Test NestedFloatsProcessor with default padding (20)."""
        processor = NestedFloatsProcessor()
        self.assertEqual(processor._padding, 20)

        # Fit on nested floats with max length 3
        samples = [
            {"values": [[1.0, 2.0, 3.0], [4.0, 5.0]]},
            {"values": [[6.0]]},
        ]
        processor.fit(samples, "values")

        # Check that max_inner_len = observed_max (3) + padding (20) = 23
        self.assertEqual(processor._max_inner_len, 23)

        # Process a sample
        result = processor.process([[1.0, 2.0], [3.0]])
        self.assertEqual(result.shape[1], 23)  # Padded to 23

    def test_nested_floats_padding_custom(self):
        """Test NestedFloatsProcessor with custom padding."""
        processor = NestedFloatsProcessor(padding=30)
        self.assertEqual(processor._padding, 30)

        # Fit on nested floats with max length 2
        samples = [
            {"values": [[1.0, 2.0], [3.0]]},
            {"values": [[4.0, 5.0]]},
        ]
        processor.fit(samples, "values")

        # Check that max_inner_len = observed_max (2) + padding (30) = 32
        self.assertEqual(processor._max_inner_len, 32)

        # Process a sample
        result = processor.process([[1.0], [2.0, 3.0]])
        self.assertEqual(result.shape[1], 32)  # Padded to 32

    def test_nested_floats_padding_zero(self):
        """Test NestedFloatsProcessor with zero padding."""
        processor = NestedFloatsProcessor(padding=0)
        self.assertEqual(processor._padding, 0)

        # Fit on nested floats with max length 4
        samples = [
            {"values": [[1.0, 2.0, 3.0, 4.0], [5.0]]},
            {"values": [[6.0, 7.0]]},
        ]
        processor.fit(samples, "values")

        # Check that max_inner_len = observed_max (4) + padding (0) = 4
        self.assertEqual(processor._max_inner_len, 4)

        # Process a sample
        result = processor.process([[1.0, 2.0], [3.0]])
        self.assertEqual(result.shape[1], 4)  # Padded to exactly 4

    def test_nested_floats_padding_additive_behavior(self):
        """Test that padding is truly additive (observed_max + padding)."""
        processor = NestedFloatsProcessor(padding=10)

        # Fit on nested floats with max length 5
        samples = [
            {"values": [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0]]},
            {"values": [[7.0, 8.0, 9.0]]},
        ]
        processor.fit(samples, "values")

        # Verify: max_inner_len should be 5 + 10 = 15
        self.assertEqual(processor._max_inner_len, 15)

        # Process samples
        result = processor.process([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        self.assertEqual(result.shape, (2, 15))  # 2 visits, padded to 15

    def test_nested_floats_padding_with_forward_fill(self):
        """Test padding interaction with forward_fill."""
        processor = NestedFloatsProcessor(forward_fill=True, padding=5)

        # Fit on nested floats with max length 2
        samples = [{"values": [[1.0, 2.0], [3.0]]}]
        processor.fit(samples, "values")

        # max_inner_len should be 2 + 5 = 7
        self.assertEqual(processor._max_inner_len, 7)

        # Process with missing values
        result = processor.process([[1.0, None], [None, 3.0]])

        # Check shape
        self.assertEqual(result.shape, (2, 7))

        # Check forward fill is applied
        # First visit: [1.0, None] -> None filled to 1.0, then padded
        self.assertAlmostEqual(result[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), 1.0, places=5)  # Forward filled

    def test_nested_floats_padding_without_forward_fill(self):
        """Test padding without forward_fill."""
        processor = NestedFloatsProcessor(forward_fill=False, padding=5)

        # Fit on nested floats with max length 2
        samples = [{"values": [[1.0, 2.0]]}]
        processor.fit(samples, "values")

        # max_inner_len should be 2 + 5 = 7
        self.assertEqual(processor._max_inner_len, 7)

        # Process with None values
        result = processor.process([[1.0, None], [None, 3.0]])

        # Check shape
        self.assertEqual(result.shape, (2, 7))

        # Without forward fill, None should become 0.0
        self.assertEqual(result[0, 1].item(), 0.0)
        self.assertEqual(result[1, 0].item(), 0.0)

    def test_nested_floats_padding_repr(self):
        """Test that processor __repr__ includes padding information."""
        processor = NestedFloatsProcessor(padding=8)
        samples = [{"values": [[1.0, 2.0], [3.0]]}]
        processor.fit(samples, "values")

        repr_str = repr(processor)
        self.assertIn("padding=8", repr_str)
        self.assertIn("max_inner_len=10", repr_str)  # 2 + 8


class TestNestedSequenceProcessorsIntegration(unittest.TestCase):
    """Integration tests for real-world drug recommendation scenarios."""

    def test_drug_recommendation_scenario(self):
        """Test realistic drug recommendation scenario with history."""
        processor = NestedSequenceProcessor(padding=0)

        # Simulate 3 patients with visit histories
        samples = [
            {
                "conditions": [["D1", "D2"], ["D3"], ["D4", "D5"]],
                "procedures": [["P1"], ["P2", "P3"], []],
            },
            {
                "conditions": [["D1"], ["D2", "D3"]],
                "procedures": [["P1", "P2"], ["P3"]],
            },
        ]

        # Fit on conditions
        processor.fit(samples, "conditions")
        result = processor.process([["D1", "D2"], [], ["D4"]])

        # Second visit (empty) should be padded
        self.assertEqual(result[1, 0].item(), processor.code_vocab["<pad>"])
        self.assertEqual(result[1, 1].item(), processor.code_vocab["<pad>"])

    def test_lab_values_scenario(self):
        """Test realistic lab values scenario with missing measurements."""
        processor = NestedFloatsProcessor(padding=0)

        # Simulate lab measurements over visits
        samples = [
            {"labs": [[98.6, 120.0, 80.0], [99.1, 125.0, 85.0]]},
            {"labs": [[98.0, 115.0, 75.0]]},
        ]

        processor.fit(samples, "labs")

        # Process with missing lab in second visit
        result = processor.process([[98.6, 120.0, 80.0], [None, 125.0, None]])

        # Second visit: None values should be forward-filled
        self.assertAlmostEqual(result[1, 0].item(), 98.6, places=5)
        self.assertEqual(result[1, 1].item(), 125.0)  # Actual value
        self.assertAlmostEqual(result[1, 2].item(), 80.0, places=5)


if __name__ == "__main__":
    unittest.main()
