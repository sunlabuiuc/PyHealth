"""
Unit tests for DeepNestedSequenceProcessor and DeepNestedFloatsProcessor.

Tests cover:
- Basic encoding and padding for 3-level nested structures
- Forward-fill for missing/empty visits within groups
- Vocabulary building for deep categorical sequences
- Edge cases (empty groups, empty sequences, single group/visit, etc.)
"""

import unittest
import torch
from pyhealth.processors import (
    DeepNestedSequenceProcessor,
    DeepNestedFloatsProcessor,
)


class TestDeepNestedSequenceProcessor(unittest.TestCase):
    """Tests for DeepNestedSequenceProcessor (categorical codes with vocab)."""

    def test_basic_encoding(self):
        """Test basic vocabulary building and encoding for 3-level nesting."""
        processor = DeepNestedSequenceProcessor()
        samples = [
            {
                "codes": [
                    [["A", "B"], ["C", "D", "E"]],  # group 1
                    [["F"]],  # group 2
                ]
            },
            {
                "codes": [
                    [["G", "H"]],  # group 1
                ]
            },
        ]
        processor.fit(samples, "codes")

        # Check vocabulary size (A–H + <unk>, <pad>)
        self.assertEqual(processor.vocab_size(), 10)

        # Check max lengths
        self.assertEqual(processor._max_middle_len, 2)  # max visits per group
        self.assertEqual(processor._max_inner_len, 3)   # max codes per visit

        # Process a single group with two visits
        result = processor.process([[["A", "B"], ["C"]]])
        self.assertEqual(result.shape, (1, 2, 3))
        self.assertEqual(result.dtype, torch.long)

        pad = processor.code_vocab["<pad>"]

        # First visit
        self.assertEqual(result[0, 0, 0].item(), processor.code_vocab["A"])
        self.assertEqual(result[0, 0, 1].item(), processor.code_vocab["B"])
        self.assertEqual(result[0, 0, 2].item(), pad)

        # Second visit (padded)
        self.assertEqual(result[0, 1, 0].item(), processor.code_vocab["C"])
        self.assertEqual(result[0, 1, 1].item(), pad)
        self.assertEqual(result[0, 1, 2].item(), pad)

    def test_unknown_codes(self):
        """Test handling of unknown codes at the deepest level."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B"]]]}]
        processor.fit(samples, "codes")

        result = processor.process([[["A", "X"]]])  # X is unknown
        self.assertEqual(result.shape, (1, 1, 2))
        self.assertEqual(result[0, 0, 1].item(), processor.code_vocab["<unk>"])

    def test_padding_empty_inner_visits(self):
        """Test padding for empty visits inside a group."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B"], ["C"], ["D", "E"]]]}]
        processor.fit(samples, "codes")

        # One group, middle visit is empty
        result = processor.process([[["A", "B"], [], ["D"]]])

        pad = processor.code_vocab["<pad>"]

        # Shape: 1 group, 3 visits, max_len=2
        self.assertEqual(result.shape, (1, 3, 2))

        # Second visit (empty) should be all padding
        self.assertEqual(result[0, 1, 0].item(), pad)
        self.assertEqual(result[0, 1, 1].item(), pad)

    def test_padding_none_inner_visits(self):
        """Test padding for None visits inside a group."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B"], ["C"]]]}]
        processor.fit(samples, "codes")

        # One group, second visit is None
        result = processor.process([[["A", "B"], None]])

        pad = processor.code_vocab["<pad>"]

        self.assertEqual(result.shape, (1, 2, 2))
        self.assertEqual(result[0, 1, 0].item(), pad)
        self.assertEqual(result[0, 1, 1].item(), pad)

    def test_empty_first_group(self):
        """Test when first group is empty (no visits)."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B"]]]}]
        processor.fit(samples, "codes")

        # First group empty, second group has one visit
        result = processor.process([[], [["A", "B"]]])

        pad = processor.code_vocab["<pad>"]

        # Shape: 2 groups, 1 visit (max_middle_len=1), 2 codes (max_inner_len=2)
        self.assertEqual(result.shape, (2, 1, 2))

        # First group should be all padding
        self.assertEqual(result[0, 0, 0].item(), pad)
        self.assertEqual(result[0, 0, 1].item(), pad)

    def test_none_first_group(self):
        """Test when first group is None (no visits)."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B"]]]}]
        processor.fit(samples, "codes")

        # First group None, second group has one visit
        result = processor.process([None, [["A", "B"]]])

        pad = processor.code_vocab["<pad>"]

        self.assertEqual(result.shape, (2, 1, 2))
        self.assertEqual(result[0, 0, 0].item(), pad)
        self.assertEqual(result[0, 0, 1].item(), pad)

    def test_empty_sequence(self):
        """Test processing completely empty deep sequence."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A"]]]}]
        processor.fit(samples, "codes")

        result = processor.process([])

        # Returns a single group with a single padded visit
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertEqual(result[0, 0, 0].item(), processor.code_vocab["<pad>"])

    def test_single_group_single_visit(self):
        """Test processing a single group with a single visit."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B", "C"]]]}]
        processor.fit(samples, "codes")

        result = processor.process([[["A", "B"]]])

        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result[0, 0, 0].item(), processor.code_vocab["A"])
        self.assertEqual(result[0, 0, 1].item(), processor.code_vocab["B"])
        self.assertEqual(result[0, 0, 2].item(), processor.code_vocab["<pad>"])

    def test_all_empty_inner_visits(self):
        """Test when ALL visits within a group are empty."""
        processor = DeepNestedSequenceProcessor()
        samples = [{"codes": [[["A", "B"], ["C"]]]}]
        processor.fit(samples, "codes")

        # One group with all visits empty
        result = processor.process([[[], []]])

        pad = processor.code_vocab["<pad>"]

        self.assertEqual(result.shape, (1, 2, 2))

        # All positions in the group should be <pad>
        for visit_idx in range(2):
            for code_idx in range(2):
                self.assertEqual(
                    result[0, visit_idx, code_idx].item(),
                    pad,
                    f"Position [0, {visit_idx}, {code_idx}] should be <pad>",
                )


class TestDeepNestedFloatsProcessor(unittest.TestCase):
    """Tests for DeepNestedFloatsProcessor (numerical values)."""

    def test_basic_processing(self):
        """Test basic numerical processing and padding in 3D."""
        processor = DeepNestedFloatsProcessor()
        samples = [
            {"values": [[[1.0, 2.0], [3.0, 4.0, 5.0]]]},
            {"values": [[[6.0]]]},
        ]
        processor.fit(samples, "values")

        # Check max lengths
        self.assertEqual(processor._max_middle_len, 2)  # visits per group
        self.assertEqual(processor._max_inner_len, 3)   # values per visit

        # Process a single group with two visits
        result = processor.process([[[1.0, 2.0], [3.0]]])

        self.assertEqual(result.shape, (1, 2, 3))
        self.assertEqual(result.dtype, torch.float)

        # Check values
        self.assertEqual(result[0, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 0, 1].item(), 2.0)
        # Padding at position 2 is forward-filled to 2.0
        self.assertEqual(result[0, 0, 2].item(), 2.0)

    def test_forward_fill_within_features(self):
        """Test forward fill for NaN values within feature dimensions."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0, 3.0]]]}]
        processor.fit(samples, "values")

        # One group, two visits
        result = processor.process([[[1.0, None, 3.0], [None, 5.0, 6.0]]])

        # First visit: None at position 1 -> forward-filled to 1.0
        self.assertEqual(result[0, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 0, 1].item(), 1.0)
        self.assertEqual(result[0, 0, 2].item(), 3.0)

        # Second visit: None at position 0 -> forward-filled from previous visit
        self.assertEqual(result[0, 1, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 1].item(), 5.0)
        self.assertEqual(result[0, 1, 2].item(), 6.0)

    def test_forward_fill_empty_visits(self):
        """Test forward fill for empty visits within a group."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0], [3.0]]]}]
        processor.fit(samples, "values")

        # One group, empty visit in the middle
        result = processor.process([[[1.0, 2.0], [], [5.0]]])

        # Second visit (empty) should be forward-filled from first visit
        self.assertEqual(result[0, 1, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 1].item(), 2.0)

    def test_forward_fill_none_visits(self):
        """Test forward fill for None visits within a group."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0]]]}]
        processor.fit(samples, "values")

        # One group, middle visit is None
        result = processor.process([[[1.0, 2.0], None, [5.0]]])

        # Second visit (None) should be forward-filled from first visit
        self.assertEqual(result[0, 1, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 1].item(), 2.0)

    def test_empty_first_visit(self):
        """Test when first visit in a group is empty (no prior history)."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0]]]}]
        processor.fit(samples, "values")

        # One group, first visit empty
        result = processor.process([[[], [3.0]]])

        # First visit should be zeros (no prior history)
        self.assertEqual(result[0, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 0, 1].item(), 0.0)

    def test_none_first_visit(self):
        """Test when first visit in a group is None (no prior history)."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0]]]}]
        processor.fit(samples, "values")

        # One group, first visit None
        result = processor.process([[None, [3.0]]])

        # First visit should be zeros (no prior history)
        self.assertEqual(result[0, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 0, 1].item(), 0.0)

    def test_empty_sequence(self):
        """Test processing completely empty deep sequence."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0]]]}]
        processor.fit(samples, "values")

        result = processor.process([])

        # Returns single group, single visit
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertTrue(torch.isnan(result[0, 0, 0]))

    def test_single_group_single_visit(self):
        """Test processing single group with single visit."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0, 3.0]]]}]
        processor.fit(samples, "values")

        result = processor.process([[[1.0, 2.0]]])

        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result[0, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 0, 1].item(), 2.0)
        # Position 2 is padded NaN, then forward-filled to 2.0
        self.assertEqual(result[0, 0, 2].item(), 2.0)

    def test_all_none_values(self):
        """Test handling when all values in the first visit are None."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0]]]}]
        processor.fit(samples, "values")

        # One group, all None in first visit
        result = processor.process([[[None, None], [3.0, 4.0]]])

        # First visit: Nones should end up as 0.0 after forward fill + nan_to_num
        self.assertEqual(result[0, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 0, 1].item(), 0.0)

        # Second visit: Normal values
        self.assertEqual(result[0, 1, 0].item(), 3.0)
        self.assertEqual(result[0, 1, 1].item(), 4.0)

    def test_mixed_valid_and_invalid_types(self):
        """Test handling of mixed valid/invalid value types in deep nesting."""
        processor = DeepNestedFloatsProcessor()
        samples = [{"values": [[[1.0, 2.0]]]}]
        processor.fit(samples, "values")

        # One group, mixed valid floats, None, and invalid strings
        result = processor.process([[[1.0, "invalid"], [None, 3.0]]])

        # First visit
        self.assertEqual(result[0, 0, 0].item(), 1.0)
        # "invalid" -> NaN -> forward-filled to 1.0
        self.assertEqual(result[0, 0, 1].item(), 1.0)

        # Second visit
        # None at position 0 should be forward-filled from previous visit
        self.assertEqual(result[0, 1, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 1].item(), 3.0)


class TestDeepNestedSequenceProcessorsIntegration(unittest.TestCase):
    """Integration tests for deep nested scenarios (e.g., episodes x visits)."""

    def test_deep_drug_recommendation_scenario(self):
        """Test realistic deep drug recommendation scenario with history."""
        processor = DeepNestedSequenceProcessor()

        # Simulate patients with episode-level visit histories
        samples = [
            {
                "conditions": [
                    [["D1", "D2"], ["D3"]],       # episode 1
                    [["D4"], ["D5"]],             # episode 2
                ],
                "procedures": [
                    [["P1"], ["P2", "P3"]],
                    [["P4"], []],
                ],
            },
            {
                "conditions": [
                    [["D1"], ["D2", "D3"]],       # single episode
                ],
                "procedures": [
                    [["P1", "P2"], ["P3"]],
                ],
            },
        ]

        # Fit on deep nested conditions
        processor.fit(samples, "conditions")

        # Process one patient with an empty visit inside a group
        value = [
            [["D1", "D2"], []],   # episode 1
            [["D4"]],             # episode 2
        ]
        result = processor.process(value)

        pad = processor.code_vocab["<pad>"]

        # First group, second visit (empty) should be padded
        self.assertEqual(result[0, 1, 0].item(), pad)
        self.assertEqual(result[0, 1, 1].item(), pad)

    def test_deep_lab_values_scenario(self):
        """Test realistic deep lab values scenario with missing measurements."""
        processor = DeepNestedFloatsProcessor()

        # Simulate lab measurements grouped by episodes, then visits
        samples = [
            {"labs": [[[98.6, 120.0, 80.0], [99.1, 125.0, 85.0]]]},
            {"labs": [[[98.0, 115.0, 75.0]]]},
        ]

        processor.fit(samples, "labs")

        # Process with missing labs in second visit of the first episode
        value = [[[98.6, 120.0, 80.0], [None, 125.0, None]]]
        result = processor.process(value)

        # Second visit: None values should be forward-filled from first visit
        self.assertAlmostEqual(result[0, 1, 0].item(), 98.6, places=5)
        self.assertEqual(result[0, 1, 1].item(), 125.0)  # actual value
        self.assertAlmostEqual(result[0, 1, 2].item(), 80.0, places=5)


if __name__ == "__main__":
    unittest.main()
