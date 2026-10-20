"""Tests for NestedMultiHotProcessor.

The processor exists to feed set-membership models (HALO) without materialising
the padded index form, whose width is set by the single longest visit in the
dataset. These tests pin the properties that make it a safe swap: the vocabulary
lays out identically to NestedSequenceProcessor, repeats collapse, and empty
visits stay empty.
"""

import unittest

import torch

from pyhealth.processors import NestedMultiHotProcessor, NestedSequenceProcessor


class TestNestedMultiHotProcessor(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"codes": [["A", "B"], ["C"]]},
            {"codes": [["B", "D", "E"]]},
        ]
        self.proc = NestedMultiHotProcessor()
        self.proc.fit(self.samples, "codes")

    def test_vocab_matches_nested_sequence(self):
        """Same vocabulary layout, so a cached vocab restores into either.

        This is what lets an existing cohort cache be reused without a rebuild.
        """
        other = NestedSequenceProcessor()
        other.fit(self.samples, "codes")
        self.assertEqual(self.proc.code_vocab, other.code_vocab)
        self.assertEqual(self.proc.code_vocab["<pad>"], 0)
        self.assertEqual(self.proc.code_vocab["<unk>"], 1)

    def test_shape_is_visits_by_vocab(self):
        out = self.proc.process([["A"], ["B", "C"], ["D"]])
        self.assertEqual(out.shape, (3, self.proc.vocab_size()))
        self.assertEqual(out.dtype, torch.float)

    def test_marks_present_codes(self):
        out = self.proc.process([["A", "C"]])
        expected = {self.proc.code_vocab["A"], self.proc.code_vocab["C"]}
        self.assertEqual(set(out[0].nonzero().flatten().tolist()), expected)

    def test_repeats_collapse(self):
        """A code charted five times is still one bit.

        HALO's encoder already did this (``= 1``, not ``+= 1``), so preserving
        it is what keeps output identical to the index pipeline.
        """
        out = self.proc.process([["A", "A", "A", "A", "A"]])
        self.assertEqual(out.max().item(), 1.0)
        self.assertEqual(out[0].sum().item(), 1.0)

    def test_empty_visit_is_all_zero(self):
        out = self.proc.process([["A"], [], ["B"]])
        self.assertEqual(out[1].sum().item(), 0.0)

    def test_empty_sample_is_one_empty_visit(self):
        """Mirrors NestedSequenceProcessor returning a single all-<pad> row."""
        self.assertEqual(self.proc.process([]).shape, (1, self.proc.vocab_size()))
        self.assertEqual(self.proc.process([]).sum().item(), 0.0)

    def test_unknown_code_maps_to_unk(self):
        out = self.proc.process([["NOT_IN_VOCAB"]])
        self.assertEqual(out[0].nonzero().flatten().tolist(),
                         [self.proc.code_vocab["<unk>"]])

    def test_pad_column_never_set(self):
        """Column 0 must stay clear, or padding becomes indistinguishable from
        a real code."""
        out = self.proc.process([["A", "B"], [], ["C"]])
        self.assertEqual(out[:, 0].sum().item(), 0.0)

    def test_none_entries_treated_as_unknown(self):
        out = self.proc.process([[None]])
        self.assertEqual(out[0].nonzero().flatten().tolist(),
                         [self.proc.code_vocab["<unk>"]])

    def test_vocab_edit_helpers(self):
        p = NestedMultiHotProcessor()
        p.fit(self.samples, "codes")
        p.add({"Z"})
        self.assertIn("Z", p.tokens())
        p.retain({"A"})
        self.assertEqual(p.tokens(), {"<pad>", "<unk>", "A"})
        # Width follows the vocabulary, so it must shrink with it.
        self.assertEqual(p.process([["A"]]).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
