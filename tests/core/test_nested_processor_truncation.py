"""Nested processors must return their fitted width for longer inputs.

When processors are fitted on a training split, a validation or test sample can
have a longer visit (or more visits per group) than anything seen in fit().
The output must still have the fitted width, otherwise samples have ragged
shapes and batching fails.
"""

import unittest

import torch

from pyhealth.processors import (
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
    NestedFloatsProcessor,
    NestedSequenceProcessor,
)

LOGGER = DEEP_LOGGER = "pyhealth.processors.base_processor"


class TestNestedSequenceTruncation(unittest.TestCase):
    def setUp(self):
        self.p = NestedSequenceProcessor()
        self.p.fit([{"v": [["a", "b"], ["c"]]}], "v")  # fitted width 2

    def test_longer_visit_is_truncated_to_fitted_width(self):
        with self.assertLogs(LOGGER, "WARNING"):
            out = self.p.process([["a", "b", "c", "d"]])
        self.assertEqual(tuple(out.shape), (1, 2))
        self.assertEqual(out.tolist(), [[self.p.code_vocab["a"], self.p.code_vocab["b"]]])

    def test_batch_of_longer_and_shorter_visits_stacks(self):
        with self.assertLogs(LOGGER, "WARNING"):
            rows = [self.p.process([["a"]]), self.p.process([["a", "b", "c"]])]
        self.assertEqual(tuple(torch.cat(rows).shape), (2, 2))

    def test_warns_once_per_processor(self):
        with self.assertLogs(LOGGER, "WARNING") as logs:
            self.p.process([["a", "b", "c"]])
            self.p.process([["a", "b", "c", "d"]])
        self.assertEqual(len(logs.records), 1)

    def test_padding_keeps_longer_visits(self):
        p = NestedSequenceProcessor(padding=2)
        p.fit([{"v": [["a", "b"]]}], "v")
        with self.assertNoLogs(LOGGER, "WARNING"):
            out = p.process([["a", "b", "c", "d"]])
        self.assertEqual(tuple(out.shape), (1, 4))


class TestNestedFloatsTruncation(unittest.TestCase):
    def test_longer_visit_is_truncated(self):
        for forward_fill in (True, False):
            with self.subTest(forward_fill=forward_fill):
                p = NestedFloatsProcessor(forward_fill=forward_fill)
                p.fit([{"v": [[1.0, 2.0]]}], "v")
                with self.assertLogs(LOGGER, "WARNING"):
                    out = p.process([[1.0, 2.0, 3.0, 4.0]])
                self.assertEqual(tuple(out.shape), (1, 2))
                self.assertEqual(out.tolist(), [[1.0, 2.0]])


class TestDeepNestedTruncation(unittest.TestCase):
    def test_codes_per_visit_are_truncated(self):
        p = DeepNestedSequenceProcessor()
        p.fit([{"v": [[["a", "b"], ["c"]]]}], "v")  # 2 visits x 2 codes
        with self.assertLogs(DEEP_LOGGER, "WARNING"):
            out = p.process([[["a", "b", "c"], ["a"]]])  # a visit with 3 codes
        self.assertEqual(tuple(out.shape), (1, 2, 2))

    def test_float_values_per_visit_are_truncated(self):
        for forward_fill in (True, False):
            with self.subTest(forward_fill=forward_fill):
                p = DeepNestedFloatsProcessor(forward_fill=forward_fill)
                p.fit([{"v": [[[1.0, 2.0], [3.0]]]}], "v")
                with self.assertLogs(DEEP_LOGGER, "WARNING"):
                    out = p.process([[[1.0, 2.0, 9.0], [3.0]]])
                self.assertEqual(tuple(out.shape), (1, 2, 2))
                self.assertEqual(out[0, 0].tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
