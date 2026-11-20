"""Tests for survival label processors."""

import unittest

import torch

from pyhealth.processors import (
    PROCESSOR_REGISTRY,
    SurvivalIntervalProcessor,
    SurvivalLabelProcessor,
)

NAN = float("nan")


class TestSurvivalIntervalProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = SurvivalIntervalProcessor()

    def test_registered(self):
        self.assertIs(
            PROCESSOR_REGISTRY["survival_interval"], SurvivalIntervalProcessor
        )

    def test_process_returns_float_tensor(self):
        out = self.processor.process([0.5, 1.25, 1])
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (3,))
        torch.testing.assert_close(out, torch.tensor([0.5, 1.25, 1.0]))

    def test_accepts_tensor_input(self):
        out = self.processor.process(torch.tensor([0.0, 2.0, 0.0]))
        torch.testing.assert_close(out, torch.tensor([0.0, 2.0, 0.0]))

    def test_rejects_malformed_intervals(self):
        for bad in (
            [1.0, 1.0, 0],
            [2.0, 1.0, 0],
            [-0.1, 1.0, 0],
            [0.0, 1.0, 2],
            [0.0, 1.0],
            [0.0, NAN, 1],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.processor.process(bad)

    def test_metadata(self):
        self.assertEqual(self.processor.size(), 3)
        self.assertFalse(self.processor.is_token())
        self.assertEqual(self.processor.dim(), (1,))
        self.assertTrue(self.processor.stores_tensor)


class TestSurvivalLabelProcessor(unittest.TestCase):
    def test_process_and_validation(self):
        processor = SurvivalLabelProcessor()
        self.assertIs(PROCESSOR_REGISTRY["survival"], SurvivalLabelProcessor)
        torch.testing.assert_close(
            processor.process([1.5, 1]), torch.tensor([1.5, 1.0])
        )
        for bad in ([0.0, 1], [-1.0, 0], [1.0, 0.5], [1.0, 0, 0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                processor.process(bad)
        self.assertEqual(processor.size(), 2)


if __name__ == "__main__":
    unittest.main()
