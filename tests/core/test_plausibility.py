"""Unit tests for plausibility metrics (Edin et al., EMNLP 2024)."""

import unittest

import numpy as np

from pyhealth.metrics.interpretability import (
    best_plausibility_threshold,
    plausibility_metrics,
    spans_to_token_ids,
)


class TestSpansToTokenIds(unittest.TestCase):
    def test_overlap_special_tokens_and_truncation(self):
        # [CLS] "chest" "pain" "fever" [SEP]
        offsets = [(0, 0), (0, 5), (6, 10), (11, 16), (0, 0)]
        self.assertEqual(spans_to_token_ids(offsets, [(6, 10)]), [2])
        self.assertEqual(spans_to_token_ids(offsets, [(3, 7)]), [1, 2])
        self.assertEqual(spans_to_token_ids(offsets, [(0, 5), (11, 16)]), [1, 3])
        self.assertEqual(spans_to_token_ids(offsets, [(40, 50)]), [])
        self.assertEqual(spans_to_token_ids(offsets, [(5, 6)]), [])


class TestPlausibilityMetrics(unittest.TestCase):
    def setUp(self):
        self.scores = [
            np.array([0.6, 0.3, 0.1, 0.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        self.evidence = [[0, 2], [3]]

    def test_hand_computed_values(self):
        out = plausibility_metrics(self.scores, self.evidence, 0.5, ks=(1, 2))
        # Pair 1 predicts {0}: tp=1, fn=1. Pair 2 predicts nothing: fn=1.
        self.assertAlmostEqual(out["precision"], 1.0)
        self.assertAlmostEqual(out["recall"], 1 / 3)
        self.assertAlmostEqual(out["f1"], 0.5)
        self.assertAlmostEqual(out["empty"], 0.5)
        # Top-1: pair 1 -> token 0 (hit); pair 2 ties -> token 0 (miss).
        self.assertAlmostEqual(out["recall@1"], 1 / 3)
        # Top-2: pair 1 -> {0, 1}; pair 2 -> {0, 1}.
        self.assertAlmostEqual(out["recall@2"], 1 / 3)

    def test_low_threshold_predicts_everything(self):
        out = plausibility_metrics(self.scores, self.evidence, 0.0)
        self.assertAlmostEqual(out["recall"], 1.0)
        self.assertAlmostEqual(out["precision"], 3 / 7)
        self.assertAlmostEqual(out["empty"], 0.0)

    def test_best_threshold(self):
        # Noise tokens score at most 0.05, so 0.05 is the smallest threshold
        # on the grid that keeps only the evidence.
        scores = [np.array([0.03, 0.94, 0.03]), np.array([0.9, 0.05, 0.05])]
        evidence = [[1], [0]]
        best = best_plausibility_threshold(scores, evidence)
        self.assertAlmostEqual(best, 0.05)
        self.assertEqual(plausibility_metrics(scores, evidence, best)["f1"], 1.0)

    def test_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            plausibility_metrics(self.scores, [[0]], 0.5)
        with self.assertRaises(ValueError):
            plausibility_metrics([], [], 0.5)


if __name__ == "__main__":
    unittest.main()
