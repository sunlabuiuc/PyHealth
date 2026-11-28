"""Tests for pyhealth.metrics.fairness_selection."""

import unittest

import numpy as np

from pyhealth.metrics.fairness_selection import (
    dto_select,
    minimax_pareto_select,
    pareto_front,
)

# rows: candidate models, columns: per-group validation AUC
SCORES = np.array(
    [
        [0.80, 0.70],
        [0.78, 0.76],
        [0.85, 0.60],
        [0.75, 0.74],
    ]
)


class TestFairnessSelection(unittest.TestCase):
    def test_dto_select(self):
        distances = np.linalg.norm(1 - SCORES / SCORES.max(axis=0), axis=1)
        self.assertEqual(dto_select(SCORES), int(np.argmin(distances)))
        self.assertEqual(dto_select(SCORES), 1)

    def test_dto_prefers_utopia(self):
        scores = np.array([[0.9, 0.5], [0.6, 0.8], [0.9, 0.8]])
        self.assertEqual(dto_select(scores), 2)

    def test_pareto_front_drops_dominated(self):
        np.testing.assert_array_equal(pareto_front(SCORES), [0, 1, 2])

    def test_pareto_front_keeps_ties(self):
        scores = np.array([[0.8, 0.8], [0.8, 0.8], [0.7, 0.7]])
        np.testing.assert_array_equal(pareto_front(scores), [0, 1])

    def test_minimax_maximises_worst_group(self):
        self.assertEqual(minimax_pareto_select(SCORES), 1)

    def test_minimax_reference_group(self):
        self.assertEqual(minimax_pareto_select(SCORES, group=0), 2)

    def test_single_candidate(self):
        self.assertEqual(dto_select(SCORES[:1]), 0)
        self.assertEqual(minimax_pareto_select(SCORES[:1]), 0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            dto_select(np.array([0.8, 0.7]))
        with self.assertRaises(ValueError):
            dto_select(np.array([[0.8, np.nan]]))
        with self.assertRaises(ValueError):
            minimax_pareto_select(SCORES, group=2)
        with self.assertRaises(ValueError):
            dto_select(np.array([[0.8, 0.0], [0.7, 0.0]]))
        with self.assertRaises(ValueError):
            minimax_pareto_select(np.array([[0.8, np.inf], [0.7, 0.6]]))


if __name__ == "__main__":
    unittest.main()
