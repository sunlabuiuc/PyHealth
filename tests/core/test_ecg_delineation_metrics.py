"""Deterministic hand-calculated ECG metric fixtures; no data downloads.

Contributors: Thi Van Trinh Tran and Zexi Li.
"""

import unittest

import numpy as np

from pyhealth.metrics.ecg_delineation import (
    assemble_ecg_sequences,
    ecg_delineation_metrics,
    segment_mean_iou,
)


class TestECGDelineationMetrics(unittest.TestCase):
    def test_point_scores_and_fixed_class_denominators(self):
        scores = ecg_delineation_metrics(
            [[0, 1, 2, 3]], [[0, 2, 2, 0]], [[False, False, False, True]],
        )
        self.assertAlmostEqual(scores["supervised_accuracy"], 2 / 3)
        self.assertAlmostEqual(scores["macro_precision"], 3 / 8)
        self.assertAlmostEqual(scores["macro_recall"], 1 / 2)
        self.assertAlmostEqual(scores["macro_f1"], 5 / 12)
        self.assertAlmostEqual(scores["macro_iou"], 3 / 8)
        self.assertEqual(sum(map(sum, scores["confusion_matrix"])), 3)
        self.assertEqual(scores["ignored_points"], 1)
        self.assertEqual(scores["per_class_iou"]["T"], 0)
        self.assertNotIn("segment_miou", scores)

    def test_ignored_predictions_cannot_change_scores(self):
        mask = [False, True, False, True]
        first = ecg_delineation_metrics([0, 1, 2, 3], [0, 0, 2, 0], mask)
        second = ecg_delineation_metrics([0, 1, 2, 3], [0, 3, 2, 3], mask)
        self.assertEqual(first, second)
        empty = ecg_delineation_metrics([[0, 1]], [[3, 3]], [[True, True]])
        self.assertEqual(empty["supervised_points"], 0)
        self.assertEqual(empty["skipped_windows"], 1)
        self.assertEqual(empty["macro_iou"], 0)

    def test_segments_differ_from_point_iou_and_ignores_break_segments(self):
        true, pred = np.ones(5, dtype=int), np.array([1, 0, 1, 0, 1])
        mean, per_class = segment_mean_iou([(true, pred, [True] * 5)], num_classes=2)
        # One true segment vs three predicted: one 1/5 match, two zeros.
        self.assertAlmostEqual(per_class[1], 1 / 15)
        self.assertAlmostEqual(mean, 1 / 30)
        point = ecg_delineation_metrics(true, pred, [False] * 5, num_classes=2)
        self.assertAlmostEqual(point["macro_iou"], 0.3)
        mean, _ = segment_mean_iou([
            (true, pred, [True, False, True, False, True]),
        ], num_classes=2)
        self.assertEqual(mean, 0.5)  # three correct segments; background absent

    def test_sequence_assembly_keeps_patient_lead_and_gap_boundaries(self):
        def entry(start):
            return (start, start + 2, [0, 1], [0, 1], [True, True])

        parts = {(1, 0): [entry(2), entry(0), entry(8)],
                 (1, 1): [entry(0)], (2, 0): [entry(0)]}
        sequences = assemble_ecg_sequences(parts)
        self.assertEqual([len(values[0]) for values in sequences], [4, 2, 2, 2])
        mean, _ = segment_mean_iou(sequences, num_classes=2)
        self.assertEqual(mean, 1)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            assemble_ecg_sequences({(1, 0): [entry(0), entry(1)]})

    def test_invalid_shapes_and_class_indices_fail(self):
        with self.assertRaises(ValueError):
            ecg_delineation_metrics([0], [4], [False])
        with self.assertRaises(ValueError):
            ecg_delineation_metrics([0, 1], [0], [False])
        with self.assertRaises(ValueError):
            segment_mean_iou([([[0]], [[0]], [[True]])])
