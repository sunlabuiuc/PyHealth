import unittest

import numpy as np
from pyhealth.metrics.segmentation import dice_score, iou_score, segmentation_metrics_fn


class TestSegmentationMetrics(unittest.TestCase):
    def setUp(self):
        # Perfect match
        self.y_true_1 = np.array([[[1, 1], [0, 0]]])
        self.y_prob_1 = np.array([[[0.9, 0.8], [0.1, 0.2]]])

        # No match
        self.y_true_2 = np.array([[[1, 1], [0, 0]]])
        self.y_prob_2 = np.array([[[0.1, 0.2], [0.9, 0.8]]])

        # Partial match (1 element out of 2 in true, 1 out of 2 in pred)
        # Intersection = 1, Union = 3
        # IoU = 1/3 = 0.333
        # Dice = 2*1 / (2+2) = 0.5
        self.y_true_3 = np.array([[[1, 1], [0, 0]]])
        self.y_prob_3 = np.array([[[0.9, 0.1], [0.8, 0.1]]])

    def test_iou_score(self):
        score1 = iou_score(self.y_true_1, self.y_prob_1)
        self.assertAlmostEqual(score1, 1.0, places=5)

        score2 = iou_score(self.y_true_2, self.y_prob_2)
        self.assertAlmostEqual(score2, 0.0, places=5)

        score3 = iou_score(self.y_true_3, self.y_prob_3)
        self.assertAlmostEqual(score3, 1.0 / 3.0, places=5)

    def test_dice_score(self):
        score1 = dice_score(self.y_true_1, self.y_prob_1)
        self.assertAlmostEqual(score1, 1.0, places=5)

        score2 = dice_score(self.y_true_2, self.y_prob_2)
        self.assertAlmostEqual(score2, 0.0, places=5)

        score3 = dice_score(self.y_true_3, self.y_prob_3)
        self.assertAlmostEqual(score3, 0.5, places=5)

    def test_segmentation_metrics_fn(self):
        results = segmentation_metrics_fn(self.y_true_3, self.y_prob_3)
        self.assertIn("iou", results)
        self.assertIn("dice", results)
        self.assertAlmostEqual(results["iou"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(results["dice"], 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
