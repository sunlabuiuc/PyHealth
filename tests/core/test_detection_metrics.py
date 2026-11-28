"""Tests for pyhealth.metrics.detection (COCO box AP)."""

import unittest

import numpy as np

from pyhealth.metrics import box_iou, detection_metrics_fn


def _split(gts, dts, n_images):
    y_true = [
        {"boxes": [b for i, c, b in gts if i == k], "labels": [c for i, c, b in gts if i == k]}
        for k in range(n_images)
    ]
    y_pred = [
        {
            "boxes": [b for i, c, b, s in dts if i == k],
            "labels": [c for i, c, b, s in dts if i == k],
            "scores": [s for i, c, b, s in dts if i == k],
        }
        for k in range(n_images)
    ]
    return y_true, y_pred


class TestDetectionMetrics(unittest.TestCase):
    def test_box_iou(self):
        iou = box_iou(np.array([[0, 0, 10, 10], [0, 0, 0, 0]]), np.array([[5, 0, 10, 10]]))
        np.testing.assert_allclose(iou, [[1 / 3], [0.0]])

    def test_matches_pycocotools(self):
        # Two classes, duplicates, a class confusion and a false positive.
        # Expected values from pycocotools 2.0 COCOeval (bbox).
        gts = [
            (0, 0, [10, 10, 40, 30]), (0, 1, [60, 50, 20, 20]),
            (1, 0, [5, 5, 50, 50]), (1, 0, [70, 70, 15, 25]),
            (2, 1, [30, 30, 30, 30]),
        ]
        dts = [
            (0, 0, [12, 11, 38, 30], 0.9), (0, 0, [10, 10, 40, 30], 0.6),
            (0, 1, [58, 52, 22, 18], 0.7), (1, 0, [8, 4, 45, 52], 0.8),
            (1, 0, [72, 66, 14, 28], 0.6), (1, 1, [5, 5, 50, 50], 0.5),
            (2, 1, [35, 28, 22, 34], 0.4), (2, 0, [0, 0, 10, 10], 0.95),
        ]
        y_true, y_pred = _split(gts, dts, 3)
        out = detection_metrics_fn(y_true, y_pred, metrics=("ap", "ap50", "ap75", "ar100"))
        expected = [0.4684488448844884, 0.7396039603960395, 0.4735973597359736, 0.6416666666666667]
        np.testing.assert_allclose(list(out.values()), expected, rtol=1e-12)

    def test_perfect_and_empty_predictions(self):
        y_true = [{"boxes": [[0, 0, 10, 10]]}, {"boxes": []}]
        perfect = [{"boxes": [[0, 0, 10, 10]], "scores": [1.0]}, {"boxes": [], "scores": []}]
        # pycocotools adds np.spacing(1) to the precision denominator.
        for value in detection_metrics_fn(y_true, perfect).values():
            self.assertAlmostEqual(value, 1.0, places=12)
        none = [{"boxes": [], "scores": []}, {"boxes": [], "scores": []}]
        self.assertEqual(detection_metrics_fn(y_true, none, metrics=("ap", "ar100")),
                         {"ap": 0.0, "ar100": 0.0})

    def test_max_detections_keeps_highest_scores(self):
        y_true = [{"boxes": [[0, 0, 10, 10]]}]
        y_pred = [{"boxes": [[50, 50, 5, 5], [0, 0, 10, 10]], "scores": [0.9, 0.1]}]
        self.assertEqual(detection_metrics_fn(y_true, y_pred, max_detections=1)["ap"], 0.0)
        self.assertEqual(detection_metrics_fn(y_true, y_pred, max_detections=2)["ap50"], 0.5)

    def test_no_ground_truth_gives_nan_and_errors(self):
        out = detection_metrics_fn([{"boxes": []}], [{"boxes": [[0, 0, 1, 1]], "scores": [1.0]}])
        self.assertTrue(np.isnan(out["ap"]))
        with self.assertRaises(ValueError):
            detection_metrics_fn([], [], metrics=("map",))
        with self.assertRaises(ValueError):
            detection_metrics_fn([{"boxes": []}], [])


if __name__ == "__main__":
    unittest.main()
