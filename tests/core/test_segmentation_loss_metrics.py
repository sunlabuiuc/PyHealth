"""Tests for segmentation losses and metrics."""

import unittest

import numpy as np
import torch

from pyhealth.metrics import (
    dice_score,
    iou_score,
    per_class_dice_score,
    per_class_segmentation_metrics,
    remap_class_mask,
    segmentation_metrics_fn,
)
from pyhealth.models import DiceLoss, SegmentationLoss


class TestSegmentationLoss(unittest.TestCase):
    def test_binary_dice_prefers_correct_logits(self):
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, :2, :] = 1
        good = torch.where(target.bool(), 10.0, -10.0)
        bad = -good
        loss = DiceLoss("segmentation_binary")
        self.assertLess(float(loss(good, target)), float(loss(bad, target)))

    def test_multiclass_loss_is_finite(self):
        logits = torch.randn(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 1, 4, 4))
        value = float(SegmentationLoss()(logits, target))
        self.assertTrue(np.isfinite(value))

    def test_partial_class_mask_ignores_unannotated_classes(self):
        torch.manual_seed(0)
        target = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.long)
        logits = torch.randn(1, 5, 2, 2)
        class_mask = [[1, 1, 0, 0, 0]]
        loss = SegmentationLoss(ignore_index=255)

        masked = loss(logits, target, class_mask=class_mask)
        baseline = loss(logits[:, :2], target)
        self.assertTrue(torch.allclose(masked, baseline, atol=1e-6))

        changed = logits.clone()
        changed[:, 2:] += 100
        changed_loss = loss(changed, target, class_mask=class_mask)
        self.assertTrue(torch.allclose(masked, changed_loss, atol=1e-6))

    def test_partial_class_mask_all_background_is_finite(self):
        logits = torch.randn(1, 3, 4, 4)
        target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        loss = SegmentationLoss(ignore_index=255)(
            logits, target, class_mask=[[1, 0, 0]]
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)

    def test_partial_class_mask_ignores_padding(self):
        logits = torch.randn(1, 5, 3, 3)
        target = torch.zeros(1, 1, 3, 3, dtype=torch.long)
        target[0, 0, 0, 0] = 1
        target[0, 0, 2, 2] = 255
        value = SegmentationLoss(ignore_index=255)(
            logits, target, class_mask=[[1, 1, 0, 0, 0]]
        )
        self.assertTrue(torch.isfinite(value))

    def test_ignore_index(self):
        logits = torch.zeros(1, 2, 3, 3)
        target = torch.full((1, 1, 3, 3), 255, dtype=torch.long)
        value = SegmentationLoss(ignore_index=255)(logits, target)
        self.assertTrue(torch.isfinite(value))


class TestSegmentationMetrics(unittest.TestCase):
    def test_empty_reference_with_false_positives_scores_zero(self):
        target = np.zeros((4, 4), dtype=np.int64)
        prob = np.zeros((4, 4), dtype=np.float32)
        prob[1:3, 1:3] = 0.9
        self.assertEqual(
            dice_score(target, prob, mode="segmentation_binary"), 0.0
        )
        self.assertEqual(
            iou_score(target, prob, mode="segmentation_binary"), 0.0
        )

    def test_empty_reference_and_prediction_scores_one(self):
        target = np.zeros((4, 4), dtype=np.int64)
        prob = np.zeros((4, 4), dtype=np.float32)
        self.assertEqual(
            dice_score(target, prob, mode="segmentation_binary"), 1.0
        )
        self.assertEqual(
            iou_score(target, prob, mode="segmentation_binary"), 1.0
        )

    def test_per_class_empty_classes_follow_the_empty_reference_rule(self):
        target = np.zeros((2, 2), dtype=np.int64)
        prob = np.zeros((3, 2, 2), dtype=np.float32)
        prob[2] = 0.9
        scores = per_class_dice_score(
            target, prob, num_classes=3, include_background=False
        )
        self.assertEqual(scores[1], 1.0)
        self.assertEqual(scores[2], 0.0)

    def test_binary_perfect_prediction(self):
        target = np.array([[0, 1], [1, 1]])
        prob = np.array([[0.0, 0.9], [0.8, 0.7]])
        self.assertAlmostEqual(
            dice_score(target, prob, mode="segmentation_binary"), 1.0
        )
        self.assertAlmostEqual(
            iou_score(target, prob, mode="segmentation_binary"), 1.0
        )

    def test_multiclass_perfect_prediction(self):
        target = np.array([[0, 1], [2, 2]])
        prob = np.array(
            [
                [[0.9, 0.1], [0.1, 0.1]],
                [[0.1, 0.8], [0.1, 0.1]],
                [[0.0, 0.1], [0.8, 0.8]],
            ]
        )
        self.assertAlmostEqual(
            dice_score(target, prob, mode="segmentation_multiclass"), 1.0
        )
        self.assertAlmostEqual(
            iou_score(target, prob, mode="segmentation_multiclass"), 1.0
        )

    def test_per_class_dice(self):
        target = np.array([[0, 1], [2, 2]])
        prob = np.array(
            [
                [[0.9, 0.1], [0.1, 0.1]],
                [[0.1, 0.8], [0.1, 0.1]],
                [[0.0, 0.1], [0.8, 0.8]],
            ]
        )
        scores = per_class_segmentation_metrics(
            [target],
            [prob],
            metrics=["dice", "iou"],
            mode="segmentation_multiclass",
            num_classes=3,
        )
        self.assertAlmostEqual(scores["dice_class_1"], 1.0)
        self.assertAlmostEqual(scores["dice_class_2"], 1.0)
        self.assertAlmostEqual(scores["iou_class_1"], 1.0)
        self.assertAlmostEqual(scores["iou_class_2"], 1.0)
        self.assertNotIn("dice_class_0", scores)

    def test_multiclass_metrics_ignore_background(self):
        target = np.zeros((4, 4), dtype=np.int64)
        target[0, :2] = 1
        # Perfect background prediction, no foreground prediction.
        prob = np.zeros((3, 4, 4), dtype=np.float32)
        prob[0] = 1.0

        self.assertAlmostEqual(
            dice_score(target, prob, mode="segmentation_multiclass"),
            0.0,
            delta=1e-5,
        )
        self.assertAlmostEqual(
            iou_score(target, prob, mode="segmentation_multiclass"), 0.0
        )
        self.assertAlmostEqual(
            segmentation_metrics_fn(
                [target], [prob], mode="segmentation_multiclass"
            )["dice"],
            0.0,
            delta=1e-5,
        )

    def test_per_class_metrics_exclude_background_by_default(self):
        target = np.zeros((4, 4), dtype=np.int64)
        target[0, :2] = 1
        prob = np.zeros((2, 4, 4), dtype=np.float32)
        prob[0] = 1.0

        scores = per_class_segmentation_metrics(
            [target], [prob], mode="segmentation_multiclass", num_classes=2
        )
        self.assertNotIn("dice_class_0", scores)
        self.assertIn("dice_class_1", scores)
        self.assertAlmostEqual(scores["dice_class_1"], 0.0, delta=1e-5)

        with_background = per_class_segmentation_metrics(
            [target],
            [prob],
            mode="segmentation_multiclass",
            num_classes=2,
            include_background=True,
        )
        self.assertIn("dice_class_0", with_background)
        self.assertGreater(with_background["dice_class_0"], 0.9)

        all_classes = per_class_dice_score(
            target, prob, mode="segmentation_multiclass", num_classes=2
        )
        self.assertEqual(sorted(all_classes), [0, 1])

    def test_remap_class_mask_non_prefix(self):
        target = np.array([[0, 1], [2, 2]])
        prob = np.array(
            [
                [[0.9, 0.1], [0.1, 0.1]],
                [[0.1, 0.8], [0.1, 0.1]],
                [[0.0, 0.1], [0.8, 0.8]],
            ]
        )
        remapped_target, remapped_prob = remap_class_mask(
            target, prob, [1, 0, 1]
        )
        np.testing.assert_array_equal(
            remapped_target, [[0, 255], [1, 1]]
        )
        self.assertEqual(remapped_prob.shape[0], 2)
        np.testing.assert_allclose(remapped_prob[0], prob[0])
        np.testing.assert_allclose(remapped_prob[1], prob[2])

    def test_per_class_metrics_ignore_out_of_range_padding(self):
        target = np.array([[0, 1], [255, 255]])
        prob = np.array(
            [
                [[0.9, 0.1], [0.9, 0.9]],
                [[0.1, 0.9], [0.1, 0.1]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        scores = per_class_segmentation_metrics(
            [target],
            [prob],
            mode="segmentation_multiclass",
            num_classes=3,
        )
        self.assertAlmostEqual(scores["dice_class_1"], 1.0)

    def test_metrics_average_over_cases(self):
        target = [np.array([[0, 1]]), np.array([[0, 0]])]
        prob = [np.array([[0.0, 0.9]]), np.array([[0.0, 0.0]])]
        scores = segmentation_metrics_fn(
            target, prob, mode="segmentation_binary"
        )
        self.assertAlmostEqual(scores["dice"], 1.0)
        self.assertAlmostEqual(scores["iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
