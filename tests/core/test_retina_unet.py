import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RetinaUNet


class TestRetinaUNet(unittest.TestCase):
    """Basic tests for the RetinaUNet model on pseudo data."""

    def setUp(self):
        """Build a minimal dataset and model for RetinaUNet tests."""
        images = torch.randn((2, 1, 64, 64))

        dummy_masks = torch.zeros((2, 1, 64, 64))
        dummy_masks[0, 0, 10:20, 10:20] = 1  # Object 1
        dummy_masks[1, 0, 10:15, 10:15] = 1  # Object 2
        dummy_masks[1, 0, 30:40, 30:40] = 1  # Object 3

        boxes = [
            [
                torch.tensor([10, 10, 20, 20])
            ],
            [
                torch.tensor([10, 10, 15, 15]),
                torch.tensor([30, 30, 40, 40])
            ],

        ]

        classes = [
            [torch.tensor([1])],
            [torch.tensor([1, 1])],
        ]

        self.samples = []

        for i in range(images.shape[0]):
            sample = {
                "patient_id": f"p_{i}",
                "visit_id": f"v_{i}",
                "images": images[i],
                "gt_seg_masks": dummy_masks[i],
                "gt_boxes_list": boxes[i],
                "gt_classes_list": classes[i]

            }

            self.samples.append(sample)

        self.input_schema = {
            "images": "tensor",
        }

        self.output_schema = {
            "gt_seg_masks": "tensor",  # Model needs this for Seg loss
            "gt_boxes_list": "raw",  # Model needs this for Detection loss
            "gt_classes_list": "raw"  # Model needs this for Classification loss
        }

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="RetinaUnetDetection",
            task_name="ObjectDetection"
        )

        self.model = RetinaUNet(
            dataset=self.dataset,
            num_classes=2,
            dim=2,
        )

    def test_model_initialization(self):
        """Test model initializes and exposes core submodules."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, RetinaUNet)
        self.assertIsNotNone(self.model.core.fpn)
        self.assertIsNotNone(self.model.core.classification_head)
        self.assertIsNotNone(self.model.core.bbox_head)
        self.assertEqual(self.model.core.dim, 2)

    def test_model_forward(self):
        """Test forward pass returns standard model keys and correct shapes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("class_loss", ret)
        self.assertIn("bbox_loss", ret)
        self.assertIn("seg_loss", ret)
        self.assertIn("det_bboxes", ret)
        self.assertIn("class_logits", ret)
        self.assertIn("bbox_deltas", ret)
        self.assertIn("seg_logits", ret)
        self.assertIn("anchors", ret)

        # Scalar losses
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["class_loss"].dim(), 0)
        self.assertEqual(ret["bbox_loss"].dim(), 0)
        self.assertEqual(ret["seg_loss"].dim(), 0)

        # (batch, total_anchors, num_classes)
        self.assertEqual(ret["class_logits"].shape, torch.Size([2, 3060, 2]))
        # (batch, total_anchors, 4) for 2D
        self.assertEqual(ret["bbox_deltas"].shape, torch.Size([2, 3060, 4]))
        # (batch, num_seg_classes, H, W)
        self.assertEqual(ret["seg_logits"].shape, torch.Size([2, 2, 64, 64]))
        # (total_anchors, 4) for 2D
        self.assertEqual(ret["anchors"].shape, torch.Size([3060, 4]))
        # det_bboxes: (K, 7) for 2D — K varies, column count is fixed
        self.assertEqual(ret["det_bboxes"].ndim, 2)
        self.assertEqual(ret["det_bboxes"].shape[1], 7)


    def test_model_accepts_missing_labels_in_eval_mode(self):
        """Test inference mode: no labels → zero losses and correct output shapes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        inference_batch = {"images": data_batch["images"]}

        self.model.eval()
        with torch.no_grad():
            ret = self.model(**inference_batch)

        self.assertIn("loss", ret)
        self.assertIn("class_loss", ret)
        self.assertIn("bbox_loss", ret)
        self.assertIn("seg_loss", ret)
        self.assertIn("det_bboxes", ret)
        self.assertIn("class_logits", ret)
        self.assertIn("bbox_deltas", ret)
        self.assertIn("seg_logits", ret)
        self.assertIn("anchors", ret)

        # All losses must be zero when no labels are provided
        self.assertEqual(ret["loss"].item(), 0.0)
        self.assertEqual(ret["class_loss"].item(), 0.0)
        self.assertEqual(ret["bbox_loss"].item(), 0.0)
        self.assertEqual(ret["seg_loss"].item(), 0.0)

        # (batch, total_anchors, num_classes)
        self.assertEqual(ret["class_logits"].shape, torch.Size([2, 3060, 2]))
        # (batch, total_anchors, 4) for 2D
        self.assertEqual(ret["bbox_deltas"].shape, torch.Size([2, 3060, 4]))
        # (batch, num_seg_classes, H, W)
        self.assertEqual(ret["seg_logits"].shape, torch.Size([2, 2, 64, 64]))
        # (total_anchors, 4) for 2D
        self.assertEqual(ret["anchors"].shape, torch.Size([3060, 4]))
        # det_bboxes: (K, 7) for 2D — K varies with NMS, column count is fixed
        self.assertEqual(ret["det_bboxes"].ndim, 2)
        self.assertEqual(ret["det_bboxes"].shape[1], 7)

    def test_model_backward(self):
        """Test backward pass computes gradients across key submodules."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        fpn_has_grad = any(
            param.requires_grad and param.grad is not None
            for param in self.model.core.fpn.parameters()
        )
        cls_head_has_grad = any(
            param.requires_grad and param.grad is not None
            for param in self.model.core.classification_head.parameters()
        )
        seg_head_has_grad = any(
            param.requires_grad and param.grad is not None
            for param in self.model.core.segmentation_head.parameters()
        )

        self.assertTrue(fpn_has_grad, "FPN parameters should receive gradients")
        self.assertTrue(
            cls_head_has_grad,
            "Classification head parameters should receive gradients"
        )
        self.assertTrue(
            seg_head_has_grad,
            "Segmentation head parameters should receive gradients"
        )


if __name__ == "__main__":
    unittest.main()
 