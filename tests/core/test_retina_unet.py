"""Unit tests for RetinaUNet.

Contributor: Tuan Nguyen
NetID: tuanmn2
Paper: Retina U-Net: Embarrassingly Simple Exploitation of Segmentation
    Supervision for Medical Object Detection
Paper link: https://proceedings.mlr.press/v116/jaeger20a/jaeger20a.pdf
Description: Synthetic unit tests covering initialization, forward pass,
    backward pass, embedding mode, and inference mode for RetinaUNet.
"""

import unittest

import torch

from pyhealth.models import RetinaUNet


class _DummyOutputProcessor:
    def size(self):
        return 1


class _DummyDataset:
    def __init__(self):
        self.input_schema = {"image": "tensor"}
        self.output_schema = {"label": "binary"}
        self.output_processors = {"label": _DummyOutputProcessor()}


class TestRetinaUNet(unittest.TestCase):
    def setUp(self):
        self.dataset = _DummyDataset()
        self.model = RetinaUNet(
            dataset=self.dataset,
            in_channels=3,
            num_classes=2,
            base_channels=8,
        )
        self.images = torch.randn(2, 3, 64, 64)
        self.boxes = [
            torch.tensor([[10.0, 10.0, 24.0, 24.0]], dtype=torch.float32),
            torch.tensor([[30.0, 28.0, 48.0, 44.0]], dtype=torch.float32),
        ]
        self.labels = [
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ]
        self.seg_target = torch.zeros(2, 64, 64, dtype=torch.long)
        self.seg_target[0, 10:24, 10:24] = 1
        self.seg_target[1, 28:44, 30:48] = 2

    def test_initialization(self):
        self.assertEqual(self.model.feature_key, "image")
        self.assertEqual(self.model.label_key, "label")
        self.assertEqual(self.model.in_channels, 3)
        self.assertEqual(self.model.num_classes, 2)

    def test_forward_train(self):
        output = self.model(
            image=self.images,
            boxes=self.boxes,
            labels=self.labels,
            seg_target=self.seg_target,
        )

        self.assertIn("loss", output)
        self.assertIn("cls_loss", output)
        self.assertIn("bbox_loss", output)
        self.assertIn("seg_loss", output)
        self.assertIn("detections", output)
        self.assertIn("seg_logit", output)
        self.assertEqual(output["cls_logits"].shape[0], 2)
        self.assertEqual(output["seg_logit"].shape, (2, 3, 64, 64))
        self.assertEqual(len(output["detections"]), 2)
        self.assertEqual(output["loss"].dim(), 0)

    def test_backward(self):
        output = self.model(
            image=self.images,
            boxes=self.boxes,
            labels=self.labels,
            seg_target=self.seg_target,
        )
        output["loss"].backward()

        has_grad = any(
            parameter.requires_grad and parameter.grad is not None
            for parameter in self.model.parameters()
        )
        self.assertTrue(has_grad)

    def test_embed_mode(self):
        output = self.model(image=self.images, embed=True)
        self.assertIn("embed", output)
        self.assertEqual(output["embed"].shape[0], 2)

    def test_forward_inference(self):
        with torch.no_grad():
            output = self.model(image=self.images)

        self.assertIn("detections", output)
        self.assertEqual(len(output["detections"]), 2)
        self.assertIn("boxes", output["detections"][0])
        self.assertIn("scores", output["detections"][0])
        self.assertIn("labels", output["detections"][0])


if __name__ == "__main__":
    unittest.main()
