"""Tests for the Grad-CAM interpretability method."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import GradCAM
from pyhealth.models import TorchvisionModel


class SimpleProbCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, **kwargs):
        x = kwargs["image"]
        feats = self.conv(x)
        logits = self.fc(self.pool(feats).flatten(1))
        return {"y_prob": torch.softmax(logits, dim=1)}


class SimpleLogitCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, **kwargs):
        x = kwargs["image"]
        feats = self.conv(x)
        logits = self.fc(self.pool(feats).flatten(1))
        return {
            "logit": logits,
            "y_prob": torch.softmax(logits, dim=1),
        }


class TorchvisionLogitShim(nn.Module):
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.model = core_model

    def forward(self, **kwargs):
        x = kwargs["image"]
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        return {"logit": self.model(x)}


class TestGradCAMToyCNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = {"image": torch.randn(2, 3, 32, 32)}

    def test_gradcam_forward_backward_shape_and_batch_support(self):
        model = SimpleLogitCNN()
        gradcam = GradCAM(model, target_layer=model.conv)

        attributions = gradcam.attribute(**self.batch)

        self.assertIn("image", attributions)
        self.assertEqual(attributions["image"].shape, (2, 32, 32))

    def test_gradcam_default_target_uses_prediction(self):
        model = SimpleProbCNN()
        expected = model(**self.batch)["y_prob"].argmax(dim=1)
        gradcam = GradCAM(model, target_layer=model.conv)

        gradcam.attribute(**self.batch)

        self.assertTrue(torch.equal(gradcam.last_target_class, expected.cpu()))

    def test_gradcam_explicit_target(self):
        model = SimpleLogitCNN(num_classes=3)
        gradcam = GradCAM(model, target_layer=model.conv)

        gradcam.attribute(class_index=1, **self.batch)

        self.assertTrue(torch.equal(gradcam.last_target_class, torch.tensor([1, 1])))

    def test_gradcam_bad_layer_path(self):
        model = SimpleLogitCNN()
        with self.assertRaises(ValueError):
            GradCAM(model, target_layer="missing.layer")

    def test_gradcam_non_spatial_layer_error(self):
        model = SimpleLogitCNN()
        gradcam = GradCAM(model, target_layer="fc")
        with self.assertRaises(ValueError):
            gradcam.attribute(**self.batch)

    def test_gradcam_normalization(self):
        model = SimpleLogitCNN()
        gradcam = GradCAM(model, target_layer=model.conv)

        cam = gradcam.attribute(normalize=True, **self.batch)["image"]

        self.assertTrue(torch.all(cam >= 0))
        self.assertTrue(torch.all(cam <= 1))

    def test_gradcam_y_prob_fallback(self):
        model = SimpleProbCNN()
        gradcam = GradCAM(model, target_layer="conv")

        attributions = gradcam.attribute(**self.batch)

        self.assertEqual(attributions["image"].shape, (2, 32, 32))

    def test_gradcam_missing_input_key(self):
        model = SimpleLogitCNN()
        gradcam = GradCAM(model, target_layer=model.conv, input_key="xray")

        with self.assertRaises(KeyError):
            gradcam.attribute(**self.batch)

    def test_gradcam_invalid_class_index_raises_value_error(self):
        model = SimpleLogitCNN(num_classes=3)
        gradcam = GradCAM(model, target_layer=model.conv)

        with self.assertRaises(ValueError):
            gradcam.attribute(class_index=5, **self.batch)

    def test_gradcam_invalid_class_index_tensor_shape_raises_value_error(self):
        model = SimpleLogitCNN(num_classes=3)
        gradcam = GradCAM(model, target_layer=model.conv)

        with self.assertRaises(ValueError):
            gradcam.attribute(class_index=torch.tensor([1]), **self.batch)

    def test_gradcam_no_grad_context_raises_runtime_error(self):
        model = SimpleLogitCNN()
        gradcam = GradCAM(model, target_layer=model.conv)

        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                gradcam.attribute(**self.batch)


class TestGradCAMTorchvisionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.samples = []
        for i in range(4):
            # Build tiny synthetic grayscale PNGs on disk for wrapper-level tests.
            img_path = os.path.join(cls.temp_dir, f"img_{i}.png")
            image = Image.fromarray(
                np.random.randint(0, 255, (64, 64), dtype=np.uint8),
                mode="L",
            )
            image.save(img_path)
            cls.samples.append(
                {
                    "patient_id": f"p{i // 2}",
                    "visit_id": f"v{i}",
                    "image": img_path,
                    "label": i % 2,
                }
            )

        cls.dataset = create_sample_dataset(
            samples=cls.samples,
            input_schema={"image": ("image", {"image_size": 64, "mode": "L"})},
            output_schema={"label": "binary"},
            dataset_name="gradcam_image_smoke",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_gradcam_torchvisionmodel_smoke(self):
        model = TorchvisionModel(
            dataset=self.dataset,
            model_name="resnet18",
            model_config={"weights": None},
        )
        model.eval()

        batch = next(iter(get_dataloader(self.dataset, batch_size=1, shuffle=False)))
        gradcam = GradCAM(model, target_layer="model.layer4.1.conv2")

        attributions = gradcam.attribute(**batch)

        self.assertIn("image", attributions)
        self.assertEqual(attributions["image"].shape, (1, 64, 64))
        self.assertTrue(torch.all(attributions["image"] >= 0))
        self.assertTrue(torch.all(attributions["image"] <= 1))

    def test_gradcam_torchvisionmodel_matches_direct_logit_path(self):
        model = TorchvisionModel(
            dataset=self.dataset,
            model_name="resnet18",
            model_config={"weights": None},
        )
        model.eval()

        batch = next(iter(get_dataloader(self.dataset, batch_size=1, shuffle=False)))
        wrapper_cam = GradCAM(
            model,
            target_layer="model.layer4.1.conv2",
        ).attribute(
            **batch
        )["image"]
        shim_cam = GradCAM(
            TorchvisionLogitShim(model.model),
            target_layer="model.layer4.1.conv2",
        ).attribute(image=batch["image"])["image"]

        self.assertTrue(torch.allclose(wrapper_cam, shim_cam, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
