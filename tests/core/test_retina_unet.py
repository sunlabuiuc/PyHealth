import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RetinaUNet


class TestRetinaUNet(unittest.TestCase):
    """Basic tests for the RetinaUNet model on pseudo data."""

    def setUp(self):
        h, w = 64, 64
        y1, x1, y2, x2 = h // 4, w // 4, h // 2, w // 2

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "image": torch.randn(1, h, w).numpy().tolist(),
                "seg": np.zeros((h, w), dtype=np.int64).tolist(),
                "bb_target": np.array([[y1, x1, y2, x2]], dtype=np.float32).tolist(),
                "roi_labels": np.array([1], dtype=np.int64).tolist(),
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "image": torch.randn(1, h, w).numpy().tolist(),
                "seg": np.zeros((h, w), dtype=np.int64).tolist(),
                "bb_target": np.array([[y1, x1, y2, x2]], dtype=np.float32).tolist(),
                "roi_labels": np.array([1], dtype=np.int64).tolist(),
            },
        ]

        self.input_schema = {
            "image": "tensor",
            "seg": "tensor",
            "bb_target": "tensor",
            "roi_labels": "tensor",
        }
        self.output_schema = {"seg": "tensor"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="retina_unet_test",
        )

        self.model = RetinaUNet(
            dataset=self.dataset,
            feature_key="image",
            seg_label_key="seg",
            box_label_key="bb_target",
            class_label_key="roi_labels",
            num_seg_classes=2,
            head_classes=2,
            dim=2,
        )

    def test_dataset_initialization(self):
        """Test dataset setup has expected fields and IDs."""
        self.assertEqual(len(self.samples), 2)
        self.assertEqual(self.samples[0]["patient_id"], "patient-0")
        self.assertEqual(self.samples[0]["visit_id"], "visit-0")
        self.assertEqual(self.samples[1]["visit_id"], "visit-1")
        self.assertIsNotNone(self.dataset)

    def test_model_initialization(self):
        """Test model initializes and exposes core submodules."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, RetinaUNet)
        self.assertIsNotNone(self.model.core.Fpn)
        self.assertIsNotNone(self.model.core.Classifier)
        self.assertIsNotNone(self.model.core.BBRegressor)
        self.assertEqual(self.model.core.cf.dim, 2)
        self.assertEqual(self.model.feature_key, "image")

    def test_model_forward(self):
        """Test forward pass returns standard model keys."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_forward_with_aux_outputs(self):
        """Test optional Retina-specific outputs."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(return_aux=True, **data_batch)

        self.assertIn("seg_preds", ret)
        self.assertIn("detections", ret)
        self.assertIn("boxes", ret)
        self.assertIn("class_logits", ret)
        self.assertIn("bbox_deltas", ret)
        self.assertIn("monitor_values", ret)

    def test_model_backward(self):
        """Test backward pass computes gradients."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_grad = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameters have gradients after backward pass")


if __name__ == "__main__":
    unittest.main()
 