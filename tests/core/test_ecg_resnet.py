"""Tests for the plain PTB-XL ResNet-18 classifier."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import PTBXLResNet18


class TestPTBXLResNet18(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        samples = [
            {
                "patient_id": "p0",
                "signal": torch.zeros(12, 256).tolist(),
                "labels": ["NORM"],
            },
            {
                "patient_id": "p1",
                "signal": torch.ones(12, 256).tolist(),
                "labels": ["MI"],
            },
        ]
        cls.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"signal": "tensor"},
            output_schema={"labels": "multilabel"},
        )
        cls.batch = next(
            iter(get_dataloader(cls.dataset, batch_size=2, shuffle=False))
        )

    def make_model(self):
        return PTBXLResNet18(
            self.dataset,
            base_width=4,
            layers=(1, 1, 1, 1),
        )

    def test_forward_shapes_and_finite_loss(self):
        model = self.make_model()
        output = model(**self.batch)

        self.assertEqual(output["logit"].shape, (2, 2))
        self.assertEqual(output["y_prob"].shape, (2, 2))
        self.assertEqual(output["y_true"].shape, (2, 2))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(
            ((output["y_prob"] >= 0) & (output["y_prob"] <= 1)).all()
        )

    def test_forward_without_labels(self):
        model = self.make_model()
        output = model(signal=self.batch["signal"])

        self.assertEqual(output["logit"].shape, (2, 2))
        self.assertNotIn("loss", output)
        self.assertNotIn("y_true", output)

    def test_gradients_reach_backbone_and_classifier(self):
        model = self.make_model()
        output = model(**self.batch)
        output["loss"].backward()

        self.assertIsNotNone(next(model.backbone.parameters()).grad)
        self.assertIsNotNone(model.head[-1].weight.grad)

    def test_rejects_wrong_number_of_leads(self):
        model = self.make_model()

        with self.assertRaises(ValueError):
            model(signal=torch.randn(2, 6, 256))


if __name__ == "__main__":
    unittest.main()
