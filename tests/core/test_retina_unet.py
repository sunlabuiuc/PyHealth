import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RetinaUNet


class TestRetinaUNet(unittest.TestCase):
    """Unit tests for RetinaUNet with synthetic image tensors."""

    def setUp(self):
        samples = []
        for idx in range(4):
            image = [
                [float((r + c + idx) % 5) / 5.0 for c in range(32)]
                for r in range(32)
            ]
            samples.append(
                {
                    "patient_id": f"patient-{idx}",
                    "visit_id": f"visit-{idx}",
                    "image": image,
                    "label": idx % 2,
                }
            )

        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"image": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="retina_unet_toy",
        )
        self.model = RetinaUNet(dataset=self.dataset, in_channels=1, base_channels=16)

    def test_initialization(self):
        self.assertEqual(self.model.feature_key, "image")
        self.assertEqual(self.model.label_key, "label")
        self.assertEqual(self.model.in_channels, 1)
        self.assertEqual(self.model.base_channels, 16)

    def test_forward_train(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            output = self.model(**batch)

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)
        self.assertIn("seg_logit", output)
        self.assertEqual(output["y_prob"].shape[0], 2)
        self.assertEqual(output["seg_logit"].shape[0], 2)
        self.assertEqual(output["seg_logit"].shape[1], 1)
        self.assertEqual(output["loss"].dim(), 0)

    def test_backward(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        output = self.model(**batch)
        output["loss"].backward()

        has_grad = any(
            parameter.requires_grad and parameter.grad is not None
            for parameter in self.model.parameters()
        )
        self.assertTrue(has_grad)

    def test_embed_mode(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True

        with torch.no_grad():
            output = self.model(**batch)

        self.assertIn("embed", output)
        self.assertEqual(output["embed"].shape[0], 2)

    def test_custom_seg_target(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        seg_target = torch.randint(0, 2, size=(2, 1, 32, 32)).float()
        batch["seg_target"] = seg_target
        output = self.model(**batch)

        self.assertIn("seg_loss", output)
        self.assertTrue(torch.isfinite(output["seg_loss"]))


if __name__ == "__main__":
    unittest.main()
