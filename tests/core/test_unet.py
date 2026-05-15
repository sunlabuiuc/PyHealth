import unittest

import torch
from pyhealth.models.unet import UNet


class DummyDataset:
    """
    A minimal mock of SampleDataset for testing initialization.
    """

    def __init__(self, samples, input_schema, output_schema):
        self.samples = samples
        self.input_schema = input_schema
        self.output_schema = output_schema

        # BaseModel's get_output_size uses dataset.output_processors
        class DummyProcessor:
            def size(self):
                return 1

        self.output_processors = {k: DummyProcessor() for k in output_schema}

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


class TestUNet(unittest.TestCase):
    def setUp(self):
        # Use smaller images (16x16)
        self.samples = [
            {
                "image": torch.randn(1, 16, 16),
                "mask": torch.randint(0, 2, (1, 16, 16)).float(),
                "patient_id": "patient-0",
            },
            {
                "image": torch.randn(1, 16, 16),
                "mask": torch.randint(0, 2, (1, 16, 16)).float(),
                "patient_id": "patient-1",
            },
        ]
        self.dataset = DummyDataset(
            samples=self.samples,
            input_schema={"image": "image"},  # mock processor name
            output_schema={"mask": "binary"},
        )
        # Use minimal filters and depth for speed
        self.model = UNet(dataset=self.dataset, base_filters=4, depth=1, n_classes=1)

    def test_initialization(self):
        self.assertEqual(self.model.n_channels, 1)
        self.assertEqual(self.model.n_classes, 1)
        self.assertEqual(self.model.mode, "segmentation")

    def test_forward_binary(self):
        batch = {
            "image": torch.stack([s["image"] for s in self.samples]),
            "mask": torch.stack([s["mask"] for s in self.samples]),
        }
        out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("logit", out)
        self.assertEqual(out["y_prob"].shape, (2, 1, 16, 16))
        self.assertEqual(out["logit"].shape, (2, 1, 16, 16))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.all((out["y_prob"] >= 0) & (out["y_prob"] <= 1)))

    def test_forward_multiclass(self):
        dataset = DummyDataset(
            samples=self.samples, input_schema={"image": "image"}, output_schema={"mask": "multiclass"}
        )
        # Minimal filters and depth
        model = UNet(dataset=dataset, base_filters=4, depth=1, n_classes=3)

        # For multiclass, mask should be indices: (B, 1, H, W)
        mask = torch.randint(0, 3, (2, 1, 16, 16))
        batch = {
            "image": torch.stack([s["image"] for s in self.samples]),
            "mask": mask,
        }
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertEqual(out["y_prob"].shape, (2, 3, 16, 16))
        self.assertEqual(out["loss"].dim(), 0)
        # Sum of probabilities across classes should be 1
        self.assertTrue(torch.allclose(out["y_prob"].sum(dim=1), torch.ones(2, 16, 16)))

    def test_forward_without_mask(self):
        batch = {
            "image": torch.stack([s["image"] for s in self.samples]),
        }
        out = self.model(**batch)
        self.assertNotIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertEqual(out["y_prob"].shape, (2, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
