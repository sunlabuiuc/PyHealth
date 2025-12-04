"""Unit tests for the MMTM model and MMTMLayer."""

import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import MMTM, MMTMLayer


class TestMMTMLayer(unittest.TestCase):
    """Tests for standalone MMTMLayer."""

    def test_layer_forward(self):
        dim_a, dim_b = 64, 128
        layer = MMTMLayer(dim_a, dim_b)

        a = torch.randn(4, dim_a)
        b = torch.randn(4, dim_b)

        a_out, b_out = layer(a, b)

        self.assertEqual(a_out.shape, (4, dim_a))
        self.assertEqual(b_out.shape, (4, dim_b))


class TestMMTMModel(unittest.TestCase):
    """Tests for the MMTM model."""

    def setUp(self):
        """Construct a minimal realistic EHR dataset with 2 modalities."""
        samples = [
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "codes": ["250.01", "414.01"],
                "procedures": ["36.15", "88.72"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v2",
                "codes": ["518.81"],
                "procedures": ["96.71"],
                "label": 0,
            },
        ]

        self.dataset = SampleDataset(
            samples=samples,
            input_schema={
                "codes": "sequence",
                "procedures": "sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="test_mmtm",
        )

        # Use the correct PyHealth loader
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        self.batch = next(iter(loader))

    def test_initialization(self):
        """Model should initialize correctly with 2 modalities."""
        model = MMTM(self.dataset, embedding_dim=32)

        self.assertIsInstance(model, MMTM)
        self.assertEqual(len(model.feature_keys), 2)
        self.assertTrue(hasattr(model, "mmtm"))
        self.assertTrue(hasattr(model, "embedding_model"))
        self.assertTrue(hasattr(model, "fc"))

    def test_forward_output_structure(self):
        """Forward pass should return loss, y_prob, y_true, logit."""
        model = MMTM(self.dataset, embedding_dim=32)
        outputs = model(**self.batch)

        self.assertIn("loss", outputs)
        self.assertIn("y_prob", outputs)
        self.assertIn("y_true", outputs)
        self.assertIn("logit", outputs)

        self.assertIsInstance(outputs["loss"], torch.Tensor)
        self.assertEqual(outputs["logit"].shape[0], 2)

    def test_backward(self):
        """Loss should be backpropagatable."""
        model = MMTM(self.dataset, embedding_dim=32)
        outputs = model(**self.batch)

        loss = outputs["loss"]
        loss.backward()

        any_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        self.assertTrue(any_grad)

    def test_model_parameters(self):
        """Model should have trainable parameters."""
        model = MMTM(self.dataset, embedding_dim=32)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)

    def test_device_property(self):
        """BaseModel exposes device properly."""
        model = MMTM(self.dataset)
        self.assertIsInstance(model.device, torch.device)


if __name__ == "__main__":
    unittest.main()
