"""Test cases for the DuETT model.

Author: Shubham Srivastava (ss253@illinois.edu)

Description:
    Unit tests for the DuETT model implementation. Tests cover model
    initialization, forward pass, backward pass, embedding extraction,
    various fusion methods, and custom hyperparameters. All tests use
    synthetic data and complete in milliseconds.
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.duett import DuETT, DuETTLayer


class TestDuETT(unittest.TestCase):
    """Test cases for the DuETT model."""

    def setUp(self):
        """Set up test data and model."""
        # Synthetic samples: T=4 time bins, V=5 variables, S=2 static
        self.samples = [
            {
                "patient_id": "patient-0",
                "ts_values": [
                    [0.5, 0.3, 0.0, 0.8, 0.1],
                    [0.2, 0.0, 0.4, 0.7, 0.3],
                    [0.0, 0.6, 0.1, 0.0, 0.5],
                    [0.9, 0.2, 0.3, 0.4, 0.0],
                ],
                "ts_counts": [
                    [1.0, 1.0, 0.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 2.0, 1.0, 0.0, 1.0],
                    [3.0, 1.0, 1.0, 1.0, 0.0],
                ],
                "static": [0.65, 1.0],
                "times": [0.25, 0.5, 0.75, 1.0],
                "mortality": 1,
            },
            {
                "patient_id": "patient-1",
                "ts_values": [
                    [0.1, 0.7, 0.2, 0.0, 0.4],
                    [0.3, 0.5, 0.0, 0.6, 0.2],
                    [0.8, 0.0, 0.5, 0.3, 0.1],
                    [0.4, 0.1, 0.7, 0.9, 0.6],
                ],
                "ts_counts": [
                    [1.0, 2.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0],
                    [2.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0, 1.0],
                ],
                "static": [0.45, 0.0],
                "times": [0.25, 0.5, 0.75, 1.0],
                "mortality": 0,
            },
            {
                "patient_id": "patient-2",
                "ts_values": [
                    [0.3, 0.4, 0.1, 0.5, 0.2],
                    [0.6, 0.2, 0.3, 0.1, 0.7],
                    [0.1, 0.8, 0.0, 0.4, 0.3],
                    [0.7, 0.3, 0.5, 0.2, 0.1],
                ],
                "ts_counts": [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0, 2.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                "static": [0.72, 1.0],
                "times": [0.25, 0.5, 0.75, 1.0],
                "mortality": 1,
            },
            {
                "patient_id": "patient-3",
                "ts_values": [
                    [0.2, 0.1, 0.6, 0.3, 0.8],
                    [0.5, 0.4, 0.2, 0.7, 0.1],
                    [0.4, 0.3, 0.8, 0.1, 0.5],
                    [0.1, 0.6, 0.4, 0.5, 0.3],
                ],
                "ts_counts": [
                    [1.0, 1.0, 2.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                "static": [0.55, 0.0],
                "times": [0.25, 0.5, 0.75, 1.0],
                "mortality": 0,
            },
        ]

        self.input_schema = {
            "ts_values": "tensor",
            "ts_counts": "tensor",
            "static": "tensor",
            "times": "tensor",
        }
        self.output_schema = {"mortality": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_duett",
        )

        self.model = DuETT(
            dataset=self.dataset,
            d_embedding=16,
            n_event_layers=1,
            n_time_layers=1,
            n_heads=2,
            dropout=0.1,
        )

    def test_model_initialization(self):
        """Test that the DuETT model initializes correctly."""
        self.assertIsInstance(self.model, DuETT)
        self.assertEqual(self.model.d_embedding, 16)
        self.assertEqual(self.model.d_time_series, 5)
        self.assertEqual(self.model.d_static, 2)
        self.assertEqual(self.model.n_event_layers, 1)
        self.assertEqual(self.model.n_time_layers, 1)
        self.assertEqual(self.model.label_key, "mortality")
        self.assertEqual(self.model.fusion_method, "rep_token")

    def test_model_forward(self):
        """Test that the forward pass produces correct output keys."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that gradients flow correctly through the model."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(
            has_gradient,
            "No parameters have gradients after backward pass",
        )

    def test_model_with_embedding(self):
        """Test that embed=True returns patient embeddings."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        self.assertEqual(ret["embed"].shape[1], 16)

    def test_custom_hyperparameters(self):
        """Test DuETT with different hyperparameter configurations."""
        model = DuETT(
            dataset=self.dataset,
            d_embedding=32,
            n_event_layers=2,
            n_time_layers=2,
            n_heads=4,
            dropout=0.5,
        )

        self.assertEqual(model.d_embedding, 32)
        self.assertEqual(model.n_event_layers, 2)
        self.assertEqual(model.n_time_layers, 2)

        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_fusion_rep_token(self):
        """Test DuETT with rep_token fusion method."""
        model = DuETT(
            dataset=self.dataset,
            d_embedding=16,
            n_heads=2,
            fusion_method="rep_token",
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)

    def test_fusion_averaging(self):
        """Test DuETT with averaging fusion method."""
        model = DuETT(
            dataset=self.dataset,
            d_embedding=16,
            n_heads=2,
            fusion_method="averaging",
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)

    def test_fusion_masked_embed(self):
        """Test DuETT with masked_embed fusion method."""
        model = DuETT(
            dataset=self.dataset,
            d_embedding=16,
            n_heads=2,
            fusion_method="masked_embed",
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)

    def test_duett_layer_standalone(self):
        """Test DuETTLayer independently from the BaseModel wrapper."""
        layer = DuETTLayer(
            d_time_series=5,
            d_static=2,
            d_embedding=16,
            n_event_layers=1,
            n_time_layers=1,
            n_heads=2,
            dropout=0.0,
        )

        x_values = torch.randn(2, 4, 5)
        x_counts = torch.ones(2, 4, 5)
        static = torch.randn(2, 2)
        times = torch.linspace(0, 1, 4).unsqueeze(0).expand(2, -1)

        emb = layer(x_values, x_counts, static, times)

        self.assertEqual(emb.shape, (2, 16))
        self.assertFalse(torch.isnan(emb).any())

    def test_multiclass_classification(self):
        """Test DuETT with multiclass classification."""
        samples = [
            {
                "patient_id": f"patient-{i}",
                "ts_values": [[0.5, 0.3], [0.1, 0.4]],
                "ts_counts": [[1.0, 1.0], [1.0, 1.0]],
                "static": [0.5, 1.0],
                "times": [0.5, 1.0],
                "label": i % 3,
            }
            for i in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "ts_values": "tensor",
                "ts_counts": "tensor",
                "static": "tensor",
                "times": "tensor",
            },
            output_schema={"label": "multiclass"},
            dataset_name="test_multiclass",
        )

        model = DuETT(
            dataset=dataset,
            d_embedding=16,
            n_heads=2,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertEqual(ret["y_prob"].shape[1], 3)

    def test_loss_finite(self):
        """Test that loss is finite and not NaN."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertFalse(torch.isnan(ret["loss"]))
        self.assertTrue(torch.isfinite(ret["loss"]))


if __name__ == "__main__":
    unittest.main()
