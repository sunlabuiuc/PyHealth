import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import HGNN


class TestHGNN(unittest.TestCase):
    """Test cases for the HGNN model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": ["proc-1", "proc-2"],
                "label": 1,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": ["proc-1"],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-2",
                "conditions": ["cond-80", "cond-12"],
                "procedures": ["proc-2"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-3",
                "conditions": ["cond-33", "cond-12"],
                "procedures": ["proc-1", "proc-2"],
                "label": 0,
            },
        ]

        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_hgnn",
        )

        self.model = HGNN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the HGNN model initializes correctly."""
        self.assertIsInstance(self.model, HGNN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(self.model.num_conv_layers, 2)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the HGNN model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
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

    def test_model_backward(self):
        """Test that the HGNN model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_model_with_embedding(self):
        """Test that the HGNN model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = len(self.model.feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test HGNN model with custom hyperparameters."""
        model = HGNN(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            num_conv_layers=3,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.num_conv_layers, 3)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        batch_size = data_batch["label"].shape[0]
        self.assertEqual(ret["y_prob"].shape, torch.Size([batch_size, 1]))
        self.assertEqual(ret["logit"].shape, torch.Size([batch_size, 1]))
        self.assertEqual(ret["y_true"].shape, torch.Size([batch_size, 1]))
        self.assertEqual(ret["loss"].shape, torch.Size([]))

    def test_loss_validity(self):
        """Test that loss is computed and is a valid number."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)

        self.assertIsInstance(ret["loss"].item(), float)
        self.assertFalse(torch.isnan(ret["loss"]))
        self.assertFalse(torch.isinf(ret["loss"]))

    def test_probability_bounds(self):
        """Test that predicted probabilities are in valid range [0, 1]."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertTrue(torch.all(ret["y_prob"] >= 0.0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1.0))

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        for batch_size in [1, 2, 4]:
            train_loader = get_dataloader(
                self.dataset, batch_size=batch_size, shuffle=True
            )
            data_batch = next(iter(train_loader))

            with torch.no_grad():
                ret = self.model(**data_batch)

            self.assertEqual(ret["y_prob"].shape[0], batch_size)
            self.assertEqual(ret["y_true"].shape[0], batch_size)
            self.assertFalse(torch.isnan(ret["loss"]))

    def test_different_conv_layers(self):
        """Test model with different numbers of convolution layers."""
        for num_layers in [1, 2, 3]:
            model = HGNN(
                dataset=self.dataset,
                embedding_dim=32,
                hidden_dim=32,
                num_conv_layers=num_layers,
            )

            train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
            data_batch = next(iter(train_loader))

            with torch.no_grad():
                ret = model(**data_batch)

            self.assertFalse(torch.isnan(ret["loss"]))
            self.assertIn("y_prob", ret)

    def test_different_embedding_dims(self):
        """Test model with different embedding dimensions."""
        for embedding_dim in [16, 32, 64]:
            model = HGNN(
                dataset=self.dataset,
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim,
                num_conv_layers=1,
            )

            train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
            data_batch = next(iter(train_loader))

            with torch.no_grad():
                ret = model(**data_batch)

            self.assertFalse(torch.isnan(ret["loss"]))

    def test_single_sample(self):
        """Test model with batch size of 1."""
        train_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertEqual(ret["y_prob"].shape[0], 1)
        self.assertEqual(ret["logit"].shape[0], 1)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_reproducibility(self):
        """Test that model produces consistent results with same seed."""
        torch.manual_seed(42)
        model1 = HGNN(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            num_conv_layers=1,
        )

        torch.manual_seed(42)
        model2 = HGNN(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            num_conv_layers=1,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            result1 = model1(**data_batch)
            result2 = model2(**data_batch)

        self.assertTrue(torch.allclose(result1["logit"], result2["logit"], atol=1e-5))

    def test_hypergraph_builder(self):
        """Test that hypergraph builder creates proper incidence matrices."""
        X = torch.randn(5, self.model.embedding_dim)
        H = self.model.hypergraph_builder(X)

        self.assertEqual(H.shape[0], 5)
        self.assertGreater(H.shape[1], 0)
        self.assertTrue(torch.all((H == 0) | (H == 1)))
        self.assertGreater(torch.sum(H), 0)

    def test_variable_sequence_lengths(self):
        """Test that model handles variable sequence lengths across features."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)

        for data_batch in train_loader:
            with torch.no_grad():
                ret = self.model(**data_batch)

            self.assertFalse(torch.isnan(ret["loss"]))
            self.assertEqual(ret["y_prob"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
