import unittest
from datetime import datetime, timedelta

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.models import MLP


def _padded_series():
    """Series of 2, 5 and 3 hourly steps; the last one is all zeros."""
    t0 = datetime(2020, 1, 1)
    samples = []
    for i, n in enumerate([2, 5, 3]):
        steps = [[float(h + 1), 2.0 * i + 1, -1.0] for h in range(n)]
        samples.append({
            "patient_id": f"patient-{i}",
            "vitals": (
                [t0 + timedelta(hours=h) for h in range(n)],
                steps if i < 2 else [[0.0, 0.0, 0.0]] * n,
            ),
            "label": i % 2,
        })
    return create_sample_dataset(
        samples=samples,
        input_schema={"vitals": "timeseries"},
        output_schema={"label": "binary"},
        dataset_name="test_padding",
    )


def _nested_multihot():
    """Two visits per patient as per-visit multi-hot rows."""
    samples = [
        {
            "patient_id": f"patient-{i}",
            "visits": [["a", "b"], ["c"] if i % 2 else ["a"]],
            "label": i % 2,
        }
        for i in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"visits": "nested_multihot"},
        output_schema={"label": "binary"},
        dataset_name="test_multihot",
    )


class TestMLP(unittest.TestCase):
    """Test cases for the MLP model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "label": 1,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",  # sequence of condition codes
            "procedures": "tensor",  # tensor of procedure values
        }
        self.output_schema = {"label": "binary"}  # binary classification

        # Create dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create model
        self.model = MLP(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the MLP model initializes correctly."""
        self.assertIsInstance(self.model, MLP)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(self.model.n_layers, 2)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the MLP model forward pass works correctly."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        # Forward pass
        with torch.no_grad():
            ret = self.model(**data_batch)

        # Check output structure
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        # Check tensor shapes
        self.assertEqual(ret["y_prob"].shape[0], 2)  # batch size
        self.assertEqual(ret["y_true"].shape[0], 2)  # batch size
        self.assertEqual(ret["logit"].shape[0], 2)  # batch size

        # Check that loss is a scalar
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the MLP model backward pass works correctly."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        # Forward pass
        ret = self.model(**data_batch)

        # Backward pass
        ret["loss"].backward()

        # Check that at least one parameter has gradients (backward working)
        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_model_with_embedding(self):
        """Test that the MLP model returns embeddings when requested."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        # Forward pass
        with torch.no_grad():
            ret = self.model(**data_batch)

        # Check that embeddings are returned
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        # Check embedding dimension (2 features * hidden_dim)
        expected_embed_dim = len(self.model.feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test MLP model with custom hyperparameters."""
        model = MLP(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            n_layers=3,
            activation="tanh",
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.n_layers, 3)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_timeseries_and_nested_codes(self):
        """A (time, features) series is embedded per step, not flattened."""
        t0 = datetime(2020, 1, 1)
        samples = [
            {
                "patient_id": f"patient-{i}",
                "vitals": (
                    [t0 + timedelta(hours=h) for h in range(4)],
                    [[float(i + h), 1.0, 2.0] for h in range(4)],
                ),
                "visits": [["a", "b"], ["c"]],
                "label": i % 2,
            }
            for i in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"vitals": "timeseries", "visits": "nested_sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_timeseries",
        )
        model = MLP(dataset=dataset, embedding_dim=8, hidden_dim=8)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        ret = model(**batch)
        ret["loss"].backward()
        self.assertEqual(ret["y_prob"].shape, (4, 1))

    def test_padded_steps_are_left_out_of_the_mean(self):
        """A short series pools the same in a padded batch as on its own."""
        dataset = _padded_series()
        model = MLP(dataset=dataset, embedding_dim=8)
        model.eval()
        linear = model.embedding_model.embedding_layers["vitals"]
        with torch.no_grad():
            batch = next(iter(get_dataloader(dataset, batch_size=3)))
            together = model(**batch, embed=True)["embed"]
            self.assertTrue(torch.isfinite(together).all())
            for i in range(len(dataset)):
                row = batch["patient_id"].index(dataset[i]["patient_id"])
                alone = next(iter(get_dataloader(dataset.subset([i]), batch_size=1)))
                torch.testing.assert_close(
                    together[row], model(**alone, embed=True)["embed"][0]
                )
                # each step is embedded on its own, then the real steps averaged
                pooled = linear(dataset[i]["vitals"]).mean(dim=0)
                torch.testing.assert_close(together[row], model.mlp["vitals"](pooled))

    def test_all_zero_series_with_an_external_mask(self):
        """A mask built from the raw values, as the interpretability methods
        build it, is all zero for an all-zero series; pooling stays finite."""
        dataset = _padded_series()
        model = MLP(dataset=dataset, embedding_dim=8, hidden_dim=8)
        model.eval()
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        raw = batch["vitals"]
        mask = (raw.abs().sum(dim=-1) != 0).float()
        self.assertFalse(mask.sum(dim=1).all())  # one row has no valid step
        with torch.no_grad():
            embedded = model.embedding_model({"vitals": raw})["vitals"]
            out = model.forward_from_embedding(vitals=(embedded, mask),
                                               label=batch["label"])
            torch.testing.assert_close(out["logit"], model(**batch)["logit"])
        attributions = IntegratedGradients(model).attribute(**batch, steps=5)
        self.assertTrue(torch.isfinite(attributions["vitals"]).all())

    def test_nested_multihot_passes_through_flattened(self):
        """EmbeddingModel has no layer for nested_multihot, so the raw
        multi-hot rows pass through and are flattened into one vector."""
        # Pins the current passthrough, not a design choice.
        dataset = _nested_multihot()
        width = 2 * dataset.input_processors["visits"].size()  # two visits
        model = MLP(dataset=dataset, embedding_dim=width, hidden_dim=8)
        model.eval()
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        with torch.no_grad():
            embed = model(**batch, embed=True)["embed"]
            flat = batch["visits"].reshape(4, -1)
            torch.testing.assert_close(embed, model.mlp["visits"](flat))

    def test_forward_from_embedding_inputs(self):
        """A bare tensor and a mask with a feature dimension are accepted;
        other ranks are rejected."""
        dataset = _padded_series()
        model = MLP(dataset=dataset, embedding_dim=8, hidden_dim=8)
        model.eval()
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        with torch.no_grad():
            embedded = model.embedding_model({"vitals": batch["vitals"]})["vitals"]
            bare = model.forward_from_embedding(vitals=embedded)
            ones = model.forward_from_embedding(vitals=(embedded, torch.ones_like(embedded)))
            # without a mask every embedded (nonzero) step counts, as with all ones
            torch.testing.assert_close(bare["logit"], ones["logit"])
            with self.assertRaises(ValueError):
                model.forward_from_embedding(vitals=embedded.unsqueeze(1))
        self.assertEqual(MLP.sum_pooling(embedded).shape, (3, 8))

if __name__ == "__main__":
    unittest.main()
