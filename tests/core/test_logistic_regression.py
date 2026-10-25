import unittest
from datetime import datetime, timedelta

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LogisticRegression


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


# Other 3D numeric inputs embedded by an nn.Linear over their last dimension;
# no step is all zeros, so every step counts in the mean.
NUMERIC_INPUTS = {
    "nested_sequence_floats": lambda i: [[1.0 + i, 2.0], [3.0, 4.0 - i]],
    "stagenet_tensor": lambda i: (
        [0.0, 1.5, 3.0],
        [[1.0, 2.0 + i, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0 - i]],
    ),
    "tensor": lambda i: [[1.0, 2.0 + i], [3.0, 4.0], [5.0 - i, 6.0]],
}


def _numeric_input(input_type):
    """Four samples of one field of the given input type."""
    samples = [
        {
            "patient_id": f"patient-{i}",
            "x": NUMERIC_INPUTS[input_type](i),
            "label": i % 2,
        }
        for i in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"x": input_type},
        output_schema={"label": "binary"},
        dataset_name=f"test_{input_type}",
    )


class TestLogisticRegression(unittest.TestCase):
    """Test cases for the LogisticRegression model."""

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
                "patient_id": "patient-1",
                "visit_id": "visit-1",
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
        self.model = LogisticRegression(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the LogisticRegression model initializes correctly."""
        self.assertIsInstance(self.model, LogisticRegression)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the LogisticRegression model forward pass works correctly."""
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
        """Test that the LogisticRegression model backward pass works correctly."""
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
        """Test that the LogisticRegression model returns embeddings when requested."""
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
        # Check embedding dimension (2 features * embedding_dim)
        expected_embed_dim = len(self.model.feature_keys) * self.model.embedding_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_embedding_dim(self):
        """Test LogisticRegression model with custom embedding dimension."""
        model = LogisticRegression(
            dataset=self.dataset,
            embedding_dim=64,
        )

        self.assertEqual(model.embedding_dim, 64)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_regression_task(self):
        """Test LogisticRegression model with a regression task."""
        # Create regression samples
        regression_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "score": 0.5,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "score": 0.8,
            },
        ]

        input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        output_schema = {"score": "regression"}

        regression_dataset = create_sample_dataset(
            samples=regression_samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_regression",
        )

        # Create model
        regression_model = LogisticRegression(dataset=regression_dataset)

        # Test forward pass
        train_loader = get_dataloader(regression_dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = regression_model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)  # For regression, this is just the raw output
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

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
        model = LogisticRegression(dataset=dataset, embedding_dim=8)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        ret = model(**batch)
        ret["loss"].backward()
        self.assertEqual(ret["y_prob"].shape, (4, 1))

    def test_other_numeric_3d_inputs_are_embedded_per_step(self):
        """nested_sequence_floats, stagenet_tensor and tensor inputs are
        embedded per step too, each pooling to the mean of its steps."""
        for input_type in NUMERIC_INPUTS:
            with self.subTest(input_type):
                dataset = _numeric_input(input_type)
                model = LogisticRegression(dataset=dataset, embedding_dim=8)
                batch = next(iter(get_dataloader(dataset, batch_size=4)))
                ret = model(**batch)
                ret["loss"].backward()
                self.assertEqual(ret["y_prob"].shape, (4, 1))
                model.eval()
                linear = model.embedding_model.embedding_layers["x"]
                schema = dataset.input_processors["x"].schema()
                with torch.no_grad():
                    embed = model(**batch, embed=True)["embed"]
                    for i in range(len(dataset)):
                        row = batch["patient_id"].index(dataset[i]["patient_id"])
                        steps = dataset[i]["x"]
                        if isinstance(steps, tuple):
                            steps = steps[schema.index("value")]
                        pooled = linear(steps).mean(dim=0)
                        torch.testing.assert_close(embed[row], pooled)

    def test_padded_steps_are_left_out_of_the_mean(self):
        """A short series pools the same in a padded batch as on its own."""
        dataset = _padded_series()
        model = LogisticRegression(dataset=dataset, embedding_dim=8)
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
                torch.testing.assert_close(together[row], pooled)

    def test_mean_pooling_without_a_valid_step(self):
        """A row whose mask is all zero is averaged over all its steps."""
        x = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        pooled = LogisticRegression.mean_pooling(x, mask)
        torch.testing.assert_close(pooled[0], x[0, [0, 2]].mean(dim=0))
        torch.testing.assert_close(pooled[1], x[1].mean(dim=0))

    def test_nested_multihot_passes_through_flattened(self):
        """Documents a limitation: EmbeddingModel has no layer for
        nested_multihot, so the raw multi-hot rows pass through and are
        flattened into one vector."""
        dataset = _nested_multihot()
        width = 2 * dataset.input_processors["visits"].size()  # two visits
        model = LogisticRegression(dataset=dataset, embedding_dim=width)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        with torch.no_grad():
            embed = model(**batch, embed=True)["embed"]
        torch.testing.assert_close(embed, batch["visits"].reshape(4, -1))

    def test_list_inputs_and_unsupported_ranks(self):
        """Inputs given as lists are converted; other ranks are rejected."""
        dataset = _padded_series()
        model = LogisticRegression(dataset=dataset, embedding_dim=8)
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        with torch.no_grad():
            as_list = model(vitals=batch["vitals"].tolist(), label=batch["label"])
            torch.testing.assert_close(as_list["logit"], model(**batch)["logit"])
            with self.assertRaises(ValueError):
                model(vitals=batch["vitals"].unsqueeze(1), label=batch["label"])

if __name__ == "__main__":
    unittest.main()
