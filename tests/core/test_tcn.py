import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TCN


class TestTCN(unittest.TestCase):
    """Test cases for the TCN model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": ["proc-1", "proc-2", "proc-3"],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": ["proc-1", "proc-2"],
                "label": 1,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",  # sequence of condition codes
            "procedures": "sequence",  # sequence of procedure codes
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
        self.model = TCN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the TCN model initializes correctly."""
        self.assertIsInstance(self.model, TCN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.num_channels, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the TCN model forward pass works correctly."""
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
        """Test that the TCN model backward pass works correctly."""
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
        """Test that the TCN model returns embeddings when requested."""
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
        # Check embedding dimension (2 features * num_channels)
        expected_embed_dim = len(self.model.feature_keys) * self.model.num_channels
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test TCN model with custom hyperparameters."""
        model = TCN(
            dataset=self.dataset,
            embedding_dim=64,
            num_channels=64,
            kernel_size=3,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.num_channels, 64)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_num_channels_as_list(self):
        """Test TCN model with num_channels as a list."""
        model = TCN(
            dataset=self.dataset,
            embedding_dim=64,
            num_channels=[64, 128, 256],
        )

        # Should use the last value in the list
        self.assertEqual(model.num_channels, 256)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_model_with_stagenet_tuple_feature(self):
        """TCN must handle tuple-schema features (StageNetProcessor).

        Regression test: StageNetProcessor emits a (time, value) tuple per
        feature. TCN previously passed the raw tuple to the embedding model
        and crashed; it must unwrap the "value" tensor.
        """
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ([0.0, 2.0, 1.3], ["c1", "c2", "c3"]),
                "conditions": ["cond-33", "cond-86"],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "codes": ([0.0, 2.0], ["c1", "c4"]),
                "conditions": ["cond-33"],
                "label": 1,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "stagenet", "conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_stagenet",
        )
        model = TCN(dataset=dataset)
        data_batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))

        ret = model(**data_batch)
        ret["loss"].backward()

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def _run(self, samples, schema):
        dataset = create_sample_dataset(
            samples=samples, input_schema=schema,
            output_schema={"label": "binary"}, dataset_name="test_shapes",
        )
        model = TCN(dataset=dataset, embedding_dim=8, num_channels=8)
        batch = next(iter(get_dataloader(dataset, batch_size=len(samples))))
        ret = model(**batch)
        ret["loss"].backward()
        self.assertEqual(tuple(ret["y_prob"].shape), (len(samples), 1))
        self.assertTrue(torch.isfinite(ret["loss"]))
        return model, dataset

    def test_one_vector_per_sample_inputs(self):
        """tensor and multi_hot fields give 2-D embeddings (one step)."""
        samples = [
            {"patient_id": f"p{i}", "vitals": [float(i), 1.0, 2.0],
             "flags": ["a", "b"][: 1 + i % 2], "label": i % 2}
            for i in range(4)
        ]
        self._run(samples, {"vitals": "tensor", "flags": "multi_hot"})

    def test_nested_codes_pool_each_visit(self):
        """nested_sequence fields give 4-D embeddings: codes per visit."""
        samples = [
            {"patient_id": "p0", "visits": [["a", "b"], ["c"], ["a", "d"]],
             "label": 0},
            {"patient_id": "p1", "visits": [["b"]], "label": 1},
        ]
        model, dataset = self._run(samples, {"visits": "nested_sequence"})
        # The short record scores the same alone as next to a longer one:
        # padded visits come after it, and the convolutions are causal.
        model.eval()
        with torch.no_grad():
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            together = model(**batch)["y_prob"][1]
            alone = model(**next(iter(get_dataloader(dataset.subset([1]),
                                                     batch_size=1))))["y_prob"][0]
        torch.testing.assert_close(together, alone)


if __name__ == "__main__":
    unittest.main()
