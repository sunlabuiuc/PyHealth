import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MultimodalRETAIN


class TestMultimodalRETAIN(unittest.TestCase):
    """Test cases for the MultimodalRETAIN model."""

    def setUp(self):
        """Set up test data and model with mixed feature types."""
        # Samples with mixed sequential and non-sequential features
        # RETAIN typically works with visit-level nested sequences
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["A", "B"], ["C"]],          # nested sequence
                "procedures": [["P1"], ["P2", "P3"]],       # nested sequence
                "demographics": ["asian", "male"],           # multi-hot
                "vitals": [120.0, 80.0, 98.6],             # tensor
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["D"], ["E", "F"]],          # nested sequence
                "procedures": [["P4"]],                      # nested sequence
                "demographics": ["white", "female"],         # multi-hot
                "vitals": [110.0, 75.0, 98.2],             # tensor
                "label": 0,
            },
        ]

        # Define input and output schemas with mixed types
        self.input_schema = {
            "conditions": "nested_sequence",   # sequential
            "procedures": "nested_sequence",   # sequential
            "demographics": "multi_hot",       # non-sequential
            "vitals": "tensor",               # non-sequential
        }
        self.output_schema = {"label": "binary"}

        # Create dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create model
        self.model = MultimodalRETAIN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the MultimodalRETAIN model initializes correctly."""
        self.assertIsInstance(self.model, MultimodalRETAIN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 4)

        # Check that features are correctly classified
        self.assertIn("conditions", self.model.sequential_features)
        self.assertIn("procedures", self.model.sequential_features)
        self.assertIn("demographics", self.model.non_sequential_features)
        self.assertIn("vitals", self.model.non_sequential_features)

        # Check that RETAIN layers are only created for sequential features
        self.assertIn("conditions", self.model.retain)
        self.assertIn("procedures", self.model.retain)
        self.assertNotIn("demographics", self.model.retain)
        self.assertNotIn("vitals", self.model.retain)

    def test_model_forward(self):
        """Test that the MultimodalRETAIN model forward pass works correctly."""
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
        """Test that the MultimodalRETAIN model backward pass works correctly."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        # Forward pass
        ret = self.model(**data_batch)

        # Backward pass
        ret["loss"].backward()

        # Check that at least one parameter has gradients
        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_model_with_embedding(self):
        """Test that the MultimodalRETAIN model returns embeddings when requested."""
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

        # Check embedding dimension
        # All features contribute embedding_dim
        expected_embed_dim = len(self.model.feature_keys) * self.model.embedding_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test MultimodalRETAIN model with custom hyperparameters."""
        model = MultimodalRETAIN(
            dataset=self.dataset,
            embedding_dim=64,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 64)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_only_sequential_features(self):
        """Test MultimodalRETAIN with only sequential features."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["A", "B"], ["C"]],
                "procedures": [["P1"], ["P2"]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["D"], ["E"]],
                "procedures": [["P3"]],
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions": "nested_sequence",
                "procedures": "nested_sequence"
            },
            output_schema={"label": "binary"},
            dataset_name="test_seq_only",
        )

        model = MultimodalRETAIN(dataset=dataset)

        # Check that all features are sequential
        self.assertEqual(len(model.sequential_features), 2)
        self.assertEqual(len(model.non_sequential_features), 0)

        # Test forward pass
        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_only_non_sequential_features(self):
        """Test MultimodalRETAIN with only non-sequential features."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "demographics": ["asian", "male"],
                "vitals": [120.0, 80.0, 98.6],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "demographics": ["white", "female"],
                "vitals": [110.0, 75.0, 98.2],
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"demographics": "multi_hot", "vitals": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_non_seq_only",
        )

        model = MultimodalRETAIN(dataset=dataset)

        # Check that all features are non-sequential
        self.assertEqual(len(model.sequential_features), 0)
        self.assertEqual(len(model.non_sequential_features), 2)

        # Test forward pass
        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_output_shapes(self):
        """Test that output shapes are correct for multimodal inputs."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertEqual(ret["y_prob"].shape, (2, 1))
        self.assertEqual(ret["y_true"].shape, (2, 1))
        self.assertEqual(ret["logit"].shape, (2, 1))

    def test_loss_is_finite(self):
        """Test that the loss is finite."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())

    def test_sequential_processor_classification(self):
        """Test that _is_sequential_processor correctly identifies processor types."""
        from pyhealth.processors import (
            MultiHotProcessor,
            NestedSequenceProcessor,
            TensorProcessor,
        )

        # Test with actual processor instances
        nested_seq_proc = NestedSequenceProcessor()
        self.assertTrue(self.model._is_sequential_processor(nested_seq_proc))

        # Create simple multi-hot processor
        multihot_proc = MultiHotProcessor()
        self.assertFalse(self.model._is_sequential_processor(multihot_proc))

        # Tensor processor
        tensor_proc = TensorProcessor()
        self.assertFalse(self.model._is_sequential_processor(tensor_proc))

    def test_with_simple_sequences(self):
        """Test MultimodalRETAIN with simple (non-nested) sequences."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ["A", "B", "C"],              # simple sequence
                "demographics": ["asian", "male"],      # multi-hot
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": ["D", "E"],                    # simple sequence
                "demographics": ["white", "female"],    # multi-hot
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence", "demographics": "multi_hot"},
            output_schema={"label": "binary"},
            dataset_name="test_simple_seq",
        )

        model = MultimodalRETAIN(dataset=dataset)

        # Check that codes is sequential
        self.assertIn("codes", model.sequential_features)
        self.assertIn("demographics", model.non_sequential_features)

        # Test forward pass
        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)


if __name__ == "__main__":
    unittest.main()

