import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MultimodalRNN


class TestMultimodalRNN(unittest.TestCase):
    """Test cases for the MultimodalRNN model."""

    def setUp(self):
        """Set up test data and model with mixed feature types."""
        # Samples with mixed sequential and non-sequential features
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80"],  # sequential
                "procedures": ["proc-12", "proc-45"],              # sequential
                "demographics": ["asian", "male", "smoker"],       # multi-hot
                "vitals": [120.0, 80.0, 98.6, 16.0],             # tensor
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-12", "cond-52"],              # sequential
                "procedures": ["proc-23"],                         # sequential
                "demographics": ["white", "female"],               # multi-hot
                "vitals": [110.0, 75.0, 98.2, 18.0],             # tensor
                "label": 0,
            },
        ]

        # Define input and output schemas with mixed types
        self.input_schema = {
            "conditions": "sequence",      # sequential
            "procedures": "sequence",      # sequential
            "demographics": "multi_hot",   # non-sequential
            "vitals": "tensor",           # non-sequential
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
        self.model = MultimodalRNN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the MultimodalRNN model initializes correctly."""
        self.assertIsInstance(self.model, MultimodalRNN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 4)

        # Check that features are correctly classified
        self.assertIn("conditions", self.model.sequential_features)
        self.assertIn("procedures", self.model.sequential_features)
        self.assertIn("demographics", self.model.non_sequential_features)
        self.assertIn("vitals", self.model.non_sequential_features)

        # Check that RNN layers are only created for sequential features
        self.assertIn("conditions", self.model.rnn)
        self.assertIn("procedures", self.model.rnn)
        self.assertNotIn("demographics", self.model.rnn)
        self.assertNotIn("vitals", self.model.rnn)

        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the MultimodalRNN model forward pass works correctly."""
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
        """Test that the MultimodalRNN model backward pass works correctly."""
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
        """Test that the MultimodalRNN model returns embeddings when requested."""
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
        # 2 sequential features * hidden_dim + 2 non-sequential features * embedding_dim
        expected_embed_dim = (
            len(self.model.sequential_features) * self.model.hidden_dim +
            len(self.model.non_sequential_features) * self.model.embedding_dim
        )
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test MultimodalRNN model with custom hyperparameters."""
        model = MultimodalRNN(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            rnn_type="LSTM",
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_only_sequential_features(self):
        """Test MultimodalRNN with only sequential features (like vanilla RNN)."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86"],
                "procedures": ["proc-12", "proc-45"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-12"],
                "procedures": ["proc-23"],
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_seq_only",
        )

        model = MultimodalRNN(dataset=dataset, hidden_dim=64)

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
        """Test MultimodalRNN with only non-sequential features (like MLP)."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "demographics": ["asian", "male", "smoker"],
                "vitals": [120.0, 80.0, 98.6, 16.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "demographics": ["white", "female"],
                "vitals": [110.0, 75.0, 98.2, 18.0],
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"demographics": "multi_hot", "vitals": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_non_seq_only",
        )

        model = MultimodalRNN(dataset=dataset, hidden_dim=64)

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

    def test_sequential_processor_classification(self):
        """Test that _is_sequential_processor correctly identifies processor types."""
        from pyhealth.processors import (
            MultiHotProcessor,
            SequenceProcessor,
            TensorProcessor,
            TimeseriesProcessor,
        )

        # Test with actual processor instances
        seq_proc = SequenceProcessor()
        self.assertTrue(self.model._is_sequential_processor(seq_proc))

        # Create simple multi-hot processor
        multihot_proc = MultiHotProcessor()
        self.assertFalse(self.model._is_sequential_processor(multihot_proc))

        # Tensor processor
        tensor_proc = TensorProcessor()
        self.assertFalse(self.model._is_sequential_processor(tensor_proc))


if __name__ == "__main__":
    unittest.main()

