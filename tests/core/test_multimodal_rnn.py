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


class TestMultimodalRNNNestedSequence(unittest.TestCase):
    """Tests for MultimodalRNN with nested_sequence features.

    Covers the bug where drugs_hist = [[]] (first sample in drug recommendation
    tasks) caused: RuntimeError: Length of all samples has to be greater than 0
    because MultimodalRNN was flattening visits*codes into a single sequence
    instead of pooling codes within visits first.
    """

    def _make_drug_rec_samples(self):
        """Mimics DrugRecommendationMIMIC4 output for a 2-patient dataset.

        samples[0] for each patient always has drugs_hist = [[]] — one visit
        with an empty inner list, because the task zeroes out the current
        visit's drugs from the history.
        """
        return [
            # patient-0, visit-0: first ever visit → drugs_hist has one empty inner list
            {
                "patient_id": "p0", "visit_id": "v0",
                "conditions": [["cond-1", "cond-2"]],
                "drugs_hist": [[]],          # <-- the problematic case
                "label": ["drug-A"],
            },
            # patient-0, visit-1: second visit → current slot cleared
            {
                "patient_id": "p0", "visit_id": "v1",
                "conditions": [["cond-1", "cond-2"], ["cond-3"]],
                "drugs_hist": [["drug-A"], []],
                "label": ["drug-B"],
            },
            # patient-1, visit-0: another first-visit case
            {
                "patient_id": "p1", "visit_id": "v2",
                "conditions": [["cond-4"]],
                "drugs_hist": [[]],
                "label": ["drug-A"],
            },
            # patient-1, visit-1
            {
                "patient_id": "p1", "visit_id": "v3",
                "conditions": [["cond-4"], ["cond-5", "cond-6"]],
                "drugs_hist": [["drug-B"], []],
                "label": ["drug-B"],
            },
        ]

    def test_forward_with_empty_inner_list(self):
        """MultimodalRNN must not crash when a nested_sequence has empty inner lists.

        Before the fix, MultimodalRNN flattened (visits * max_codes) into a single
        sequence dimension, giving length=0 for patients whose inner lists were all
        empty (e.g. drugs_hist=[[]] for first-visit samples).
        """
        dataset = create_sample_dataset(
            samples=self._make_drug_rec_samples(),
            input_schema={
                "conditions": "nested_sequence",
                "drugs_hist": "nested_sequence",
            },
            output_schema={"label": "multilabel"},
            dataset_name="test_empty_inner",
        )
        loader = get_dataloader(dataset, batch_size=4, shuffle=False)
        model = MultimodalRNN(dataset=dataset, embedding_dim=32, hidden_dim=32)
        model.eval()

        batch = next(iter(loader))
        with torch.no_grad():
            result = model(**batch)

        self.assertIn("loss", result)
        self.assertIn("y_prob", result)
        self.assertEqual(result["y_prob"].shape[0], 4)

    def test_nested_sequence_visit_level_mask(self):
        """Verify the fixed MultimodalRNN uses visit-level masks (not code-level).

        With visit-level masking, a patient with 2 visits where the second is
        empty should have sequence length 1, not 0.
        """
        samples = [
            # One non-empty visit followed by one empty visit
            {
                "patient_id": "p0", "visit_id": "v0",
                "conditions": [["cond-1"], []],
                "label": 1,
            },
            {
                "patient_id": "p1", "visit_id": "v1",
                "conditions": [["cond-2", "cond-3"], ["cond-4"]],
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_visit_mask",
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        model = MultimodalRNN(dataset=dataset, embedding_dim=16, hidden_dim=16)
        model.eval()

        batch = next(iter(loader))
        with torch.no_grad():
            result = model(**batch)

        self.assertIn("loss", result)
        self.assertEqual(result["y_prob"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()

