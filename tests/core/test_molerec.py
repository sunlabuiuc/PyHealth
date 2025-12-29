import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MoleRec


class TestMoleRec(unittest.TestCase):
    """Test cases for the MoleRec model."""

    def setUp(self):
        """Set up test data and model."""
        try:
            import pkg_resources
            pkg_resources.require(["ogb>=1.3.5"])
        except Exception:
            self.skipTest("ogb package not available")
        
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-1", "cond-2"], ["cond-3", "cond-4"]],
                "procedures": [["proc-1"], ["proc-2", "proc-3"]],
                "drugs": ["drug-1", "drug-2", "drug-3"],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["cond-5", "cond-6"]],
                "procedures": [["proc-4"]],
                "drugs": ["drug-2", "drug-4"],
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        self.output_schema = {"drugs": "multilabel"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = MoleRec(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the MoleRec model initializes correctly."""
        self.assertIsInstance(self.model, MoleRec)
        self.assertEqual(self.model.embedding_dim, 64)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.num_rnn_layers, 1)
        self.assertEqual(self.model.num_gnn_layers, 4)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "drugs")

    def test_forward_input_format(self):
        """Test that the dataloader provides tensor inputs."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertIsInstance(data_batch["conditions"], torch.Tensor)
        self.assertIsInstance(data_batch["procedures"], torch.Tensor)
        self.assertIsInstance(data_batch["drugs"], torch.Tensor)

    def test_model_forward(self):
        """Test that the MoleRec model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the MoleRec model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_loss_is_finite(self):
        """Test that the loss is finite."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        batch_size = 2
        num_labels = len(self.dataset.output_processors["drugs"].label_vocab)

        self.assertEqual(ret["y_prob"].shape, (batch_size, num_labels))
        self.assertEqual(ret["y_true"].shape, (batch_size, num_labels))


if __name__ == "__main__":
    unittest.main()

