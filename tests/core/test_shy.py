import unittest

import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SHy

CODES = ["504", "505", "506", "I1234", "I1235"]


class TestSHy(unittest.TestCase):
    """Basic tests for the simplified SHy model."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "prev_diag": [[CODES[0], CODES[1]], [CODES[2]]],
                "label": [CODES[0], CODES[2]],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "prev_diag": [[CODES[2]], [CODES[3], CODES[4]]],
                "label": [CODES[1], CODES[3]],
            },
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"prev_diag": "nested_sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="test_shy_simple",
        )

        self.model = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hidden_dim=32,
            num_phenotypes=2,
            num_layers=1,
        )

    def test_initialization(self):
        """Check model parameters are correctly set."""
        self.assertIsInstance(self.model, SHy)
        self.assertEqual(self.model.embedding_dim, 16)
        self.assertEqual(self.model.hidden_dim, 32)
        self.assertEqual(self.model.num_phenotypes, 2)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(self.model.classifier.out_features, self.model.output_size)

    def test_forward_pass(self):
        """Test forward pass runs without crashing."""
        dataloader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))

        with torch.no_grad():
            output = self.model(**batch)

        self.assertIn("logits", output)
        self.assertIn("y_prob", output)

    def test_output_shapes(self):
        """Ensure correct output tensor shapes."""
        dataloader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))

        with torch.no_grad():
            output = self.model(**batch)

        logits = output["logits"]
        y_prob = output["y_prob"]

        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(y_prob.shape[0], 2)
        self.assertEqual(logits.dim(), 2)
        self.assertEqual(y_prob.dim(), 2)

        # Probabilities must be in [0, 1]
        self.assertTrue(torch.all(y_prob >= 0))
        self.assertTrue(torch.all(y_prob <= 1))

    def test_gradients(self):
        """Ensure model is trainable (backprop works)."""
        dataloader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))

        output = self.model(**batch)
        loss = output["logits"].sum()

        loss.backward()

        has_grad = any(
            p.grad is not None for p in self.model.parameters()
        )

        self.assertTrue(has_grad)

    def test_single_visit(self):
        """Test edge case: model handles a single visit sequence."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "prev_diag": [[CODES[0], CODES[1]]],
                "label": [CODES[0]],
            }
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"prev_diag": "nested_sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="single_visit_test",
        )

        model = SHy(dataset=dataset)

        dataloader = get_dataloader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(dataloader))

        with torch.no_grad():
            output = model(**batch)

        self.assertEqual(output["logits"].shape[0], 1)

if __name__ == "__main__":
    unittest.main()
