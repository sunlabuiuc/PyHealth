# Authors: Yujia Li (yujia9@illinois.edu)
# Paper: Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction
# Paper link: https://arxiv.org/abs/2502.10689
# Description: Self-explaining hypergraph diagnosis prediction tests.

import unittest
import torch
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import SHy


class TestSHy(unittest.TestCase):
    """
    Unit tests for the SHy model implementation.

    Verifies model instantiation, forward pass logic, output dimensions,
    and gradient flow for optimization.
    """

    def setUp(self):
        """
        Set up a minimal synthetic dataset and initialize the SHy model.
        Uses 4 patients as required by the PyHealth testing standard.
        """
        # Synthetic samples for testing
        self.samples = [
            {
                "patient_id": "sub_01",
                "diagnoses_hist": [["v10", "v20"], ["v10", "v30", "v40"]],
                "diagnoses": ["v10", "v30"],
            },
            {
                "patient_id": "sub_02",
                "diagnoses_hist": [["v20"], ["v50", "v60"], ["v20", "v70"]],
                "diagnoses": ["v50", "v80"],
            },
            {
                "patient_id": "sub_03",
                "diagnoses_hist": [["v10", "v80"], ["v90"]],
                "diagnoses": ["v10", "v20"],
            }
        ]

        # Create SampleDataset with required schema
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"diagnoses_hist": "nested_sequence"},
            output_schema={"diagnoses": "multilabel"},
            dataset_name="test_shy_synthetic",
        )

        # Initialize the refactored SHy model
        self.model = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hgnn_dim=16,
            hgnn_layers=1,
            num_tp=2,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
        )

    def test_initialization(self):
        """Checks if the model components are correctly initialized."""
        # Verify phenotype extractors count matches num_tp
        self.assertEqual(len(self.model.extractors), 2)
        # Check if the output layer has the correct label dimension
        expected_label_size = self.model.output_size
        self.assertEqual(self.model.output_layer.out_features, expected_label_size)

    def test_forward_pass_and_shapes(self):
        """Validates forward pass output keys and tensor shapes."""
        # Extract features for a batch
        diagnoses_hist = [s["diagnoses_hist"] for s in self.samples]

        # Run forward pass
        output = self.model(diagnoses_hist=diagnoses_hist)

        # Verify output structure
        self.assertIn("logits", output)
        self.assertIn("y_prob", output)

        # Verify output shapes: [batch_size, num_labels]
        batch_size = len(self.samples)
        num_labels = len(self.model.get_label_info("diagnoses"))
        self.assertEqual(output["logits"].shape, (batch_size, num_labels))
        self.assertEqual(output["y_prob"].shape, (batch_size, num_labels))

    def test_gradient_computation(self):
        """Verifies if gradients flow through the model during loss calculation."""
        # Mock ground truth labels (multilabel format)
        num_labels = len(self.model.get_label_info("diagnoses"))
        diagnoses_labels = torch.randint(0, 2, (len(self.samples), num_labels)).float()

        # Run forward with labels to trigger loss calculation
        diagnoses_hist = [s["diagnoses_hist"] for s in self.samples]
        output = self.model(diagnoses_hist=diagnoses_hist, diagnoses=diagnoses_labels)

        self.assertIn("loss", output)
        loss = output["loss"]

        # Perform backward pass
        loss.backward()

        # Check if code embeddings and weight matrices have gradients
        self.assertIsNotNone(self.model.code_embeddings.weight.grad)
        # Check an arbitrary weight in the first GNN layer
        self.assertIsNotNone(self.model.gnn_stack.linear_transform.weight.grad)

    def test_probability_range(self):
        """Ensures y_prob values are valid probabilities between 0 and 1."""
        diagnoses_hist = [s["diagnoses_hist"] for s in self.samples]
        output = self.model(diagnoses_hist=diagnoses_hist)

        y_prob = output["y_prob"]
        self.assertTrue(torch.all(y_prob >= 0))
        self.assertTrue(torch.all(y_prob <= 1))


if __name__ == "__main__":
    unittest.main()
