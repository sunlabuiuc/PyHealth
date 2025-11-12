import unittest
from typing import Dict, Type, Union

import torch
import numpy as np

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import MICRON
from pyhealth.processors.base_processor import FeatureProcessor


class TestMICRON(unittest.TestCase):
    """Test cases for the MICRON model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["E11.9", "I10"],  # diabetes, hypertension
                "procedures": ["0DJD8ZZ"],       # surgical procedure
                "drugs": ["A10BA02", "C09AA05"],  # metformin, ramipril
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["J45.901"],        # asthma
                "procedures": ["0BBJ4ZX"],        # bronchoscopy
                "drugs": ["R03BA02"],            # budesonide
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["J45.901", "R05"], # asthma, cough
                "procedures": ["0BBJ4ZX"],        # bronchoscopy
                "drugs": ["R03BA02", "R05DA04"], # budesonide, codeine
            },
        ]

        # Schema definition for PyHealth 2.0
        self.input_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "conditions": "sequence",
            "procedures": "sequence",
        }
        self.output_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "drugs": "multilabel"
        }

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = MICRON(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the MICRON model initializes correctly."""
        self.assertIsInstance(self.model, MICRON)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "drugs")

        # Test that the MICRON layer is initialized correctly
        self.assertEqual(
            self.model.micron.input_size,
            self.model.embedding_dim * len(self.model.feature_keys)
        )
        self.assertEqual(self.model.micron.hidden_size, self.model.hidden_dim)

    def test_model_forward(self):
        """Test that the MICRON forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        # Check required outputs
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

        # Check shapes
        batch_size = 2
        num_drugs = self.model.micron.num_labels
        self.assertEqual(ret["y_prob"].shape, (batch_size, num_drugs))
        self.assertEqual(ret["y_true"].shape, (batch_size, num_drugs))
        self.assertEqual(ret["loss"].dim(), 0)  # scalar loss

        # Check value ranges
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))
        self.assertTrue(torch.all(torch.logical_or(ret["y_true"] == 0, ret["y_true"] == 1)))

    def test_model_backward(self):
        """Test that the MICRON backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        # Check that gradients are computed
        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        """Test that the MICRON returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        expected_seq_len = max(len(data_batch["conditions"][0]), len(data_batch["procedures"][0]))
        expected_feature_dim = self.model.embedding_dim * len(self.model.feature_keys)
        self.assertEqual(ret["embed"].shape[1], expected_seq_len)
        self.assertEqual(ret["embed"].shape[2], expected_feature_dim)

    def test_custom_hyperparameters(self):
        """Test MICRON with custom hyperparameters."""
        model = MICRON(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            lam=0.2,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.micron.lam, 0.2)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_ddi_adjacency_matrix(self):
        """Test the drug-drug interaction adjacency matrix generation."""
        ddi_matrix = self.model.generate_ddi_adj()
        
        # Check matrix properties
        self.assertIsInstance(ddi_matrix, torch.Tensor)
        num_drugs = self.model.micron.num_labels
        self.assertEqual(ddi_matrix.shape, (num_drugs, num_drugs))
        
        # Check matrix is symmetric
        self.assertTrue(torch.allclose(ddi_matrix, ddi_matrix.t()))
        
        # Check diagonal is zero (no self-interactions)
        self.assertTrue(torch.all(torch.diag(ddi_matrix) == 0))
        
        # Check values are binary
        self.assertTrue(torch.all(torch.logical_or(ddi_matrix == 0, ddi_matrix == 1)))

    def test_reconstruction_loss(self):
        """Test the reconstruction loss computation."""
        batch_size = 2
        seq_len = 3
        num_drugs = 4
        
        # Create dummy data
        logits = torch.randn(batch_size, seq_len, num_drugs)
        logits_residual = torch.randn(batch_size, seq_len - 1, num_drugs)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        rec_loss = self.model.micron.compute_reconstruction_loss(
            logits, logits_residual, mask
        )
        
        self.assertEqual(rec_loss.dim(), 0)  # scalar loss
        self.assertTrue(rec_loss >= 0)  # non-negative loss


if __name__ == "__main__":
    unittest.main()