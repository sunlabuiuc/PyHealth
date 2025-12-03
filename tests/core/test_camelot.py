"""Test cases for the CAMELOT model."""

import unittest
import torch

from pyhealth.datasets import SampleDataset
from pyhealth.models import CAMELOTModule


class TestCAMELOT(unittest.TestCase):
    """Test cases for the CAMELOT model."""

    def setUp(self):
        """Set up test data and model."""
        # Create a minimal dataset for testing
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-1", "cond-2"], ["cond-3", "cond-4"]],
                "procedures": [["proc-1"], ["proc-2", "proc-3"]],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["cond-5", "cond-6"]],
                "procedures": [["proc-4"]],
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

    def test_model_initialization(self):
        """Test that the CAMELOT model initializes correctly."""
        model = CAMELOTModule(
            task="ihm",
            modeltype="TS",
            dataset=self.dataset,
        )
        self.assertIsInstance(model, CAMELOTModule)
        self.assertEqual(model.task, "ihm")
        self.assertEqual(model.modeltype, "TS")
        self.assertEqual(model.K, 10)  # default num_clusters

    def test_model_initialization_with_params(self):
        """Test that the CAMELOT model initializes with custom parameters."""
        model = CAMELOTModule(
            task="ihm",
            modeltype="TS",
            num_clusters=5,
            hidden_dim=32,
            dataset=self.dataset,
        )
        self.assertEqual(model.K, 5)
        self.assertEqual(model.hidden_dim, 32)

    def test_model_forward(self):
        """Test forward pass for CAMELOT model."""
        model = CAMELOTModule(
            task="ihm",
            modeltype="TS",
            dataset=self.dataset,
        )
        
        # Create dummy input data
        batch_size = 2
        seq_len = 48
        reg_ts = torch.randn(batch_size, seq_len, 30)  # orig_reg_d_ts=30
        
        # Test forward pass without labels (inference)
        with torch.no_grad():
            output = model(reg_ts=reg_ts)
            self.assertEqual(output.shape, (batch_size,))
        
        # Test forward pass with labels (training)
        labels = torch.randint(0, 2, (batch_size,))
        loss = model(reg_ts=reg_ts, labels=labels)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_model_forward_pheno(self):
        """Test forward pass for phenotyping task."""
        model = CAMELOTModule(
            task="pheno",
            modeltype="TS",
            dataset=self.dataset,
        )
        
        batch_size = 2
        seq_len = 48
        reg_ts = torch.randn(batch_size, seq_len, 30)
        labels = torch.randint(0, 2, (batch_size, 25))  # 25 phenotypes
        
        loss = model(reg_ts=reg_ts, labels=labels)
        self.assertIsInstance(loss, torch.Tensor)

    def test_model_parameters(self):
        """Test that model parameters are properly initialized."""
        model = CAMELOTModule(
            task="ihm",
            modeltype="TS",
            dataset=self.dataset,
        )
        
        # Check that model has parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)
        
        # Check specific components
        self.assertTrue(hasattr(model, "Encoder"))
        self.assertTrue(hasattr(model, "Identifier"))
        self.assertTrue(hasattr(model, "predictor"))
        self.assertTrue(hasattr(model, "cluster_rep_set"))
        self.assertEqual(model.cluster_rep_set.shape, (model.K, model.hidden_dim))

    def test_model_device(self):
        """Test that model device property works."""
        model = CAMELOTModule(
            task="ihm",
            modeltype="TS",
            dataset=self.dataset,
        )
        
        # Check device property exists (from BaseModel)
        self.assertTrue(hasattr(model, "device"))
        device = model.device
        self.assertIsInstance(device, torch.device)

    def test_cluster_losses(self):
        """Test that cluster loss functions work correctly."""
        model = CAMELOTModule(
            task="ihm",
            modeltype="TS",
            dataset=self.dataset,
        )
        
        # Test l_pat_dist
        clusters_prob = torch.softmax(torch.randn(4, model.K), dim=-1)
        loss = model.l_pat_dist(clusters_prob)
        self.assertIsInstance(loss, torch.Tensor)
        
        # Test l_clus_dist
        loss = model.l_clus_dist(clusters_prob)
        self.assertIsInstance(loss, torch.Tensor)
        
        # Test l_clus
        cluster_reps = torch.randn(model.K, model.hidden_dim)
        loss = model.l_clus(cluster_reps)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()

