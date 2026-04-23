import unittest
import torch
import numpy as np
from typing import Dict
from pyhealth.models import ClinicalTSFTransformer

class TestClinicalTSFTransformer(unittest.TestCase):
    """Unit tests for ClinicalTSFTransformer using structured sample data."""

    def setUp(self) -> None:
        """Sets up the model and structured sample data."""
        self.feature_size = 131
        self.batch_size = 2
        self.seq_len = 24
        
        # 1. Create Structured Sample Data
        # We create a "Sepsis" pattern (increasing heart rate, decreasing BP)
        # and a "Healthy" pattern (stable values).
        x = torch.zeros((self.batch_size, self.seq_len, self.feature_size))
        
        # Patient 0: Sepsis (Trend upwards in feature index 0)
        x[0, :, 0] = torch.linspace(70, 120, self.seq_len) 
        # Patient 1: Healthy (Stable around 70)
        x[1, :, 0] = torch.full((self.seq_len,), 70.0)
        
        y = torch.tensor([1, 0]) # Labels matching the patterns
        
        self.sample_batch = {"x": x, "y": y}

        # 2. Mock PyHealth Dataset
        class MockDataset:
            def __init__(self):
                self.input_info = {"x": {"type": torch.Tensor}}
                self.output_info = {"y": {"type": torch.Tensor}}
        
        self.model = ClinicalTSFTransformer(
            dataset=MockDataset(),
            feature_size=self.feature_size,
            nhead=1,
            num_layers=1
        )

    def test_logic_and_shapes(self):
        """Verifies model output shapes and non-random loss on sample data."""
        output = self.model(**self.sample_batch)
        
        # Check Shapes
        self.assertEqual(output["y_prob"].shape, (self.batch_size, 1))
        self.assertEqual(output["reconstruction"].shape, self.sample_batch["x"].shape)
        
        # Check Loss
        loss = output["loss"]
        self.assertFalse(torch.isnan(loss), "Loss is NaN")
        self.assertGreater(loss.item(), 0, "Loss should be positive")

    def test_reconstruction_fidelity(self):
        """Checks if the reconstruction head output is differentiable against inputs."""
        output = self.model(**self.sample_batch)
        recon = output["reconstruction"]
        
        # If the model is learning to forecast, the reconstruction should 
        # eventually converge toward the input 'x'.
        # We check if we can compute a gradient from the reconstruction error.
        recon_loss = torch.nn.MSELoss()(recon, self.sample_batch["x"])
        recon_loss.backward()
        
        self.assertIsNotNone(self.model.forecasting_head.weight.grad)

if __name__ == "__main__":
    unittest.main()