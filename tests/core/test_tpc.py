"""
Unit tests for TPC (Temporal Pointwise Convolution) model.

Tests include:
- Model initialization with various configurations
- Forward pass with synthetic data
- Output shape validation
- Gradient computation (backward pass)
- MC Dropout uncertainty estimation
- Custom hyperparameters
"""

import unittest
import torch
import numpy as np

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TPC, MSLELoss, MaskedMSELoss


class TestTPC(unittest.TestCase):
    """Test cases for the TPC model and its components."""

    def setUp(self):
        """Set up test data with MINIMAL synthetic samples."""
        # Create minimal synthetic data - 2 samples only!
        # TPC expects specific input format from RemainingLOSMIMIC4 task
        
        F = 4  # Number of clinical features (keep SMALL for fast tests)
        T = 10  # Sequence length (keep SHORT)
        
        # Sample 1: Normal case
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                # timeseries shape: (2F+2, T) = (10, 10)
                # [elapsed(1), values(F), decay(F), hour_of_day(1)]
                "timeseries": torch.randn(2 * F + 2, T),
                # static features: [age, sex]
                "static": torch.tensor([65.0, 1.0]),
                # diagnosis codes (will be processed by SequenceProcessor)
                "conditions": ["icd_A01", "icd_B02"],
                # remaining LoS labels in hours
                "los": torch.rand(T) * 48,  # 0-48 hours
            },
            # Sample 2: Another normal case
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "timeseries": torch.randn(2 * F + 2, T),
                "static": torch.tensor([72.0, 0.0]),
                "conditions": ["icd_A01"],
                "los": torch.rand(T) * 24,  # 0-24 hours
            },
        ]

        # Define schema matching TPC requirements
        self.input_schema = {
            "timeseries": "tensor",
            "static": "tensor",
            "conditions": "sequence",
        }
        self.output_schema = {"los": "tensor"}

        # Create synthetic dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="tpc_test",
        )

        # Create model with SMALL configuration for fast testing
        self.model = TPC(
            dataset=self.dataset,
            n_layers=2,  # Minimal layers
            temp_kernels=[4, 4],  # Small kernel counts
            point_sizes=[8, 8],  # Small hidden sizes
            diagnosis_size=16,  # Small diagnosis encoding
            last_linear_size=16,  # Small final layer
            time_before_pred=3,  # Minimum history
        )

    def test_model_initialization(self):
        """Test that TPC initializes correctly with all parameters."""
        self.assertIsInstance(self.model, TPC)
        
        # Check configuration parameters
        self.assertEqual(self.model.n_layers, 2)
        self.assertEqual(len(self.model.temp_kernels), 2)
        self.assertEqual(len(self.model.point_sizes), 2)
        self.assertEqual(self.model.diagnosis_size, 16)
        self.assertEqual(self.model.last_linear_size, 16)
        self.assertEqual(self.model.time_before_pred, 3)
        
        # Check feature keys
        self.assertIn("timeseries", self.model.feature_keys)
        self.assertIn("static", self.model.feature_keys)
        self.assertIn("conditions", self.model.feature_keys)
        
        # Check label key
        self.assertEqual(self.model.label_key, "los")
        
        # Check mode
        self.assertEqual(self.model.mode, "regression")

    def test_model_forward_pass(self):
        """Test TPC forward pass produces correct output structure."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            output = self.model(**data_batch)

        # Check required output keys
        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)
        
        # Check output types
        self.assertIsInstance(output["loss"], torch.Tensor)
        self.assertIsInstance(output["y_prob"], torch.Tensor)
        self.assertIsInstance(output["y_true"], torch.Tensor)
        
        # Check loss is scalar
        self.assertEqual(output["loss"].dim(), 0, "Loss should be scalar")

    def test_output_shapes(self):
        """Test that TPC produces correct output shapes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            output = self.model(**data_batch)

        # With time_before_pred=3 and T=10, we get 7 prediction timesteps per sample
        # Flattened predictions = batch_size * (T - time_before_pred)
        B = 2
        T = 10
        expected_flat_predictions = B * (T - self.model.time_before_pred)
        
        # Check flattened output shapes (default behavior)
        self.assertEqual(
            output["y_prob"].shape[0],
            expected_flat_predictions,
            "y_prob should be flattened predictions"
        )
        self.assertEqual(
            output["y_true"].shape[0],
            expected_flat_predictions,
            "y_true should be flattened labels"
        )

    def test_output_shapes_full_sequence(self):
        """Test that TPC can return full sequences when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            output = self.model(return_full_sequence=True, **data_batch)

        B = 2
        T = 10
        post_hist = T - self.model.time_before_pred
        
        # Check full sequence output shapes
        self.assertEqual(
            output["y_prob"].shape,
            (B, post_hist),
            f"y_prob should be (batch, post_hist) = ({B}, {post_hist})"
        )
        self.assertEqual(
            output["y_true"].shape,
            (B, post_hist),
            f"y_true should be (batch, post_hist) = ({B}, {post_hist})"
        )
        self.assertIn("mask", output, "Should include mask in full sequence mode")

    def test_backward_pass(self):
        """Test that gradients are computed correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Forward pass
        output = self.model(**data_batch)
        loss = output["loss"]
        
        # Backward pass
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(
            has_gradient,
            "Model parameters should have gradients after backward pass"
        )

    def test_custom_hyperparameters(self):
        """Test TPC with different hyperparameter configurations."""
        # Test with MSLE loss
        model_msle = TPC(
            dataset=self.dataset,
            n_layers=1,
            temp_kernels=[8],
            point_sizes=[12],
            use_msle=True,
            apply_exp=True,
        )
        
        self.assertEqual(model_msle.n_layers, 1)
        self.assertIsInstance(model_msle.loss_fn, MSLELoss)
        self.assertTrue(model_msle.apply_exp)
        
        # Test with MSE loss
        model_mse = TPC(
            dataset=self.dataset,
            n_layers=1,
            temp_kernels=[8],
            point_sizes=[12],
            use_msle=False,
            apply_exp=False,
        )
        
        self.assertIsInstance(model_mse.loss_fn, MaskedMSELoss)
        self.assertFalse(model_mse.apply_exp)
        
        # Verify both models can run forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            out_msle = model_msle(**data_batch)
            out_mse = model_mse(**data_batch)
        
        self.assertIn("loss", out_msle)
        self.assertIn("loss", out_mse)

    def test_mc_dropout_uncertainty(self):
        """Test Monte Carlo Dropout uncertainty estimation (ablation study feature)."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # Run MC Dropout with small sample count for speed
        mc_samples = 5
        with torch.no_grad():
            uncertainty_output = self.model.predict_with_uncertainty(
                mc_samples=mc_samples,
                **data_batch
            )

        # Check required output keys
        self.assertIn("mean", uncertainty_output)
        self.assertIn("std", uncertainty_output)
        self.assertIn("samples", uncertainty_output)
        
        # Check shapes
        B = 2
        T = 10
        post_hist = T - self.model.time_before_pred
        
        self.assertEqual(
            uncertainty_output["mean"].shape,
            (B, post_hist),
            "Mean should have shape (batch, post_hist)"
        )
        self.assertEqual(
            uncertainty_output["std"].shape,
            (B, post_hist),
            "Std should have shape (batch, post_hist)"
        )
        self.assertEqual(
            uncertainty_output["samples"].shape,
            (mc_samples, B, post_hist),
            f"Samples should have shape ({mc_samples}, {B}, {post_hist})"
        )
        
        # Verify uncertainty values are reasonable
        self.assertTrue(
            torch.all(uncertainty_output["std"] >= 0),
            "Standard deviation should be non-negative"
        )

    def test_loss_functions(self):
        """Test custom loss functions (MSLELoss and MaskedMSELoss)."""
        B, T = 2, 10
        
        # Create synthetic predictions and targets
        y_hat = torch.rand(B, T) * 10  # Predictions 0-10 hours
        y = torch.rand(B, T) * 10  # True values 0-10 hours
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, :3] = False  # Mask out first 3 timesteps
        seq_length = mask.sum(dim=1)
        
        # Test MSLELoss
        msle_loss_fn = MSLELoss()
        msle_loss = msle_loss_fn(y_hat, y, mask, seq_length, sum_losses=False)
        
        self.assertIsInstance(msle_loss, torch.Tensor)
        self.assertEqual(msle_loss.dim(), 0, "Loss should be scalar")
        self.assertTrue(msle_loss >= 0, "MSLE should be non-negative")
        
        # Test MaskedMSELoss
        mse_loss_fn = MaskedMSELoss()
        mse_loss = mse_loss_fn(y_hat, y, mask, seq_length, sum_losses=False)
        
        self.assertIsInstance(mse_loss, torch.Tensor)
        self.assertEqual(mse_loss.dim(), 0, "Loss should be scalar")
        self.assertTrue(mse_loss >= 0, "MSE should be non-negative")

    def test_minimal_config(self):
        """Test TPC with absolute minimum configuration."""
        # Single layer, smallest possible model
        tiny_model = TPC(
            dataset=self.dataset,
            n_layers=1,
            temp_kernels=[2],
            point_sizes=[4],
            diagnosis_size=8,
            last_linear_size=8,
            time_before_pred=2,
        )
        
        self.assertEqual(tiny_model.n_layers, 1)
        
        # Verify it can still run
        train_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))
        
        tiny_model.eval()  # Set to eval mode for batch_size=1
        with torch.no_grad():
            output = tiny_model(**data_batch)
        
        self.assertIn("loss", output)

    def test_edge_case_short_sequence(self):
        """Test TPC behavior with very short sequences (edge case)."""
        # Create dataset with minimum viable sequence length
        F = 4
        T = 10  # Enough for time_before_pred=3 and some predictions
        
        short_samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "timeseries": torch.randn(2 * F + 2, T),
                "static": torch.tensor([65.0, 1.0]),
                "conditions": ["icd_A01"],
                "los": torch.rand(T) * 48,
            }
        ]
        
        short_dataset = create_sample_dataset(
            samples=short_samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="tpc_short",
        )
        
        short_model = TPC(
            dataset=short_dataset,
            n_layers=1,
            temp_kernels=[4],
            point_sizes=[8],
            time_before_pred=3,
        )
        
        loader = get_dataloader(short_dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(loader))
        
        short_model.eval()  # Set to eval mode for batch_size=1
        with torch.no_grad():
            output = short_model(**data_batch)
        
        # Should still produce valid output
        self.assertIn("loss", output)
        self.assertIn("y_prob", output)


class TestTPCLossFunctions(unittest.TestCase):
    """Separate test class for TPC loss functions."""
    
    def test_msle_loss_properties(self):
        """Test mathematical properties of MSLE loss."""
        msle = MSLELoss()
        
        # Test that identical predictions give ~zero loss
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_hat = y.clone()
        mask = torch.ones_like(y, dtype=torch.bool)
        seq_length = torch.tensor([3, 3])
        
        loss = msle(y_hat, y, mask, seq_length)
        self.assertLess(loss.item(), 1e-5, "Loss should be near zero for identical predictions")
    
    def test_masked_mse_loss_masking(self):
        """Test that MaskedMSELoss correctly ignores masked values."""
        mse = MaskedMSELoss()
        
        # Create data where masked values have large errors
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_hat = torch.tensor([[1.0, 2.0, 100.0], [4.0, 5.0, 200.0]])  # Large errors at end
        mask = torch.tensor([[True, True, False], [True, True, False]])  # Mask the errors
        seq_length = torch.tensor([2, 2])
        
        loss = mse(y_hat, y, mask, seq_length)
        
        # Loss should be small because errors are masked
        self.assertLess(loss.item(), 1.0, "Masked values should not contribute to loss")


if __name__ == "__main__":
    unittest.main()
