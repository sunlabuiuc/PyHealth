import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SPESTransformer


class TestSPESTransformer(unittest.TestCase):
    """Test cases for the SPESTransformer model."""

    def setUp(self):
        """Set up test data and model."""
        torch.manual_seed(11)
        # Create minimal synthetic data: [C, 2, T+1].
        # Transformer's MLP/conv branches expect long enough T for fixed slices.
        # Distances are at index 0 of the time dimension.
        # We give one sample 5 channels and another 3 channels to test padding.
        sample0_tensor = torch.randn(5, 2, 509)
        sample1_tensor = torch.randn(3, 2, 509)
        
        # Inject positive distances so valid channels are identified
        sample0_tensor[:, 0, 0] = torch.abs(torch.randn(5)) + 1.0
        sample1_tensor[:, 0, 0] = torch.abs(torch.randn(3)) + 1.0

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "spes_responses": sample0_tensor.tolist(),
                "soz_label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "spes_responses": sample1_tensor.tolist(),
                "soz_label": 0,
            },
        ]

        self.input_schema = {
            "spes_responses": "tensor",
        }
        self.output_schema = {"soz_label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_spes",
        )

        # Use small architecture configurations for fast execution
        self.model = SPESTransformer(
            dataset=self.dataset,
            mean=True,
            std=True,
            conv_embedding=True,
            mlp_embedding=True,
            dropout_rate=0.5,
            num_layers=1,         # Tiny model
            embedding_dim=16,     # Tiny embedding
            random_channels=2,    # Small number of channels
            noise_std=0.1
        )

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, SPESTransformer)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("spes_responses", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "soz_label")
        self.assertEqual(self.model.mode, "binary")

    def test_model_forward(self):
        """Test that the model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_output_probabilities_in_range(self):
        """Test that predicted probabilities are bounded in [0, 1]."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        with torch.no_grad():
            ret = self.model(**data_batch)
        self.assertTrue(torch.all(ret["y_prob"] >= 0.0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1.0))

    def test_missing_spes_responses_raises_error(self):
        """Test that missing model feature input raises an error."""
        with self.assertRaises((KeyError, ValueError, TypeError)):
            self.model(soz_label=torch.tensor([0, 1]))

    def test_repeated_eval_forward_outputs_are_finite(self):
        """Test repeated eval forwards produce finite values with valid shapes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        self.model.eval()
        with torch.no_grad():
            ret1 = self.model(**data_batch)
            ret2 = self.model(**data_batch)
        self.assertEqual(ret1["logit"].shape, ret2["logit"].shape)
        self.assertTrue(torch.isfinite(ret1["logit"]).all())
        self.assertTrue(torch.isfinite(ret2["logit"]).all())
        self.assertTrue(torch.isfinite(ret1["y_prob"]).all())
        self.assertTrue(torch.isfinite(ret2["y_prob"]).all())

    def test_model_configurations(self):
        """Test the model when toggling mean/std or embedding types."""
        # Test without conv embedding
        model_no_conv = SPESTransformer(
            dataset=self.dataset,
            conv_embedding=False,
            mlp_embedding=True,
            num_layers=1,
            embedding_dim=16,
            random_channels=None # Test without random sampling
        )
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            ret = model_no_conv(**data_batch)
        self.assertEqual(ret["logit"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
