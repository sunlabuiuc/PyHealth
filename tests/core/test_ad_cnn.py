import math
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SimpleADCNN, BaseModel


class TestSimpleADCNN(unittest.TestCase):
    """Test cases for the SimpleADCNN model."""

    def setUp(self):
        """Set up a tiny synthetic dataset and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "mri": torch.randn(1, 4, 6, 8).tolist(),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "mri": torch.randn(1, 4, 6, 8).tolist(),
                "label": 0,
            },
        ]

        self.input_schema = {"mri": "tensor"}
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = SimpleADCNN(
            dataset=self.dataset,
            conv_channels=(4, 8),
            dense_dim=8,
        )

    def test_model_initialization(self):
        """Test that SimpleADCNN initializes correctly and inherits BaseModel."""
        self.assertIsInstance(self.model, SimpleADCNN)
        self.assertIsInstance(self.model, BaseModel)
        self.assertEqual(self.model.conv_channels, (4, 8))
        self.assertEqual(self.model.dense_dim, 8)
        self.assertEqual(self.model.dropout, 0.4)
        self.assertEqual(self.model.in_channels, 1)
        self.assertEqual(self.model.kernel_size, 3)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("mri", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the forward pass returns correct keys and shapes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
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
        """Test that gradients flow through all trainable parameters."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_custom_hyperparameters(self):
        """Test SimpleADCNN with non-default hyperparameters."""
        model = SimpleADCNN(
            dataset=self.dataset,
            conv_channels=(8, 16, 32),
            dropout=0.2,
            dense_dim=16,
            kernel_size=5,
        )

        self.assertEqual(model.conv_channels, (8, 16, 32))
        self.assertEqual(model.dropout, 0.2)
        self.assertEqual(model.dense_dim, 16)
        self.assertEqual(model.kernel_size, 5)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_different_input_shapes(self):
        """Test that the model handles different volume dimensions."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "mri": torch.randn(1, 6, 8, 10).tolist(),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "mri": torch.randn(1, 6, 8, 10).tolist(),
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"mri": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_shape",
        )

        model = SimpleADCNN(dataset=dataset, conv_channels=(4, 8), dense_dim=8)

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_rejects_multiple_features(self):
        """Test that SimpleADCNN raises on datasets with multiple input features."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "mri": torch.randn(1, 4, 6, 8).tolist(),
                "extra": torch.randn(1, 4, 6, 8).tolist(),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "mri": torch.randn(1, 4, 6, 8).tolist(),
                "extra": torch.randn(1, 4, 6, 8).tolist(),
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"mri": "tensor", "extra": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_multi",
        )

        with self.assertRaises(ValueError):
            SimpleADCNN(dataset=dataset, conv_channels=(4, 8), dense_dim=8)

    def test_rejects_invalid_hyperparameters(self):
        """Test that invalid model hyperparameters raise clear errors."""
        invalid_cases = [
            {"conv_channels": ()},
            {"conv_channels": (4, 0, 8)},
            {"in_channels": 0},
            {"kernel_size": 2},
            {"dropout": -0.1},
            {"dropout": 1.1},
            {"dense_dim": 0},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SimpleADCNN(dataset=self.dataset, **kwargs)

    def test_multi_channel_input(self):
        """Test that in_channels > 1 works with matching input data."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "mri": torch.randn(3, 4, 6, 8).tolist(),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "mri": torch.randn(3, 4, 6, 8).tolist(),
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"mri": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_multichan",
        )

        model = SimpleADCNN(
            dataset=dataset,
            in_channels=3,
            conv_channels=(4, 8),
            dense_dim=8,
        )

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)

    def test_channel_mismatch_rejection(self):
        """Test that 4D input raises when in_channels != 1."""
        model = SimpleADCNN(
            dataset=self.dataset,
            in_channels=3,
            conv_channels=(4, 8),
            dense_dim=8,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        # self.dataset has (1, 4, 6, 8) volumes — 4D after batch dim
        # Model has in_channels=3, so the unsqueeze guard should reject
        with self.assertRaises(ValueError):
            model(**data_batch)

    def test_forward_rejects_invalid_input_shape(self):
        """Test that invalid MRI tensor shapes raise clear errors."""
        with self.assertRaises(ValueError):
            self.model(
                mri=torch.randn(2, 4, 6),
                label=torch.tensor([[1.0], [0.0]]),
            )

    def test_he_initialization(self):
        """Spot-check that conv weights follow He-uniform distribution."""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv3d):
                fan_in = (
                    m.in_channels
                    * m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                )
                # He-uniform bound = sqrt(6 / fan_in)
                expected_bound = math.sqrt(6.0 / fan_in)
                weight_max = m.weight.data.abs().max().item()
                # Weights should be within the He-uniform bound
                # (with a small tolerance for floating point)
                self.assertLessEqual(weight_max, expected_bound + 1e-6)
                # Weights should not be all zeros
                self.assertGreater(m.weight.data.abs().sum().item(), 0.0)
                break  # only check the first conv layer


if __name__ == "__main__":
    unittest.main()