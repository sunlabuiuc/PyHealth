import unittest
import torch
import os

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import VAE


class TestVAE(unittest.TestCase):
    """Test cases for the VAE model."""

    def test_model_initialization_image(self):
        """Test VAE initialization for image mode."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "image": torch.randn(1, 128, 128),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "image": torch.randn(1, 128, 128),
                "label": 0,
            },
        ]

        input_schema = {"image": "tensor"}
        output_schema = {"label": "binary"}

        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_image",
        )

        model = VAE(
            dataset=dataset,
            feature_keys=["image"],
            label_key="label",
            mode="binary",
            input_type="image",
            input_channel=1,  # assuming grayscale
            input_size=128,  # use 128
            hidden_dim=64,
        )

        self.assertIsInstance(model, VAE)
        self.assertEqual(model.input_type, "image")
        self.assertEqual(model.hidden_dim, 64)

    def test_model_forward_image(self):
        """Test VAE forward pass for image mode."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "image": torch.rand(1, 128, 128),  # dummy image 0-1
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "image": torch.rand(1, 128, 128),
                "label": 0,
            },
        ]

        input_schema = {"image": "tensor"}
        output_schema = {"label": "binary"}

        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_image",
        )

        model = VAE(
            dataset=dataset,
            feature_keys=["image"],
            label_key="label",
            mode="binary",
            input_type="image",
            input_channel=1,
            input_size=128,
            hidden_dim=64,
        )

        train_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

    def test_model_initialization_timeseries(self):
        """Test VAE initialization for timeseries mode."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86"],
                "label": 1.0,  # dummy
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33"],
                "label": 0.5,
            },
        ]

        input_schema = {"conditions": "sequence"}
        output_schema = {"label": "regression"}

        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_ts",
        )

        model = VAE(
            dataset=dataset,
            feature_keys=["conditions"],
            label_key="label",
            mode="regression",
            input_type="timeseries",
            hidden_dim=64,
        )

        self.assertIsInstance(model, VAE)
        self.assertEqual(model.input_type, "timeseries")
        self.assertTrue(hasattr(model, "embedding_model"))

    def test_model_forward_timeseries(self):
        """Test VAE forward pass for timeseries mode."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86"],
                "label": 1.0,
            },
        ]

        input_schema = {"conditions": "sequence"}
        output_schema = {"label": "regression"}

        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_ts",
        )

        model = VAE(
            dataset=dataset,
            feature_keys=["conditions"],
            label_key="label",
            mode="regression",
            input_type="timeseries",
            hidden_dim=64,
        )

        train_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

    def test_conditional_image_vae(self):
        """Test VAE with conditional features for image mode."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "image": torch.rand(1, 128, 128),
                "conditions": ["cond-33"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "image": torch.randn(1, 128, 128),
                "conditions": ["cond-86"],
                "label": 0,
            },
        ]

        input_schema = {"image": "tensor", "conditions": "sequence"}
        output_schema = {"label": "binary"}

        dataset = SampleDataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_cond",
        )

        model = VAE(
            dataset=dataset,
            feature_keys=["image"],
            label_key="label",
            mode="binary",
            input_type="image",
            input_channel=1,
            input_size=128,
            hidden_dim=64,
            conditional_feature_keys=["conditions"],
        )

        self.assertTrue(hasattr(model, "embedding_model"))

        train_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)


if __name__ == "__main__":
    unittest.main()