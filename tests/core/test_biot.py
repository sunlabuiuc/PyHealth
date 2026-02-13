import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BIOT


class TestBIOT(unittest.TestCase):
    """Test cases for the BIOT model."""

    def setUp(self):
        """Set up test data and model."""
        n_channels = 18
        n_time = 10
        n_fft = 200
        hop_length = 100
        n_samples = n_fft * n_time  # 2000

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
                "label": 0,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-0",
                "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
                "label": 2,
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-0",
                "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
                "label": 3,
            },
        ]

        self.input_schema = {
            "signal": "tensor",
        }
        self.output_schema = {"label": "multiclass"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_biot",
        )

        self.model = BIOT(
            dataset=self.dataset,
            emb_size=256,
            heads=8,
            depth=4,
            n_fft=200,
            hop_length=100,
            n_classes=6,
            n_channels=18,
        )

    def test_model_initialization(self):
        """Test that the BIOT model initializes correctly."""
        self.assertIsInstance(self.model, BIOT)
        self.assertIsNotNone(self.model.biot)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("signal", self.model.feature_keys)

    def test_model_forward(self):
        """Test that the BIOT forward pass works correctly."""
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
        self.assertEqual(ret["logit"].shape[1], 6)  # n_classes
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the BIOT backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_different_batch_sizes(self):
        """Test BIOT with different batch sizes."""
        for batch_size in [1, 2, 4]:
            train_loader = get_dataloader(self.dataset, batch_size=batch_size, shuffle=False)
            data_batch = next(iter(train_loader))

            with torch.no_grad():
                ret = self.model(**data_batch)

            actual_batch = min(batch_size, len(self.samples))
            self.assertEqual(ret["y_prob"].shape[0], actual_batch)
            self.assertEqual(ret["y_true"].shape[0], actual_batch)

    def test_model_output_probabilities(self):
        """Test that output probabilities are valid."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        y_prob = ret["y_prob"]
        # Probabilities should be between 0 and 1
        self.assertTrue(torch.all(y_prob >= 0), "Probabilities contain negative values")
        self.assertTrue(torch.all(y_prob <= 1), "Probabilities exceed 1")

    def test_missing_signal_raises_error(self):
        """Test that missing 'signal' input raises ValueError."""
        with self.assertRaises((ValueError, KeyError)):
            self.model(label=torch.tensor([0, 1]))

    def test_model_different_n_classes(self):
        """Test BIOT with different number of classes."""
        model_binary = BIOT(
            dataset=self.dataset,
            emb_size=256,
            heads=8,
            depth=4,
            n_fft=200,
            hop_length=100,
            n_classes=2,
            n_channels=18,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_binary(**data_batch)

        self.assertEqual(ret["logit"].shape[1], 2)

    def test_model(self):
        """Test BIOT"""
        model_small = BIOT(
            dataset=self.dataset,
            emb_size=256,
            heads=4,
            depth=2,
            n_fft=200,
            hop_length=100,
            n_classes=6,
            n_channels=18,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_small(**data_batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[1], 6)


if __name__ == "__main__":
    unittest.main()