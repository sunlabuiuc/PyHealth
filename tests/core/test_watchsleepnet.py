import unittest
import numpy as np
import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import WatchSleepNet


SEQ_LEN = 4
SEQ_SAMPLE_SIZE = 750
NUM_CLASSES = 3


def make_dataset(num_samples=4, num_classes=NUM_CLASSES, seed=0):
    """Create a small synthetic dataset for WatchSleepNet testing."""
    rng = np.random.RandomState(seed)
    samples = []
    for i in range(num_samples):
        samples.append(
            {
                "patient_id": f"patient-{i}",
                "record_id": f"patient-{i}-0",
                "signal": rng.randn(SEQ_LEN, SEQ_SAMPLE_SIZE).astype(np.float32),
                "label": i % num_classes,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="test_watchsleepnet",
    )


class TestWatchSleepNet(unittest.TestCase):
    """Test cases for the WatchSleepNet model."""

    def setUp(self):
        """Set up test dataset and model."""
        self.dataset = make_dataset(num_samples=4)
        self.model = WatchSleepNet(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that WatchSleepNet initializes with correct attributes."""
        self.assertIsInstance(self.model, WatchSleepNet)
        self.assertIsNotNone(self.model.feature_extractor)
        self.assertIsNotNone(self.model.tcn)
        self.assertIsNotNone(self.model.lstm)
        self.assertIsNotNone(self.model.attention)
        self.assertIsNotNone(self.model.classifier)
        self.assertEqual(self.model.classifier.out_features, NUM_CLASSES)

    def test_forward_input_format(self):
        """Test that the batch dict has the expected keys and tensor shapes."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        self.assertIn("signal", batch)
        self.assertIn("label", batch)
        self.assertEqual(len(batch["signal"].shape), 3)
        self.assertEqual(len(batch["label"].shape), 1)

    def test_model_forward(self):
        """Test that the forward pass returns correct keys and shapes."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["y_prob"].shape, (4, NUM_CLASSES))
        self.assertEqual(ret["y_true"].shape, (4,))

    def test_model_backward(self):
        """Test that gradients flow through the model on a backward pass."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_output_shapes(self):
        """Test that output tensor shapes are correct."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        batch_size = batch["label"].shape[0]
        self.assertEqual(ret["y_prob"].shape, (batch_size, NUM_CLASSES))
        self.assertEqual(ret["y_true"].shape, (batch_size,))
        self.assertEqual(ret["loss"].shape, ())

    def test_custom_hyperparameters(self):
        """Test WatchSleepNet initializes and runs with custom hyperparameters."""
        model = WatchSleepNet(
            dataset=self.dataset,
            lstm_hidden_size=64,
            lstm_num_layers=1,
            num_heads=4,
            tcn_kernel_size=3,
        )
        self.assertEqual(model.classifier.out_features, NUM_CLASSES)

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape, (4, NUM_CLASSES))

    def test_loss_is_finite(self):
        """Test that the loss is a finite value."""
        torch.manual_seed(42)
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_y_true_matches_labels(self):
        """Test that y_true matches the input labels."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        expected = batch["label"]
        self.assertTrue(torch.equal(ret["y_true"].cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()
