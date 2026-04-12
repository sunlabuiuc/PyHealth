"""
Unit tests for Wav2Sleep model.
Requirement: Fast, performant, and uses synthetic data.
"""
import unittest
import torch
from pyhealth.models import Wav2Sleep


class TestWav2Sleep(unittest.TestCase):
    def setUp(self):
        class MockDataset:
            def __init__(self):
                self.input_schema = {
                    "ecg": {"type": float},
                    "ppg": {"type": float}
                }

                self.output_schema = {
                    "label": {"type": int}
                }

        self.dataset = MockDataset()
        self.feature_keys = ["ecg", "ppg"]
        self.label_key = "label"

        self.model = Wav2Sleep(
            dataset=self.dataset,
            feature_keys=self.feature_keys,
            label_key=self.label_key,
            mode="multiclass",
            embedding_dim=128,
            nhead=4,
            num_layers=1
        )

        self.model.total_num_classes = 5

    def test_forward_pass(self):
        """Test if the forward pass works and returns correct shapes."""
        batch_size = 2
        seq_len = 10  # number of epochs
        signal_len = 100  # simplified signal length

        # Create synthetic tensors
        data = {
            "ecg": torch.randn(batch_size, seq_len, signal_len),
            "ppg": torch.randn(batch_size, seq_len, signal_len),
            "label": torch.randint(0, 5, (batch_size, seq_len))
        }

        output = self.model(**data)

        # Check keys
        self.assertIn("loss", output)
        self.assertIn("y_prob", output)

        # Check output shape [B, T, C]
        self.assertEqual(output["y_prob"].shape, (batch_size, seq_len, 5))

        # Check if loss is a scalar
        self.assertEqual(output["loss"].dim(), 0)


if __name__ == "__main__":
    unittest.main()
