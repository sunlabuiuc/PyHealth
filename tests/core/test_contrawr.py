import unittest
import numpy as np
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import ContraWR


class TestContraWR(unittest.TestCase):
    """Test cases for the ContraWR model."""

    def setUp(self):
        """Set up small synthetic signal samples and dataset."""
        rnd = np.random.RandomState(0)
        n_channels = 2
        length = 200  # must be >= n_fft to compute STFT
        # two samples, each with (n_channels, length) numpy arrays
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "epoch_signal": rnd.randn(n_channels, length).astype(np.float32),
                "label": 0,
            },
            {
                "patient_id": "p1",
                "visit_id": "v0",
                "epoch_signal": rnd.randn(n_channels, length).astype(np.float32),
                "label": 1,
            },
        ]

        self.input_schema = {"epoch_signal": "tensor"}
        self.output_schema = {"label": "multiclass"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_contrawr",
        )

        # use smaller n_fft so that short signals produce reasonable spectrogram dims
        self.model = ContraWR(dataset=self.dataset, n_fft=32)

    def test_model_initialization(self):
        """Model initializes and has expected attributes."""
        self.assertIsInstance(self.model, ContraWR)
        self.assertEqual(self.model.n_fft, 32)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertEqual(len(self.model.label_keys), 1)

    def test_model_forward(self):
        """Forward pass returns required keys with proper shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        # batch size matches
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Backward computes gradients for some parameters."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient)

    def test_model_with_embedding(self):
        """Request embeddings and verify shape."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        # embedding returned by model is 2D (batch, emb_size)
        self.assertEqual(ret["embed"].dim(), 2)

    def test_custom_hyperparameters(self):
        """Initialize with custom hyperparameters and run forward."""
        model = ContraWR(
            dataset=self.dataset, embedding_dim=64, hidden_dim=64, n_fft=16
        )
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 64)
        self.assertEqual(model.n_fft, 16)

        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_torch_stft(self):
        """Test the STFT transformation."""
        # shape (1, n_channels, length)
        x = torch.tensor(self.samples[0]["epoch_signal"][np.newaxis, ...])  #

        spectrogram = self.model.torch_stft(x)

        # Expected shape: (batch, channels, freq, time_steps)
        freq = self.model.n_fft // 2 + 1  # 17
        hop_length = self.model.n_fft // 4  # 8
        length = self.samples[0]["epoch_signal"].shape[1]  # 200
        time_steps = (length - self.model.n_fft) // hop_length + 1  # 22
        self.assertEqual(spectrogram.shape, (1, 2, freq, time_steps))


if __name__ == "__main__":
    unittest.main()
