import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ShiftLSTM
from pyhealth.models.shift_lstm import ShiftLSTMLayer


class TestShiftLSTM(unittest.TestCase):
    """Test cases for the ShiftLSTM model and synthetic generator."""

    def setUp(self):
        base = datetime(2020, 1, 1)
        self.samples = []
        for i in range(4):
            timestamps = [base + timedelta(hours=t) for t in range(6)]
            values = np.array(
                [
                    [float(i + t), float((i + 1) * (t + 1) % 5), float(t % 2)]
                    for t in range(6)
                ],
                dtype=float,
            )
            self.samples.append(
                {
                    "patient_id": f"patient-{i}",
                    "visit_id": f"visit-{i}",
                    "signal": (timestamps, values),
                    "label": i % 2,
                }
            )

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"signal": "timeseries"},
            output_schema={"label": "binary"},
            dataset_name="synthetic_shift_test",
        )
        self.model = ShiftLSTM(
            dataset=self.dataset,
            embedding_dim=16,
            hidden_dim=8,
            num_segments=2,
            dropout=0.0,
        )

    def _load_synthetic_module(self):
        module_path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "synthetic"
            / "shift_lstm_synthetic_data.py"
        )
        spec = importlib.util.spec_from_file_location(
            "shift_lstm_synthetic_data", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_model_initialization(self):
        """Test that the ShiftLSTM model initializes correctly."""
        self.assertIsInstance(self.model, ShiftLSTM)
        self.assertEqual(self.model.embedding_dim, 16)
        self.assertEqual(self.model.hidden_dim, 8)
        self.assertEqual(self.model.num_segments, 2)
        self.assertEqual(self.model.label_key, "label")
        self.assertIn("signal", self.model.shift_lstm)
        self.assertEqual(len(self.model.shift_lstm["signal"].cells), 2)

    def test_model_forward(self):
        """Test that the ShiftLSTM forward pass works correctly."""
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
        """Test that the ShiftLSTM backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(has_gradient, "No gradients found after backward pass.")

    def test_shift_lstm_layer_shapes(self):
        """Test the low-level ShiftLSTMLayer output shapes and segment count."""
        layer = ShiftLSTMLayer(
            input_size=4, hidden_size=3, num_segments=3, dropout=0.0
        )
        x = torch.randn(2, 5, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.int64)

        outputs, last_outputs = layer(x, mask)

        self.assertEqual(outputs.shape, (2, 5, 3))
        self.assertEqual(last_outputs.shape, (2, 3))
        self.assertEqual(len(layer.cells), 3)

    def test_synthetic_generator_outputs(self):
        """Test synthetic data generator shapes and distribution normalization."""
        synth = self._load_synthetic_module()
        config = synth.SyntheticConfig(N=8, T=12, d=3, l=4, delta=0.2, seed=7)
        bundle = synth.generate_synthetic_arrays(config)

        self.assertEqual(bundle["x"].shape, (8, 12, 3))
        self.assertEqual(bundle["y"].shape, (8, 12, 1))
        self.assertEqual(bundle["k_dist"].shape, (12, 4))
        self.assertEqual(bundle["d_dist"].shape, (12, 3))

        # For valid prediction timesteps, temporal and feature weights should sum to 1.
        self.assertTrue(
            np.allclose(bundle["k_dist"][config.l :].sum(axis=1), 1.0, atol=1e-5)
        )
        self.assertTrue(
            np.allclose(bundle["d_dist"][config.l :].sum(axis=1), 1.0, atol=1e-5)
        )

        samples = synth.to_pyhealth_samples(bundle["x"], bundle["y"])
        self.assertEqual(len(samples), 8)
        self.assertIn("signal", samples[0])
        self.assertIn("label", samples[0])

    def test_synthetic_bundle_save_with_tempdir(self):
        """Test saving synthetic bundles via TemporaryDirectory with cleanup."""
        synth = self._load_synthetic_module()
        config = synth.SyntheticConfig(N=6, T=10, d=3, l=4, delta=0.1, seed=11)
        bundle = synth.generate_synthetic_arrays(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic_bundle.npz"
            saved_path = synth.save_synthetic_bundle(bundle, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(saved_path.exists())

            loaded = np.load(saved_path, allow_pickle=False)
            self.assertEqual(tuple(loaded["x"].shape), (6, 10, 3))
            self.assertEqual(tuple(loaded["y"].shape), (6, 10, 1))
            self.assertEqual(tuple(loaded["k_dist"].shape), (10, 4))
            self.assertEqual(tuple(loaded["d_dist"].shape), (10, 3))


if __name__ == "__main__":
    unittest.main()
