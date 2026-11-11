"""Tests for SensorLSTM using tiny synthetic sensor windows."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.sensor_lstm import SensorLSTM

NUM_SAMPLES = 4
STEPS = 10
CHANNELS = 9
NUM_CLASSES = 3


class TestSensorLSTM(unittest.TestCase):
    """Tests instantiation, forward pass, shapes and gradients."""

    def setUp(self):
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        labels = [0, 1, 2, 0]
        samples = [
            {
                "patient_id": f"p{i}",
                "signal": rng.normal(size=(STEPS, CHANNELS)),
                "source_torso": rng.normal(size=(STEPS, CHANNELS)),
                "label": labels[i],
            }
            for i in range(NUM_SAMPLES)
        ]
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"signal": "tensor", "source_torso": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="sensor-test",
        )
        loader = get_dataloader(self.dataset, batch_size=NUM_SAMPLES)
        self.batch = next(iter(loader))

    def test_initialization(self):
        model = SensorLSTM(self.dataset, hidden_dim=8)
        self.assertEqual(model.input_key, "signal")
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.hidden_dim, 8)

    def test_forward_shapes_and_keys(self):
        model = SensorLSTM(self.dataset, hidden_dim=8)
        with torch.no_grad():
            out = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)
        self.assertEqual(tuple(out["logit"].shape), (NUM_SAMPLES, NUM_CLASSES))
        self.assertEqual(tuple(out["y_prob"].shape), (NUM_SAMPLES, NUM_CLASSES))
        self.assertEqual(out["loss"].dim(), 0)

    def test_backward_produces_gradients(self):
        model = SensorLSTM(self.dataset, hidden_dim=8)
        model(**self.batch)["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_input_key_selects_the_stream(self):
        torch.manual_seed(1)
        model = SensorLSTM(self.dataset, input_key="signal", hidden_dim=8)
        model.eval()
        with torch.no_grad():
            on_signal = model(**self.batch)["logit"]
            swapped = dict(self.batch)
            swapped["signal"] = self.batch["source_torso"]
            on_source = model(**swapped)["logit"]
        self.assertFalse(torch.allclose(on_signal, on_source))

        source_model = SensorLSTM(
            self.dataset, input_key="source_torso", hidden_dim=8
        )
        self.assertEqual(source_model.input_key, "source_torso")

    def test_bidirectional_and_stacked(self):
        model = SensorLSTM(
            self.dataset,
            hidden_dim=8,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
        )
        with torch.no_grad():
            out = model(**self.batch)
        self.assertEqual(tuple(out["logit"].shape), (NUM_SAMPLES, NUM_CLASSES))

    def test_embed_flag(self):
        model = SensorLSTM(self.dataset, hidden_dim=8)
        batch = dict(self.batch)
        batch["embed"] = True
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(tuple(out["embed"].shape), (NUM_SAMPLES, 8))

    def test_prediction_without_label(self):
        model = SensorLSTM(self.dataset, hidden_dim=8)
        batch = {k: v for k, v in self.batch.items() if k != "label"}
        with torch.no_grad():
            out = model(**batch)
        self.assertNotIn("loss", out)
        self.assertIn("y_prob", out)

    def test_invalid_input_key_raises(self):
        with self.assertRaises(ValueError):
            SensorLSTM(self.dataset, input_key="not_a_feature")


if __name__ == "__main__":
    unittest.main()
