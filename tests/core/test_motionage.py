"""Tests for the MotionAge model, using small synthetic activity windows."""

import os
import tempfile
import unittest
from typing import Any

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MotionAge, MotionAgeLayer

SEQ_LEN = 12


def _make_samples(num: int = 4) -> list[dict[str, Any]]:
    """Builds synthetic windows: 12 epochs, last 3 not worn."""
    samples = []
    for i in range(num):
        samples.append(
            {
                "patient_id": f"p{i // 2}",
                "intensity": [0.1 * (i + 1) + 0.01 * t for t in range(SEQ_LEN)],
                "hour": [(6 + t) % 24 for t in range(SEQ_LEN)],
                "day": [i % 7] * SEQ_LEN,
                "wear_mask": [1.0] * (SEQ_LEN - 3) + [0.0] * 3,
                "static": [50.0 + i, float(i % 2), float("nan") if i == 0 else 25.0],
                "mortality": i % 2,
            }
        )
    return samples


SCHEMA = {
    "intensity": "tensor",
    "hour": "tensor",
    "day": "tensor",
    "wear_mask": "tensor",
    "static": "tensor",
}


class TestMotionAge(unittest.TestCase):
    """Unit tests for MotionAge and MotionAgeLayer."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = create_sample_dataset(
            samples=_make_samples(),
            input_schema=SCHEMA,
            output_schema={"mortality": "binary"},
            dataset_name="test_motionage",
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def setUp(self):
        torch.manual_seed(0)

    def _model(self, **kwargs: Any) -> MotionAge:
        params = {"hidden_dim": 8, "num_layers": 2}
        params.update(kwargs)
        return MotionAge(self.dataset, **params)

    def test_initialization(self):
        model = self._model(static_key="static")
        self.assertIsInstance(model, MotionAge)
        self.assertEqual(model.label_key, "mortality")
        self.assertEqual(model.static_dim, 3)
        self.assertEqual(model.encoder.hidden_dim, 8)
        self.assertIsNone(self._model().covariate_encoder)

    def test_forward_activity_only(self):
        ret = self._model()(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, ret)
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["y_prob"].shape, (4, 1))
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(torch.all((ret["y_prob"] >= 0) & (ret["y_prob"] <= 1)))

    def test_forward_with_static_handles_nan(self):
        ret = self._model(static_key="static")(**self.batch, embed=True)
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["embed"].shape, (4, 8 + 32))
        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_forward_without_label(self):
        batch = {k: v for k, v in self.batch.items() if k != "mortality"}
        ret = self._model()(**batch)
        self.assertNotIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape, (4, 1))

    def test_backward_computes_gradients(self):
        model = self._model(static_key="static")
        model(**self.batch)["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and name != "_dummy_param":
                self.assertIsNotNone(param.grad, name)
        self.assertGreater(model.encoder.rnn.weight_ih_l0.grad.abs().sum(), 0)
        self.assertGreater(model.encoder.hour_embedding.weight.grad.abs().sum(), 0)

    def test_masked_pooling_ignores_non_wear_epochs(self):
        layer = MotionAgeLayer(hidden_dim=4, num_layers=1, pooling="masked")
        layer.eval()
        intensity = torch.rand(1, 6)
        hour = torch.arange(6).unsqueeze(0)
        day = torch.zeros(1, 6, dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        changed = intensity.clone()
        changed[0, 3:] = 100.0
        with torch.no_grad():
            a = layer(intensity, hour, day, mask)
            b = layer(changed, hour, day, mask)
        # A unidirectional RNN never sees later epochs earlier, so values in
        # non-wear epochs after the last worn epoch cannot change the result.
        self.assertTrue(torch.allclose(a, b))

    def test_pooling_modes_differ(self):
        intensity = torch.rand(2, 6)
        hour = torch.randint(0, 24, (2, 6))
        day = torch.randint(0, 7, (2, 6))
        mask = torch.tensor([[1, 1, 0, 0, 1, 1]] * 2).float()
        outputs = {}
        for pooling in ("masked", "mean", "last"):
            torch.manual_seed(0)
            layer = MotionAgeLayer(hidden_dim=4, num_layers=1, pooling=pooling)
            with torch.no_grad():
                outputs[pooling] = layer(intensity, hour, day, mask)
            self.assertEqual(outputs[pooling].shape, (2, 4))
        self.assertFalse(torch.allclose(outputs["masked"], outputs["mean"]))
        self.assertFalse(torch.allclose(outputs["masked"], outputs["last"]))

    def test_save_and_load_state(self):
        model = self._model(static_key="static")
        model.eval()
        with torch.no_grad():
            before = model(**self.batch)["y_prob"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "motionage.pt")
            torch.save(model.state_dict(), path)
            torch.manual_seed(123)
            loaded = self._model(static_key="static")
            loaded.load_state_dict(torch.load(path))
        loaded.eval()
        with torch.no_grad():
            after = loaded(**self.batch)["y_prob"]
        self.assertTrue(torch.allclose(before, after))

    def test_lstm_and_no_mask(self):
        model = self._model(rnn_type="lstm", mask_key=None)
        self.assertEqual(model(**self.batch)["logit"].shape, (4, 1))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self._model(static_key="missing")
        with self.assertRaises(ValueError):
            self._model(pooling="max")
        with self.assertRaises(ValueError):
            self._model(rnn_type="transformer")
        layer = MotionAgeLayer(hidden_dim=4, num_layers=1)
        with self.assertRaises(ValueError):
            layer(torch.rand(1, 3), torch.tensor([[0, 1, 24]]), torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            layer(torch.rand(1, 3), torch.zeros(1, 2), torch.zeros(1, 3))

    def test_aggregate_participant_risk(self):
        risk = MotionAge.aggregate_participant_risk(
            torch.tensor([[0.2], [0.4], [0.9]]), ["a", "a", "b"]
        )
        self.assertAlmostEqual(risk["a"], 0.3, places=6)
        self.assertAlmostEqual(risk["b"], 0.9, places=6)
        with self.assertRaises(ValueError):
            MotionAge.aggregate_participant_risk(torch.tensor([0.1]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
