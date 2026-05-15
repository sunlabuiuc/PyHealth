"""Tests for EEGGraphConvNet.

Uses minimal synthetic tensor samples — no real EEG data required.
All tests complete in milliseconds.
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import EEGGraphConvNet


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

def _make_sample(patient_id: str, label: int) -> dict:
    """Return a single synthetic EEG graph sample."""
    rng = np.random.default_rng(abs(hash(patient_id)) % (2**32))
    return {
        "patient_id":    patient_id,
        "node_features": rng.random((8, 6)).astype(np.float32),
        "adj_matrix":    rng.random((8, 8)).astype(np.float32),
        "label":         label,
    }


_SAMPLES = [
    _make_sample("p001", 0),
    _make_sample("p002", 1),
    _make_sample("p003", 0),
    _make_sample("p004", 1),
]

_INPUT_SCHEMA  = {"node_features": "tensor", "adj_matrix": "tensor"}
_OUTPUT_SCHEMA = {"label": "binary"}


def _make_dataset():
    return create_sample_dataset(
        samples=_SAMPLES,
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        dataset_name="test_eeg_gcnn",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEEGGraphConvNet(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = _make_dataset()
        self.model   = EEGGraphConvNet(dataset=self.dataset)
        self.model.eval()

    def _get_batch(self) -> dict:
        from pyhealth.datasets import get_dataloader
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        return next(iter(loader))

    # --- initialisation ---------------------------------------------------

    def test_model_instantiates(self) -> None:
        self.assertIsInstance(self.model, EEGGraphConvNet)

    def test_label_keys(self) -> None:
        self.assertEqual(self.model.label_keys, ["label"])

    def test_feature_keys_contain_inputs(self) -> None:
        for key in ("node_features", "adj_matrix"):
            self.assertIn(key, self.model.feature_keys)

    # --- forward pass -----------------------------------------------------

    def test_forward_output_keys(self) -> None:
        batch = self._get_batch()
        with torch.no_grad():
            out = self.model(**batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

    def test_forward_output_shapes(self) -> None:
        batch = self._get_batch()
        with torch.no_grad():
            out = self.model(**batch)
        batch_size = len(_SAMPLES)
        self.assertEqual(out["y_prob"].shape[0], batch_size)
        self.assertEqual(out["y_true"].shape[0], batch_size)
        self.assertEqual(out["logit"].shape[0],  batch_size)

    def test_forward_loss_is_scalar(self) -> None:
        batch = self._get_batch()
        with torch.no_grad():
            out = self.model(**batch)
        self.assertEqual(out["loss"].dim(), 0)

    def test_forward_y_prob_in_01(self) -> None:
        batch = self._get_batch()
        with torch.no_grad():
            out = self.model(**batch)
        probs = out["y_prob"].cpu().numpy()
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    # --- backward pass ----------------------------------------------------

    def test_backward_produces_gradients(self) -> None:
        self.model.train()
        batch = self._get_batch()
        out = self.model(**batch)
        out["loss"].backward()
        has_grad = any(
            p.grad is not None
            for p in self.model.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
