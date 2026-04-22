"""
Unit tests for the BiLSTMECG model.

Covers:
  - model initialisation and attribute checks
  - forward pass output keys and shapes
  - backward pass (gradient flow)
  - embed flag (not applicable — model returns the standard four-key dict)
  - paper-aligned variants (lstm_d1_h64 and lstm_d3_h128)
  - custom hyperparameters
  - all three output modes (multilabel, multiclass, binary)
  - variable-length input signals

Authors:
    Anurag Dixit - anuragd2@illinois.edu
    Kent Spillner - kspillne@illinois.edu
    John Wells - jtwells2@illinois.edu
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.bilstm_ecg import BiLSTMECG

# ---------------------------------------------------------------------------
# Shared fixture helpers  (mirrors test_resnet_ecg.py conventions)
# ---------------------------------------------------------------------------

_N_LEADS  = 12
_LENGTH   = 1000   # 10 s @ 100 Hz — typical PTB-XL low-rate recording
_N_LABELS = 5      # number of multilabel classes


def _make_samples(n: int, rng: np.random.RandomState,
                  label_mode: str = "multilabel") -> list:
    """Return ``n`` synthetic ECG samples for the given label mode.

    For multilabel, PyHealth's MultiLabelProcessor expects a list of active
    class indices, not a fixed-length binary vector.  Every class in
    ``range(_N_LABELS)`` is forced to appear at least once across the dataset
    so the full vocabulary is established.
    """
    samples = []
    for i in range(n):
        if label_mode == "multilabel":
            active = [j for j in range(_N_LABELS) if rng.randint(0, 2)]
            forced = i % _N_LABELS
            if forced not in active:
                active.append(forced)
            label = sorted(active)
        elif label_mode == "multiclass":
            label = int(rng.randint(0, 3))
        else:  # binary
            label = int(rng.randint(0, 2))
        samples.append({
            "patient_id": f"p{i}",
            "visit_id":   "v0",
            "signal":     rng.randn(_N_LEADS, _LENGTH).astype(np.float32),
            "label":      label,
        })
    return samples


def _make_dataset(samples: list, label_mode: str):
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": label_mode},
        dataset_name=f"test_bilstm_{label_mode}",
    )


def _make_model(dataset, **kwargs) -> BiLSTMECG:
    """Construct a BiLSTMECG with the mandatory constructor arguments."""
    return BiLSTMECG(
        dataset=dataset,
        feature_keys=["signal"],
        label_key="label",
        mode=dataset.output_schema["label"],
        **kwargs,
    )


def _assert_forward_output(tc: unittest.TestCase, ret: dict,
                           batch_size: int, n_classes: int) -> None:
    """Assert the standard PyHealth forward-output contract."""
    tc.assertIn("loss",   ret)
    tc.assertIn("y_prob", ret)
    tc.assertIn("y_true", ret)
    tc.assertIn("logit",  ret)
    tc.assertEqual(ret["loss"].dim(), 0)
    tc.assertEqual(ret["y_prob"].shape[0], batch_size)
    tc.assertEqual(ret["y_prob"].shape[1], n_classes)
    tc.assertEqual(ret["y_true"].shape[0], batch_size)
    tc.assertEqual(ret["logit"].shape[0],  batch_size)
    tc.assertEqual(ret["logit"].shape[1],  n_classes)
    tc.assertTrue(torch.isfinite(ret["loss"]))


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class TestBiLSTMECG(unittest.TestCase):
    """Tests for BiLSTMECG."""

    def setUp(self):
        rng = np.random.RandomState(0)
        samples = _make_samples(5, rng, "multilabel")
        self.dataset = _make_dataset(samples, "multilabel")
        self.model   = _make_model(self.dataset)
        self.batch   = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))

    # -- initialisation -------------------------------------------------------

    def test_initialization(self):
        self.assertIsInstance(self.model, BiLSTMECG)
        self.assertEqual(self.model.feature_key, "signal")
        self.assertEqual(self.model.label_key, "label")
        # Default paper variant: 1 layer, hidden_size=64
        self.assertIsInstance(self.model.lstm, torch.nn.LSTM)
        self.assertTrue(self.model.lstm.bidirectional)
        self.assertEqual(self.model.lstm.hidden_size, 64)
        self.assertEqual(self.model.lstm.num_layers, 1)
        self.assertEqual(self.model.lstm.input_size, _N_LEADS)
        # FC head maps hidden*2 → n_classes
        self.assertIsInstance(self.model.fc, torch.nn.Linear)
        self.assertEqual(self.model.fc.in_features, 64 * 2)
        self.assertEqual(self.model.fc.out_features, _N_LABELS)

    def test_lstm_is_bidirectional(self):
        """Bidirectional flag is set and output dim is 2 × hidden_size."""
        self.assertTrue(self.model.lstm.bidirectional)
        x = torch.randn(2, _N_LEADS, _LENGTH)
        # permute to (B, T, C) as forward does
        out, _ = self.model.lstm(x.permute(0, 2, 1))
        self.assertEqual(out.shape, (2, _LENGTH, 64 * 2))

    def test_pooling_over_all_timesteps(self):
        """AdaptiveAvgPool1d(1) reduces the time dimension to a single vector."""
        x = torch.randn(2, _N_LEADS, _LENGTH)
        out, _ = self.model.lstm(x.permute(0, 2, 1))   # (B, T, hidden*2)
        pooled = self.model.pool(out.permute(0, 2, 1)).squeeze(-1)
        self.assertEqual(pooled.shape, (2, 64 * 2))

    # -- forward --------------------------------------------------------------

    def test_forward_multilabel(self):
        with torch.no_grad():
            ret = self.model(**self.batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)
        # multilabel y_prob must be in [0, 1] (sigmoid output)
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))

    def test_forward_multiclass(self):
        rng = np.random.RandomState(1)
        n_classes = 4
        samples = _make_samples(4, rng, "multiclass")
        for i, s in enumerate(samples):
            s["label"] = i % n_classes
        ds = _make_dataset(samples, "multiclass")
        model = _make_model(ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=n_classes)
        # multiclass y_prob rows sum to ~1 (softmax output)
        self.assertTrue(torch.allclose(ret["y_prob"].sum(dim=1),
                                       torch.ones(4), atol=1e-5))

    def test_forward_binary(self):
        rng = np.random.RandomState(2)
        samples = _make_samples(4, rng, "binary")
        for i, s in enumerate(samples):
            s["label"] = i % 2
        ds = _make_dataset(samples, "binary")
        model = _make_model(ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=1)
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))

    # -- backward -------------------------------------------------------------

    def test_backward(self):
        ret = self.model(**self.batch)
        ret["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameters received gradients")

    def test_lstm_weights_receive_gradients(self):
        """LSTM weight matrices (not just the FC head) receive gradients."""
        ret = self.model(**self.batch)
        ret["loss"].backward()
        lstm_params_with_grad = [
            name for name, p in self.model.lstm.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        self.assertGreater(len(lstm_params_with_grad), 0,
                           "No LSTM parameters received gradients")

    # -- paper-aligned variants -----------------------------------------------

    def test_paper_variant_lstm_d1_h64(self):
        """lstm_d1_h64: 1 layer, hidden_size=64 (paper best variant)."""
        model = _make_model(self.dataset, hidden_size=64, n_layers=1)
        self.assertEqual(model.lstm.hidden_size, 64)
        self.assertEqual(model.lstm.num_layers, 1)
        self.assertEqual(model.fc.in_features, 128)
        batch = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)

    def test_paper_variant_lstm_d3_h128(self):
        """lstm_d3_h128: 3 layers, hidden_size=128."""
        model = _make_model(self.dataset, hidden_size=128, n_layers=3)
        self.assertEqual(model.lstm.hidden_size, 128)
        self.assertEqual(model.lstm.num_layers, 3)
        self.assertEqual(model.fc.in_features, 256)
        batch = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)

    # -- dropout behaviour ----------------------------------------------------

    def test_dropout_disabled_for_single_layer(self):
        """PyTorch raises a UserWarning if dropout > 0 with num_layers=1;
        the implementation guards against this by passing 0.0 in that case."""
        model = _make_model(self.dataset, n_layers=1, dropout=0.5)
        # PyTorch stores the effective dropout on the module
        self.assertEqual(model.lstm.dropout, 0.0)

    def test_dropout_enabled_for_multi_layer(self):
        """Dropout is applied between layers when n_layers > 1."""
        model = _make_model(self.dataset, n_layers=2, dropout=0.3)
        self.assertAlmostEqual(model.lstm.dropout, 0.3)

    # -- custom hyperparameters -----------------------------------------------

    def test_custom_hyperparameters(self):
        model = _make_model(self.dataset, hidden_size=32, n_layers=2, dropout=0.1)
        self.assertEqual(model.lstm.hidden_size, 32)
        self.assertEqual(model.lstm.num_layers, 2)
        self.assertEqual(model.fc.in_features, 64)   # 32 * 2 (bidirectional)
        batch = next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[1], _N_LABELS)

    # -- variable-length input ------------------------------------------------

    def test_variable_signal_length(self):
        """Model handles different signal lengths without retraining
        because AdaptiveAvgPool1d(1) is length-agnostic."""
        for length in [500, 1000, 2500]:
            signal = torch.randn(2, _N_LEADS, length)
            batch = {
                "signal": signal,
                "label":  self.batch["label"][:2],
            }
            with torch.no_grad():
                ret = self.model(**batch)
            self.assertEqual(ret["logit"].shape, (2, _N_LABELS),
                             f"Wrong shape for signal length {length}")

    def test_high_rate_signal_length(self):
        """5000-sample input (10 s @ 500 Hz, the paper's high-rate setting)."""
        signal = torch.randn(4, _N_LEADS, 5000)
        batch = {"signal": signal, "label": self.batch["label"]}
        with torch.no_grad():
            ret = self.model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)


if __name__ == "__main__":
    unittest.main()
