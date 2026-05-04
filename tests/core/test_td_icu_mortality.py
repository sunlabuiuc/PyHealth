"""Unit tests for pyhealth.models.td_icu_mortality.

Converted from pytest to unittest
so it runs under the standard ``python -m unittest`` runner used by PyHealth CI.

Performance strategy:
  - setUpClass reuses the same model across tests within a class. Tests that
    mutate weights operate on disposable copies or restore from a snapshot.
  - Tiny config: n_features=4, hidden_dim=2, cnn_layers=1, seq_len=8,
    batch_size=2. A forward pass at this size is a fraction of a ms.
  - Synthetic tensor batches are pre-computed once per class.

Run with:
    python -m unittest tests/core/test_td_icu_mortality.py -v
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Dict, List

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models.td_icu_mortality import (
    CNNLSTMPredictor,
    MaxPool1D,
    TDICUMortalityModel,
    Transpose,
    WeightedBCELoss,
)

# ---------------------------------------------------------------------------
# Tiny config
# ---------------------------------------------------------------------------

N_FEATURES = 4
HIDDEN_DIM = 2
CNN_LAYERS = 1
SEQ_LEN = 8
BATCH_SIZE = 2
LABEL_KEY = "mortality"
FEATURE_KEYS = [
    "timepoints", "values", "features", "delta_time", "delta_value",
]
FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_scaling(feature_names: List[str]) -> Dict:
    scaling: Dict = {"mean": {}, "std": {}}
    scaling["mean"]["timepoints"] = torch.tensor([0.0])
    scaling["std"]["timepoints"] = torch.tensor([1.0])
    for key in ["values", "delta_time", "delta_value"]:
        scaling["mean"][key] = {f: torch.tensor([0.0]) for f in feature_names}
        scaling["std"][key] = {f: torch.tensor([1.0]) for f in feature_names}
    return scaling


def _make_current_window(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    n_features: int = N_FEATURES,
    pad_head: int = 1,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    features = torch.randint(0, n_features, (batch_size, seq_len), generator=g)
    if pad_head > 0:
        features[:, :pad_head] = -1
    timepoints = torch.randn(batch_size, seq_len, generator=g)
    values = torch.randn(batch_size, seq_len, generator=g)
    dt = torch.randn(batch_size, seq_len, generator=g)
    dv = torch.randn(batch_size, seq_len, generator=g)
    if pad_head > 0:
        pad_mask = features == -1
        timepoints[pad_mask] = float("nan")
        values[pad_mask] = float("nan")
        dt[pad_mask] = -1.0
        dv[pad_mask] = float("nan")
    return {
        "timepoints": timepoints,
        "values": values,
        "features": features,
        "delta_time": dt,
        "delta_value": dv,
    }


def _make_td_batch(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    n_features: int = N_FEATURES,
    terminal_mask: torch.Tensor = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    cur = _make_current_window(batch_size, seq_len, n_features, seed=seed)
    nxt = _make_current_window(batch_size, seq_len, n_features, seed=seed + 1)
    batch = dict(cur)
    for key, val in nxt.items():
        batch[f"next_{key}"] = val
    if terminal_mask is None:
        g = torch.Generator().manual_seed(seed + 2)
        terminal_mask = torch.randint(0, 2, (batch_size,), generator=g).float()
    batch["isterminal"] = terminal_mask.view(-1, 1)
    return batch


def _make_targets(batch_size: int = BATCH_SIZE, seed: int = 0) -> Dict:
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (batch_size,), generator=g).float().view(-1, 1)
    return {LABEL_KEY: y}


def _make_empty_dataset() -> SampleEHRDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SampleEHRDataset(samples=[], dataset_name="td_icu_unit_test")


def _make_predictor(scaling=None, feature_names=None) -> CNNLSTMPredictor:
    if scaling is None:
        scaling = _build_scaling(FEATURE_NAMES)
    if feature_names is None:
        feature_names = FEATURE_NAMES
    torch.manual_seed(42)
    return CNNLSTMPredictor(
        n_features=N_FEATURES,
        features=feature_names,
        output_dim=1,
        scaling=scaling,
        cnn_layers=CNN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
        device="cpu",
    )


def _make_td_model(scaling=None, feature_names=None) -> TDICUMortalityModel:
    if scaling is None:
        scaling = _build_scaling(FEATURE_NAMES)
    if feature_names is None:
        feature_names = FEATURE_NAMES
    torch.manual_seed(42)
    return TDICUMortalityModel(
        dataset=_make_empty_dataset(),
        feature_keys=FEATURE_KEYS,
        label_key=LABEL_KEY,
        mode="binary",
        n_features=N_FEATURES,
        hidden_dim=HIDDEN_DIM,
        cnn_layers=CNN_LAYERS,
        dropout=0.1,
        output_dim=1,
        scaling=scaling,
        features_vocab=feature_names,
        td_alpha=0.99,
        pos_weight=None,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# TestHelperModules
# ---------------------------------------------------------------------------

class TestHelperModules(unittest.TestCase):
    """MaxPool1D, Transpose, WeightedBCELoss."""

    def test_maxpool1d_preserves_all_nan_window(self):
        pool = MaxPool1D(2, 2)
        x = torch.tensor([[[1.0, 2.0, float("nan"), float("nan")]]])
        out = pool(x)
        self.assertEqual(out.shape, (1, 1, 2))
        self.assertEqual(out[0, 0, 0].item(), 2.0)
        self.assertTrue(torch.isnan(out[0, 0, 1]))

    def test_maxpool1d_mixed_window_ignores_nan(self):
        pool = MaxPool1D(2, 2)
        x = torch.tensor([[[3.0, float("nan")]]])
        out = pool(x)
        self.assertEqual(out[0, 0, 0].item(), 3.0)
        self.assertFalse(torch.isnan(out[0, 0, 0]))

    def test_transpose_swaps_dims(self):
        t = Transpose(1, 2)
        x = torch.randn(2, 3, 5)
        self.assertTrue(torch.equal(t(x), x.transpose(1, 2)))

    def test_weighted_bce_matches_builtin(self):
        loss_fn = WeightedBCELoss(pos_weight=None)
        logits = torch.zeros(3, 1)
        targets = torch.tensor([[1.0], [0.0], [1.0]])
        expected = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        self.assertTrue(torch.allclose(loss_fn(logits, targets), expected))

    def test_weighted_bce_with_pos_weight(self):
        no_w = WeightedBCELoss(pos_weight=None)
        w = WeightedBCELoss(pos_weight=torch.tensor([2.0]))
        logits = torch.zeros(4, 1)
        targets = torch.ones(4, 1)
        self.assertGreater(w(logits, targets).item(), no_w(logits, targets).item())


# ---------------------------------------------------------------------------
# TestCNNLSTMPredictorInstantiation
# ---------------------------------------------------------------------------

class TestCNNLSTMPredictorInstantiation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = _make_predictor()

    def test_instantiation_succeeds(self):
        self.assertIsInstance(self.predictor, CNNLSTMPredictor)
        self.assertEqual(self.predictor.n_features, N_FEATURES)
        self.assertEqual(self.predictor.hidden_dim, HIDDEN_DIM)
        self.assertEqual(self.predictor.cnn_layers, CNN_LAYERS)

    def test_has_five_embedding_heads(self):
        expected = {"time", "value", "feature", "delta_time", "delta_value"}
        self.assertEqual(set(self.predictor.embedding_net.keys()), expected)

    def test_feature_embedding_dims(self):
        emb = self.predictor.embedding_net["feature"]
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.num_embeddings, N_FEATURES)
        self.assertEqual(emb.embedding_dim, HIDDEN_DIM)

    def test_scalar_embeddings_are_mlps(self):
        for name in ["time", "value", "delta_time", "delta_value"]:
            net = self.predictor.embedding_net[name]
            self.assertIsInstance(net, nn.Sequential)
            self.assertEqual(net[0].in_features, 1)
            self.assertEqual(net[-1].out_features, HIDDEN_DIM)

    def test_scaling_buffers_registered(self):
        buffers = dict(self.predictor.named_buffers())
        for name in [
            "mean_values", "std_values",
            "mean_delta_time", "std_delta_time",
            "mean_delta_value", "std_delta_value",
            "mean_timepoints", "std_timepoints",
        ]:
            self.assertIn(name, buffers)

    def test_lstm_hidden_is_8x_base(self):
        self.assertEqual(self.predictor.lstm.hidden_size, HIDDEN_DIM * 8)
        self.assertEqual(self.predictor.lstm.num_layers, 2)

    def test_parameter_count_reasonable(self):
        n = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        self.assertGreater(n, 0)
        self.assertLess(n, 100_000)


# ---------------------------------------------------------------------------
# TestCNNLSTMPredictorForward
# ---------------------------------------------------------------------------

class TestCNNLSTMPredictorForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = _make_predictor()
        cls.predictor.eval()

    def test_forward_returns_tuple(self):
        out = self.predictor(**_make_current_window())
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_probs_shape(self):
        probs, _ = self.predictor(**_make_current_window())
        self.assertEqual(probs.shape, (BATCH_SIZE, 1))

    def test_logits_shape(self):
        _, logits = self.predictor(**_make_current_window())
        self.assertEqual(logits.shape, (BATCH_SIZE, 1))

    def test_probs_in_unit_interval(self):
        probs, _ = self.predictor(**_make_current_window())
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))

    def test_no_nan_in_output(self):
        probs, logits = self.predictor(**_make_current_window())
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_handles_no_padding(self):
        probs, _ = self.predictor(**_make_current_window(pad_head=0))
        self.assertTrue(torch.isfinite(probs).all())

    def test_handles_heavy_padding(self):
        probs, _ = self.predictor(**_make_current_window(pad_head=SEQ_LEN - 2))
        self.assertTrue(torch.isfinite(probs).all())

    def test_probs_match_sigmoid_logits(self):
        probs, logits = self.predictor(**_make_current_window())
        self.assertTrue(torch.allclose(probs, torch.sigmoid(logits), atol=1e-6))


# ---------------------------------------------------------------------------
# TestCNNLSTMPredictorGradients
# ---------------------------------------------------------------------------

class TestCNNLSTMPredictorGradients(unittest.TestCase):

    def setUp(self):
        self.predictor = _make_predictor()
        self.predictor.train()

    def test_backward_produces_gradients(self):
        _, logits = self.predictor(**_make_current_window())
        logits.sum().backward()
        grads = [p.grad for p in self.predictor.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_all_trainable_params_receive_gradient(self):
        _, logits = self.predictor(**_make_current_window())
        logits.sum().backward()
        missing = [
            name for name, p in self.predictor.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        self.assertFalse(missing)

    def test_gradients_are_finite(self):
        _, logits = self.predictor(**_make_current_window())
        logits.sum().backward()
        for name, p in self.predictor.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)


# ---------------------------------------------------------------------------
# TestSoftUpdate
# ---------------------------------------------------------------------------

class TestSoftUpdate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        scaling = _build_scaling(FEATURE_NAMES)
        torch.manual_seed(1)
        cls.tgt = _make_predictor(scaling=scaling)
        torch.manual_seed(2)
        cls.src = _make_predictor(scaling=scaling)
        # Snapshot BOTH — test_alpha_half_is_midpoint mutates src.dense weights
        cls.tgt_initial = copy.deepcopy(cls.tgt.state_dict())
        cls.src_initial = copy.deepcopy(cls.src.state_dict())

    def setUp(self):
        # Restore both to initial state before every test
        self.tgt.load_state_dict(copy.deepcopy(self.tgt_initial))
        self.src.load_state_dict(copy.deepcopy(self.src_initial))

    def test_alpha_zero_copies_source(self):
        self.tgt.soft_update(self.src, alpha=0.0)
        for k, v in self.src.state_dict().items():
            if v.dtype.is_floating_point:
                self.assertTrue(torch.allclose(self.tgt.state_dict()[k], v))

    def test_alpha_one_leaves_target(self):
        self.tgt.soft_update(self.src, alpha=1.0)
        for k, v in self.tgt.state_dict().items():
            if v.dtype.is_floating_point:
                self.assertTrue(torch.allclose(self.tgt_initial[k], v))

    def test_alpha_half_is_midpoint(self):
        with torch.no_grad():
            self.tgt.dense[1].weight.fill_(0.0)
            self.src.dense[1].weight.fill_(2.0)
        self.tgt.soft_update(self.src, alpha=0.5)
        self.assertTrue(torch.allclose(
            self.tgt.dense[1].weight,
            torch.full_like(self.tgt.dense[1].weight, 1.0),
        ))

    def test_soft_update_has_no_grad(self):
        self.tgt.soft_update(self.src, alpha=0.9)
        for p in self.tgt.parameters():
            self.assertIsNone(p.grad)


# ---------------------------------------------------------------------------
# TestTDICUMortalityModelInstantiation
# ---------------------------------------------------------------------------

class TestTDICUMortalityModelInstantiation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_td_model()
        cls.empty_dataset = _make_empty_dataset()
        cls.scaling = _build_scaling(FEATURE_NAMES)

    def test_instantiation_succeeds(self):
        self.assertIsInstance(self.model, TDICUMortalityModel)
        self.assertEqual(self.model.label_key, LABEL_KEY)
        self.assertEqual(self.model.mode, "binary")
        self.assertAlmostEqual(self.model.td_alpha, 0.99)

    def test_inherits_from_basemodel(self):
        from pyhealth.models import BaseModel
        self.assertIsInstance(self.model, BaseModel)

    def test_has_online_and_target_nets(self):
        self.assertIsInstance(self.model.online_net, CNNLSTMPredictor)
        self.assertIsInstance(self.model.target_net, CNNLSTMPredictor)

    def test_both_loss_functions_exist(self):
        self.assertIsInstance(self.model.supervised_loss, WeightedBCELoss)
        self.assertIsInstance(self.model.td_loss, WeightedBCELoss)

    def test_td_loss_has_no_pos_weight(self):
        self.assertIsNone(self.model.td_loss.loss_fn.pos_weight)

    def test_non_binary_mode_raises(self):
        with self.assertRaises(ValueError):
            TDICUMortalityModel(
                dataset=self.empty_dataset,
                feature_keys=FEATURE_KEYS,
                label_key=LABEL_KEY,
                mode="multiclass",
                n_features=N_FEATURES,
                hidden_dim=HIDDEN_DIM,
                cnn_layers=CNN_LAYERS,
                scaling=self.scaling,
                features_vocab=FEATURE_NAMES,
            )

    def test_missing_scaling_raises(self):
        with self.assertRaises(ValueError):
            TDICUMortalityModel(
                dataset=self.empty_dataset,
                feature_keys=FEATURE_KEYS,
                label_key=LABEL_KEY,
                mode="binary",
                n_features=N_FEATURES,
                hidden_dim=HIDDEN_DIM,
                cnn_layers=CNN_LAYERS,
                scaling=None,
                features_vocab=FEATURE_NAMES,
            )


# ---------------------------------------------------------------------------
# TestBaseModelInterface
# ---------------------------------------------------------------------------

class TestBaseModelInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_td_model()

    def test_prepare_labels_shape(self):
        labels = torch.tensor([0, 1, 1, 0])
        out = self.model.prepare_labels(labels)
        self.assertEqual(out.shape, (4, 1))
        self.assertEqual(out.dtype, torch.float32)

    def test_prepare_labels_idempotent_from_batched(self):
        labels = torch.tensor([[0], [1], [1], [0]])
        out = self.model.prepare_labels(labels)
        self.assertEqual(out.shape, (4, 1))
        self.assertEqual(out.dtype, torch.float32)

    def test_get_loss_function_returns_module(self):
        loss_fn = self.model.get_loss_function()
        self.assertIsInstance(loss_fn, nn.Module)
        logits = torch.zeros(2, 1)
        targets = torch.tensor([[1.0], [0.0]])
        result = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(result))


# ---------------------------------------------------------------------------
# TestTDICUMortalityModelForward
# ---------------------------------------------------------------------------

class TestTDICUMortalityModelForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_td_model()
        cls.model.eval()
        cls.batch = _make_td_batch(seed=0)
        cls.targets = _make_targets(seed=0)

    def setUp(self):
        # Mirrors pytest's autouse _eval_mode fixture — ensures eval() is set
        # before every test, since train_td=True tests may alter model state
        self.model.eval()

    def test_returns_all_expected_keys(self):
        out = self.model(self.batch, targets=None, train_td=False)
        self.assertEqual(set(out.keys()), {"loss", "y_prob", "y_true", "logit"})
        self.assertIsNone(out["loss"])
        self.assertIsNone(out["y_true"])

    def test_with_targets_produces_loss(self):
        out = self.model(self.batch, targets=self.targets, train_td=False)
        self.assertIsNotNone(out["loss"])
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_y_prob_shape(self):
        out = self.model(self.batch, targets=None, train_td=False)
        self.assertEqual(out["y_prob"].shape, (BATCH_SIZE, 1))

    def test_logit_shape(self):
        out = self.model(self.batch, targets=None, train_td=False)
        self.assertEqual(out["logit"].shape, (BATCH_SIZE, 1))

    def test_probs_in_unit_interval(self):
        out = self.model(self.batch, targets=None, train_td=False)
        self.assertTrue(torch.all(out["y_prob"] >= 0.0))
        self.assertTrue(torch.all(out["y_prob"] <= 1.0))

    def test_supervised_and_td_mode_differ(self):
        batch = _make_td_batch(terminal_mask=torch.zeros(BATCH_SIZE))
        sup_loss = self.model(batch, targets=self.targets, train_td=False)["loss"]
        td_loss = self.model(batch, targets=self.targets, train_td=True)["loss"]
        self.assertFalse(torch.allclose(sup_loss, td_loss))

    def test_all_terminal_reduces_to_supervised(self):
        batch = _make_td_batch(terminal_mask=torch.ones(BATCH_SIZE))
        sup = self.model(batch, targets=self.targets, train_td=False)["loss"]
        td = self.model(batch, targets=self.targets, train_td=True)["loss"]
        self.assertTrue(torch.allclose(sup, td, atol=1e-5))


# ---------------------------------------------------------------------------
# TestTDTargetComputation
# ---------------------------------------------------------------------------

class TestTDTargetComputation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_td_model()
        cls.model.eval()
        cls.batch = _make_td_batch(seed=0)
        cls.targets = _make_targets(seed=0)

    def test_td_target_shape(self):
        td = self.model.compute_td_target(self.batch, self.targets)
        self.assertEqual(td.shape, (BATCH_SIZE, 1))

    def test_td_target_in_unit_interval(self):
        td = self.model.compute_td_target(self.batch, self.targets)
        self.assertTrue(torch.all(td >= 0.0) and torch.all(td <= 1.0))

    def test_td_target_detached(self):
        td = self.model.compute_td_target(self.batch, self.targets)
        self.assertFalse(td.requires_grad)


# ---------------------------------------------------------------------------
# TestTDICUMortalityModelGradients
# ---------------------------------------------------------------------------

class TestTDICUMortalityModelGradients(unittest.TestCase):
    # Each test gets a fresh model — no shared state, no snapshot needed
    # since _make_td_model() always uses torch.manual_seed(42)

    def setUp(self):
        self.model = _make_td_model()
        self.model.train()
        self.batch = _make_td_batch(seed=0)
        self.targets = _make_targets(seed=0)

    def test_supervised_backward(self):
        out = self.model(self.batch, targets=self.targets, train_td=False)
        out["loss"].backward()
        grads = [p.grad for p in self.model.online_net.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_td_backward(self):
        out = self.model(self.batch, targets=self.targets, train_td=True)
        out["loss"].backward()
        grads = [p.grad for p in self.model.online_net.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_target_net_no_gradient_in_td_mode(self):
        out = self.model(self.batch, targets=self.targets, train_td=True)
        out["loss"].backward()
        for name, p in self.model.target_net.named_parameters():
            self.assertIsNone(p.grad, f"target_net.{name} got a gradient")


# ---------------------------------------------------------------------------
# TestSoftUpdateTarget
# ---------------------------------------------------------------------------

class TestSoftUpdateTarget(unittest.TestCase):

    def test_target_changes_after_training_step(self):
        model = _make_td_model()
        model.train()
        batch = _make_td_batch(seed=0)
        targets = _make_targets(seed=0)

        target_before = {
            k: v.detach().clone()
            for k, v in model.target_net.state_dict().items()
        }
        optim = torch.optim.AdamW(model.online_net.parameters(), lr=1e-1)
        out = model(batch, targets=targets, train_td=True)
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        optim.step()
        model.soft_update_target()

        target_after = model.target_net.state_dict()
        changed = any(
            not torch.allclose(target_before[k], target_after[k], atol=1e-6)
            for k in target_before
            if target_before[k].dtype.is_floating_point
        )
        self.assertTrue(changed)


# ---------------------------------------------------------------------------
# TestMCDropoutConfidence
# ---------------------------------------------------------------------------

class TestMCDropoutConfidence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_td_model()
        cls.model.eval()
        cls.batch = _make_td_batch(seed=0)

    def setUp(self):
        # Mirrors pytest's autouse _eval_mode fixture
        self.model.eval()

    def test_returns_all_expected_keys(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        expected = {
            "mortality_prob", "confidence_std",
            "ci_95_lower", "ci_95_upper",
            "is_high_confidence", "is_low_confidence",
        }
        self.assertEqual(set(out.keys()), expected)

    def test_output_shapes(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        for key in ["mortality_prob", "confidence_std", "ci_95_lower", "ci_95_upper"]:
            self.assertEqual(out[key].shape, (BATCH_SIZE,), key)

    def test_mortality_prob_in_unit_interval(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        self.assertTrue(torch.all(out["mortality_prob"] >= 0.0))
        self.assertTrue(torch.all(out["mortality_prob"] <= 1.0))

    def test_std_non_negative(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        self.assertTrue(torch.all(out["confidence_std"] >= 0.0))

    def test_ci_in_unit_interval_and_contains_mean(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        self.assertTrue(torch.all(out["ci_95_lower"] >= 0.0))
        self.assertTrue(torch.all(out["ci_95_upper"] <= 1.0))
        self.assertTrue(torch.all(out["ci_95_lower"] <= out["mortality_prob"]))
        self.assertTrue(torch.all(out["mortality_prob"] <= out["ci_95_upper"]))

    def test_confidence_flags_are_bool(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        self.assertEqual(out["is_high_confidence"].dtype, torch.bool)
        self.assertEqual(out["is_low_confidence"].dtype, torch.bool)

    def test_confidence_flags_mutually_exclusive(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=3)
        both = out["is_high_confidence"] & out["is_low_confidence"]
        self.assertFalse(both.any())

    def test_does_not_alter_train_eval_of_non_dropout_modules(self):
        self.model.eval()
        _ = self.model.predict_with_confidence(self.batch, n_mc_samples=2)
        for m in self.model.online_net.modules():
            if isinstance(m, nn.BatchNorm1d):
                self.assertFalse(
                    m.training,
                    "BatchNorm should stay in eval during MC dropout",
                )

    def test_no_gradients_from_mc_sampling(self):
        out = self.model.predict_with_confidence(self.batch, n_mc_samples=2)
        self.assertFalse(out["mortality_prob"].requires_grad)
        self.assertFalse(out["confidence_std"].requires_grad)


# ---------------------------------------------------------------------------
# TestCheckpointIO
# ---------------------------------------------------------------------------

class TestCheckpointIO(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="td_icu_test_"))
        self.model = _make_td_model()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_state_dict_roundtrip(self):
        ckpt_path = self.tmp_dir / "model.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        self.assertTrue(ckpt_path.exists())

        original = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            for p in self.model.online_net.parameters():
                p.add_(1.0)

        restored = torch.load(ckpt_path, weights_only=True)
        self.model.load_state_dict(restored)
        loaded = self.model.state_dict()
        for k, v in original.items():
            self.assertTrue(torch.allclose(v, loaded[k]))

    def test_tmp_dir_is_writable(self):
        self.assertTrue(self.tmp_dir.exists())
        self.assertTrue(self.tmp_dir.is_dir())
        (self.tmp_dir / "sanity.txt").write_text("ok")
        self.assertEqual((self.tmp_dir / "sanity.txt").read_text(), "ok")

    def test_tmp_dir_is_unique(self):
        self.assertIn("td_icu_test_", str(self.tmp_dir))


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = _make_predictor()
        cls.predictor.eval()

    def test_same_input_same_output(self):
        cur = _make_current_window(seed=42)
        with torch.no_grad():
            p1, _ = self.predictor(**cur)
            p2, _ = self.predictor(**cur)
        self.assertTrue(torch.allclose(p1, p2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
