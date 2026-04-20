"""Unit tests for pyhealth.models.td_icu_mortality.

Self-contained: no ``conftest.py`` required. An ``atexit`` handler at
module level prints a total-runtime summary when pytest exits.

Performance strategy (to meet the <5 s suite-runtime rubric):
  - Module-scoped fixtures reuse the same model across tests. Tests that
    mutate weights (optimizer steps, soft_update) operate on disposable
    copies or restore from a snapshot afterwards.
  - Tiny config: n_features=4, hidden_dim=2, cnn_layers=1, seq_len=8,
    batch_size=2. A forward pass at this size is a fraction of a ms.
  - Synthetic tensor batches are pre-computed once per module.
  - Instantiation-heavy tests use the shared fixture instead of creating
    fresh models.

Run with:
    pytest tests/test_td_icu_mortality.py -v
"""

from __future__ import annotations

import atexit
import copy
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import warnings

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 


from pyhealth.datasets import SampleEHRDataset
from pyhealth.models.td_icu_mortality import (
    CNNLSTMPredictor,
    MaxPool1D,
    TDICUMortalityModel,
    Transpose,
    WeightedBCELoss,
)


# ---------------------------------------------------------------------------
# Self-contained timing summary
#
# Pytest's session-level hooks must live in conftest.py to be discovered.
# To keep this test file standalone, we instead:
#   1. Collect per-test durations via a function-scoped autouse fixture
#      (fixtures ARE discovered from test modules).
#   2. Print the summary in an ``atexit`` callback, which fires reliably
#      at interpreter shutdown regardless of pytest's hook machinery.
# ---------------------------------------------------------------------------

_TEST_DURATIONS: List[Tuple[str, float]] = []


@pytest.fixture(autouse=True)
def _record_test_duration(request):
    """Time each test and push (nodeid, seconds) into ``_TEST_DURATIONS``."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _TEST_DURATIONS.append((request.node.nodeid, elapsed))


def _print_timing_summary() -> None:
    """Print a compact timing report. Runs at interpreter shutdown."""
    if not _TEST_DURATIONS:
        return
    durations = sorted(_TEST_DURATIONS, key=lambda x: -x[1])
    total = sum(d for _, d in durations)
    out = sys.stderr
    print("\n" + "=" * 35 + " timing summary " + "=" * 35, file=out)
    print(
        f"total runtime:    {total * 1000:.1f} ms across "
        f"{len(durations)} tests",
        file=out,
    )
    print(
        f"mean per test:    {total / len(durations) * 1000:.1f} ms",
        file=out,
    )
    print("slowest 5 tests:", file=out)
    for name, d in durations[:5]:
        print(f"  {d * 1000:>7.1f} ms  {name}", file=out)
    slow = [d for _, d in durations if d > 0.5]
    if slow:
        print(
            f"warning: {len(slow)} test(s) exceeded 500 ms", file=out,
        )


atexit.register(_print_timing_summary)


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


# ---------------------------------------------------------------------------
# Data helpers (cheap, no fixture overhead)
# ---------------------------------------------------------------------------


def _build_scaling(feature_names: List[str]) -> Dict:
    """Minimal scaling dict with zero means and unit stds."""
    scaling: Dict = {"mean": {}, "std": {}}
    scaling["mean"]["timepoints"] = torch.tensor([0.0])
    scaling["std"]["timepoints"] = torch.tensor([1.0])
    for key in ["values", "delta_time", "delta_value"]:
        scaling["mean"][key] = {
            f: torch.tensor([0.0]) for f in feature_names
        }
        scaling["std"][key] = {
            f: torch.tensor([1.0]) for f in feature_names
        }
    return scaling


def _make_current_window(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    n_features: int = N_FEATURES,
    pad_head: int = 1,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build a synthetic current-state window with head padding."""
    g = torch.Generator().manual_seed(seed)
    features = torch.randint(
        0, n_features, (batch_size, seq_len), generator=g
    )
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
    """Build a full TD batch (current + next + isterminal)."""
    cur = _make_current_window(batch_size, seq_len, n_features, seed=seed)
    nxt = _make_current_window(
        batch_size, seq_len, n_features, seed=seed + 1,
    )
    batch = dict(cur)
    for key, val in nxt.items():
        batch[f"next_{key}"] = val
    if terminal_mask is None:
        g = torch.Generator().manual_seed(seed + 2)
        terminal_mask = torch.randint(
            0, 2, (batch_size,), generator=g,
        ).float()
    batch["isterminal"] = terminal_mask.view(-1, 1)
    return batch


def _make_targets(batch_size: int = BATCH_SIZE, seed: int = 0) -> Dict:
    """Synthetic binary labels."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (batch_size,), generator=g).float().view(-1, 1)
    return {LABEL_KEY: y}


# ---------------------------------------------------------------------------
# Module-scoped fixtures (built once per file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_names() -> List[str]:
    return [f"feat_{i}" for i in range(N_FEATURES)]


@pytest.fixture(scope="module")
def scaling(feature_names) -> Dict:
    return _build_scaling(feature_names)


@pytest.fixture(scope="module")
def shared_predictor(scaling, feature_names) -> CNNLSTMPredictor:
    """Shared CNNLSTMPredictor for read-only tests."""
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


@pytest.fixture(scope="module")
def predictor_initial_state(shared_predictor) -> Dict:
    """Snapshot of shared_predictor's state right after construction.

    Tests that mutate weights must restore from this snapshot in their
    teardown, otherwise subsequent tests get mutated state.
    """
    return copy.deepcopy(shared_predictor.state_dict())


@pytest.fixture()
def fresh_predictor(
    shared_predictor, predictor_initial_state,
) -> CNNLSTMPredictor:
    """A predictor that resets to the initial state after each test."""
    shared_predictor.load_state_dict(predictor_initial_state)
    yield shared_predictor
    shared_predictor.load_state_dict(predictor_initial_state)




@pytest.fixture(scope="module")
def empty_dataset() -> SampleEHRDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SampleEHRDataset(samples=[], dataset_name="td_icu_unit_test")


@pytest.fixture(scope="module")
def shared_td_model(empty_dataset, scaling, feature_names):
    """Shared TDICUMortalityModel for read-only tests."""
    torch.manual_seed(42)
    return TDICUMortalityModel(
        dataset=empty_dataset,
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


@pytest.fixture(scope="module")
def td_model_initial_state(shared_td_model) -> Dict:
    """Snapshot of shared_td_model's state after construction."""
    return copy.deepcopy(shared_td_model.state_dict())


@pytest.fixture()
def fresh_td_model(shared_td_model, td_model_initial_state):
    """TDICUMortalityModel that resets to initial state each test."""
    shared_td_model.load_state_dict(td_model_initial_state)
    yield shared_td_model
    shared_td_model.load_state_dict(td_model_initial_state)


@pytest.fixture(scope="module")
def shared_batch() -> Dict[str, torch.Tensor]:
    """A synthetic batch shared across tests that only need to read it."""
    return _make_td_batch(seed=0)


@pytest.fixture(scope="module")
def shared_targets() -> Dict[str, torch.Tensor]:
    """Labels paired with ``shared_batch``."""
    return _make_targets(seed=0)


@pytest.fixture()
def tmp_ckpt_dir():
    """Per-test temp dir, automatically cleaned up."""
    d = tempfile.mkdtemp(prefix="td_icu_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestHelperModules
# ---------------------------------------------------------------------------


class TestHelperModules:
    """MaxPool1D, Transpose, WeightedBCELoss."""

    def test_maxpool1d_preserves_all_nan_window(self):
        pool = MaxPool1D(2, 2)
        x = torch.tensor([[[1.0, 2.0, float("nan"), float("nan")]]])
        out = pool(x)
        assert out.shape == (1, 1, 2)
        assert out[0, 0, 0] == 2.0
        assert torch.isnan(out[0, 0, 1])

    def test_maxpool1d_mixed_window_ignores_nan(self):
        pool = MaxPool1D(2, 2)
        x = torch.tensor([[[3.0, float("nan")]]])
        out = pool(x)
        assert out[0, 0, 0] == 3.0
        assert not torch.isnan(out[0, 0, 0])

    def test_transpose_swaps_dims(self):
        t = Transpose(1, 2)
        x = torch.randn(2, 3, 5)
        assert torch.equal(t(x), x.transpose(1, 2))

    def test_weighted_bce_matches_builtin(self):
        loss_fn = WeightedBCELoss(pos_weight=None)
        logits = torch.zeros(3, 1)
        targets = torch.tensor([[1.0], [0.0], [1.0]])
        expected = nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )
        assert torch.allclose(loss_fn(logits, targets), expected)

    def test_weighted_bce_with_pos_weight(self):
        no_w = WeightedBCELoss(pos_weight=None)
        w = WeightedBCELoss(pos_weight=torch.tensor([2.0]))
        logits = torch.zeros(4, 1)
        targets = torch.ones(4, 1)
        assert w(logits, targets) > no_w(logits, targets)


# ---------------------------------------------------------------------------
# TestCNNLSTMPredictorInstantiation (all read-only, use shared_predictor)
# ---------------------------------------------------------------------------


class TestCNNLSTMPredictorInstantiation:
    """Construction details of the predictor."""

    def test_instantiation_succeeds(self, shared_predictor):
        assert isinstance(shared_predictor, CNNLSTMPredictor)
        assert shared_predictor.n_features == N_FEATURES
        assert shared_predictor.hidden_dim == HIDDEN_DIM
        assert shared_predictor.cnn_layers == CNN_LAYERS

    def test_has_five_embedding_heads(self, shared_predictor):
        expected = {
            "time", "value", "feature", "delta_time", "delta_value",
        }
        assert set(shared_predictor.embedding_net.keys()) == expected

    def test_feature_embedding_dims(self, shared_predictor):
        emb = shared_predictor.embedding_net["feature"]
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == N_FEATURES
        assert emb.embedding_dim == HIDDEN_DIM

    def test_scalar_embeddings_are_mlps(self, shared_predictor):
        for name in ["time", "value", "delta_time", "delta_value"]:
            net = shared_predictor.embedding_net[name]
            assert isinstance(net, nn.Sequential)
            assert net[0].in_features == 1
            assert net[-1].out_features == HIDDEN_DIM

    def test_scaling_buffers_registered(self, shared_predictor):
        buffers = dict(shared_predictor.named_buffers())
        for name in [
            "mean_values", "std_values",
            "mean_delta_time", "std_delta_time",
            "mean_delta_value", "std_delta_value",
            "mean_timepoints", "std_timepoints",
        ]:
            assert name in buffers

    def test_lstm_hidden_is_8x_base(self, shared_predictor):
        assert shared_predictor.lstm.hidden_size == HIDDEN_DIM * 8
        assert shared_predictor.lstm.num_layers == 2

    def test_parameter_count_reasonable(self, shared_predictor):
        n = sum(
            p.numel() for p in shared_predictor.parameters()
            if p.requires_grad
        )
        assert 0 < n < 100_000


# ---------------------------------------------------------------------------
# TestCNNLSTMPredictorForward (read-only: uses shared_predictor in eval)
# ---------------------------------------------------------------------------


class TestCNNLSTMPredictorForward:
    """Predictor forward pass properties."""

    @pytest.fixture(autouse=True)
    def _eval_mode(self, shared_predictor):
        shared_predictor.eval()
        yield

    def test_forward_returns_tuple(self, shared_predictor):
        out = shared_predictor(**_make_current_window())
        assert isinstance(out, tuple) and len(out) == 2

    def test_probs_shape(self, shared_predictor):
        probs, _ = shared_predictor(**_make_current_window())
        assert probs.shape == (BATCH_SIZE, 1)

    def test_logits_shape(self, shared_predictor):
        _, logits = shared_predictor(**_make_current_window())
        assert logits.shape == (BATCH_SIZE, 1)

    def test_probs_in_unit_interval(self, shared_predictor):
        probs, _ = shared_predictor(**_make_current_window())
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_no_nan_in_output(self, shared_predictor):
        probs, logits = shared_predictor(**_make_current_window())
        assert torch.isfinite(probs).all()
        assert torch.isfinite(logits).all()

    def test_handles_no_padding(self, shared_predictor):
        probs, _ = shared_predictor(
            **_make_current_window(pad_head=0)
        )
        assert torch.isfinite(probs).all()

    def test_handles_heavy_padding(self, shared_predictor):
        probs, _ = shared_predictor(
            **_make_current_window(pad_head=SEQ_LEN - 2)
        )
        assert torch.isfinite(probs).all()

    def test_probs_match_sigmoid_logits(self, shared_predictor):
        probs, logits = shared_predictor(**_make_current_window())
        assert torch.allclose(probs, torch.sigmoid(logits), atol=1e-6)


# ---------------------------------------------------------------------------
# TestCNNLSTMPredictorGradients (mutates - uses fresh_predictor)
# ---------------------------------------------------------------------------


class TestCNNLSTMPredictorGradients:
    """Gradient correctness."""

    def test_backward_produces_gradients(self, fresh_predictor):
        fresh_predictor.train()
        _, logits = fresh_predictor(**_make_current_window())
        logits.sum().backward()
        grads = [
            p.grad for p in fresh_predictor.parameters()
            if p.requires_grad
        ]
        assert any(g is not None for g in grads)

    def test_all_trainable_params_receive_gradient(self, fresh_predictor):
        fresh_predictor.train()
        _, logits = fresh_predictor(**_make_current_window())
        logits.sum().backward()
        missing = [
            name for name, p in fresh_predictor.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing

    def test_gradients_are_finite(self, fresh_predictor):
        fresh_predictor.train()
        _, logits = fresh_predictor(**_make_current_window())
        logits.sum().backward()
        for name, p in fresh_predictor.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# TestSoftUpdate - uses two disposable predictors
# ---------------------------------------------------------------------------


class TestSoftUpdate:
    """EMA rule edge cases."""

    @pytest.fixture(scope="class")
    def pair(self, scaling, feature_names):
        """Two tiny predictors used only for soft_update tests."""
        torch.manual_seed(1)
        a = CNNLSTMPredictor(
            n_features=N_FEATURES, features=feature_names,
            output_dim=1, scaling=scaling,
            cnn_layers=CNN_LAYERS, hidden_dim=HIDDEN_DIM,
            dropout=0.1, device="cpu",
        )
        torch.manual_seed(2)
        b = CNNLSTMPredictor(
            n_features=N_FEATURES, features=feature_names,
            output_dim=1, scaling=scaling,
            cnn_layers=CNN_LAYERS, hidden_dim=HIDDEN_DIM,
            dropout=0.1, device="cpu",
        )
        # Snapshot initial state of `a` so each test can restore it
        return a, b, copy.deepcopy(a.state_dict())

    def test_alpha_zero_copies_source(self, pair):
        tgt, src, initial = pair
        tgt.load_state_dict(initial)
        tgt.soft_update(src, alpha=0.0)
        for k, v in src.state_dict().items():
            if v.dtype.is_floating_point:
                assert torch.allclose(tgt.state_dict()[k], v)
        tgt.load_state_dict(initial)

    def test_alpha_one_leaves_target(self, pair):
        tgt, src, initial = pair
        tgt.load_state_dict(initial)
        tgt.soft_update(src, alpha=1.0)
        for k, v in tgt.state_dict().items():
            if v.dtype.is_floating_point:
                assert torch.allclose(initial[k], v)

    def test_alpha_half_is_midpoint(self, pair):
        tgt, src, initial = pair
        tgt.load_state_dict(initial)
        with torch.no_grad():
            tgt.dense[1].weight.fill_(0.0)
            src.dense[1].weight.fill_(2.0)
        tgt.soft_update(src, alpha=0.5)
        assert torch.allclose(
            tgt.dense[1].weight,
            torch.full_like(tgt.dense[1].weight, 1.0),
        )
        tgt.load_state_dict(initial)

    def test_soft_update_has_no_grad(self, pair):
        tgt, src, initial = pair
        tgt.load_state_dict(initial)
        tgt.soft_update(src, alpha=0.9)
        for p in tgt.parameters():
            assert p.grad is None
        tgt.load_state_dict(initial)


# ---------------------------------------------------------------------------
# TestTDICUMortalityModelInstantiation (read-only)
# ---------------------------------------------------------------------------


class TestTDICUMortalityModelInstantiation:
    """TD wrapper construction."""

    def test_instantiation_succeeds(self, shared_td_model):
        assert isinstance(shared_td_model, TDICUMortalityModel)
        assert shared_td_model.label_key == LABEL_KEY
        assert shared_td_model.mode == "binary"
        assert shared_td_model.td_alpha == 0.99

    def test_inherits_from_basemodel(self, shared_td_model):
        from pyhealth.models import BaseModel
        assert isinstance(shared_td_model, BaseModel)

    def test_has_online_and_target_nets(self, shared_td_model):
        assert isinstance(shared_td_model.online_net, CNNLSTMPredictor)
        assert isinstance(shared_td_model.target_net, CNNLSTMPredictor)

    def test_both_loss_functions_exist(self, shared_td_model):
        assert isinstance(shared_td_model.supervised_loss, WeightedBCELoss)
        assert isinstance(shared_td_model.td_loss, WeightedBCELoss)

    def test_td_loss_has_no_pos_weight(self, shared_td_model):
        """TD loss must be unweighted - pos_weight breaks continuous targets."""
        assert shared_td_model.td_loss.loss_fn.pos_weight is None

    def test_non_binary_mode_raises(
        self, empty_dataset, scaling, feature_names,
    ):
        with pytest.raises(ValueError):
            TDICUMortalityModel(
                dataset=empty_dataset,
                feature_keys=FEATURE_KEYS,
                label_key=LABEL_KEY,
                mode="multiclass",
                n_features=N_FEATURES,
                hidden_dim=HIDDEN_DIM,
                cnn_layers=CNN_LAYERS,
                scaling=scaling,
                features_vocab=feature_names,
            )

    def test_missing_scaling_raises(self, empty_dataset, feature_names):
        with pytest.raises(ValueError):
            TDICUMortalityModel(
                dataset=empty_dataset,
                feature_keys=FEATURE_KEYS,
                label_key=LABEL_KEY,
                mode="binary",
                n_features=N_FEATURES,
                hidden_dim=HIDDEN_DIM,
                cnn_layers=CNN_LAYERS,
                scaling=None,
                features_vocab=feature_names,
            )


# ---------------------------------------------------------------------------
# TestBaseModelInterface (read-only)
# ---------------------------------------------------------------------------


class TestBaseModelInterface:
    """Required abstract method implementations."""

    def test_prepare_labels_shape(self, shared_td_model):
        labels = torch.tensor([0, 1, 1, 0])
        out = shared_td_model.prepare_labels(labels)
        assert out.shape == (4, 1)
        assert out.dtype == torch.float32

    def test_prepare_labels_idempotent_from_batched(self, shared_td_model):
        labels = torch.tensor([[0], [1], [1], [0]])
        out = shared_td_model.prepare_labels(labels)
        assert out.shape == (4, 1)
        assert out.dtype == torch.float32

    def test_get_loss_function_returns_module(self, shared_td_model):
        loss_fn = shared_td_model.get_loss_function()
        assert isinstance(loss_fn, nn.Module)
        logits = torch.zeros(2, 1)
        targets = torch.tensor([[1.0], [0.0]])
        result = loss_fn(logits, targets)
        assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# TestTDICUMortalityModelForward (read-only, uses shared_td_model)
# ---------------------------------------------------------------------------


class TestTDICUMortalityModelForward:
    """Forward pass under supervised + TD modes."""

    @pytest.fixture(autouse=True)
    def _eval_mode(self, shared_td_model):
        shared_td_model.eval()
        yield

    def test_returns_all_expected_keys(self, shared_td_model, shared_batch):
        out = shared_td_model(shared_batch, targets=None, train_td=False)
        assert set(out.keys()) == {"loss", "y_prob", "y_true", "logit"}
        assert out["loss"] is None
        assert out["y_true"] is None

    def test_with_targets_produces_loss(
        self, shared_td_model, shared_batch, shared_targets,
    ):
        out = shared_td_model(
            shared_batch, targets=shared_targets, train_td=False,
        )
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"])

    def test_y_prob_shape(self, shared_td_model, shared_batch):
        out = shared_td_model(shared_batch, targets=None, train_td=False)
        assert out["y_prob"].shape == (BATCH_SIZE, 1)

    def test_logit_shape(self, shared_td_model, shared_batch):
        out = shared_td_model(shared_batch, targets=None, train_td=False)
        assert out["logit"].shape == (BATCH_SIZE, 1)

    def test_probs_in_unit_interval(self, shared_td_model, shared_batch):
        out = shared_td_model(shared_batch, targets=None, train_td=False)
        assert torch.all(out["y_prob"] >= 0.0)
        assert torch.all(out["y_prob"] <= 1.0)

    def test_supervised_and_td_mode_differ(
        self, shared_td_model, shared_targets,
    ):
        """Non-terminal transitions -> two losses should differ."""
        batch = _make_td_batch(terminal_mask=torch.zeros(BATCH_SIZE))
        sup_loss = shared_td_model(
            batch, targets=shared_targets, train_td=False,
        )["loss"]
        td_loss = shared_td_model(
            batch, targets=shared_targets, train_td=True,
        )["loss"]
        assert not torch.allclose(sup_loss, td_loss)

    def test_all_terminal_reduces_to_supervised(
        self, shared_td_model, shared_targets,
    ):
        """All-terminal -> TD loss == supervised loss."""
        batch = _make_td_batch(terminal_mask=torch.ones(BATCH_SIZE))
        sup = shared_td_model(
            batch, targets=shared_targets, train_td=False,
        )["loss"]
        td = shared_td_model(
            batch, targets=shared_targets, train_td=True,
        )["loss"]
        assert torch.allclose(sup, td, atol=1e-5)


# ---------------------------------------------------------------------------
# TestTDTargetComputation (read-only)
# ---------------------------------------------------------------------------


class TestTDTargetComputation:
    """TD target rule behavior."""

    def test_td_target_shape(
        self, shared_td_model, shared_batch, shared_targets,
    ):
        td = shared_td_model.compute_td_target(shared_batch, shared_targets)
        assert td.shape == (BATCH_SIZE, 1)

    def test_td_target_in_unit_interval(
        self, shared_td_model, shared_batch, shared_targets,
    ):
        td = shared_td_model.compute_td_target(shared_batch, shared_targets)
        assert torch.all(td >= 0.0) and torch.all(td <= 1.0)

    def test_td_target_detached(
        self, shared_td_model, shared_batch, shared_targets,
    ):
        td = shared_td_model.compute_td_target(shared_batch, shared_targets)
        assert not td.requires_grad


# ---------------------------------------------------------------------------
# TestTDICUMortalityModelGradients (mutates - uses fresh_td_model)
# ---------------------------------------------------------------------------


class TestTDICUMortalityModelGradients:
    """Gradient flow + target-net no-grad invariant."""

    def test_supervised_backward(
        self, fresh_td_model, shared_batch, shared_targets,
    ):
        fresh_td_model.train()
        out = fresh_td_model(
            shared_batch, targets=shared_targets, train_td=False,
        )
        out["loss"].backward()
        grads = [
            p.grad for p in fresh_td_model.online_net.parameters()
            if p.requires_grad
        ]
        assert any(g is not None for g in grads)

    def test_td_backward(
        self, fresh_td_model, shared_batch, shared_targets,
    ):
        fresh_td_model.train()
        out = fresh_td_model(
            shared_batch, targets=shared_targets, train_td=True,
        )
        out["loss"].backward()
        grads = [
            p.grad for p in fresh_td_model.online_net.parameters()
            if p.requires_grad
        ]
        assert any(g is not None for g in grads)

    def test_target_net_no_gradient_in_td_mode(
        self, fresh_td_model, shared_batch, shared_targets,
    ):
        """Target net must never receive gradient flow."""
        fresh_td_model.train()
        out = fresh_td_model(
            shared_batch, targets=shared_targets, train_td=True,
        )
        out["loss"].backward()
        for name, p in fresh_td_model.target_net.named_parameters():
            assert p.grad is None, f"target_net.{name} got a gradient"


# ---------------------------------------------------------------------------
# TestSoftUpdateTarget (mutates - uses fresh_td_model)
# ---------------------------------------------------------------------------


class TestSoftUpdateTarget:
    """End-to-end optimizer step + target update."""

    def test_target_changes_after_training_step(
        self, fresh_td_model, shared_batch, shared_targets,
    ):
        target_before = {
            k: v.detach().clone()
            for k, v in fresh_td_model.target_net.state_dict().items()
        }
        optim = torch.optim.AdamW(
            fresh_td_model.online_net.parameters(), lr=1e-1,
        )
        out = fresh_td_model(
            shared_batch, targets=shared_targets, train_td=True,
        )
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        optim.step()
        fresh_td_model.soft_update_target()
        target_after = fresh_td_model.target_net.state_dict()
        changed = any(
            not torch.allclose(target_before[k], target_after[k], atol=1e-6)
            for k in target_before
            if target_before[k].dtype.is_floating_point
        )
        assert changed


# ---------------------------------------------------------------------------
# TestMCDropoutConfidence - NEW: covers the extension
# ---------------------------------------------------------------------------


class TestMCDropoutConfidence:
    """Monte Carlo dropout confidence extension."""

    @pytest.fixture(autouse=True)
    def _eval_mode(self, shared_td_model):
        shared_td_model.eval()
        yield

    def test_returns_all_expected_keys(self, shared_td_model, shared_batch):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        expected = {
            "mortality_prob", "confidence_std",
            "ci_95_lower", "ci_95_upper",
            "is_high_confidence", "is_low_confidence",
        }
        assert set(out.keys()) == expected

    def test_output_shapes(self, shared_td_model, shared_batch):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        for key in [
            "mortality_prob", "confidence_std",
            "ci_95_lower", "ci_95_upper",
        ]:
            assert out[key].shape == (BATCH_SIZE,), key

    def test_mortality_prob_in_unit_interval(
        self, shared_td_model, shared_batch,
    ):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        assert torch.all(out["mortality_prob"] >= 0.0)
        assert torch.all(out["mortality_prob"] <= 1.0)

    def test_std_non_negative(self, shared_td_model, shared_batch):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        assert torch.all(out["confidence_std"] >= 0.0)

    def test_ci_in_unit_interval_and_contains_mean(
        self, shared_td_model, shared_batch,
    ):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        assert torch.all(out["ci_95_lower"] >= 0.0)
        assert torch.all(out["ci_95_upper"] <= 1.0)
        assert torch.all(out["ci_95_lower"] <= out["mortality_prob"])
        assert torch.all(out["mortality_prob"] <= out["ci_95_upper"])

    def test_confidence_flags_are_bool(self, shared_td_model, shared_batch):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        assert out["is_high_confidence"].dtype == torch.bool
        assert out["is_low_confidence"].dtype == torch.bool

    def test_confidence_flags_mutually_exclusive(
        self, shared_td_model, shared_batch,
    ):
        """A sample cannot be both high-conf and low-conf with the default
        thresholds (high_thresh=0.005 < low_thresh=0.01)."""
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=3,
        )
        both = out["is_high_confidence"] & out["is_low_confidence"]
        assert not both.any()

    def test_does_not_alter_train_eval_of_non_dropout_modules(
        self, shared_td_model, shared_batch,
    ):
        """After predict_with_confidence, BatchNorm layers stay in eval."""
        shared_td_model.eval()
        _ = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=2,
        )
        for m in shared_td_model.online_net.modules():
            if isinstance(m, nn.BatchNorm1d):
                assert not m.training, (
                    "BatchNorm should stay in eval during MC dropout"
                )

    def test_no_gradients_from_mc_sampling(
        self, shared_td_model, shared_batch,
    ):
        out = shared_td_model.predict_with_confidence(
            shared_batch, n_mc_samples=2,
        )
        assert not out["mortality_prob"].requires_grad
        assert not out["confidence_std"].requires_grad


# ---------------------------------------------------------------------------
# TestCheckpointIO (uses shared_td_model, snapshot is saved to tmp dir)
# ---------------------------------------------------------------------------


class TestCheckpointIO:
    """Save/load round trip."""

    def test_state_dict_roundtrip(
        self, fresh_td_model, tmp_ckpt_dir,
    ):
        ckpt_path = tmp_ckpt_dir / "model.pt"
        torch.save(fresh_td_model.state_dict(), ckpt_path)
        assert ckpt_path.exists()

        original = {
            k: v.detach().clone()
            for k, v in fresh_td_model.state_dict().items()
        }
        with torch.no_grad():
            for p in fresh_td_model.online_net.parameters():
                p.add_(1.0)

        restored = torch.load(ckpt_path, weights_only=True)
        fresh_td_model.load_state_dict(restored)
        loaded = fresh_td_model.state_dict()
        for k, v in original.items():
            assert torch.allclose(v, loaded[k])

    def test_tmp_dir_is_writable(self, tmp_ckpt_dir):
        assert tmp_ckpt_dir.exists() and tmp_ckpt_dir.is_dir()
        (tmp_ckpt_dir / "sanity.txt").write_text("ok")
        assert (tmp_ckpt_dir / "sanity.txt").read_text() == "ok"

    def test_tmp_dir_is_unique(self, tmp_ckpt_dir):
        assert "td_icu_test_" in str(tmp_ckpt_dir)


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Fixed-seed reproducibility (uses shared_predictor in eval)."""

    def test_same_input_same_output(self, shared_predictor):
        shared_predictor.eval()
        cur = _make_current_window(seed=42)
        with torch.no_grad():
            p1, _ = shared_predictor(**cur)
            p2, _ = shared_predictor(**cur)
        # Same input + eval mode (no dropout) -> identical output
        assert torch.allclose(p1, p2, atol=1e-6)
