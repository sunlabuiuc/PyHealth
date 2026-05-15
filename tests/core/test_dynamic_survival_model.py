"""Unit tests for DynamicSurvivalModel.

These tests are self-contained: they stub the PyHealth Dataset/BaseModel
dependencies so the tests run with only torch and numpy installed.
No external datasets (MIMIC-III, MIMIC-IV, eICU) or a full PyHealth
installation are required.

When running inside a complete PyHealth environment (Python ≥ 3.12 with
``pip install -e .``), you can also import the full stack and all assertions
still hold.

Run from the repository root::

    python -m pytest tests/core/test_dynamic_survival_model.py -v
"""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub the minimal PyHealth symbols needed for the import chain
# ---------------------------------------------------------------------------

class _SampleDataset:
    """Minimal stub for pyhealth.datasets.SampleDataset."""
    feature_keys = ["timeseries"]
    label_keys   = ["label"]
    input_schema  = {"timeseries": "tensor"}
    output_schema = {"label": "binary"}


class _BaseModel(nn.Module):
    """Minimal stub for pyhealth.models.BaseModel."""
    def __init__(self, dataset: Any, **kw: Any) -> None:
        super().__init__()
        self.dataset      = dataset
        self.feature_keys = list(getattr(dataset, "input_schema",  {}).keys())
        self.label_keys   = list(getattr(dataset, "output_schema", {}).keys())
        self._dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self._dummy_param.device


def _make_pyhealth_stubs() -> None:
    """Insert lightweight stubs so dynamic_survival_model.py can be imported
    without a full PyHealth installation."""
    # pyhealth root
    ph = sys.modules.get("pyhealth") or types.ModuleType("pyhealth")
    sys.modules["pyhealth"] = ph

    # pyhealth.datasets
    ph_ds = sys.modules.get("pyhealth.datasets") or types.ModuleType("pyhealth.datasets")
    ph_ds.SampleDataset = _SampleDataset
    ph_ds.create_sample_dataset = lambda **kw: _SampleDataset()
    ph_ds.get_dataloader = lambda ds, **kw: []
    setattr(ph, "datasets", ph_ds)
    sys.modules["pyhealth.datasets"] = ph_ds

    # pyhealth.models
    ph_models = sys.modules.get("pyhealth.models") or types.ModuleType("pyhealth.models")
    ph_models.BaseModel = _BaseModel
    setattr(ph, "models", ph_models)
    sys.modules["pyhealth.models"] = ph_models

_make_pyhealth_stubs()

# Import our model file directly, bypassing the pyhealth package __init__
import importlib.util as _ilu, pathlib as _pl
_repo   = _pl.Path(__file__).parents[2]           # .../pyhealth repo root
_target = _repo / "pyhealth" / "models" / "dynamic_survival_model.py"

# Register all intermediate pyhealth sub-modules so the file's top-level
# imports resolve to our stubs rather than a missing real package.
import sys as _sys, types as _types
for _name in (
    "pyhealth",
    "pyhealth.datasets",
    "pyhealth.models",
):
    if _name not in _sys.modules:
        _sys.modules[_name] = _types.ModuleType(_name)

# Ensure the stubs we built earlier are reachable
_make_pyhealth_stubs()

_spec = _ilu.spec_from_file_location("_dsa_model_module", _target)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DynamicSurvivalModel = _mod.DynamicSurvivalModel


# ---------------------------------------------------------------------------
# Minimal stub dataset
# ---------------------------------------------------------------------------

class _FakeDataset:
    input_schema  = {"timeseries": "tensor"}
    output_schema = {"label": "binary"}


N_FEAT   = 8
T        = 20
HORIZON  = 6
B        = 4
H_DIM    = 32
EMB_DIM  = 16


def _model(**kw: Any) -> DynamicSurvivalModel:
    defaults: Dict[str, Any] = dict(
        dataset      = _FakeDataset(),
        input_dim    = N_FEAT,
        hidden_dim   = H_DIM,
        embedding_dim= EMB_DIM,
        num_layers   = 1,
        horizon      = HORIZON,
        l1_reg       = 0.01,
    )
    defaults.update(kw)
    return DynamicSurvivalModel(**defaults)


def _batch(include_label: bool = True) -> Dict[str, torch.Tensor]:
    d = {"timeseries": torch.randn(B, T, N_FEAT)}
    if include_label:
        d["label"] = torch.randint(0, 2, (B,)).float()
    return d


# ---------------------------------------------------------------------------
# Tests: initialisation
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):
    def test_parameter_count_positive(self) -> None:
        self.assertGreater(_model().count_parameters(), 0)

    def test_invalid_encoder_raises(self) -> None:
        with self.assertRaises(ValueError):
            _model(encoder_type="rnn")

    def test_all_backbones_instantiate(self) -> None:
        _model(encoder_type="gru")
        _model(encoder_type="lstm")
        _model(encoder_type="transformer", hidden_dim=32, num_heads=4)


# ---------------------------------------------------------------------------
# Tests: forward — shapes and values
# ---------------------------------------------------------------------------

class TestForwardWithLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.m = _model()
        self.m.eval()
        self.b = _batch()

    def _fwd(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.m(**self.b)

    def test_keys(self) -> None:
        out = self._fwd()
        for k in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(k, out)

    def test_y_prob_shape(self) -> None:
        self.assertEqual(self._fwd()["y_prob"].shape, (B, 1))

    def test_y_prob_range(self) -> None:
        p = self._fwd()["y_prob"]
        self.assertTrue((p >= 0).all() and (p <= 1).all())

    def test_loss_scalar_and_finite(self) -> None:
        loss = self._fwd()["loss"]
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_logit_same_shape_as_y_prob(self) -> None:
        out = self._fwd()
        self.assertEqual(out["logit"].shape, out["y_prob"].shape)


class TestForwardNoLabel(unittest.TestCase):
    """Inference mode — no label key in batch."""

    def test_no_loss_no_y_true(self) -> None:
        m = _model()
        m.eval()
        with torch.no_grad():
            out = m(**_batch(include_label=False))
        self.assertIn("y_prob",  out)
        self.assertNotIn("loss",   out)
        self.assertNotIn("y_true", out)


# ---------------------------------------------------------------------------
# Tests: backward
# ---------------------------------------------------------------------------

class TestBackward(unittest.TestCase):
    def test_gradients_flow(self) -> None:
        m   = _model()
        out = m(**_batch())
        out["loss"].backward()
        grads = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0 for g in grads))


# ---------------------------------------------------------------------------
# Tests: encoders
# ---------------------------------------------------------------------------

class TestEncoders(unittest.TestCase):
    def _run(self, enc: str, **kw: Any) -> Dict[str, torch.Tensor]:
        m = _model(encoder_type=enc, **kw)
        m.eval()
        with torch.no_grad():
            return m(**_batch())

    def test_gru_y_prob_shape(self) -> None:
        self.assertEqual(self._run("gru")["y_prob"].shape, (B, 1))

    def test_lstm_y_prob_shape(self) -> None:
        self.assertEqual(self._run("lstm")["y_prob"].shape, (B, 1))

    def test_transformer_y_prob_shape(self) -> None:
        out = self._run("transformer", hidden_dim=32, num_heads=4)
        self.assertEqual(out["y_prob"].shape, (B, 1))

    def test_gru_finite_loss(self) -> None:
        self.assertTrue(torch.isfinite(self._run("gru")["loss"]))

    def test_lstm_finite_loss(self) -> None:
        self.assertTrue(torch.isfinite(self._run("lstm")["loss"]))

    def test_transformer_finite_loss(self) -> None:
        out = self._run("transformer", hidden_dim=32, num_heads=4)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_transformer_gradients_flow(self) -> None:
        m = _model(encoder_type="transformer", hidden_dim=32, num_heads=4)
        out = m(**_batch())
        out["loss"].backward()
        grads = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0 for g in grads))


# ---------------------------------------------------------------------------
# Tests: bias initialisation
# ---------------------------------------------------------------------------

class TestBiasInit(unittest.TestCase):
    def test_bias_shifts_output(self) -> None:
        m     = _model()
        rates = np.full(HORIZON, 0.05, dtype=np.float32)
        m.initialise_bias(rates)
        m.eval()
        with torch.no_grad():
            out = m(timeseries=torch.zeros(1, T, N_FEAT))
        expected = 1.0 - (1.0 - 0.05) ** HORIZON
        self.assertAlmostEqual(out["y_prob"].item(), expected, delta=0.05)


if __name__ == "__main__":
    unittest.main()
