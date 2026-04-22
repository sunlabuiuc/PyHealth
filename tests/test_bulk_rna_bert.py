"""Tests for BulkRNABert model and Cox loss.

Uses synthetic tensors only. All tests complete in milliseconds.

``pyhealth.models.__init__`` imports every model (including optional
``transformers``). These tests only need ``bulk_rna_bert``; register a
minimal ``pyhealth.models`` package so submodule import does not execute
that barreled ``__init__`` (see PyHealth ``tests/core`` patterns that
import submodules directly).
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "pyhealth" / "models"


def _load_bulk_rna_bert_module():
    """Import ``pyhealth.models.bulk_rna_bert`` without loading ``models/__init__.py``."""
    if "pyhealth.models" not in sys.modules:
        import pyhealth

        _pkg = types.ModuleType("pyhealth.models")
        _pkg.__path__ = [str(_MODELS_DIR)]
        sys.modules["pyhealth.models"] = _pkg
    return importlib.import_module("pyhealth.models.bulk_rna_bert")


_brb = _load_bulk_rna_bert_module()
BulkRNABert = _brb.BulkRNABert
cox_partial_likelihood_loss = _brb.cox_partial_likelihood_loss


class TestBulkRNABert:

    N_GENES = 50
    N_BINS = 64
    BATCH = 2
    EMB = 64
    N_CLASSES = 5

    def _make_model(self, mode="classification", use_ia3=False):
        return BulkRNABert(
            dataset=None,
            n_genes=self.N_GENES,
            n_bins=self.N_BINS,
            embedding_dim=self.EMB,
            n_layers=2,
            n_heads=4,
            ffn_dim=128,
            dropout=0.0,
            mlp_hidden=(32, 16),
            mode=mode,
            n_classes=self.N_CLASSES,
            use_ia3=use_ia3,
        )

    def _token_ids(self):
        return torch.randint(0, self.N_BINS, (self.BATCH, self.N_GENES))

    def test_instantiation_classification(self):
        assert self._make_model("classification") is not None

    def test_instantiation_survival(self):
        assert self._make_model("survival") is not None

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            BulkRNABert(dataset=None, n_genes=10, mode="invalid")

    def test_encode_output_shape(self):
        model = self._make_model()
        z = model.encode(self._token_ids())
        assert z.shape == (self.BATCH, self.EMB)

    def test_classification_forward_output_shapes(self):
        model = self._make_model("classification")
        labels = torch.randint(0, self.N_CLASSES, (self.BATCH,))
        out = model(token_ids=self._token_ids(), cancer_type=labels)
        assert out["logit"].shape == (self.BATCH, self.N_CLASSES)
        assert out["y_prob"].shape == (self.BATCH, self.N_CLASSES)
        assert out["loss"].ndim == 0

    def test_classification_forward_no_labels(self):
        model = self._make_model("classification")
        out = model(token_ids=self._token_ids())
        assert "logit" in out
        assert "loss" not in out

    def test_survival_forward_output_shapes(self):
        model = self._make_model("survival")
        times = torch.rand(self.BATCH) * 1000
        events = torch.randint(0, 2, (self.BATCH,)).float()
        out = model(token_ids=self._token_ids(), survival_time=times, event=events)
        assert out["logit"].shape == (self.BATCH,)

    def test_gradients_flow_classification(self):
        model = self._make_model("classification")
        labels = torch.randint(0, self.N_CLASSES, (self.BATCH,))
        out = model(token_ids=self._token_ids(), cancer_type=labels)
        out["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_gradients_flow_survival(self):
        model = self._make_model("survival")
        times = torch.tensor([500.0, 200.0])
        events = torch.tensor([1.0, 0.0])
        out = model(token_ids=self._token_ids(), survival_time=times, event=events)
        out["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_y_prob_sums_to_one(self):
        model = self._make_model("classification")
        out = model(token_ids=self._token_ids())
        prob_sum = out["y_prob"].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(self.BATCH), atol=1e-5)

    def test_mlm_forward(self):
        model = self._make_model("classification")
        token_ids = self._token_ids()
        mask = torch.zeros(self.BATCH, self.N_GENES, dtype=torch.bool)
        mask[:, :5] = True
        targets = torch.randint(0, self.N_BINS, (self.BATCH, self.N_GENES))
        loss = model.forward_mlm(token_ids, mask, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestCoxLoss:

    def test_loss_is_scalar(self):
        loss = cox_partial_likelihood_loss(
            torch.tensor([0.5, -0.3, 1.0, -1.0]),
            torch.tensor([400.0, 300.0, 200.0, 100.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        assert loss.ndim == 0

    def test_loss_is_finite(self):
        loss = cox_partial_likelihood_loss(
            torch.randn(8),
            torch.rand(8) * 1000 + 1,
            torch.randint(0, 2, (8,)).float(),
        )
        assert torch.isfinite(loss)

    def test_no_events_returns_zero(self):
        loss = cox_partial_likelihood_loss(
            torch.randn(4),
            torch.tensor([100.0, 200.0, 300.0, 400.0]),
            torch.zeros(4),
        )
        assert loss.item() == pytest.approx(0.0)

    def test_correct_ranking_reduces_loss(self):
        times = torch.tensor([100.0, 200.0])
        events = torch.tensor([1.0, 1.0])
        loss_good = cox_partial_likelihood_loss(
            torch.tensor([2.0, 1.0]), times, events
        )
        loss_bad = cox_partial_likelihood_loss(
            torch.tensor([1.0, 2.0]), times, events
        )
        assert loss_good.item() < loss_bad.item()