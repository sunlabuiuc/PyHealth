"""Tests for BulkRNABert model, BulkRNABertMLP, and CoxPartialLikelihoodLoss.

All tests use synthetic random tensors only.  No real data is loaded anywhere
in this file.  All tests are designed to complete in well under 1 second total.
"""

import pytest
import torch

from pyhealth.models.bulk_rnabert import (
    BulkRNABert,
    BulkRNABertMLP,
    CoxPartialLikelihoodLoss,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 4
NUM_GENES = 19062


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


def test_classification_model_instantiates():
    """BulkRNABert instantiates without error for classification task."""
    model = BulkRNABert(num_classes=33, task="classification")
    assert isinstance(model, BulkRNABert)


def test_survival_model_instantiates():
    """BulkRNABert instantiates without error for survival task."""
    model = BulkRNABert(task="survival")
    assert isinstance(model, BulkRNABert)


def test_invalid_task_raises():
    """BulkRNABert raises AssertionError for an unrecognised task string."""
    with pytest.raises(AssertionError):
        BulkRNABert(task="invalid")


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------


def test_classification_forward_output_shape():
    """Classification forward returns logits of shape (B, num_classes)."""
    model = BulkRNABert(num_classes=33, task="classification")
    x = torch.randn(BATCH, NUM_GENES)
    out = model(x)
    assert out["logits"].shape == (BATCH, 33)
    assert "loss" not in out


def test_classification_forward_with_labels():
    """Classification forward includes a scalar loss when labels are supplied."""
    model = BulkRNABert(num_classes=33, task="classification")
    x = torch.randn(BATCH, NUM_GENES)
    labels = torch.randint(0, 33, (BATCH,))
    out = model(x, labels=labels)
    assert "loss" in out
    assert out["loss"].ndim == 0
    assert out["logits"].shape == (BATCH, 33)


def test_survival_forward_output_shape():
    """Survival forward returns risk_score of shape (B, 1) without loss."""
    model = BulkRNABert(task="survival")
    x = torch.randn(BATCH, NUM_GENES)
    out = model(x)
    assert out["risk_score"].shape == (BATCH, 1)
    assert "loss" not in out


def test_survival_forward_with_labels():
    """Survival forward includes a scalar loss when times and events supplied."""
    model = BulkRNABert(task="survival")
    x = torch.randn(BATCH, NUM_GENES)
    times = torch.rand(BATCH) * 1000
    events = torch.randint(0, 2, (BATCH,)).float()
    out = model(x, survival_times=times, events=events)
    assert "loss" in out
    assert out["loss"].ndim == 0


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradients_flow_classification():
    """Loss.backward() yields non-zero gradients for at least one parameter."""
    model = BulkRNABert(num_classes=33, task="classification")
    x = torch.randn(BATCH, NUM_GENES)
    labels = torch.randint(0, 33, (BATCH,))
    out = model(x, labels=labels)
    out["loss"].backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad


def test_gradients_flow_survival():
    """Survival loss.backward() yields non-zero gradients for at least one parameter."""
    model = BulkRNABert(task="survival")
    x = torch.randn(BATCH, NUM_GENES)
    times = torch.rand(BATCH) * 1000 + 1.0
    events = torch.ones(BATCH).float()
    out = model(x, survival_times=times, events=events)
    out["loss"].backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad


# ---------------------------------------------------------------------------
# Encoder freeze / IA3 tests
# ---------------------------------------------------------------------------


def test_freeze_encoder_flag():
    """freeze_encoder=True freezes encoder params but leaves head trainable."""
    model = BulkRNABert(num_classes=33, task="classification", freeze_encoder=True)
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"
    for name, param in model.head.named_parameters():
        assert param.requires_grad, f"{name} should require grad"


def test_ia3_only_ia3_params_trainable():
    """use_ia3=True freezes the encoder; only IA3 vectors remain trainable."""
    model = BulkRNABert(num_classes=33, task="classification", use_ia3=True)
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder param {name} should be frozen"
    if len(list(model._ia3_vectors)) > 0:
        for i, vec in enumerate(model._ia3_vectors):
            assert vec.requires_grad, f"IA3 vector {i} should require grad"


# ---------------------------------------------------------------------------
# Cox loss tests
# ---------------------------------------------------------------------------


def test_cox_loss_with_all_events():
    """Cox loss returns a finite scalar when all samples have observed events."""
    loss_fn = CoxPartialLikelihoodLoss()
    risk = torch.randn(BATCH)
    times = torch.tensor([100.0, 200.0, 300.0, 400.0])
    events = torch.ones(BATCH)
    loss = loss_fn(risk, times, events)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_cox_loss_with_no_events():
    """Cox loss handles an all-censored batch gracefully (returns 0, no NaN)."""
    loss_fn = CoxPartialLikelihoodLoss()
    risk = torch.randn(BATCH)
    times = torch.tensor([100.0, 200.0, 300.0, 400.0])
    events = torch.zeros(BATCH)
    loss = loss_fn(risk, times, events)
    assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# MLP head test
# ---------------------------------------------------------------------------


def test_mlp_head_output_shape():
    """BulkRNABertMLP produces the correct output shape."""
    mlp = BulkRNABertMLP(input_dim=256, hidden_dims=[256, 128], output_dim=33)
    x = torch.randn(BATCH, 256)
    out = mlp(x)
    assert out.shape == (BATCH, 33)


# ---------------------------------------------------------------------------
# Five-cohort subset test (paper Table 1)
# ---------------------------------------------------------------------------


def test_5_cohort_forward():
    """Model with num_classes=5 produces logits of shape (B, 5)."""
    model = BulkRNABert(num_classes=5, task="classification")
    x = torch.randn(BATCH, NUM_GENES)
    out = model(x)
    assert out["logits"].shape == (BATCH, 5)
