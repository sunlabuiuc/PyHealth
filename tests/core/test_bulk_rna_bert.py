import numpy as np
import pytest
import torch

from pyhealth.models import BulkRNABert
from pyhealth.models.bulk_rna_bert import cox_loss, SurvivalHead, tokenize


def test_tokenize_basic():
    ids = tokenize(np.array([[0.0, 1.0, 1e9]]), normalization_factor=1.0)
    assert ids.tolist() == [[0, 19, 63]]


def test_head_shape():
    assert SurvivalHead()(torch.zeros(3, 256)).shape == (3,)


def test_forward_cached_mode():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(gene_expression=torch.randn(4, 256))
    assert out["logit"].shape == (4, 1)
    assert out["y_prob"].shape == (4, 1)
    assert torch.isfinite(out["logit"]).all()


def test_forward_with_labels_returns_loss():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(
        gene_expression=torch.randn(4, 256),
        survival_time=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        cohort=torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    )
    assert "loss" in out
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_gradients_only_in_head():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(
        gene_expression=torch.randn(4, 256),
        survival_time=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
    )
    out["loss"].backward()
    assert all(p.grad is not None for p in model.head.parameters())


def test_cox_loss_matches_brute_force():
    rng = np.random.default_rng(0)
    risk = rng.normal(size=40).astype(np.float32)
    time = rng.integers(1, 10, size=40).astype(np.float32)
    event = rng.integers(0, 2, size=40).astype(np.float32)
    got = cox_loss(torch.tensor(risk), torch.tensor(time), torch.tensor(event)).item()
    terms = [risk[i] - np.log(np.exp(risk[time >= time[i]]).sum())
             for i in range(len(time)) if event[i]]
    assert abs(got - (-np.mean(terms))) < 1e-4


def test_cohort_restricted_cox_loss():
    rng = np.random.default_rng(1)
    risk = rng.normal(size=30).astype(np.float32)
    time = rng.uniform(1, 100, size=30).astype(np.float32)
    event = rng.integers(0, 2, size=30).astype(np.float32)
    group = np.repeat([0, 1, 2], 10)
    terms = []
    for g in range(3):
        m = group == g
        r, t, e = risk[m], time[m], event[m]
        terms += [r[i] - np.log(np.exp(r[t >= t[i]]).sum()) for i in range(len(t)) if e[i]]
    got = cox_loss(torch.tensor(risk), torch.tensor(time), torch.tensor(event),
                   torch.tensor(group)).item()
    assert abs(got - (-np.mean(terms))) < 1e-4