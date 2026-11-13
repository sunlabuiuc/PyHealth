import numpy as np
import pytest
import torch
import torch.nn as nn

from pyhealth.models import BulkRNABert
from pyhealth.models.bulk_rna_bert import cox_loss, SurvivalHead, tokenize, N_GENES


def test_tokenize_basic():
    x = np.zeros((1, N_GENES), dtype=np.float64)
    x[0, 0] = 0.0
    x[0, 1] = 1.0
    x[0, 2] = 1e9
    ids = tokenize(x, normalization_factor=1.0)
    assert ids[0, 0] == 0      
    assert ids[0, 1] == 19      
    assert ids[0, 2] == 63      
    assert (ids[0, 3:] == 0).all()  

def test_tokenize_rejects_negative():
    x = np.zeros((1, N_GENES))
    x[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        tokenize(x)


def test_tokenize_rejects_non_finite():
    x = np.zeros((1, N_GENES))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tokenize(x)


def test_tokenize_rejects_wrong_gene_count():
    with pytest.raises(ValueError, match="19_062|19062|genes"):
        tokenize(np.zeros((1, 100)))

def test_head_shape():
    assert SurvivalHead()(torch.zeros(3, 256)).shape == (3,)


def test_forward_cached_mode():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(embedding=torch.randn(4, 256))
    assert out["logit"].shape == (4, 1)
    assert out["y_prob"].shape == (4, 1)
    assert out["risk_score"].shape == (4, 1)
    assert torch.isfinite(out["logit"]).all()

def test_forward_rejects_wrong_embedding_shape():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    with pytest.raises(ValueError, match="256"):
        model.forward(embedding=torch.randn(4, 128))

def test_forward_with_labels_returns_loss():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(
        embedding=torch.randn(4, 256),
        duration=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        cohort=torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    )
    assert "loss" in out
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_gradients_only_in_head():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(
        embedding=torch.randn(4, 256),
        duration=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
    )
    out["loss"].backward()
    assert all(p.grad is not None for p in model.head.parameters())


def test_gradients_flow_to_head_in_cached_mode():
    """With cached inputs, only the head should receive gradients."""
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    out = model.forward(
        embedding=torch.randn(4, 256),
        duration=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
    )
    out["loss"].backward()
    assert all(p.grad is not None for p in model.head.parameters())
    # No encoder is present in cached mode.
    assert model.encoder is None


def test_encoder_frozen_in_train_mode():
    """model.train() must not unfreeze the encoder or put it in training mode."""
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return self.fc(x)

    fake = FakeEncoder()
    for p in fake.parameters():
        p.requires_grad_(False)
    fake.eval()
    model.encoder = fake

    model.train()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_cox_loss_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    risk = rng.normal(size=40).astype(np.float32)
    duration = rng.integers(1, 10, size=40).astype(np.float32)
    event = rng.integers(0, 2, size=40).astype(np.float32)
    got = cox_loss(torch.tensor(risk), torch.tensor(duration), torch.tensor(event)).item()
    terms = [
        risk[i] - np.log(np.exp(risk[duration >= duration[i]]).sum())
        for i in range(len(duration)) if event[i]
    ]
    assert abs(got - (-np.mean(terms))) < 1e-4


def test_cox_loss_no_events():
    risk = torch.randn(10)
    time = torch.rand(10) * 100
    event = torch.zeros(10)
    loss = cox_loss(risk, time, event)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0

def test_cox_loss_matches_trajsurv_implementation():
    """Global Cox loss delegates to the shared TrajSurv implementation."""
    from pyhealth.models.trajsurv import cox_partial_likelihood_loss
    rng = np.random.default_rng(2)
    risk = torch.tensor(rng.normal(size=30).astype(np.float32))
    duration = torch.tensor(rng.uniform(1, 100, size=30).astype(np.float32))
    event = torch.tensor(rng.integers(0, 2, size=30).astype(np.float32))
    assert torch.allclose(
        cox_loss(risk, duration, event),
        cox_partial_likelihood_loss(risk, duration, event),
    )

def test_cohort_restricted_cox_loss():
    rng = np.random.default_rng(1)
    risk = rng.normal(size=30).astype(np.float32)
    duration = rng.uniform(1, 100, size=30).astype(np.float32)
    event = rng.integers(0, 2, size=30).astype(np.float32)
    group = np.repeat([0, 1, 2], 10)
    terms = []
    for g in range(3):
        m = group == g
        r, t, e = risk[m], duration[m], event[m]
        terms += [
            r[i] - np.log(np.exp(r[t >= t[i]]).sum())
            for i in range(len(t)) if e[i]
        ]
    got = cox_loss(
        torch.tensor(risk), torch.tensor(duration), torch.tensor(event),
        torch.tensor(group),
    ).item()
    assert abs(got - (-np.mean(terms))) < 1e-4


def test_cox_loss_mode_default_is_global():
    model = BulkRNABert(dataset=None, use_cached_embeddings=True)
    assert model.cox_loss_mode == "global"


def test_cox_loss_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="global.*cohort|cohort.*global"):
        BulkRNABert(use_cached_embeddings=True, cox_loss_mode="bogus")


def test_cox_loss_mode_cohort_without_labels_raises():
    model = BulkRNABert(
        dataset=None, use_cached_embeddings=True, cox_loss_mode="cohort",
    )
    with pytest.raises(ValueError, match="cohort"):
        model.forward(
            embedding=torch.randn(4, 256),
            duration=torch.tensor([[1.0]] * 4),
            event=torch.tensor([[1.0]] * 4),
        )


def test_cox_loss_mode_cohort_matches_explicit_call():
    """The model's cohort-mode loss should equal an explicit cox_loss call."""
    model = BulkRNABert(
        dataset=None, use_cached_embeddings=True, cox_loss_mode="cohort",
    )
    embedding = torch.randn(8, 256)
    dur = torch.rand(8) * 100
    ev = torch.randint(0, 2, (8,)).float()
    co = torch.randint(0, 3, (8,))
    out = model.forward(
        embedding=embedding,
        duration=dur.unsqueeze(-1),
        event=ev.unsqueeze(-1),
        cohort=co.unsqueeze(-1).float(),
    )
    expected = cox_loss(out["logit"].squeeze(-1), dur, ev, co)
    assert torch.allclose(out["loss"], expected)