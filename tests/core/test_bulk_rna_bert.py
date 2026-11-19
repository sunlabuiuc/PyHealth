import numpy as np
import pytest
import torch

from pyhealth.models import BulkRNABert
from pyhealth.models.bulk_rna_bert import SurvivalHead, cox_loss
from pyhealth.models.trajsurv import cox_partial_likelihood_loss


def test_head_shape():
    assert SurvivalHead()(torch.zeros(3, 256)).shape == (3,)


def test_forward_cached_mode():
    model = BulkRNABert()
    out = model(embedding=torch.randn(4, 256))
    assert out["logit"].shape == (4, 1)
    assert out["risk_score"].shape == (4, 1)
    assert out["y_prob"].shape == (4, 1)
    assert torch.isfinite(out["logit"]).all()


def test_forward_rejects_wrong_embedding_shape():
    model = BulkRNABert()
    with pytest.raises(ValueError, match="256"):
        model(embedding=torch.randn(4, 128))


def test_forward_with_labels_returns_loss():
    model = BulkRNABert()
    out = model(
        embedding=torch.randn(4, 256),
        duration=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
    )
    assert "loss" in out
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_gradients_flow_to_head():
    model = BulkRNABert()
    out = model(
        embedding=torch.randn(4, 256),
        duration=torch.tensor([[100.0], [200.0], [150.0], [50.0]]),
        event=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
    )
    out["loss"].backward()
    assert all(p.grad is not None for p in model.head.parameters())


def test_cox_loss_matches_trajsurv():
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
        mask = group == g
        r, t, e = risk[mask], duration[mask], event[mask]
        terms += [
            r[i] - np.log(np.exp(r[t >= t[i]]).sum())
            for i in range(len(t))
            if e[i]
        ]

    got = cox_loss(
        torch.tensor(risk),
        torch.tensor(duration),
        torch.tensor(event),
        torch.tensor(group),
    ).item()
    assert abs(got - (-np.mean(terms))) < 1e-4


def test_cox_loss_no_events():
    risk = torch.randn(10)
    duration = torch.rand(10) * 100
    event = torch.zeros(10)
    loss = cox_loss(risk, duration, event)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_global_loss_ignores_cohort():
    model = BulkRNABert(cox_loss_mode="global")
    embedding = torch.randn(8, 256)
    duration = torch.arange(1, 9).float()
    event = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1]).float()
    cohort = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0]).float()

    with_cohort = model(
        embedding=embedding, duration=duration, event=event, cohort=cohort
    )
    without_cohort = model(
        embedding=embedding, duration=duration, event=event
    )
    assert torch.allclose(with_cohort["loss"], without_cohort["loss"])


def test_cohort_loss_differs_from_global():
    torch.manual_seed(0)
    m_global = BulkRNABert(cox_loss_mode="global")
    torch.manual_seed(0)
    m_cohort = BulkRNABert(cox_loss_mode="cohort")

    embedding = torch.randn(30, 256)
    duration = torch.arange(1, 31).float()
    event = torch.randint(0, 2, (30,)).float()
    cohort = torch.tensor([0] * 20 + [1] * 8 + [2] * 2)

    with torch.no_grad():
        l_global = m_global(
            embedding=embedding,
            duration=duration,
            event=event,
            cohort=cohort,
        )["loss"]
        l_cohort = m_cohort(
            embedding=embedding,
            duration=duration,
            event=event,
            cohort=cohort,
        )["loss"]
    assert not torch.allclose(l_global, l_cohort)


def test_cox_loss_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="global.*cohort|cohort.*global"):
        BulkRNABert(cox_loss_mode="bogus")


def test_cox_loss_mode_cohort_without_labels_raises():
    model = BulkRNABert(cox_loss_mode="cohort")
    with pytest.raises(ValueError, match="cohort"):
        model(
            embedding=torch.randn(4, 256),
            duration=torch.tensor([[1.0]] * 4),
            event=torch.tensor([[1.0]] * 4),
        )
