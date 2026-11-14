"""Fast, deterministic synthetic tests for the PyHealth CET model."""

from __future__ import annotations

import math

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CET


def _dataset(n_events: int = 2):
    samples = []
    for i in range(4):
        samples.append({
            "patient_id": f"synthetic_{i}",
            "features": [float(i) / 4, 0.2 - i / 5, 0.5, 1.0],
            "duration": [0.5 + 0.25 * i + 0.1 * j for j in range(n_events)],
            "event": [float((i + j) % 2) for j in range(n_events)],
        })
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="cet_synthetic_test",
        in_memory=True,
    )


def _batch(dataset):
    return next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))


def test_init_and_forward_shapes():
    torch.manual_seed(3)
    ds = _dataset(n_events=3)
    model = CET(ds, hidden_dim=8, n_samples=5, dropout=0)
    result = model(**_batch(ds))
    assert model.input_dim == 4 and model.n_events == 3
    assert model.mode is None
    assert result["loss"].ndim == 0
    assert result["logit"].shape == (4, 3)
    assert result["y_prob"].shape == (4, 3)
    assert result["y_true"].shape == (4, 3)
    assert result["time_pred"].shape == (4, 3)
    assert torch.isfinite(result["loss"])
    assert torch.all((result["y_prob"] > 0) & (result["y_prob"] < 1))
    assert torch.all(result["time_pred"] > 0)


def test_one_target_and_unlabeled_inference():
    ds = _dataset(n_events=1)
    model = CET(ds, hidden_dim=8, n_samples=4)
    out = model(**_batch(ds))
    assert out["y_prob"].shape == (4, 1)
    only_features = model(features=torch.rand(2, 4))
    assert set(only_features) == {"logit", "y_prob", "time_pred"}


def test_finite_backprop_to_both_subnetworks():
    torch.manual_seed(9)
    ds = _dataset()
    model = CET(ds, hidden_dim=8, n_samples=8, dropout=0)
    result = model(**_batch(ds))
    result["loss"].backward()
    for layer in (model.encoder[0], model.occurrence_head,
                  model.decoder[0], model.time_mean, model.time_logvar):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()


def test_log_normal_terms_against_analytic_normal():
    duration = torch.tensor([[[0.5], [1.0], [2.0]]])
    mu = torch.zeros_like(duration)
    logvar = torch.zeros_like(duration)
    actual_pdf, actual_surv = CET.lognormal_negative_terms(duration, mu, logvar)
    standard = torch.distributions.Normal(0, 1)
    log_time = torch.log(duration + 1e-4)
    expected_pdf = -(standard.log_prob(log_time) - log_time)
    expected_surv = -torch.special.log_ndtr(-log_time)
    assert torch.allclose(actual_pdf, expected_pdf, atol=1e-6)
    assert torch.allclose(actual_surv, expected_surv, atol=1e-6)


def test_censored_likelihood_terms_by_hand():
    model = CET(_dataset(n_events=1), hidden_dim=4, n_samples=2, fpr=0.7)
    logits = torch.zeros(2, 1)
    c = torch.tensor([[[0.25], [0.75]], [[0.25], [0.75]]])
    mu = torch.zeros(2, 2, 1)
    logvar = torch.zeros_like(mu)
    duration = torch.ones(2, 1)
    event = torch.tensor([[1.], [0.]])
    observed_pdf, censored_surv = CET.lognormal_negative_terms(
        duration.unsqueeze(1), mu, logvar,
    )
    penalty = -math.log(0.7) + math.log(2)
    expected = torch.cat((
        observed_pdf[:1] + (1 - c[:1]) * penalty,
        c[1:] * censored_surv[1:],
    ), dim=0).mean()
    actual = model.censored_nll(logits, c, mu, logvar, duration, event)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_shared_sample_noise_and_reproducible_seed():
    torch.manual_seed(8)
    model = CET(_dataset(), hidden_dim=4, n_samples=3)
    logits = torch.tensor([[0.1, 0.5], [0.1, 0.5]])
    torch.manual_seed(12)
    first = model.sample_occurrence(logits, 3)
    torch.manual_seed(12)
    second = model.sample_occurrence(logits, 3)
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])
    assert first.shape == (2, 3, 2)


def test_timing_is_conditioned_on_occurrence_vector():
    model = CET(_dataset(), hidden_dim=4, n_samples=3, dropout=0)
    model.eval()
    with torch.no_grad():
        linear = model.decoder[0]
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, model.input_dim] = 2.0
        model.time_mean.weight.zero_()
        model.time_mean.bias.zero_()
        model.time_mean.weight[0, 0] = 1.0
    x = torch.zeros(2, 4)
    c0 = torch.zeros(2, 3, 2)
    c1 = c0.clone()
    c1[..., 0] = 1
    mu0, _ = model.conditional_time_params(x, c0)
    mu1, _ = model.conditional_time_params(x, c1)
    assert torch.all(mu1[..., 0] > mu0[..., 0])


@pytest.mark.parametrize("config", [
    {"hidden_dim": 0}, {"n_samples": 0}, {"temperature": 0},
    {"dropout": 1}, {"fpr": 0}, {"n_events": 0},
    {"logvar_bounds": (5, 2)},
])
def test_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        CET(_dataset(), **config)


def test_rejects_bad_inputs():
    model = CET(_dataset(), hidden_dim=4, n_samples=2)
    batch = _batch(_dataset())
    with pytest.raises(KeyError):
        model(duration=batch["duration"], event=batch["event"])
    with pytest.raises(KeyError):
        model(features=batch["features"], duration=batch["duration"])
    with pytest.raises(ValueError, match="shape"):
        model(features=torch.ones(4, 5))
    with pytest.raises(ValueError, match="positive"):
        model(features=batch["features"], duration=torch.zeros(4, 2),
              event=batch["event"])
    with pytest.raises(ValueError, match="0/1"):
        model(features=batch["features"], duration=batch["duration"],
              event=torch.full((4, 2), 2.))
    with pytest.raises(ValueError, match="finite"):
        model(features=torch.full((4, 4), float("nan")))


def test_state_dict_roundtrip_and_prediction_mode(tmp_path):
    torch.manual_seed(22)
    ds = _dataset()
    model = CET(ds, hidden_dim=8, n_samples=7, dropout=0)
    model.eval()
    batch = _batch(ds)
    torch.manual_seed(2)
    original = model(**batch)
    checkpoint = tmp_path / "cet.pt"
    torch.save(model.state_dict(), checkpoint)
    recovered = CET(ds, hidden_dim=8, n_samples=7, dropout=0)
    recovered.load_state_dict(torch.load(checkpoint, weights_only=True))
    recovered.eval()
    torch.manual_seed(2)
    result = recovered(**batch)
    assert torch.equal(original["y_prob"], result["y_prob"])
    assert torch.equal(original["time_pred"], result["time_pred"])
    model.train()
    times = model.predict_event_time(batch["features"], n_samples=5)
    assert times.shape == (4, 2)
    assert model.training
    observed = model.posterior_occurrence(
        batch["features"], batch["duration"], batch["event"], 5,
    )
    assert torch.all((observed >= 0) & (observed <= 1))
    assert torch.all(observed[batch["event"].bool()] == 1)
    assert model.training


def test_one_optimizer_step():
    ds = _dataset()
    model = CET(ds, hidden_dim=8, n_samples=5, dropout=0)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.occurrence_head.weight.detach().clone()
    optim.zero_grad()
    model(**_batch(ds))["loss"].backward()
    optim.step()
    assert not torch.equal(before, model.occurrence_head.weight)


def test_forward_from_embedding_matches_forward():
    """Embedding entry point agrees with forward under matched RNG state."""
    torch.manual_seed(41)
    ds = _dataset()
    model = CET(ds, hidden_dim=8, n_samples=6, dropout=0)
    model.eval()
    batch = _batch(ds)
    torch.manual_seed(17)
    expected = model(**batch)
    torch.manual_seed(17)
    actual = model.forward_from_embedding(
        batch["features"], duration=batch["duration"], event=batch["event"]
    )
    for key in ("logit", "y_prob", "time_pred", "loss", "y_true"):
        assert torch.allclose(actual[key], expected[key], atol=1e-7)


def test_predict_event_time_known_constant_median():
    """Predicted conditional median matches a fixed log-normal model."""
    model = CET(_dataset(), hidden_dim=8, n_samples=5, dropout=0)
    with torch.no_grad():
        model.time_mean.weight.zero_()
        model.time_mean.bias.fill_(math.log(3.5))
    features = torch.zeros(3, model.input_dim)
    predicted = model.predict_event_time(features, n_samples=7)
    assert predicted.shape == (3, 2)
    assert torch.allclose(predicted, torch.full_like(predicted, 3.5), atol=1e-5)


def test_posterior_occurrence_matches_constant_time_case():
    """Posterior is one for observed events and follows Bayes for censoring."""
    model = CET(_dataset(), hidden_dim=8, n_samples=5, dropout=0)
    model.eval()
    with torch.no_grad():
        model.occurrence_head.weight.zero_()
        model.occurrence_head.bias.fill_(math.log(0.25 / 0.75))
        model.time_mean.weight.zero_()
        model.time_mean.bias.zero_()
        model.time_logvar.weight.zero_()
        model.time_logvar.bias.zero_()
    x = torch.zeros(2, model.input_dim)
    t = torch.ones(2, model.n_events)
    observed = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    estimated = model.posterior_occurrence(x, t, observed, n_samples=5)
    _, neg_log_surv = CET.lognormal_negative_terms(
        torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)
    )
    surv = torch.exp(-neg_log_surv).item()
    expected_censored = (0.25 * surv) / (0.25 * surv + 0.75)
    assert torch.all(estimated[observed.bool()] == 1.0)
    assert torch.allclose(
        estimated[~observed.bool()],
        torch.full_like(estimated[~observed.bool()], expected_censored),
        atol=1e-6,
    )


def test_sample_median_matches_quantile():
    torch.manual_seed(42)

    for n_samples in (1, 7, 8, 32, 100):
        mu = torch.randn(5, n_samples, 3)

        expected = torch.quantile(
            torch.exp(mu.clamp(max=40)),
            0.5,
            dim=1,
        )
        actual = CET._sample_median_time(mu)

        torch.testing.assert_close(
            actual, expected, rtol=1e-6, atol=1e-6
        )


def test_sample_median_large_batch():
    # More than 2**24 total elements.
    # 16,778 patients, 100 samples, 10 events.
    with torch.no_grad():
        mu = torch.zeros((16_778, 100, 10))
        result = CET._sample_median_time(mu)

    assert result.shape == (16_778, 10)
    torch.testing.assert_close(
        result, torch.ones_like(result)
    )