# Contributors: Jenny Nguyen (lmn4@illinois.edu),
#   Jordan Alwan (jzalwan2@illinois.edu), Karthik Pullela (kpull2@illinois.edu)
"""Fast synthetic tests for :class:`pyhealth.models.DeepCoxMixturesScriptEM`.

All data here is synthetic; no real or demo dataset is used.
"""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.deep_cox_mixtures_script_em import (
    DeepCoxMixturesScriptEM,
    DeepCoxMixturesTorch,
)


def _make_dataset(n: int = 8):
    torch.manual_seed(0)
    samples = [
        {
            "patient_id": f"p{i}",
            "features": [float(i % 3), float(i) / n, -0.5 + 0.1 * i],
            "duration": [1.0 + float(i)],
            "event": [1.0 if i % 2 == 0 else 0.0],
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="deep_cox_mixtures_test",
    )


def _batch(dataset, batch_size: int = 8):
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _tensors(batch):
    def _as_tensor(value):
        return value[0] if isinstance(value, tuple) else value

    return (
        _as_tensor(batch["features"]).float(),
        _as_tensor(batch["duration"]).float().reshape(-1),
        _as_tensor(batch["event"]).float().reshape(-1),
    )


def test_instantiation_and_configuration():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(
        dataset,
        k=3,
        layers=[4],
        gamma=10.0,
        time_grid=torch.linspace(1.0, 8.0, 5),
    )
    assert model.input_dim == 3
    assert model.k == 3
    assert model.mode is None
    assert model.time_grid.shape == (5,)
    assert model.breslow_splines is None


def test_network_shapes_gate_sums_and_hazard_clamp():
    net = DeepCoxMixturesTorch(input_dim=3, k=3, gamma=2.0, layers=[4])
    log_gate, log_hazards = net(torch.randn(6, 3))
    assert log_gate.shape == (6, 3)
    assert log_hazards.shape == (6, 3)
    assert torch.allclose(log_gate.exp().sum(dim=1), torch.ones(6), atol=1e-5)
    assert torch.all(log_hazards <= 2.0)
    assert torch.all(log_hazards >= -2.0)


def test_linear_encoder_when_layers_empty():
    net = DeepCoxMixturesTorch(input_dim=3, k=2, layers=[])
    log_gate, log_hazards = net(torch.randn(4, 3))
    assert log_gate.shape == (4, 2)
    assert log_hazards.shape == (4, 2)


def test_forward_requires_fitted_baselines():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
    with pytest.raises(RuntimeError, match="No baselines fitted"):
        model(**_batch(dataset))


def test_fit_baselines_predict_survival_and_forward():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(
        dataset,
        k=2,
        layers=[4],
        time_grid=torch.linspace(1.0, 8.0, 5),
    )
    model.eval()
    features, durations, events = _tensors(_batch(dataset))
    splines = model.fit_baselines(features, durations, events)
    assert set(splines) == {0, 1}
    assert model.breslow_splines is not None

    times = torch.tensor([1.0, 4.0, 8.0])
    survival = model.predict_survival(features, times)
    assert survival.shape == (8, 3)
    assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0)
    # Survival curves are non-increasing in time.
    assert torch.all(survival[:, 1:] <= survival[:, :-1] + 1e-6)

    output = model(**_batch(dataset))
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    assert output["logit"].shape == (8, 2)
    assert output["y_prob"].shape == (8, 5)
    assert output["y_true"].shape == (8, 2)
    assert output["log_gate_prob"].shape == (8, 2)
    assert output["log_hazard_ratios"].shape == (8, 2)


def test_gate_probabilities_sum_to_one():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(dataset, k=3, layers=[4])
    probs = model.predict_latent_z(torch.randn(5, 3))
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)


def test_posteriors_sum_to_one():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
    features, durations, events = _tensors(_batch(dataset))
    model.fit_baselines(features, durations, events)
    posteriors = model.compute_posteriors(features, durations, events)
    assert posteriors.shape == (8, 2)
    assert torch.allclose(posteriors.exp().sum(dim=1), torch.ones(8), atol=1e-5)


def test_q_loss_and_soft_q_loss_have_gradients():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
    features, durations, events = _tensors(_batch(dataset))
    posteriors = torch.log(torch.full((8, 2), 0.5))
    for loss in (
        model.q_loss(features, durations, events, posteriors),
        model.q_loss(features, durations, events, posteriors, typ="hard"),
        model.soft_q_loss(features, durations, events, posteriors),
    ):
        assert torch.isfinite(loss)
        model.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "expected gradients to flow to the network"
        assert all(torch.isfinite(g).all() for g in grads)


def test_empty_group_gets_degenerate_spline():
    dataset = _make_dataset()
    model = DeepCoxMixturesScriptEM(dataset, k=3, layers=[4])
    features, durations, events = _tensors(_batch(dataset))
    # Argmax posteriors that assign every patient to group 0.
    posteriors = torch.tensor([[0.0, -100.0, -100.0]] * 8, dtype=torch.float32)
    splines = model.fit_baselines(
        features, durations, events, posteriors=posteriors, typ="hard"
    )
    assert set(splines) == {0, 1, 2}
    survival = model.predict_survival(features, torch.tensor([1.0, 4.0, 8.0]))
    assert survival.shape == (8, 3)
    assert torch.all(torch.isfinite(survival))
