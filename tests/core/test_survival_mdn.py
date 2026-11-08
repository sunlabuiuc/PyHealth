# Contributor: Neil Hajela (nhajela2@illinois.edu)
"""Fast synthetic tests for :class:`pyhealth.models.SurvivalMDN`."""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.survival_mdn import SurvivalMDN


def _make_dataset():
    samples = [
        {
            "patient_id": "p0",
            "features": [0.0, 1.0, -0.5, 0.2],
            "duration": [0.8],
            "event": [1.0],
        },
        {
            "patient_id": "p1",
            "features": [1.0, 0.2, 0.4, -0.1],
            "duration": [1.5],
            "event": [0.0],
        },
        {
            "patient_id": "p2",
            "features": [-0.5, 0.7, 0.1, 1.2],
            "duration": [2.2],
            "event": [1.0],
        },
        {
            "patient_id": "p3",
            "features": [0.3, -0.4, 1.0, 0.6],
            "duration": [3.1],
            "event": [0.0],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="survival_mdn_test",
    )


def _batch(dataset):
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    return next(iter(loader))


def test_instantiation_and_configuration():
    dataset = _make_dataset()
    model = SurvivalMDN(
        dataset,
        hidden_dim=8,
        num_components=3,
        time_grid=torch.linspace(0.1, 3.0, 11),
    )
    assert model.input_dim == 4
    assert model.num_components == 3
    assert model.mode is None
    assert model.time_grid.shape == (11,)


def test_forward_shapes_and_distribution_constraints():
    dataset = _make_dataset()
    model = SurvivalMDN(
        dataset,
        hidden_dim=8,
        num_components=3,
        time_grid=torch.linspace(0.1, 3.0, 11),
    )
    model.eval()
    output = model(**_batch(dataset))

    assert output["loss"].ndim == 0
    assert output["logit"].shape == (4, 9)
    assert output["y_prob"].shape == (4, 11)
    assert output["y_true"].shape == (4, 2)
    assert output["mixture_weights"].shape == (4, 3)
    assert output["mixture_means"].shape == (4, 3)
    assert output["mixture_scales"].shape == (4, 3)
    assert torch.allclose(
        output["mixture_weights"].sum(dim=-1),
        torch.ones(4),
        atol=1e-6,
    )
    assert torch.all(output["mixture_scales"] > 0)
    assert torch.all((output["y_prob"] >= 0) & (output["y_prob"] <= 1))
    assert torch.all(output["y_prob"][:, 1:] <= output["y_prob"][:, :-1])


def test_gradient_computation_is_finite():
    dataset = _make_dataset()
    model = SurvivalMDN(dataset, hidden_dim=8, num_components=3)
    model.train()
    output = model(**_batch(dataset))
    output["loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_exp_transform_produces_valid_survival_curves():
    dataset = _make_dataset()
    model = SurvivalMDN(
        dataset,
        hidden_dim=8,
        num_components=3,
        time_transform="exp",
        time_grid=torch.linspace(0.1, 3.0, 11),
    )
    model.eval()
    output = model(**_batch(dataset))
    assert torch.isfinite(output["loss"])
    assert torch.all(output["y_prob"][:, 1:] <= output["y_prob"][:, :-1])


def test_invalid_event_and_duration_are_rejected():
    dataset = _make_dataset()
    model = SurvivalMDN(dataset, hidden_dim=8, num_components=3)
    model.eval()
    batch = _batch(dataset)

    bad_event = dict(batch)
    bad_event["event"] = torch.tensor([[0.0], [1.0], [2.0], [0.0]])
    with pytest.raises(ValueError, match="event"):
        model(**bad_event)

    bad_duration = dict(batch)
    bad_duration["duration"] = torch.tensor([[0.8], [0.0], [2.2], [3.1]])
    with pytest.raises(ValueError, match="positive"):
        model(**bad_duration)


@pytest.mark.parametrize("transform", ["softplus", "exp"])
@pytest.mark.parametrize("observed", [0.0, 1.0])
def test_analytical_likelihood(transform, observed):
    """Check event/censor terms and Jacobian against an independent Normal."""
    from pyhealth.models.survival_mdn import _MDNParameters

    model = SurvivalMDN(
        _make_dataset(), hidden_dim=4, num_components=1, time_transform=transform
    )
    t = torch.tensor([0.5, 1.5])
    zeros = torch.zeros(2, 1)
    params = _MDNParameters(zeros, zeros, torch.ones(2, 1), zeros, zeros)
    z = torch.log(torch.expm1(t)) if transform == "softplus" else torch.log(t)
    jac = -torch.log1p(-torch.exp(-t)) if transform == "softplus" else -torch.log(t)
    normal = torch.distributions.Normal(0.0, 1.0)
    expected = -(normal.log_prob(z) + jac if observed else torch.log(normal.cdf(-z)))
    actual = model.network.censored_nll(
        params, t, torch.full((2,), observed), reduction="none"
    )
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("field", ["duration", "event", "features"])
def test_nonfinite_input(field):
    model = SurvivalMDN(_make_dataset(), hidden_dim=4, num_components=2)
    model.eval()
    batch = {
        "features": torch.ones(2, 4),
        "duration": torch.ones(2),
        "event": torch.ones(2),
    }
    batch[field].flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        model(**batch)


@pytest.mark.parametrize(
    "config",
    [
        {"hidden_dim": 0},
        {"num_components": 0},
        {"num_hidden_layers": 0},
        {"min_scale": float("nan")},
        {"time_transform": "invalid"},
        {"time_grid": torch.tensor([])},
        {"time_grid": torch.tensor([float("inf")])},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        SurvivalMDN(_make_dataset(), **config)


def test_state_roundtrip_and_embedding(tmp_path):
    ds = _make_dataset()
    model = SurvivalMDN(ds, hidden_dim=4, num_components=2)
    model.eval()
    batch = _batch(ds)
    expected = model(**batch)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = SurvivalMDN(ds, hidden_dim=4, num_components=2)
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    assert torch.equal(expected["y_prob"], restored(**batch)["y_prob"])
    x = batch["features"]
    if isinstance(x, tuple):
        x = x[0]
    assert torch.equal(
        model.forward_from_embedding(x, **batch)["loss"], expected["loss"]
    )
    with torch.no_grad():
        assert torch.equal(
            model.predict_survival(x, model.time_grid), expected["y_prob"]
        )


def test_missing_and_mismatched_fields():
    model = SurvivalMDN(_make_dataset(), hidden_dim=4, num_components=2)
    model.eval()
    with pytest.raises(KeyError, match="Missing"):
        model(features=torch.ones(2, 4))
    with pytest.raises(ValueError, match="shape"):
        model(features=torch.ones(2, 3), duration=torch.ones(2), event=torch.ones(2))
    with pytest.raises(ValueError, match="batch size"):
        model(features=torch.ones(2, 4), duration=torch.ones(3), event=torch.ones(2))


@pytest.mark.parametrize("transform", ["softplus", "exp"])
def test_multicomponent_distribution_against_scalar_reference(transform):
    """Check weighted mixture math and grid broadcasting independently."""
    import math

    from pyhealth.models.survival_mdn import (
        _MDNParameters,
        _SurvivalMDNDistribution,
    )

    model = SurvivalMDN(
        _make_dataset(), hidden_dim=4, num_components=3, time_transform=transform
    )
    weights = torch.tensor(
        [[0.15, 0.25, 0.60], [0.55, 0.35, 0.10]], dtype=torch.float64
    )
    means = torch.tensor([[-1.2, 0.3, 1.8], [0.8, -0.4, 1.5]], dtype=torch.float64)
    scales = torch.tensor([[0.7, 1.3, 0.9], [1.1, 0.6, 1.7]], dtype=torch.float64)
    params = _MDNParameters(weights.log(), means, scales, weights.log(), scales)
    distribution = _SurvivalMDNDistribution(params, model.network.time_transform)

    def reference(row, time):
        latent = (
            math.log(math.expm1(time)) if transform == "softplus" else math.log(time)
        )
        jacobian = 1 / (-math.expm1(-time)) if transform == "softplus" else 1 / time
        density, cdf, survival = 0.0, 0.0, 0.0
        for weight, mean, scale in zip(weights[row], means[row], scales[row]):
            weight, mean, scale = float(weight), float(mean), float(scale)
            z = (latent - mean) / scale
            density += weight * math.exp(-z * z / 2) / (scale * math.sqrt(2 * math.pi))
            cdf += weight * math.erfc(-z / math.sqrt(2)) / 2
            survival += weight * math.erfc(z / math.sqrt(2)) / 2
        return density * jacobian, cdf, survival

    durations = torch.tensor([0.45, 2.3], dtype=torch.float64)
    expected = torch.tensor(
        [reference(row, float(time)) for row, time in enumerate(durations)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(distribution.log_prob(durations), expected[:, 0].log())
    torch.testing.assert_close(distribution.log_cdf(durations), expected[:, 1].log())
    torch.testing.assert_close(
        distribution.log_survival(durations), expected[:, 2].log()
    )
    for events in (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])):
        expected_nll = -torch.where(events.bool(), expected[:, 0], expected[:, 2]).log()
        actual = model.network.censored_nll(params, durations, events, reduction="none")
        torch.testing.assert_close(actual, expected_nll)

    times = torch.tensor([0.05, 0.4, 1.5, 4.0, 9.0], dtype=torch.float64)
    expected_grid = torch.tensor(
        [[reference(row, float(time))[2] for time in times] for row in range(2)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(model.network.survival_at(params, times), expected_grid)


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_predict_survival_is_inference_only_and_restores_modes(mode):
    model = SurvivalMDN(
        _make_dataset(),
        hidden_dim=4,
        num_components=3,
        reference_initialization=False,
    )
    model.train(mode != "eval")
    batch_norm = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
    if mode == "mixed":
        batch_norm.eval()
    previous_modes = [module.training for module in model.modules()]
    previous_state = {key: value.clone() for key, value in model.state_dict().items()}
    features = torch.tensor(
        [[0.0, 1.0, -0.5, 0.2], [4.0, -2.0, 3.0, 1.0]], requires_grad=True
    )
    prediction = model.predict_survival(features, model.time_grid)
    assert not prediction.requires_grad
    assert [module.training for module in model.modules()] == previous_modes
    for key, value in model.state_dict().items():
        assert torch.equal(value, previous_state[key]), key

    # A single-patient call must work and agree with the same patient's batch result.
    single = model.predict_survival(features[:1], model.time_grid)
    torch.testing.assert_close(single[0], prediction[0])
    assert [module.training for module in model.modules()] == previous_modes
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_predict_survival_restores_modes_after_error(mode):
    model = SurvivalMDN(_make_dataset(), hidden_dim=4, num_components=2)
    model.train(mode != "eval")
    if mode == "mixed":
        next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)).eval()
    previous_modes = [module.training for module in model.modules()]
    previous_state = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="positive"):
        model.predict_survival(torch.ones(2, 4), torch.tensor([0.0]))
    assert [module.training for module in model.modules()] == previous_modes
    for key, value in model.state_dict().items():
        assert torch.equal(value, previous_state[key]), key
    assert torch.is_grad_enabled()
