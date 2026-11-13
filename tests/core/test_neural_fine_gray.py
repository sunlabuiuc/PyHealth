"""Fast synthetic tests for :class:`pyhealth.models.NeuralFineGray`."""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.neural_fine_gray import NeuralFineGray, PositiveLinear


def _make_dataset():
    samples = [
        {
            "patient_id": "p0",
            "features": [0.0, 1.0, -0.5, 0.2],
            "duration": [0.8],
            "event": [1],
        },
        {
            "patient_id": "p1",
            "features": [1.0, 0.2, 0.4, -0.1],
            "duration": [1.5],
            "event": [0],
        },
        {
            "patient_id": "p2",
            "features": [-0.5, 0.7, 0.1, 1.2],
            "duration": [2.2],
            "event": [2],
        },
        {
            "patient_id": "p3",
            "features": [0.3, -0.4, 1.0, 0.6],
            "duration": [3.1],
            "event": [0],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="neural_fine_gray_test",
    )


def _batch(dataset):
    return next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))


def _model(**kwargs):
    return NeuralFineGray(
        _make_dataset(),
        num_risks=2,
        hidden_dims=(8,),
        monotonic_hidden_dims=(8,),
        dropout=0.0,
        time_grid=torch.linspace(0.1, 4.0, 9),
        **kwargs,
    )


@pytest.mark.parametrize("transform", ["square", "softplus", "exp"])
def test_positive_linear_has_nonnegative_weights(transform):
    layer = PositiveLinear(3, 2, weight_transform=transform)
    assert torch.all(layer.weight >= 0)
    output = layer(torch.randn(5, 3)).sum()
    output.backward()
    assert layer.raw_weight.grad is not None
    assert torch.isfinite(layer.raw_weight.grad).all()


def test_forward_shapes_and_probability_constraints():
    model = _model()
    model.eval()
    output = model(**_batch(model.dataset))
    assert output["loss"].ndim == 0
    assert output["logit"].shape == (4, 2)
    assert output["cif"].shape == (4, 2, 9)
    assert output["survival"].shape == (4, 9)
    assert output["y_true"].shape == (4, 2)
    assert output["hazard_derivative"].shape == (4, 2)
    assert torch.all((output["cif"] >= 0) & (output["cif"] <= 1))
    assert torch.all(output["cif"].sum(dim=1) <= 1 + 1e-6)
    assert torch.all((output["survival"] >= 0) & (output["survival"] <= 1))
    assert torch.allclose(output["balance"].sum(dim=1), torch.ones(4))


def test_cif_is_zero_at_origin_and_monotonic_in_time():
    model = _model()
    features = torch.randn(6, 4)
    times = torch.tensor([0.0, 0.1, 0.4, 1.0, 3.0])
    cif = model.predict_cif(features, times)
    assert torch.allclose(cif[:, :, 0], torch.zeros(6, 2), atol=1e-7)
    assert torch.all(cif[:, :, 1:] >= cif[:, :, :-1] - 1e-7)


def test_sumonet_ablation_can_have_nonzero_cif_at_origin():
    model = _model(cdf_mode="sumonet")
    cif = model.predict_cif(torch.randn(3, 4), torch.tensor([0.0]))
    assert torch.all(cif > 0)
    assert torch.all(cif.sum(dim=1) <= 1 + 1e-6)


def test_autograd_hazard_derivative_matches_finite_difference():
    model = _model()
    model.eval()
    batch = _batch(model.dataset)
    output = model(**batch)
    features = batch["features"]
    if isinstance(features, tuple):
        features = features[0]
    duration = batch["duration"].reshape(-1).float()
    with torch.no_grad():
        embedding, _, _ = model.network.encode(features.float())
        epsilon = 1e-3
        plus = model.network.cumulative_hazard(
            embedding, duration + epsilon, model.cdf_mode
        )
        minus = model.network.cumulative_hazard(
            embedding, duration - epsilon, model.cdf_mode
        )
        finite_difference = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(
        output["hazard_derivative"], finite_difference, rtol=2e-3, atol=2e-3
    )


def test_gradient_computation_is_finite():
    model = _model()
    model.train()
    output = model(**_batch(model.dataset))
    output["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("time_scale", [1.0, 2.0])
def test_exact_likelihood_against_constant_hazard_reference(time_scale):
    model = _model(time_scale=time_scale)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    batch = _batch(model.dataset)
    output = model(**batch)
    duration = batch["duration"].reshape(-1).float()
    event = batch["event"].reshape(-1).long()
    rate = torch.tensor(2.0).log() / time_scale
    log_likelihood = -duration * rate
    log_likelihood[event > 0] += torch.log(torch.tensor(0.5) * rate)
    expected = -log_likelihood.mean()
    torch.testing.assert_close(output["loss"], expected)


def test_state_roundtrip_and_prediction_restores_mode(tmp_path):
    model = _model()
    model.train()
    previous_modes = [module.training for module in model.modules()]
    prediction = model.predict_cif(torch.ones(2, 4), torch.tensor([0.5, 1.0]))
    assert [module.training for module in model.modules()] == previous_modes
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = _model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(
        prediction,
        restored.predict_cif(torch.ones(2, 4), torch.tensor([0.5, 1.0])),
    )


@pytest.mark.parametrize(
    "config",
    [
        {"num_risks": 0},
        {"cdf_mode": "invalid"},
        {"weight_transform": "invalid"},
        {"monotonic_activation": "relu"},
        {"dropout": 1.0},
        {"min_density": 0.0},
        {"time_scale": 0.0},
        {"time_grid": torch.tensor([])},
        {"time_grid": torch.tensor([float("nan")])},
    ],
)
def test_invalid_configuration(config):
    defaults = {
        "dataset": _make_dataset(),
        "num_risks": 2,
        "hidden_dims": (4,),
        "monotonic_hidden_dims": (4,),
    }
    defaults.update(config)
    with pytest.raises(ValueError):
        NeuralFineGray(**defaults)


@pytest.mark.parametrize("field", ["features", "duration", "event"])
def test_nonfinite_inputs_are_rejected(field):
    model = _model()
    batch = {
        "features": torch.ones(3, 4),
        "duration": torch.ones(3),
        "event": torch.tensor([0.0, 1.0, 2.0]),
    }
    batch[field].flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        model(**batch)


def test_bad_causes_and_shapes_are_rejected():
    model = _model()
    with pytest.raises(KeyError, match="Missing"):
        model(features=torch.ones(2, 4))
    with pytest.raises(ValueError, match="shape"):
        model(
            features=torch.ones(2, 3),
            duration=torch.ones(2),
            event=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="between"):
        model(
            features=torch.ones(2, 4),
            duration=torch.ones(2),
            event=torch.tensor([0, 3]),
        )
    with pytest.raises(ValueError, match="integer"):
        model(
            features=torch.ones(2, 4),
            duration=torch.ones(2),
            event=torch.tensor([0.0, 1.5]),
        )
