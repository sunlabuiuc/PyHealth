"""Unit tests for Deep Cox Mixtures."""

import numpy as np
import pytest
import torch

from pyhealth.models import DeepCoxMixtures
from pyhealth.models.deep_cox_mixtures import (
    BreslowBaseline,
    cox_partial_loss,
)


@pytest.fixture
def synthetic_data():
    """Create a deterministic synthetic survival dataset."""
    torch.manual_seed(42)

    features = torch.randn(10, 4)

    duration = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
        ]
    )

    event = torch.tensor(
        [
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
        ]
    )

    return features, duration, event


@pytest.fixture
def model():
    """Initialize a lightweight DCM model."""
    torch.manual_seed(42)

    return DeepCoxMixtures(
        dataset=None,
        input_dim=4,
        num_components=3,
        hidden_dims=(8,),
    )


def test_model_initialization(model):
    """Verify model configuration."""
    assert model.input_dim == 4
    assert model.num_components == 3
    assert model.mode is None


def test_forward_shapes(model, synthetic_data):
    """Verify output tensor dimensions."""
    features, _, _ = synthetic_data

    output = model(features)

    assert output["log_pi"].shape == (10, 3)
    assert output["log_risk"].shape == (10, 3)
    assert output["mixture_probs"].shape == (10, 3)


def test_mixture_probabilities(model, synthetic_data):
    """Verify that mixture probabilities sum to one."""
    features, _, _ = synthetic_data

    probabilities = model(features)["mixture_probs"]

    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(len(features)),
    )


def test_cox_loss():
    """Verify Cox loss against an analytical calculation."""
    log_risk = torch.zeros(3)

    duration = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([1.0, 1.0, 1.0])

    loss = cox_partial_loss(
        log_risk,
        duration,
        event,
    )

    expected = (np.log(3) + np.log(2) + np.log(1)) / 3

    assert loss.item() == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_cox_loss_tied_times():
    """Verify Breslow handling of tied event times."""
    log_risk = torch.zeros(3)

    duration = torch.tensor([1.0, 1.0, 2.0])
    event = torch.tensor([1.0, 1.0, 1.0])

    loss = cox_partial_loss(
        log_risk,
        duration,
        event,
    )

    expected = 2 * np.log(3) / 3

    assert loss.item() == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_cox_loss_all_censored():
    """Verify behavior when no events are observed."""
    log_risk = torch.zeros(
        3,
        requires_grad=True,
    )

    duration = torch.tensor([1.0, 2.0, 3.0])
    event = torch.zeros(3)

    loss = cox_partial_loss(
        log_risk,
        duration,
        event,
    )

    assert loss.item() == pytest.approx(0.0)

    loss.backward()

    assert log_risk.grad is not None


def test_breslow_estimator():
    """Compare cumulative hazards with known values."""
    baseline = BreslowBaseline()

    baseline.fit(
        duration=np.array([10, 20, 30, 40, 50]),
        event=np.array([1, 0, 1, 1, 0]),
        log_risk=np.zeros(5),
    )

    times = np.array([0, 10, 30, 40])

    expected = np.array(
        [
            0.0,
            1 / 5,
            1 / 5 + 1 / 3,
            1 / 5 + 1 / 3 + 1 / 2,
        ]
    )

    np.testing.assert_allclose(
        baseline.predict_cumulative_hazard(times),
        expected,
        atol=1e-6,
    )


def test_baseline_monotonicity():
    """Verify valid baseline survival and hazard curves."""
    baseline = BreslowBaseline()

    baseline.fit(
        np.array([10, 20, 30, 40, 50]),
        np.array([1, 0, 1, 1, 0]),
        np.zeros(5),
    )

    times = np.linspace(0, 50, 100)

    survival = baseline.predict_survival(times)
    hazard = baseline.predict_hazard(times)

    assert np.all(survival >= 0)
    assert np.all(survival <= 1)
    assert np.all(np.diff(survival) <= 1e-8)
    assert np.all(hazard >= 0)


def test_em_training(model, synthetic_data):
    """Verify successful approximate EM training."""
    features, duration, event = synthetic_data

    history = model.fit_em(
        features,
        duration,
        event,
        epochs=3,
        seed=42,
        verbose=False,
    )

    assert len(history) == 3
    assert np.isfinite(history).all()
    assert model.baselines_fitted


def test_posterior_probabilities(model, synthetic_data):
    """Verify normalized E-step posterior probabilities."""
    features, duration, event = synthetic_data

    model.fit_em(
        features,
        duration,
        event,
        epochs=1,
        verbose=False,
    )

    log_pi, log_risk = model.encode(features)

    posterior = model.e_step(
        log_pi,
        log_risk,
        duration,
        event,
    )

    assert posterior.shape == (10, 3)
    assert torch.isfinite(posterior).all()
    assert torch.all(posterior >= 0)

    torch.testing.assert_close(
        posterior.sum(dim=1),
        torch.ones(10),
    )


def test_gradient_computation(model, synthetic_data):
    """Verify gradient propagation through both heads."""
    features, duration, event = synthetic_data

    model.fit_em(
        features,
        duration,
        event,
        epochs=1,
        verbose=False,
    )

    model.zero_grad()

    log_pi, log_risk = model.encode(features)

    posterior = model.e_step(
        log_pi,
        log_risk,
        duration,
        event,
    )

    assignments = torch.multinomial(
        posterior,
        num_samples=1,
    ).squeeze(1)

    loss = model.m_step_loss(
        log_pi,
        log_risk,
        posterior,
        assignments,
        duration,
        event,
    )

    loss.backward()

    for module in [model.encoder, model.gate, model.risk]:
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_survival_prediction(model, synthetic_data):
    """Verify survival output dimensions and bounds."""
    features, duration, event = synthetic_data

    model.fit_em(
        features,
        duration,
        event,
        epochs=3,
        verbose=False,
    )

    survival = model.predict_survival(
        features,
        horizons=[0, 10, 30, 60, 90],
    )

    assert survival.shape == (10, 5)
    assert torch.isfinite(survival).all()
    assert torch.all(survival >= 0)
    assert torch.all(survival <= 1)

    torch.testing.assert_close(
        survival[:, 0],
        torch.ones(10),
    )

    assert torch.all(survival[:, 1:] <= survival[:, :-1] + 1e-6)


def test_single_component(synthetic_data):
    """Verify single-component deep Cox behavior."""
    features, duration, event = synthetic_data

    model = DeepCoxMixtures(
        dataset=None,
        input_dim=4,
        num_components=1,
        hidden_dims=(8,),
    )

    output = model(features)

    torch.testing.assert_close(
        output["mixture_probs"],
        torch.ones(10, 1),
    )

    model.fit_em(
        features,
        duration,
        event,
        epochs=2,
        verbose=False,
    )

    survival = model.predict_survival(
        features,
        horizons=[0, 30, 60],
    )

    assert survival.shape == (10, 3)
    assert torch.isfinite(survival).all()


def test_invalid_components():
    """Reject invalid numbers of mixture components."""
    with pytest.raises(ValueError):
        DeepCoxMixtures(
            dataset=None,
            input_dim=4,
            num_components=0,
        )


def test_invalid_events(model, synthetic_data):
    """Reject nonbinary event indicators."""
    features, duration, event = synthetic_data

    event[0] = 2.0

    with pytest.raises(ValueError):
        model.fit_em(
            features,
            duration,
            event,
            epochs=1,
            verbose=False,
        )


def test_prediction_before_training(
    model,
    synthetic_data,
):
    """Reject survival prediction without fitted baselines."""
    features, _, _ = synthetic_data

    with pytest.raises(RuntimeError):
        model.predict_survival(
            features,
            horizons=[10, 20],
        )


def test_checkpoint_persistence(
    model,
    synthetic_data,
    tmp_path,
):
    """Verify survival predictions survive checkpoint reload."""

    features, duration, event = synthetic_data

    model.fit_em(
        features,
        duration,
        event,
        epochs=3,
        seed=42,
        verbose=False,
    )

    horizons = [0, 10, 30, 60, 90]

    expected = model.predict_survival(
        features,
        horizons,
    )

    checkpoint = tmp_path / "dcm_checkpoint.pt"

    torch.save(
        model.state_dict(),
        checkpoint,
    )

    restored = DeepCoxMixtures(
        dataset=None,
        input_dim=4,
        num_components=3,
        hidden_dims=(8,),
    )

    restored.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    )

    actual = restored.predict_survival(
        features,
        horizons,
    )

    assert restored.baselines_fitted

    assert restored.training_history == model.training_history

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_untrained_checkpoint(model):
    """Verify checkpoints also support untrained models."""

    restored = DeepCoxMixtures(
        dataset=None,
        input_dim=4,
        num_components=3,
        hidden_dims=(8,),
    )

    restored.load_state_dict(model.state_dict())

    assert not restored.baselines_fitted

    assert all(baseline.event_times is None for baseline in restored.baselines)


def test_forward_output_interface(model, synthetic_data):
    """Verify conventional and DCM-specific output keys."""
    features, _, _ = synthetic_data

    output = model(features=features)

    expected_keys = {
        "logit",
        "y_prob",
        "log_pi",
        "mixture_probs",
        "log_risk",
    }

    assert expected_keys.issubset(output.keys())

    torch.testing.assert_close(
        output["logit"],
        output["log_risk"],
    )

    torch.testing.assert_close(
        output["y_prob"],
        output["mixture_probs"],
    )


def test_forward_keyword_arguments(model, synthetic_data):
    """Verify forward accepts additional dataset fields."""
    features, duration, event = synthetic_data

    output = model(
        features=features,
        duration=duration,
        event=event,
    )

    assert output["logit"].shape == (10, 3)
    assert output["y_prob"].shape == (10, 3)

    assert torch.isfinite(output["logit"]).all()
    assert torch.isfinite(output["y_prob"]).all()
