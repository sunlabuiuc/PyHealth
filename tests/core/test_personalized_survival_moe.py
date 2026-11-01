"""Tiny synthetic correctness tests; no dataset downloads or persistent files."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from pyhealth.models import PersonalizedSurvivalMoE


@pytest.fixture
def model():
    torch.manual_seed(7)
    return PersonalizedSurvivalMoE(None, 3, 5, hidden_dim=8, num_experts=2)


def test_instantiation(model):
    from pyhealth.models import BaseModel

    assert isinstance(model, BaseModel)
    assert model.input_dim == 3
    assert model.num_time_bins == 5
    assert model.num_experts == 2
    assert model.mode is None


def test_distribution_and_gradients(model):
    out = model(features=torch.randn(4, 3), time_bin=[0, 1, 3, 4], event=[1, 0, 1, 0])
    assert out["pmf"].shape == (4, 5)
    assert out["logit"].shape == (4, 5)
    assert out["mixture_weights"].shape == (4, 2)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["survival"].shape == (4, 4)
    assert out["expert_pmf"].shape == (4, 2, 5)
    assert out["y_true"].shape == (4, 2)
    for key in ["pmf", "expert_pmf", "mixture_weights"]:
        assert torch.allclose(
            out[key].sum(-1), torch.ones_like(out[key].sum(-1)), atol=1e-6
        )
    assert torch.all((out["survival"] >= 0) & (out["survival"] <= 1))
    assert torch.all(out["survival"][:, 1:] <= out["survival"][:, :-1])
    mixture = (out["mixture_weights"].unsqueeze(-1) * out["expert_pmf"]).sum(1)
    assert torch.allclose(out["pmf"], mixture, atol=1e-6)
    out["loss"].backward()
    assert any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in model.parameters()
    )
    for name, param in model.named_parameters():
        if name != "_dummy_param":
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name


def test_hand_computed_censoring_likelihood():
    log_p = torch.tensor([[0.2, 0.3, 0.5]] * 3, dtype=torch.double).log()
    loss = PersonalizedSurvivalMoE.negative_log_likelihood(log_p, [0, 1, 2], [1, 0, 0])
    expected = -(math.log(0.2) + math.log(0.8) + math.log(0.5)) / 3
    assert loss.item() == pytest.approx(expected)
    assert PersonalizedSurvivalMoE.negative_log_likelihood(
        log_p, [0] * 3, [0] * 3
    ).item() == pytest.approx(0)


@pytest.mark.parametrize("event", [[0] * 4, [1] * 4])
def test_all_censored_or_observed(model, event):
    out = model(features=torch.zeros(4, 3), time_bin=[0, 1, 2, 4], event=event)
    assert torch.isfinite(out["loss"])
    assert torch.all((out["survival"] >= 0) & (out["survival"] <= 1))
    out["loss"].backward()


@pytest.mark.parametrize(
    "options",
    [
        {"dropout": -0.1},
        {"dropout": 1},
        {"num_experts": 3},
        {"num_time_bins": 1},
        {"input_dim": 0},
    ],
)
def test_invalid_configuration(options):
    config = {
        "dataset": None,
        "input_dim": 3,
        "num_time_bins": 5,
        "hidden_dim": 8,
        "num_experts": 2,
    }
    config.update(options)
    with pytest.raises(ValueError):
        PersonalizedSurvivalMoE(**config)


@pytest.mark.parametrize(
    "times,events",
    [
        ([-1, 0], [1, 0]),
        ([0, 5], [1, 0]),
        ([0.5, 1], [1, 0]),
        ([0, float("nan")], [1, 0]),
        ([0, 1], [2, 0]),
        ([0], [1, 0]),
    ],
)
def test_invalid_labels(model, times, events):
    with pytest.raises(ValueError):
        model(features=torch.zeros(2, 3), time_bin=times, event=events)


def test_inference_validation_and_serialization(model):
    model.eval()
    x = torch.randn(2, 3)
    out = model(features=x)
    assert "loss" not in out
    with pytest.raises(ValueError):
        model(features=x, event=[1, 0])
    with pytest.raises(ValueError):
        model(features=torch.full((2, 3), float("nan")))
    with pytest.raises(ValueError):
        model(features=torch.zeros(2, 4))
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = PersonalizedSurvivalMoE(None, 3, 5, hidden_dim=8, num_experts=2)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        restored.eval()
        assert torch.equal(restored(features=x)["pmf"], out["pmf"])
    assert not Path(directory).exists()


def test_dropout_train_and_eval():
    model = PersonalizedSurvivalMoE(
        None,
        3,
        5,
        hidden_dim=8,
        num_experts=2,
        dropout=0.5,
    )
    x = torch.randn(4, 3)
    model.eval()
    assert torch.equal(model(features=x)["pmf"], model(features=x)["pmf"])
    model.train()
    assert not torch.equal(model(features=x)["pmf"], model(features=x)["pmf"])


def test_extreme_scores_stable(model):
    with torch.no_grad():
        for head in model.experts:
            head.bias.copy_(torch.tensor([1000.0, -1000.0, 1000.0, -1000.0, 0.0]))
    out = model(features=torch.zeros(2, 3), time_bin=[1, 4], event=[1, 0])
    assert torch.isfinite(out["loss"])
    assert torch.all((out["survival"] >= 0) & (out["survival"] <= 1))
    out["loss"].backward()
    assert torch.isfinite(model.experts[0].bias.grad).all()


def test_pyhealth_sample_dataset_and_dataloader():
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    dataset = create_sample_dataset(
        samples=[
            {
                "patient_id": str(i),
                "features": [0.0, float(i), 1.0],
                "time_bin": [i],
                "event": [i % 2],
            }
            for i in range(4)
        ],
        input_schema={"features": "tensor"},
        output_schema={"time_bin": "tensor", "event": "tensor"},
        in_memory=True,
    )
    model = PersonalizedSurvivalMoE(dataset, 3, 5, hidden_dim=8, num_experts=2)
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
    assert model.mode is None
    assert torch.isfinite(model(**batch)["loss"])


def test_categorical_embeddings_and_gradients():
    model = PersonalizedSurvivalMoE(
        None,
        3,
        5,
        hidden_dim=8,
        num_experts=2,
        num_hidden_layers=1,
        categorical_cardinalities=(3, 2),
        embedding_dim=4,
    )
    features = torch.randn(4, 3)
    categories = torch.tensor([[0, 0], [1, 1], [2, 0], [1, 1]])
    out = model(
        features=features,
        categories=categories,
        time_bin=[0, 1, 3, 4],
        event=[1, 0, 1, 0],
    )
    assert out["pmf"].shape == (4, 5)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for embedding in model.embeddings:
        assert embedding.weight.grad is not None
        assert torch.isfinite(embedding.weight.grad).all()
        assert embedding.weight.grad.abs().sum() > 0
    for invalid in [categories[:, :1], categories + 5, categories.float() + 0.5]:
        with pytest.raises(ValueError):
            model(features=features, categories=invalid)
