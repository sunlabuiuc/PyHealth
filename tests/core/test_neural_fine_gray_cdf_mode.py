"""Tests for the NeuralFineGray ``cdf_mode`` ablation, with tiny synthetic tensors."""

import pytest
import torch

from pyhealth.models import NeuralFineGray

FEATURES = torch.tensor([[0.2, -0.1, 0.5], [-1.0, 0.3, 0.0], [0.7, 0.7, -0.4]])
TIMES = torch.tensor([0.3, 0.6, 0.9])
EVENTS = torch.tensor([1, 2, 0])


def _model(cdf_mode: str) -> NeuralFineGray:
    torch.manual_seed(0)
    return NeuralFineGray(
        None, 3, hidden_dims=(4,), survival_dims=(4,), cdf_mode=cdf_mode
    ).eval()


def test_default_is_paper_parameterization():
    assert _model("nfg").cdf_mode == "nfg"
    torch.manual_seed(0)
    default = NeuralFineGray(None, 3, hidden_dims=(4,), survival_dims=(4,)).eval()
    with torch.no_grad():
        assert torch.equal(
            default.predict_cif(FEATURES, [0.5]),
            _model("nfg").predict_cif(FEATURES, [0.5]),
        )


def test_invalid_cdf_mode():
    with pytest.raises(ValueError, match="cdf_mode"):
        NeuralFineGray(None, 3, cdf_mode="weibull")


def test_only_nfg_starts_at_zero():
    with torch.no_grad():
        assert torch.all(_model("nfg").predict_cif(FEATURES, [0.0]) == 0)
        assert torch.all(_model("sumonet").predict_cif(FEATURES, [0.0]) > 0)


def test_sumonet_is_monotonic_and_bounded():
    grid = torch.linspace(0, 2, 9)
    with torch.no_grad():
        cif = _model("sumonet").predict_cif(FEATURES, grid)
    assert torch.all(cif[:, 1:] >= cif[:, :-1])
    assert torch.all(cif.sum(dim=-1) <= 1)


def test_sumonet_likelihood_and_gradients():
    model = _model("sumonet").train()
    output = model(FEATURES, TIMES, EVENTS)
    assert output["cif"].shape == (3, 2)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_modes_give_different_likelihoods():
    with torch.no_grad():
        nfg = _model("nfg")(FEATURES, TIMES, EVENTS)["loss"]
        sumonet = _model("sumonet")(FEATURES, TIMES, EVENTS)["loss"]
    assert not torch.isclose(nfg, sumonet)
