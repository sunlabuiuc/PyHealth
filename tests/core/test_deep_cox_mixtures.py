# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""DeepCoxMixtures on tiny synthetic data.

Tests without a marker need only core dependencies: they replace the EM fit by
hand-made baseline splines. Tests that fit the model or compare against the
authors' code skip when ``auton_survival``/``sksurv`` are not installed.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scipy.interpolate import UnivariateSpline

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepCoxMixtures
from pyhealth.models._dcm_reference import (
    DCMReference,
    DeepCoxMixturesNetwork,
    predict_mixture_survival,
)
from pyhealth.trainer import Trainer

INPUT_DIM = 3
HORIZONS = [1.5, 2.5, 3.5]
CONFIGS = [(1, 4), (2, 8), (3, 16), (6, 4)]


def needs_author_code() -> Any:
    """Import the optional authors' DCM utilities or skip the calling test."""
    pytest.importorskip("sksurv", reason="optional dependency of DCM training")
    return pytest.importorskip(
        "auton_survival.models.dcm.dcm_utilities",
        reason="optional auton-survival package (see DeepCoxMixtures docs)",
    )


def make_samples(n: int = 6) -> list[dict[str, Any]]:
    """Return n synthetic samples with 3 features, a time and an event."""
    rng = np.random.default_rng(0)
    return [
        {
            "patient_id": f"p{i}",
            "features": rng.normal(size=INPUT_DIM).round(3).tolist(),
            "time": [float(i + 1)],
            "event": [float(i % 2)],
        }
        for i in range(n)
    ]


def make_dataset(n: int = 6) -> Any:
    """Build an in-memory SampleDataset of synthetic survival samples."""
    return create_sample_dataset(
        samples=make_samples(n),
        input_schema={"features": "tensor"},
        output_schema={"time": "tensor", "event": "tensor"},
        dataset_name="synthetic_dcm",
    )


def analytic_baselines(k: int) -> dict[int, UnivariateSpline]:
    """Return k decreasing linear baseline survival splines (no EM fit)."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    return {
        i: UnivariateSpline(
            times, np.linspace(1.0, 0.4 - 0.05 * i, len(times)), k=1, s=0, ext=3
        )
        for i in range(k)
    }


def fitted_model(k: int = 2, width: int = 4, seed: int = 7) -> DeepCoxMixtures:
    """Return a model with hand-made baselines standing in for an EM fit."""
    model = DeepCoxMixtures(input_dim=INPUT_DIM, k=k, width=width, seed=seed)
    model.baselines = analytic_baselines(k)
    model.horizons = np.asarray(HORIZONS)
    return model


def batch(n: int = 6) -> dict[str, torch.Tensor]:
    """Return a dense batch with features, time and event tensors."""
    samples = make_samples(n)
    return {
        "features": torch.tensor([s["features"] for s in samples]),
        "time": torch.tensor([s["time"] for s in samples]),
        "event": torch.tensor([s["event"] for s in samples]),
    }


def synthetic_arrays(n: int, seed: int = 3) -> tuple[np.ndarray, ...]:
    """Return features, positive times and events for a tiny EM fit."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, INPUT_DIM)).astype(np.float32)
    time_ = rng.exponential(10 * np.exp(-0.5 * x[:, 0])) + 1
    event = (rng.random(n) < 0.7).astype(int)
    return x, time_, event


# ---------------------------------------------------------------- core


@pytest.mark.parametrize("k,width", CONFIGS)
def test_instantiation_from_dataset(k: int, width: int) -> None:
    """The model infers input_dim from the dataset and has mode None."""
    model = DeepCoxMixtures(make_dataset(), k=k, width=width)
    assert model.input_dim == INPUT_DIM
    assert model.mode is None and model.fit_outside_trainer
    assert model.feature_keys == ["features"]
    assert set(model.label_keys) == {"time", "event"}
    assert model.network.gate.weight.shape == (k, width)
    assert model.network.expert.weight.shape == (k, width)
    assert model.network.embedding[0].weight.shape == (width, INPUT_DIM)


def test_instantiation_errors() -> None:
    """Missing input_dim or unknown schema keys raise ValueError."""
    with pytest.raises(ValueError, match="input_dim"):
        DeepCoxMixtures()
    with pytest.raises(ValueError, match="input_schema"):
        DeepCoxMixtures(make_dataset(), feature_key="missing")
    with pytest.raises(ValueError, match="output_schema"):
        DeepCoxMixtures(make_dataset(), time_key="missing")
    with pytest.raises(ValueError, match="positive int"):
        DeepCoxMixturesNetwork(INPUT_DIM, k=0)


def test_construction_keeps_global_rng() -> None:
    """Seeding at construction does not change the global torch generator."""
    state = torch.get_rng_state()
    a = DeepCoxMixtures(input_dim=INPUT_DIM, seed=1)
    b = DeepCoxMixtures(input_dim=INPUT_DIM, seed=1)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(a.network.gate.weight, b.network.gate.weight)


@pytest.mark.parametrize("k,width", CONFIGS)
def test_forward_output_shapes(k: int, width: int) -> None:
    """forward returns the BaseModel keys with the documented shapes."""
    model = fitted_model(k, width)
    out = model(**batch(6))
    assert out["logit"].shape == (6, 2 * k)
    assert out["log_weights"].shape == (6, k)
    assert out["log_hazard_ratios"].shape == (6, k)
    assert out["y_prob"].shape == (6, len(HORIZONS))
    assert out["y_true"].shape == (6, 2)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert torch.allclose(out["log_weights"].exp().sum(1), torch.ones(6))
    assert out["log_hazard_ratios"].abs().max() <= 10
    prob = out["y_prob"].numpy()
    assert ((prob >= 0) & (prob <= 1)).all()
    assert (np.diff(prob, axis=1) <= 0).all()


def test_forward_without_labels_has_no_loss() -> None:
    """Without time/event in the batch, forward returns predictions only."""
    out = fitted_model()(features=batch()["features"])
    assert "loss" not in out and "y_true" not in out
    assert out["y_prob"].shape == (6, len(HORIZONS))


@pytest.mark.parametrize("k,width", CONFIGS)
def test_loss_gradients_reach_every_layer(k: int, width: int) -> None:
    """loss.backward gives finite, nonzero gradients for all weights."""
    model = fitted_model(k, width)
    model(**batch(8))["loss"].backward()
    for name, param in model.network.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert model.network.expert.weight.grad.abs().sum() > 0
    assert model.network.embedding[0].weight.grad.abs().sum() > 0
    if k > 1:
        assert model.network.gate.weight.grad.abs().sum() > 0


def test_single_component_loss_is_cox_partial_likelihood() -> None:
    """With k=1 the loss equals the mean negative Cox partial log-likelihood."""
    model = fitted_model(k=1, width=4)
    data = batch(6)
    out = model(**data)
    risk = out["log_hazard_ratios"][:, 0].detach().numpy().astype(float)
    t = data["time"].reshape(-1).numpy()
    e = data["event"].reshape(-1).numpy()
    expected = -sum(
        risk[i] - np.log(np.exp(risk[t >= t[i]]).sum())
        for i in range(len(t))
        if e[i] == 1
    ) / len(t)
    assert out["loss"].item() == pytest.approx(expected, rel=1e-5)


def test_dataloader_batch_and_trainer_evaluate() -> None:
    """A PyHealth dataloader batch runs through forward and Trainer.evaluate."""
    dataset = make_dataset(6)
    model = DeepCoxMixtures(dataset, k=2, width=4, horizons=HORIZONS)
    model.baselines = analytic_baselines(2)
    loader = get_dataloader(dataset, batch_size=3, shuffle=False)
    out = model(**next(iter(loader)))
    assert out["y_prob"].shape == (3, len(HORIZONS))
    trainer = Trainer(model=model, device="cpu", enable_logging=False)
    assert np.isfinite(trainer.evaluate(loader)["loss"])
    y_true, y_prob, _ = trainer.inference(loader)
    assert y_true.shape == (6, 2) and y_prob.shape == (6, len(HORIZONS))
    with pytest.raises(TypeError, match="Trainer.train"):
        trainer.train(train_dataloader=loader, epochs=1)


def test_unfitted_and_missing_horizons_raise() -> None:
    """Prediction before fitting, or with no horizons, fails clearly."""
    model = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4)
    with pytest.raises(RuntimeError, match="fit_survival"):
        model(**batch())
    with pytest.raises(RuntimeError, match="fit_survival"):
        model.predict_survival(batch()["features"], HORIZONS)
    model.baselines = analytic_baselines(2)
    with pytest.raises(ValueError, match="horizons"):
        model(**batch())
    with pytest.raises(ValueError, match="horizons"):
        model.predict_survival(batch()["features"])


@pytest.mark.parametrize(
    "horizons", [[3.0, 1.0], [1.0, 1.0], [float("nan")], [float("inf")], [-1.0], []]
)
def test_invalid_horizons_raise(horizons: list[float]) -> None:
    """Unsorted, repeated, nonfinite, negative or empty horizons raise."""
    with pytest.raises(ValueError, match="Horizons"):
        fitted_model().predict_survival(batch()["features"], horizons)


def test_invalid_batches_raise() -> None:
    """Bad feature shapes, nonfinite features and bad targets raise."""
    model, data = fitted_model(), batch()
    with pytest.raises(ValueError, match="shape"):
        model(features=torch.zeros(4, INPUT_DIM + 1))
    with pytest.raises(ValueError, match="finite"):
        model(features=torch.full((2, INPUT_DIM), float("nan")))
    with pytest.raises(ValueError, match="missing field"):
        model(time=data["time"])
    with pytest.raises(ValueError, match="batch targets"):
        model(**{**data, "time": torch.zeros(6, 1)})
    with pytest.raises(ValueError, match="length"):
        model(**{**data, "event": torch.ones(2, 1)})


def test_fit_survival_rejects_bad_inputs_before_training() -> None:
    """Shape and target checks run before any optional import or training."""
    model = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4)
    x, t, e = synthetic_arrays(5)
    with pytest.raises(ValueError, match="input_dim"):
        model.fit_survival(x[:, :2], t, e)
    with pytest.raises(ValueError, match="training targets"):
        model.fit_survival(x, np.r_[0.0, t[1:]], e)
    with pytest.raises(ValueError, match="training targets"):
        model.fit_survival(x, t, np.r_[2, e[1:]])
    with pytest.raises(ValueError, match="finite"):
        model.fit_survival(np.full_like(x, np.nan), t, e)
    with pytest.raises(ValueError, match="Validation feature"):
        model.fit_survival(x, t, e, validation=(x[:, :2], t, e))
    with pytest.raises(ValueError, match="Validation feature"):
        model.fit_survival(x, t, e, validation=(x, t[:3], e))
    with pytest.raises(ValueError, match="validation targets"):
        model.fit_survival(x, t, e, validation=(x, -t, e))
    with pytest.raises(ValueError, match="Horizons"):
        model.fit_survival(x, t, e, horizons=[2.0, 1.0])


def test_state_dict_round_trip_with_weights_only_load(tmp_path: Path) -> None:
    """Splines, horizons and weights survive torch.save/load(weights_only=True)."""
    model = fitted_model(k=3, width=4)
    model.fit_audit = {"epochs": 2, "validation_q_losses": [1.0, 0.5]}
    x = batch()["features"]
    expected = model.predict_survival(x)
    path = tmp_path / "dcm.pt"
    torch.save(model.state_dict(), path)
    restored = DeepCoxMixtures(input_dim=INPUT_DIM, k=3, width=4, seed=99)
    restored.load_state_dict(torch.load(path, weights_only=True))
    np.testing.assert_array_equal(restored.predict_survival(x), expected)
    np.testing.assert_array_equal(restored.horizons, HORIZONS)
    assert restored.fit_audit == model.fit_audit


def test_trainer_checkpoint_round_trip(tmp_path: Path) -> None:
    """Trainer.save_ckpt/load_ckpt (weights_only=True) restores predictions."""
    model = fitted_model(k=2, width=4)
    path = str(tmp_path / "best.ckpt")
    Trainer(model=model, device="cpu", enable_logging=False).save_ckpt(path)
    restored = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4, seed=5)
    Trainer(model=restored, device="cpu", enable_logging=False).load_ckpt(path)
    x = batch()["features"]
    np.testing.assert_array_equal(
        restored.predict_survival(x), model.predict_survival(x)
    )


def test_unfitted_state_round_trip() -> None:
    """An unfitted model's state dict loads into another unfitted model."""
    model = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4)
    other = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4, seed=3)
    other.load_state_dict(model.state_dict())
    assert other.baselines is None and other.horizons is None
    bad = model.state_dict()
    bad["_extra_state"] = {**bad["_extra_state"], "version": 0}
    with pytest.raises(ValueError, match="version"):
        other.load_state_dict(bad)


def test_missing_optional_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without auton_survival, fit_survival raises an ImportError with a hint."""
    import sys

    monkeypatch.setitem(sys.modules, "auton_survival", None)
    model = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4)
    x, t, e = synthetic_arrays(8)
    with pytest.raises(ImportError, match="pyhealth\\[dcm\\]"):
        model.fit_survival(x, t, e)


# ------------------------------------------- optional: authors' package


def test_network_matches_author_network() -> None:
    """Same seed gives the authors' initial weights and identical outputs."""
    needs_author_code()
    from auton_survival.models.dcm.dcm_torch import DeepCoxMixturesTorch

    for k, width in CONFIGS:
        torch.manual_seed(11)
        author = DeepCoxMixturesTorch(INPUT_DIM, k=k, layers=[width], gamma=10)
        torch.manual_seed(11)
        native = DeepCoxMixturesNetwork(INPUT_DIM, k=k, layers=[width], gamma=10)
        assert author.state_dict().keys() == native.state_dict().keys()
        for key, value in author.state_dict().items():
            assert torch.equal(value, native.state_dict()[key]), key
        x = 3 * torch.randn(16, INPUT_DIM)
        for a, b in zip(author(x), native(x)):
            assert torch.equal(a, b)


def test_curves_and_posteriors_match_author_code() -> None:
    """Native curves equal, and E-step posteriors match, the authors' code."""
    utils = needs_author_code()
    model = fitted_model(k=3, width=4)
    data = batch(8)
    x = data["features"]
    native = predict_mixture_survival(model.network, model.baselines, x, HORIZONS)
    author = utils.predict_survival((model.network, model.baselines), x, HORIZONS)
    np.testing.assert_array_equal(native, author)
    t, e = data["time"].reshape(-1), data["event"].reshape(-1)
    with torch.no_grad():
        gates, risks = model.network(x)
        expected = utils.e_step(model.network, model.baselines, x, t, e)
    got = model._posteriors(gates, risks, t.numpy(), e.numpy())
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)


def test_tiny_fit_survival_run() -> None:
    """One EM epoch on 60 rows fits, predicts and restores global state."""
    needs_author_code()
    x, t, e = synthetic_arrays(60)
    model = DeepCoxMixtures(input_dim=INPUT_DIM, k=2, width=4, seed=0)
    np_state, torch_state = np.random.get_state()[1].copy(), torch.get_rng_state()
    threads = torch.get_num_threads()
    started = time.monotonic()
    model.fit_survival(
        x[:48], t[:48], e[:48], validation=(x[48:], t[48:], e[48:]),
        horizons=[2.0, 5.0], epochs=1, capture_output=True,
    )  # fmt: skip
    assert time.monotonic() - started < 5
    assert model.fit_audit["epochs"] == 1
    assert len(model.baselines) == 2
    curves = model.predict_survival(x[48:])
    assert curves.shape == (12, 2) and np.isfinite(curves).all()
    assert (np.diff(curves, axis=1) <= 0).all()
    assert torch.isfinite(model(features=torch.from_numpy(x[48:]))["y_prob"]).all()
    np.testing.assert_array_equal(np.random.get_state()[1], np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads


def test_fit_is_deterministic_and_matches_author_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refits are identical, and training with the authors' network is too."""
    needs_author_code()
    from auton_survival.models.dcm.dcm_torch import DeepCoxMixturesTorch

    import pyhealth.models._dcm_reference as reference

    x, t, e = synthetic_arrays(60)
    y = np.empty(60, dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = e.astype(bool), t

    def fit_curves() -> np.ndarray:
        """Fit one epoch with a fixed seed and return held-out curves."""
        learner = DCMReference(k=2, width=4, seed=5)
        learner.fit(x[:48], y[:48], epochs=1, refit=True, capture_output=True)
        return learner.predict_survival(x[48:], [2.0, 5.0])

    native = fit_curves()
    np.testing.assert_array_equal(native, fit_curves())
    monkeypatch.setattr(reference, "DeepCoxMixturesNetwork", DeepCoxMixturesTorch)
    np.testing.assert_array_equal(native, fit_curves())


def test_fit_from_dataloader() -> None:
    """fit() collects PyHealth dataloader batches and fits one epoch."""
    needs_author_code()
    x, t, e = synthetic_arrays(40)
    dataset = create_sample_dataset(
        samples=[
            {"patient_id": f"p{i}", "features": x[i].tolist(),
             "time": [float(t[i])], "event": [float(e[i])]}
            for i in range(40)
        ],  # fmt: skip
        input_schema={"features": "tensor"},
        output_schema={"time": "tensor", "event": "tensor"},
    )
    model = DeepCoxMixtures(dataset, k=2, width=4)
    loader = get_dataloader(dataset, batch_size=16)
    model.fit(loader, horizons=[2.0, 5.0], epochs=1)
    assert model(**next(iter(loader)))["y_prob"].shape == (16, 2)
