"""Focused synthetic regressions for T21-T38 (no clinical data needed).
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from pyhealth.models.td_cnn_lstm import TDCNNLSTM, _SequenceEncoder


def make_dataset():
    """Provide the schema BaseModel uses without external sample files."""
    input_keys = ("timepoints", "values", "features", "delta_time", "delta_value")
    return SimpleNamespace(
        input_schema={key: "raw" for key in input_keys},
        output_schema={"label": "binary"},
    )


def make_state(batch_size=2, sequence_length=4, valid_lengths=None, prefix=""):
    """Construct aligned ICU event channels with optional trailing padding."""
    if valid_lengths is None:
        valid_lengths = [sequence_length] * batch_size
    timestamps = torch.full((batch_size, sequence_length), float("nan"))
    values = torch.full_like(timestamps, float("nan"))
    delta_time = torch.full_like(timestamps, float("nan"))
    delta_value = torch.full_like(timestamps, float("nan"))
    features = torch.full((batch_size, sequence_length), -1, dtype=torch.long)
    for row, count in enumerate(valid_lengths):
        timestamps[row, :count] = torch.arange(count - 1, -1, -1).float()
        values[row, :count] = torch.arange(count).float() / 10.0
        features[row, :count] = torch.arange(count) % 3
        delta_time[row, :count] = 1.0
        delta_value[row, :count] = 0.0
    return {
        prefix + "timepoints": timestamps,
        prefix + "values": values,
        prefix + "features": features,
        prefix + "delta_time": delta_time,
        prefix + "delta_value": delta_value,
    }


@pytest.mark.parametrize("valid_length", [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15])
def test_encoder_is_invariant_to_right_padding(valid_length):
    """Adding only trailing padding cannot change a patient's embedding."""
    torch.manual_seed(10)
    encoder = _SequenceEncoder(hidden_dim=4, dropout=0.0).eval()
    embedded = torch.randn(1, 4, valid_length)
    mask = torch.ones(1, valid_length, dtype=torch.bool)
    with torch.no_grad():
        expected = encoder(embedded, mask)
        for extra in [1, 2, 3, 4, 5, 8, 19]:
            padded = F.pad(embedded, (0, extra))
            padded_mask = F.pad(mask, (0, extra), value=False)
            actual = encoder(padded, padded_mask)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_encoder_shorter_patient_unaffected_by_longer_batchmate():
    """Another patient's length must not change this patient's embedding."""
    torch.manual_seed(11)
    encoder = _SequenceEncoder(hidden_dim=4, dropout=0.0).eval()
    short = torch.randn(1, 4, 2)
    long = torch.randn(1, 4, 17)
    with torch.no_grad():
        expected = encoder(short, torch.ones(1, 2, dtype=torch.bool))
        batched = torch.cat((F.pad(short, (0, 15)), long), dim=0)
        mask = torch.cat(
            (torch.tensor([[True, True] + [False] * 15]),
             torch.ones(1, 17, dtype=torch.bool)),
            dim=0,
        )
        actual = encoder(batched, mask)[0:1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch_size,sequence_length", [(2, 5), (3, 4)])
def test_forward_rejects_mismatched_successor_shape(batch_size, sequence_length):
    """Both batch size and sequence length must match between paired states."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    current = make_state(batch_size=2, sequence_length=4)
    successor = make_state(
        batch_size=batch_size,
        sequence_length=sequence_length,
        prefix="next_",
    )
    with pytest.raises(ValueError, match="paired states must have identical"):
        model(**current, **successor)
    with pytest.raises(ValueError, match="paired states must have identical"):
        model._target_probability({**current, **successor})


def test_forward_accepts_matched_successor_and_empty_successor():
    """An all-padded next state is legal, even when another row is valid."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    current = make_state(2, 5, valid_lengths=[2, 5])
    successor = make_state(2, 5, valid_lengths=[0, 3], prefix="next_")
    with torch.no_grad():
        output = model(**current, **successor)
        probabilities, has_next = model._target_probability({**current, **successor})
    assert output["logit"].shape == (2, 1)
    assert torch.isfinite(output["y_prob"]).all()
    assert has_next.tolist() == [False, True]
    assert torch.isnan(probabilities[0, 0])
    assert torch.isfinite(probabilities[1, 0])


def test_forward_rejects_incomplete_successor():
    """Providing one next-state channel requires all five channels."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    batch = make_state()
    batch["next_values"] = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Missing required input"):
        model(**batch)


def test_single_event_and_right_padding_invariant_at_model_level():
    """The full predictor gives equal eval scores with arbitrary right padding."""
    torch.manual_seed(12)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    minimal = make_state(batch_size=1, sequence_length=2)
    extended = make_state(batch_size=1, sequence_length=8, valid_lengths=[2])
    with torch.no_grad():
        short_out = model(**minimal)
        long_out = model(**extended)
    torch.testing.assert_close(
        short_out["logit"], long_out["logit"], atol=1e-6, rtol=1e-6
    )


def test_target_initially_frozen_and_always_eval():
    """TD target is synchronized but cannot receive gradients or train-mode BN."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4)
    assert model.target is not None
    assert not model.target.training
    assert all(not parameter.requires_grad for parameter in model.target.parameters())
    for online_value, target_value in zip(model.online.state_dict().values(),
                                          model.target.state_dict().values()):
        torch.testing.assert_close(online_value, target_value)
    model.train()
    assert model.online.training
    assert not model.target.training


# ---------------------------------------------------------------------------
# T30 + T31: temporal-difference and supervised loss calculations
# ---------------------------------------------------------------------------
def test_td_loss_uses_successor_risks_and_28_day_terminal_labels(monkeypatch):
    """Only states with a successor bootstrap; terminal states use truth."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4)
    online_logits = torch.tensor([[-1.0], [0.0], [2.0]], requires_grad=True)
    observed = torch.tensor([0.0, 1.0, 0.0])
    successor_risks = torch.tensor([[0.2], [float("nan")], [0.8]],
                                   requires_grad=True)

    def fake_target_probability(batch, current_shape=None):
        assert current_shape == torch.Size([3, 4])
        return successor_risks, torch.tensor([True, False, True])

    monkeypatch.setattr(model, "_target_probability", fake_target_probability)
    loss = model._compute_loss(
        online_logits,
        {"label": observed},
        current_shape=torch.Size([3, 4]),
    )
    expected_targets = torch.tensor([[0.2], [1.0], [0.8]])
    expected_loss = F.binary_cross_entropy_with_logits(
        online_logits, expected_targets
    )
    torch.testing.assert_close(loss, expected_loss)
    assert loss.ndim == 0
    loss.backward()
    assert online_logits.grad is not None
    assert torch.isfinite(online_logits.grad).all()
    assert successor_risks.grad is None


def test_td_loss_all_terminal_uses_observed_labels():
    """Missing successors do not imply zero mortality or missing targets."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    current = make_state(batch_size=2, sequence_length=4)
    empty_successor = make_state(
        batch_size=2, sequence_length=4, valid_lengths=[0, 0], prefix="next_"
    )
    labels = torch.tensor([1, 0])
    logits = torch.tensor([[-0.4], [0.9]], requires_grad=True)
    loss = model._compute_loss(logits, {**current, **empty_successor,
                                        "label": labels})
    expected_loss = F.binary_cross_entropy_with_logits(
        logits, labels.float().unsqueeze(1)
    )
    torch.testing.assert_close(loss, expected_loss)


def test_td_loss_uses_real_frozen_target_and_only_online_gradients():
    """Paired successors are scored with the frozen model without backprop."""
    torch.manual_seed(25)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    current = make_state(2, 4, valid_lengths=[4, 2])
    successor = make_state(2, 4, valid_lengths=[3, 0], prefix="next_")
    batch = {**current, **successor, "label": torch.tensor([0, 1])}

    logits = model(**current)["logit"]
    target_risks, has_next = model._target_probability(batch)
    assert has_next.tolist() == [True, False]
    expected_targets = torch.where(
        has_next.unsqueeze(1), target_risks, torch.tensor([[0.0], [1.0]])
    )
    expected_loss = F.binary_cross_entropy_with_logits(logits, expected_targets)
    actual_loss = model._compute_loss(logits, batch)
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()

    assert any(p.grad is not None for p in model.online.parameters())
    assert all(p.grad is None for p in model.target.parameters())


def test_supervised_loss_matches_bce_and_has_no_target_network():
    """Supervised mode trains only against observed 28-day labels."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    assert model.target is None
    logits = torch.tensor([[-0.7], [1.2]], requires_grad=True)
    labels = torch.tensor([[1], [0]])
    actual = model._compute_loss(logits, {"label": labels})
    expected = F.binary_cross_entropy_with_logits(logits, labels.float())
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_supervised_loss_does_not_need_successor_states():
    """A supervised batch requires only an observed 28-day label."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    logits = torch.tensor([[0.1], [-0.3]], requires_grad=True)
    assert torch.isfinite(model._compute_loss(logits, {"label": [0, 1]}))


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]),
        torch.tensor([0.0, 0.5]),
        torch.tensor([0.0, 1.000000001], dtype=torch.float64),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([0.0, 1.0, 0.0]),
    ],
)
def test_loss_rejects_invalid_28_day_mortality_labels(labels):
    """Reject unlabeled, five-horizon, nonbinary and nonfinite targets."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    logits = torch.tensor([[0.0], [0.0]])
    with pytest.raises(ValueError, match="Observed 28-day mortality"):
        model._compute_loss(logits, {"label": labels})


def test_loss_requires_labels_and_one_logit_per_state():
    """Loss code rejects missing labels and invalid output shapes."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    with pytest.raises(ValueError, match="Missing observed 28-day"):
        model._compute_loss(torch.zeros(2, 1), {})
    with pytest.raises(ValueError, match="logits must have shape"):
        model._compute_loss(torch.zeros(2, 5), {"label": [0, 1]})


def test_td_loss_rejects_missing_or_mismatched_successor_state():
    """TD bootstrap requires complete, correctly shaped next-state channels."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    current = make_state(2, 4)
    logit = torch.tensor([[0.1], [0.2]])
    with pytest.raises(ValueError, match="Missing required input"):
        model._compute_loss(logit, {**current, "label": [0, 1]})
    wrong_shape = make_state(2, 5, prefix="next_")
    with pytest.raises(ValueError, match="paired states must have identical"):
        model._compute_loss(
            logit, {**current, **wrong_shape, "label": [0, 1]}
        )


def test_labeled_forward_returns_standard_pyhealth_keys_in_supervised_mode():
    """A labeled supervised batch returns observed labels and BCE loss."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    ).eval()
    batch = {**make_state(2, 4), "label": torch.tensor([0, 1])}
    outputs = model(**batch)

    assert set(outputs) == {"loss", "y_prob", "y_true", "logit"}
    assert outputs["logit"].shape == (2, 1)
    assert outputs["y_prob"].shape == (2, 1)
    assert outputs["y_true"].shape == (2, 1)
    assert outputs["loss"].ndim == 0
    torch.testing.assert_close(outputs["y_true"], batch["label"].float()[:, None])
    torch.testing.assert_close(outputs["y_prob"], outputs["logit"].sigmoid())
    torch.testing.assert_close(
        outputs["loss"],
        F.binary_cross_entropy_with_logits(outputs["logit"], outputs["y_true"]),
    )
    outputs["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.online.parameters())


def test_supervised_loss_does_not_train_labels():
    """Observed labels are detached targets, not trainable parameters."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    logits = torch.tensor([[0.3], [-1.0]], requires_grad=True)
    labels = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = model._compute_loss(logits, {"label": labels})
    loss.backward()
    assert logits.grad is not None
    assert labels.grad is None


# ---------------------------------------------------------------------------
# T32: forward contract in both modes
# ---------------------------------------------------------------------------
def test_td_forward_returns_observed_truth_and_bootstrapped_loss():
    """The TD loss can use successor risks without altering observed y_true."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    batch = {
        **make_state(3, 5, valid_lengths=[5, 2, 4]),
        **make_state(3, 5, valid_lengths=[2, 0, 3], prefix="next_"),
        "label": torch.tensor([1, 0, 1]),
    }
    outputs = model(**batch)

    assert set(outputs) == {"loss", "y_prob", "y_true", "logit"}
    assert outputs["logit"].shape == (3, 1)
    assert outputs["y_prob"].shape == (3, 1)
    assert outputs["y_true"].shape == (3, 1)
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])
    torch.testing.assert_close(outputs["y_prob"], outputs["logit"].sigmoid())
    torch.testing.assert_close(outputs["y_true"], batch["label"].float()[:, None])

    next_risks, has_next = model._target_probability(batch)
    assert has_next.tolist() == [True, False, True]
    targets = torch.where(has_next[:, None], next_risks, outputs["y_true"])
    torch.testing.assert_close(
        outputs["loss"],
        F.binary_cross_entropy_with_logits(outputs["logit"], targets),
    )
    assert not torch.equal(targets, outputs["y_true"])
    outputs["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.online.parameters())
    assert all(parameter.grad is None for parameter in model.target.parameters())


def test_td_labeled_forward_all_terminal_falls_back_to_labels():
    """TD states without eligible successors use their observed 28-day labels."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    batch = {
        **make_state(2, 5),
        **make_state(2, 5, valid_lengths=[0, 0], prefix="next_"),
        "label": torch.tensor([[0], [1]]),
    }
    outputs = model(**batch)
    torch.testing.assert_close(
        outputs["loss"],
        F.binary_cross_entropy_with_logits(outputs["logit"], outputs["y_true"]),
    )


def test_td_labeled_forward_requires_complete_successor_state():
    """TD loss cannot silently train without successor information."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    batch = {**make_state(2, 4), "label": [1, 0]}
    with pytest.raises(ValueError, match="Missing required input"):
        model(**batch)
    batch.update(make_state(2, 5, prefix="next_"))
    with pytest.raises(ValueError, match="paired states must have identical"):
        model(**batch)


def test_supervised_labeled_forward_needs_no_successor():
    """Supervised batches use only their observed labels."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    ).eval()
    outputs = model(**make_state(2, 4), label=[0, 1])
    assert outputs["y_true"].tolist() == [[0.0], [1.0]]
    assert torch.isfinite(outputs["loss"])


def test_unlabeled_forward_returns_only_predictions_in_both_modes():
    """Unlabeled inference remains supported without loss or label keys."""
    for objective in ("td", "supervised"):
        model = TDCNNLSTM(
            make_dataset(), n_features=3, hidden_dim=4, objective=objective
        ).eval()
        outputs = model(**make_state(batch_size=1, sequence_length=4))
        assert set(outputs) == {"logit", "y_prob"}
        assert outputs["y_prob"].shape == (1, 1)


# ---------------------------------------------------------------------------
# T33: target-network EMA after optimization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.0, 0.75, 0.99, 1.0])
def test_update_target_handles_parameters_float_buffers_and_integer_buffers(alpha):
    """Every float tensor uses EMA; BatchNorm integer counts copy exactly."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, target_alpha=alpha
    )
    with torch.no_grad():
        model.online.embedding.feature_embedding.weight.add_(2.0)
        model.online.embedding.norm.running_mean.add_(3.0)
        model.online.decoder[0].num_batches_tracked.add_(7)

    online_before = {
        key: value.clone() for key, value in model.online.state_dict().items()
    }
    target_before = {
        key: value.clone() for key, value in model.target.state_dict().items()
    }
    model.train()
    model.update_target()

    assert not model.target.training
    assert all(not parameter.requires_grad for parameter in model.target.parameters())
    for key, actual in model.target.state_dict().items():
        if torch.is_floating_point(actual):
            expected = alpha * target_before[key] + (1 - alpha) * online_before[key]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        else:
            torch.testing.assert_close(actual, online_before[key])
        torch.testing.assert_close(model.online.state_dict()[key], online_before[key])


def test_update_target_rejects_supervised_objective():
    """A supervised model does not have a target predictor to update."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    with pytest.raises(RuntimeError, match="supervised"):
        model.update_target()


def test_target_changes_only_after_explicit_post_optimizer_update():
    """A TD optimizer step changes only online weights until target update."""
    torch.manual_seed(40)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, target_alpha=0.5
    )
    batch = {
        **make_state(2, 5),
        **make_state(2, 5, valid_lengths=[3, 0], prefix="next_"),
        "label": torch.tensor([1, 0]),
    }
    optimizer = torch.optim.AdamW(model.online.parameters(), lr=0.01)
    target_before = {
        key: value.clone() for key, value in model.target.state_dict().items()
    }

    optimizer.zero_grad()
    outputs = model(**batch)
    outputs["loss"].backward()
    assert all(parameter.grad is None for parameter in model.target.parameters())
    optimizer.step()

    # we confirm the optimizer step alone never updates the target
    for key, actual in model.target.state_dict().items():
        torch.testing.assert_close(actual, target_before[key])

    online_after_step = {
        key: value.clone() for key, value in model.online.state_dict().items()
    }
    model.update_target()
    key = "embedding.feature_embedding.weight"
    expected = target_before[key] * 0.5 + online_after_step[key] * 0.5
    torch.testing.assert_close(model.target.state_dict()[key], expected)
    assert not model.target.training


# ---------------------------------------------------------------------------
# T34: explicit AdamW optimizer and post-step TD target synchronization
# ---------------------------------------------------------------------------


_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "mimic4_mortality_td_cnn_lstm.py"
)
_EXAMPLE_SPEC = importlib.util.spec_from_file_location(
    "td_cnn_lstm_example_helpers", _EXAMPLE_PATH
)
assert _EXAMPLE_SPEC is not None and _EXAMPLE_SPEC.loader is not None
_example_helpers = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_example_helpers)
make_reference_optimizer = _example_helpers.make_reference_optimizer
training_step = _example_helpers.training_step


def test_reference_adamw_uses_released_code_formulas_and_online_only():
    """AdamW defaults match train.py, excluding frozen target parameters."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4)
    n_training_batches = 17
    optimizer = make_reference_optimizer(model, n_training_batches)
    expected_lr = 1.0 / math.sqrt(model.n_online_trainable_parameters)
    expected_decay = 1.0 / (expected_lr * n_training_batches)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(expected_lr)
    assert group["weight_decay"] == pytest.approx(expected_decay)
    assert {id(p) for p in group["params"]} == {
        id(p) for p in model.online.parameters() if p.requires_grad
    }
    assert not {id(p) for p in group["params"]} & {
        id(p) for p in model.target.parameters()
    }


def test_reference_optimizer_supports_documented_overrides():
    """Any deviation from reference hyperparameters can be passed explicitly."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    optimizer = make_reference_optimizer(
        model, 20, learning_rate=0.002, weight_decay=0.01
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    optimizer = make_reference_optimizer(model, 20, learning_rate=0.005)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(10.0)


@pytest.mark.parametrize("n_batches", [0, -1, 1.5, True])
def test_reference_optimizer_rejects_invalid_batch_counts(n_batches):
    """The reference formula is undefined for zero or invalid batch counts."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4)
    with pytest.raises((TypeError, ValueError), match="n_training_batches"):
        make_reference_optimizer(model, n_batches)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
        ("learning_rate", -0.1),
        ("learning_rate", True),
        ("weight_decay", -1.0),
        ("weight_decay", float("inf")),
        ("weight_decay", True),
    ],
)
def test_reference_optimizer_rejects_invalid_hyperparameters(parameter, value):
    """Reject invalid AdamW overrides instead of silently changing training."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4)
    with pytest.raises(ValueError, match=parameter):
        make_reference_optimizer(model, 10, **{parameter: value})


@pytest.mark.parametrize("objective", ["td", "supervised"])
def test_training_step_orders_gradients_optimizer_and_target_update(
    monkeypatch, objective
):
    """We must finish optimizer.step before synchronizing a TD target."""
    torch.manual_seed(37)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4,
        dropout=0.0, objective=objective, target_alpha=0.8,
    )
    optimizer = make_reference_optimizer(model, 100, learning_rate=0.001)
    batch = {**make_state(2, 4), "label": torch.tensor([0, 1])}
    if objective == "td":
        batch.update(make_state(2, 4, valid_lengths=[0, 2], prefix="next_"))

    # we initialize stale gradients to verify zero_grad is called
    for parameter in model.online.parameters():
        parameter.grad = torch.ones_like(parameter)

    events = []
    original_step = optimizer.step

    def checked_step(*args, **kwargs):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.online.parameters()
        )
        events.append("optimizer")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", checked_step)
    original_update = model.update_target

    def checked_update():
        assert events == ["optimizer"]
        events.append("target")
        return original_update()

    monkeypatch.setattr(model, "update_target", checked_update)
    before_online = next(model.online.parameters()).detach().clone()
    if objective == "td":
        before_target = next(model.target.parameters()).detach().clone()
    outputs = training_step(model, optimizer, batch)

    assert set(outputs) == {"loss", "y_prob", "y_true", "logit"}
    assert all(not value.requires_grad for value in outputs.values())
    assert torch.isfinite(outputs["loss"])
    assert not torch.equal(next(model.online.parameters()), before_online)
    if objective == "td":
        assert events == ["optimizer", "target"]
        torch.testing.assert_close(
            next(model.target.parameters()),
            0.8 * before_target + 0.2 * next(model.online.parameters()),
        )
        assert not model.target.training
        assert all(not p.requires_grad for p in model.target.parameters())
    else:
        assert events == ["optimizer"]


def test_training_step_requires_observed_labels():
    """Do not attempt backward when forward returns inference-only outputs."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    )
    optimizer = make_reference_optimizer(model, 5)
    with pytest.raises(ValueError, match="observed mortality label"):
        training_step(model, optimizer, make_state(2, 4))


# ---------------------------------------------------------------------------
# T35: constructor errors, CPU masks, padded states and finite forward shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "settings,error_type",
    [
        ({"n_features": 0}, ValueError),
        ({"n_features": True}, TypeError),
        ({"hidden_dim": 0}, ValueError),
        ({"hidden_dim": 3.5}, TypeError),
        ({"dropout": -0.1}, ValueError),
        ({"dropout": 1.0}, ValueError),
        ({"dropout": float("nan")}, ValueError),
        ({"use_delta_embeddings": 0}, TypeError),
        ({"objective": "invalid"}, ValueError),
        ({"target_alpha": -0.1}, ValueError),
        ({"target_alpha": 1.1}, ValueError),
    ],
)
def test_constructor_rejects_invalid_settings(settings, error_type):
    """Model configuration errors must be caught before building layers."""
    config = {"n_features": 3, "hidden_dim": 4, **settings}
    with pytest.raises(error_type):
        TDCNNLSTM(make_dataset(), **config)


@pytest.mark.parametrize("mode", ["td", "supervised"])
def test_eval_inference_returns_finite_scores_for_single_and_padded_states(mode):
    """One valid event or heavily padded sequences work in eval mode on CPU."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective=mode
    ).eval()
    for batch_size, lengths, sequence_length in [
        (1, [1], 1), (1, [1], 15), (2, [1, 11], 15),
    ]:
        outputs = model(**make_state(batch_size, sequence_length, lengths))
        assert set(outputs) == {"logit", "y_prob"}
        assert outputs["logit"].shape == (batch_size, 1)
        assert outputs["y_prob"].shape == (batch_size, 1)
        assert all(value.device.type == "cpu" for value in outputs.values())
        assert all(torch.isfinite(value).all() for value in outputs.values())
        assert torch.all((outputs["y_prob"] >= 0) & (outputs["y_prob"] <= 1))


def test_all_padded_current_is_rejected_in_both_modes():
    """Empty ICU states cannot produce a valid current-state prediction."""
    batch = make_state(2, 4, valid_lengths=[0, 4])
    for objective in ("td", "supervised"):
        model = TDCNNLSTM(
            make_dataset(), n_features=3, hidden_dim=4, objective=objective
        ).eval()
        with pytest.raises(ValueError, match="all padding"):
            model(**batch)


def test_cpu_event_and_independent_delta_masks_align_after_sorting():
    """We sort all channels together and mask missing deltas independently."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    state = {
        "timepoints": torch.tensor([[0.0, float("nan"), 20.0, 10.0]]),
        "values": torch.tensor([[3.0, float("nan"), 1.0, 2.0]]),
        "features": torch.tensor([[2, -1, 0, 1]]),
        "delta_time": torch.tensor([[0.0, float("nan"), float("nan"), 5.0]]),
        "delta_value": torch.tensor([[float("nan"), float("nan"), 0.2, 0.3]]),
    }
    prepared, masks, has_events = model._prepare_state(state)
    assert prepared["values"].device.type == "cpu"
    assert has_events.tolist() == [True]
    assert masks["event"].tolist() == [[True, True, True, False]]
    assert masks["delta_time"].tolist() == [[False, True, True, False]]
    assert masks["delta_value"].tolist() == [[True, True, False, False]]
    torch.testing.assert_close(prepared["timepoints"][0, :3],
                               torch.tensor([20.0, 10.0, 0.0]))
    assert prepared["features"].tolist() == [[0, 1, 2, -1]]


def test_valid_feature_zero_and_missing_delta_remain_distinct_from_padding():
    """Zero feature IDs and valid zero values are not padding sentinels."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    inputs = make_state(2, 4, valid_lengths=[2, 3])
    inputs["delta_time"][0, 0] = float("nan")
    inputs["delta_value"][1, 1] = float("nan")
    _, masks, _ = model._prepare_state(inputs)
    assert masks["event"][0, 0]
    assert not masks["delta_time"][0, 0]
    assert not masks["delta_value"][1, 1]
    outputs = model(**inputs)
    assert torch.isfinite(outputs["logit"]).all()


def test_model_training_rejects_singleton_batch_due_to_decoder_batchnorm():
    """One-example batches are allowed for eval but not training BatchNorm."""
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    ).train()
    with pytest.raises(ValueError, match="batch size of at least 2"):
        model(**make_state(1, 2), label=torch.tensor([1]))


def test_per_feature_scaling_preserves_missing_values_and_padding():
    """Feature-specific normalization uses training statistics on CPU."""
    scaling = {
        "timepoints": {"mean": 10.0, "std": 2.0},
        "values": {"mean": [1.0, 5.0, 9.0], "std": [2.0, 4.0, 2.0]},
        "delta_time": {"mean": [1.0, 2.0, 3.0], "std": [1.0, 2.0, 1.0]},
        "delta_value": {"mean": [0.0, 1.0, 2.0], "std": [1.0, 2.0, 1.0]},
    }
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, scaling=scaling
    ).eval()
    inputs = {
        "timepoints": [[20.0, 10.0, float("nan")]],
        "values": [[3.0, 9.0, float("nan")]],
        "features": [[0, 1, -1]],
        "delta_time": [[float("nan"), 4.0, float("nan")]],
        "delta_value": [[0.0, 3.0, float("nan")]],
    }
    prepared, masks, _ = model._prepare_state(inputs)
    torch.testing.assert_close(prepared["timepoints"][0, :2],
                               torch.tensor([5.0, 0.0]))
    torch.testing.assert_close(prepared["values"][0, :2],
                               torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(prepared["delta_time"][0, 1],
                               torch.tensor(1.0))
    torch.testing.assert_close(prepared["delta_value"][0, :2],
                               torch.tensor([0.0, 1.0]))
    assert torch.isnan(prepared["values"][0, 2])
    assert torch.isnan(prepared["delta_time"][0, 0])
    assert masks["event"].tolist() == [[True, True, False]]


# ---------------------------------------------------------------------------
# T36: end-to-end TD/supervised correctness and gradient isolation
# ---------------------------------------------------------------------------


def test_t36_td_forward_matches_manual_mixed_targets_and_logit_gradients():
    """Mixed successor/terminal TD loss uses detached risks and correct gradients."""
    torch.manual_seed(61)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    batch = {
        **make_state(3, 6, valid_lengths=[6, 3, 2]),
        **make_state(3, 6, valid_lengths=[5, 0, 4], prefix="next_"),
        "label": torch.tensor([1, 0, 1]),
    }

    outputs = model(**batch)
    next_risks, has_next = model._target_probability(batch)
    assert has_next.tolist() == [True, False, True]
    torch.testing.assert_close(
        outputs["y_true"], torch.tensor([[1.0], [0.0], [1.0]])
    )
    expected_targets = torch.where(has_next[:, None], next_risks, outputs["y_true"])
    expected_loss = F.binary_cross_entropy_with_logits(
        outputs["logit"], expected_targets
    )
    torch.testing.assert_close(outputs["loss"], expected_loss)

    outputs["logit"].retain_grad()
    outputs["loss"].backward()
    expected_logit_gradient = (
        outputs["y_prob"].detach() - expected_targets
    ) / len(batch["label"])
    torch.testing.assert_close(
        outputs["logit"].grad, expected_logit_gradient, atol=1e-6, rtol=1e-6
    )
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.online.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target.parameters())


def test_t36_supervised_forward_matches_manual_bce_and_logit_gradients():
    """Supervised mode trains on observed labels, without TD bootstrapping."""
    torch.manual_seed(62)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    ).eval()
    labels = torch.tensor([0, 1, 0])
    outputs = model(**make_state(3, 6, valid_lengths=[2, 6, 3]), label=labels)
    assert model.target is None
    assert set(outputs) == {"loss", "y_true", "y_prob", "logit"}
    torch.testing.assert_close(outputs["y_true"], labels.float()[:, None])
    torch.testing.assert_close(
        outputs["loss"],
        F.binary_cross_entropy_with_logits(outputs["logit"], outputs["y_true"]),
    )

    outputs["logit"].retain_grad()
    outputs["loss"].backward()
    expected_logit_gradient = (
        outputs["y_prob"].detach() - labels.float()[:, None]
    ) / len(labels)
    torch.testing.assert_close(
        outputs["logit"].grad, expected_logit_gradient, atol=1e-6, rtol=1e-6
    )
    assert any(parameter.grad is not None for parameter in model.online.parameters())


def test_t36_all_terminal_td_equals_supervised_with_same_online_weights():
    """For all-terminal batches TD and supervised must agree even on gradients."""
    torch.manual_seed(63)
    td_model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    supervised_model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    ).eval()
    supervised_model.online.load_state_dict(td_model.online.state_dict())
    current = make_state(3, 5, valid_lengths=[2, 5, 3])
    batch = {**current, "label": torch.tensor([0, 1, 1])}
    successors = make_state(3, 5, valid_lengths=[0, 0, 0], prefix="next_")

    td_outputs = td_model(**batch, **successors)
    supervised_outputs = supervised_model(**batch)
    for key in ("logit", "y_prob", "loss", "y_true"):
        torch.testing.assert_close(td_outputs[key], supervised_outputs[key])

    td_outputs["loss"].backward()
    supervised_outputs["loss"].backward()
    for (name_td, parameter_td), (name_sup, parameter_sup) in zip(
        td_model.online.named_parameters(),
        supervised_model.online.named_parameters(),
    ):
        assert name_td == name_sup
        assert parameter_td.grad is not None
        torch.testing.assert_close(parameter_td.grad, parameter_sup.grad)
    assert all(p.grad is None for p in td_model.target.parameters())


def test_t36_observed_labels_do_not_replace_nonterminal_td_targets():
    """Nonterminal labels affect y_true, but terminal labels affect TD loss."""
    torch.manual_seed(64)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    current = make_state(2, 5)
    successors = make_state(2, 5, valid_lengths=[3, 0], prefix="next_")
    batch = {**current, **successors}

    original = model(**batch, label=[0, 0])
    changed_nonterminal = model(**batch, label=[1, 0])
    torch.testing.assert_close(original["logit"], changed_nonterminal["logit"])
    torch.testing.assert_close(original["loss"], changed_nonterminal["loss"])
    assert original["y_true"].tolist() == [[0.0], [0.0]]
    assert changed_nonterminal["y_true"].tolist() == [[1.0], [0.0]]

    changed_terminal = model(**batch, label=[0, 1])
    next_risks, has_next = model._target_probability(batch)
    assert has_next.tolist() == [True, False]
    expected_targets = torch.where(
        has_next[:, None], next_risks, changed_terminal["y_true"]
    )
    torch.testing.assert_close(
        changed_terminal["loss"],
        F.binary_cross_entropy_with_logits(
            changed_terminal["logit"], expected_targets
        ),
    )
    assert changed_terminal["y_true"].tolist() == [[0.0], [1.0]]


def test_t36_td_training_backward_cannot_change_target_weights_or_bn():
    """Training-mode forward/backward updates online BN, never frozen target."""
    torch.manual_seed(65)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4)
    model.train()
    target_before = {
        key: tensor.detach().clone()
        for key, tensor in model.target.state_dict().items()
    }
    online_bn_before = model.online.embedding.norm.num_batches_tracked.item()
    batch = {
        **make_state(3, 6, valid_lengths=[4, 6, 2]),
        **make_state(3, 6, valid_lengths=[5, 0, 2], prefix="next_"),
        "label": torch.tensor([1, 0, 1]),
    }

    outputs = model(**batch)
    outputs["loss"].backward()
    assert model.online.embedding.norm.num_batches_tracked.item() > online_bn_before
    assert not model.target.training
    assert any(parameter.grad is not None for parameter in model.online.parameters())
    assert all(parameter.grad is None for parameter in model.target.parameters())
    for key, tensor in model.target.state_dict().items():
        torch.testing.assert_close(tensor, target_before[key])


def test_t36_all_terminal_td_never_invokes_target_predictor(monkeypatch):
    """Fully missing next states bypass target inference and use observed truth."""
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=4).eval()
    batch = {
        **make_state(2, 4),
        **make_state(2, 4, valid_lengths=[0, 0], prefix="next_"),
        "label": torch.tensor([1, 0]),
    }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Empty successor states must not be encoded")

    monkeypatch.setattr(model.target, "forward", fail_if_called)
    outputs = model(**batch)
    torch.testing.assert_close(
        outputs["loss"],
        F.binary_cross_entropy_with_logits(outputs["logit"], outputs["y_true"]),
    )


def test_t36_supervised_loss_is_independent_of_supplied_successor_states():
    """Supervised training must never bootstrap when valid next states exist."""
    torch.manual_seed(66)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=4, objective="supervised"
    ).eval()
    current = make_state(2, 5, valid_lengths=[2, 5])
    labels = torch.tensor([1, 0])
    successor = make_state(2, 5, valid_lengths=[3, 0], prefix="next_")

    without_successor = model(**current, label=labels)
    with_successor = model(**current, **successor, label=labels)
    for key in ("logit", "y_prob", "y_true", "loss"):
        torch.testing.assert_close(with_successor[key], without_successor[key])

# ---------------------------------------------------------------------------
# T37: delta ablation, embedding gradients, checkpoints and target EMA
# ---------------------------------------------------------------------------


def make_t37_state():
    """Create heterogeneous, fully observed events for embedding tests."""
    batch = make_state(4, 6, valid_lengths=[6, 5, 4, 3])
    for row in range(4):
        for index in range(6 - row):
            batch["timepoints"][row, index] *= 5.0
            batch["values"][row, index] += row * 0.4
            batch["delta_time"][row, index] = 2.0 + index + row
            batch["delta_value"][row, index] = (index - row) * 0.2
    return batch


def test_t37_ablation_removes_only_delta_embedding_branches():
    """No-delta models omit both scalar branches without changing the encoder."""
    full = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=8)
    reduced = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8,
        use_delta_embeddings=False,
    )

    assert full.online.embedding.scalar_keys == (
        "timepoints", "values", "delta_time", "delta_value"
    )
    assert reduced.online.embedding.scalar_keys == ("timepoints", "values")
    assert set(full.online.embedding.scalar_embeddings) == {
        "time", "value", "delta_time", "delta_value"
    }
    assert set(reduced.online.embedding.scalar_embeddings) == {"time", "value"}
    assert reduced.n_online_trainable_parameters < full.n_online_trainable_parameters
    assert type(reduced.online.encoder) is type(full.online.encoder)
    assert type(reduced.online.decoder) is type(full.online.decoder)
    assert reduced.target is not None
    assert "delta_time" not in reduced.target.embedding.scalar_embeddings
    assert "delta_value" not in reduced.target.embedding.scalar_embeddings


def test_t37_no_delta_predictions_ignore_delta_values_and_missingness():
    """Removing delta branches makes inference independent of both delta inputs."""
    torch.manual_seed(71)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8,
        use_delta_embeddings=False,
    ).eval()
    original = make_t37_state()
    changed = {key: value.clone() for key, value in original.items()}
    valid = torch.isfinite(changed["values"])
    changed["delta_time"][valid] *= 100.0
    changed["delta_value"][valid] -= 100.0
    changed["delta_time"][0, 0] = float("nan")
    changed["delta_value"][1, 1] = float("nan")

    with torch.no_grad():
        original_logits = model(**original)["logit"]
        changed_logits = model(**changed)["logit"]
    torch.testing.assert_close(original_logits, changed_logits, atol=0, rtol=0)


@pytest.mark.parametrize("channel", ["delta_time", "delta_value"])
def test_t37_full_embedding_responds_to_each_delta_channel(channel):
    """The complete event embedding responds independently to both deltas."""
    torch.manual_seed(72)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=8).eval()
    embedding = model.online.embedding

    # we set positive weights so neither delta MLP has dead ReLU units
    with torch.no_grad():
        network = embedding.scalar_embeddings[channel]
        for layer in (network[0], network[2]):
            layer.weight.fill_(0.25)
            layer.bias.fill_(0.5)

    first = make_t37_state()
    second = {key: value.clone() for key, value in first.items()}
    valid = torch.isfinite(second["values"])
    second[channel][valid] += 4.0

    first_state, first_masks, _ = model._prepare_state(first)
    second_state, second_masks, _ = model._prepare_state(second)
    with torch.no_grad():
        first_embedding = embedding(first_state, first_masks)
        second_embedding = embedding(second_state, second_masks)
    assert not torch.allclose(first_embedding, second_embedding)


def test_t37_full_training_backpropagates_through_all_five_branches():
    """Supervised loss trains every selected scalar embedding and feature IDs."""
    torch.manual_seed(73)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8, dropout=0.0,
        objective="supervised",
    ).train()
    for network in model.online.embedding.scalar_embeddings.values():
        # we keep the first ReLU active for all finite synthetic scalars
        with torch.no_grad():
            network[0].weight.fill_(0.1)
            network[0].bias.fill_(2.0)

    outputs = model(**make_t37_state(), label=torch.tensor([0, 1, 0, 1]))
    outputs["loss"].backward()
    feature_gradient = model.online.embedding.feature_embedding.weight.grad
    assert feature_gradient is not None
    assert float(feature_gradient.abs().sum()) > 0

    for name in ("time", "value", "delta_time", "delta_value"):
        network = model.online.embedding.scalar_embeddings[name]
        for layer in (network[0], network[2]):
            assert layer.weight.grad is not None, name
            assert torch.isfinite(layer.weight.grad).all(), name
            assert float(layer.weight.grad.abs().sum()) > 0, name


def test_t37_ablation_backpropagates_only_through_retained_branches():
    """No-delta training keeps feature, time and value gradients intact."""
    torch.manual_seed(74)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8, dropout=0.0,
        use_delta_embeddings=False, objective="supervised",
    ).train()
    for network in model.online.embedding.scalar_embeddings.values():
        with torch.no_grad():
            network[0].weight.fill_(0.1)
            network[0].bias.fill_(2.0)

    outputs = model(**make_t37_state(), label=torch.tensor([0, 1, 0, 1]))
    outputs["loss"].backward()
    assert model.online.embedding.feature_embedding.weight.grad is not None
    for name in ("time", "value"):
        network = model.online.embedding.scalar_embeddings[name]
        assert network[0].weight.grad is not None
        assert network[2].weight.grad is not None
        assert float(network[2].weight.grad.abs().sum()) > 0
    assert "delta_time" not in model.online.embedding.scalar_embeddings
    assert "delta_value" not in model.online.embedding.scalar_embeddings


@pytest.mark.parametrize(
    "objective,use_delta_embeddings",
    [("td", True), ("supervised", False)],
)
def test_t37_checkpoint_roundtrip_preserves_predictions_and_scaling(
    tmp_path, objective, use_delta_embeddings
):
    """Save/load restores weights, normalization, and optional target state."""
    scaling = {
        "timepoints": {"mean": 3.0, "std": 2.0},
        "values": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 3.0, 4.0]},
        "delta_time": {"mean": [1.0, 2.0, 1.0], "std": [1.0, 2.0, 3.0]},
        "delta_value": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 2.0]},
    }
    torch.manual_seed(75)
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8,
        objective=objective, use_delta_embeddings=use_delta_embeddings,
        scaling=scaling, target_alpha=0.8,
    ).eval()
    if model.target is not None:
        with torch.no_grad():
            model.online.embedding.feature_embedding.weight.add_(0.3)
            model.online.embedding.norm.running_mean.add_(0.5)
            model.online.decoder[0].num_batches_tracked.add_(4)
        model.update_target()

    current = make_t37_state()
    successor = make_state(4, 6, valid_lengths=[4, 2, 0, 1], prefix="next_")
    with torch.no_grad():
        before = model(**current)
        if model.target is not None:
            before_target = model._target_probability({**current, **successor})

    checkpoint = tmp_path / "td_cnn_lstm_state.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8,
        objective=objective, use_delta_embeddings=use_delta_embeddings,
        target_alpha=0.8,
    ).eval()
    saved_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(saved_state)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert restored.values_mean.tolist() == scaling["values"]["mean"]
    assert restored.timepoints_std.item() == scaling["timepoints"]["std"]
    with torch.no_grad():
        after = restored(**current)
    torch.testing.assert_close(after["logit"], before["logit"], atol=0, rtol=0)
    torch.testing.assert_close(after["y_prob"], before["y_prob"], atol=0, rtol=0)
    if model.target is not None:
        after_target = restored._target_probability({**current, **successor})
        torch.testing.assert_close(after_target[0], before_target[0], equal_nan=True)
        torch.testing.assert_close(after_target[1], before_target[1])
        assert not restored.target.training
        assert all(not p.requires_grad for p in restored.target.parameters())
    else:
        assert restored.target is None


def test_t37_target_starts_equal_but_is_not_parameter_aliased():
    """Online and target networks begin identical but store separate tensors."""
    torch.manual_seed(76)
    model = TDCNNLSTM(make_dataset(), n_features=3, hidden_dim=8)
    for key, online_value in model.online.state_dict().items():
        target_value = model.target.state_dict()[key]
        torch.testing.assert_close(target_value, online_value, atol=0, rtol=0)
        assert online_value.data_ptr() != target_value.data_ptr(), key
    with torch.no_grad():
        model.online.embedding.feature_embedding.weight.add_(1.0)
    assert not torch.equal(
        model.online.embedding.feature_embedding.weight,
        model.target.embedding.feature_embedding.weight,
    )
    assert not model.target.training
    assert all(not p.requires_grad for p in model.target.parameters())


def test_t37_repeated_target_updates_match_analytic_ema():
    """Two target updates obey repeated EMA, including BatchNorm buffers."""
    alpha = 0.8
    model = TDCNNLSTM(
        make_dataset(), n_features=3, hidden_dim=8, target_alpha=alpha
    )
    with torch.no_grad():
        model.online.embedding.feature_embedding.weight.add_(2.0)
        model.online.embedding.norm.running_mean.fill_(3.0)
        model.online.decoder[0].num_batches_tracked.add_(5)

    target_before = {
        key: value.clone() for key, value in model.target.state_dict().items()
    }
    online_before = {
        key: value.clone() for key, value in model.online.state_dict().items()
    }
    for step in (1, 2):
        model.update_target()
        for key, actual in model.target.state_dict().items():
            if torch.is_floating_point(actual):
                expected = (
                    (alpha ** step) * target_before[key]
                    + (1 - alpha ** step) * online_before[key]
                )
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            else:
                torch.testing.assert_close(actual, online_before[key])
        for key, actual in model.online.state_dict().items():
            torch.testing.assert_close(actual, online_before[key])
        assert not model.target.training
        assert all(not p.requires_grad for p in model.target.parameters())


# ---------------------------------------------------------------------------
# T38: deterministic, one-step synthetic training in both objective modes
# ---------------------------------------------------------------------------


def _run_t38_synthetic_training_step(objective):
    """Run exactly one CPU training step from a repeatable synthetic state."""
    # we isolate random initialization from the rest of the test suite
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        model = TDCNNLSTM(
            make_dataset(),
            n_features=3,
            hidden_dim=4,
            dropout=0.0,
            objective=objective,
            target_alpha=0.9,
        ).cpu()

        # we exercise mixed sequence lengths and both observed mortality classes
        batch = {
            **make_state(2, 5, valid_lengths=[5, 2]),
            "label": torch.tensor([0, 1]),
        }
        if objective == "td":
            # we include one valid successor and one terminal fallback
            batch.update(
                make_state(2, 5, valid_lengths=[3, 0], prefix="next_")
            )

        online_before = {
            key: value.detach().clone()
            for key, value in model.online.state_dict().items()
        }
        target_before = (
            {
                key: value.detach().clone()
                for key, value in model.target.state_dict().items()
            }
            if model.target is not None
            else None
        )

        # we override decay for a tiny one-batch test, not a reproduction run
        optimizer = make_reference_optimizer(
            model, n_training_batches=1, learning_rate=0.001, weight_decay=0.0
        )
        outputs = training_step(model, optimizer, batch)
        return model, batch, online_before, target_before, outputs


@pytest.mark.parametrize("objective", ["td", "supervised"])
def test_t38_deterministic_one_step_synthetic_training(objective):
    """One optimizer step trains each mode reproducibly without clinical data."""
    first = _run_t38_synthetic_training_step(objective)
    second = _run_t38_synthetic_training_step(objective)
    model, batch, online_before, target_before, outputs = first
    repeated_model, _, _, _, repeated_outputs = second

    # we verify standard PyHealth outputs after an actual backward/AdamW step
    assert set(outputs) == {"loss", "y_prob", "y_true", "logit"}
    assert outputs["loss"].ndim == 0
    assert all(not value.requires_grad for value in outputs.values())
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert outputs["logit"].shape == (2, 1)
    assert outputs["y_prob"].shape == (2, 1)
    assert outputs["y_true"].shape == (2, 1)
    assert torch.equal(outputs["y_true"].flatten(), batch["label"].float())
    assert bool(((outputs["y_prob"] >= 0) & (outputs["y_prob"] <= 1)).all())

    # we verify the optimizer modified trainable weights and gradients are finite
    assert any(
        not torch.equal(parameter.detach(), online_before[name])
        for name, parameter in model.online.named_parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool((parameter.grad != 0).any())
        for parameter in model.online.parameters()
    )

    if objective == "td":
        # we verify the post-step target tracks online weights but has no grads
        assert model.target is not None and target_before is not None
        weight_name = "embedding.feature_embedding.weight"
        expected = (
            0.9 * target_before[weight_name]
            + 0.1 * model.online.state_dict()[weight_name]
        )
        torch.testing.assert_close(
            model.target.state_dict()[weight_name], expected, atol=1e-7, rtol=1e-6
        )
        assert not model.target.training
        assert all(
            parameter.grad is None and not parameter.requires_grad
            for parameter in model.target.parameters()
        )
    else:
        # we verify supervised training needs no target network or next states
        assert model.target is None and target_before is None
        assert not any(key.startswith("next_") for key in batch)

    # we verify a fresh run with the same seed produces identical results
    for key in outputs:
        torch.testing.assert_close(
            outputs[key], repeated_outputs[key], atol=0, rtol=0
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value, repeated_model.state_dict()[name], atol=0, rtol=0
        )
