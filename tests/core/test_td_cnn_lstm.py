"""Focused synthetic regressions for T21-T33 (no clinical data needed).

Place this file in ``tests/core/test_td_cnn_lstm.py`` in the PyHealth checkout.
"""

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
