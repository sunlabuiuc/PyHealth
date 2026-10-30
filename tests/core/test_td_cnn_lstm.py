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
