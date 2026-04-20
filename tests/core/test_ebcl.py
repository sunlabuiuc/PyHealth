"""Tests for EBCL model."""

import torch

from pyhealth.models.ebcl import EBCL


def _make_dummy_batch(
    batch_size: int = 4,
    seq_len: int = 8,
    input_dim: int = 16,
):
    left_x = torch.randn(batch_size, seq_len, input_dim)
    right_x = torch.randn(batch_size, seq_len, input_dim)
    left_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    right_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return left_x, right_x, left_mask, right_mask


def test_ebcl_init():
    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=32,
        projection_dim=32,
        num_layers=2,
        num_heads=4,
    )
    assert isinstance(model, EBCL)
    assert model.input_dim == 16
    assert model.hidden_dim == 32
    assert model.projection_dim == 32


def test_ebcl_forward_shapes():
    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=32,
        projection_dim=24,
    )
    left_x, right_x, left_mask, right_mask = _make_dummy_batch(input_dim=16)

    out = model(
        left_x=left_x,
        right_x=right_x,
        left_mask=left_mask,
        right_mask=right_mask,
    )

    assert "left_emb" in out
    assert "right_emb" in out
    assert "loss" in out
    assert "contrastive_loss" in out

    assert out["left_emb"].shape == (4, 24)
    assert out["right_emb"].shape == (4, 24)
    assert out["patient_emb"].shape == (4, 48)
    assert out["loss"].ndim == 0


def test_ebcl_forward_with_classifier_binary():
    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=32,
        projection_dim=16,
        classifier_out_dim=1,
    )
    left_x, right_x, left_mask, right_mask = _make_dummy_batch(input_dim=16)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    out = model(
        left_x=left_x,
        right_x=right_x,
        left_mask=left_mask,
        right_mask=right_mask,
        y=y,
    )

    assert "logit" in out
    assert "y_prob" in out
    assert "supervised_loss" in out
    assert out["logit"].shape == (4, 1)
    assert out["y_prob"].shape == (4, 1)
    assert out["loss"].ndim == 0


def test_ebcl_backward_pass():
    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=32,
        projection_dim=16,
    )
    left_x, right_x, left_mask, right_mask = _make_dummy_batch(input_dim=16)

    out = model(
        left_x=left_x,
        right_x=right_x,
        left_mask=left_mask,
        right_mask=right_mask,
    )
    out["loss"].backward()

    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(grads) > 0


def test_ebcl_masked_forward():
    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=32,
        projection_dim=16,
    )
    left_x, right_x, left_mask, right_mask = _make_dummy_batch(input_dim=16)

    left_mask[0, -3:] = False
    right_mask[1, -2:] = False

    out = model(
        left_x=left_x,
        right_x=right_x,
        left_mask=left_mask,
        right_mask=right_mask,
    )

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["contrastive_loss"])


def test_ebcl_invalid_hidden_dim_num_heads():
    try:
        EBCL(
            dataset=None,
            input_dim=16,
            hidden_dim=30,
            projection_dim=16,
            num_heads=4,
        )
        assert False, "Expected ValueError for incompatible hidden_dim and num_heads"
    except ValueError:
        assert True


def test_ebcl_encode_single_sequence():
    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=32,
        projection_dim=20,
    )
    x = torch.randn(3, 10, 16)
    mask = torch.ones(3, 10, dtype=torch.bool)

    emb = model._encode_sequence(x, mask)
    assert emb.shape == (3, 20)
    assert torch.isfinite(emb).all()