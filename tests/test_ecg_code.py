"""Fast focused unit tests for ECGCODE model forward pass and gradients."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from pyhealth.models.ecg_code import ECGCODE


class _DummySampleDataset:
    """Minimal dataset shim compatible with BaseModel."""

    input_schema = {"signal": "tensor"}
    output_schema = {"mask": "tensor"}


def _make_model(interval_size: int = 16, width_mult: float = 0.25) -> ECGCODE:
    # width_mult=0.25 keeps tests very fast while preserving behavior.
    return ECGCODE(
        dataset=cast(Any, _DummySampleDataset()),
        interval_size=interval_size,
        width_mult=width_mult,
    )


def test_instantiation_and_invalid_interval() -> None:
    model = _make_model(interval_size=16)
    assert model.interval_size == 16

    with pytest.raises(ValueError, match="interval_size must be positive"):
        _make_model(interval_size=0)


def test_extract_tensor_supports_tensor_and_tuple() -> None:
    model = _make_model()
    x = torch.randn(2, 64)

    assert model._extract_tensor(x) is x
    assert model._extract_tensor((None, x, "ignored")) is x

    with pytest.raises(
        ValueError, match="Expected a tensor or tuple containing a tensor"
    ):
        model._extract_tensor(("a", "b"))


def test_normalize_signal_shape_variants() -> None:
    model = _make_model()

    x1 = torch.randn(64)  # [T]
    y1 = model._normalize_signal_shape(x1)
    assert y1.shape == (1, 1, 64)

    x2 = torch.randn(2, 64)  # [B, T]
    y2 = model._normalize_signal_shape(x2)
    assert y2.shape == (2, 1, 64)

    x3 = torch.randn(2, 64, 1)  # [B, T, C] -> transpose to [B, C, T]
    y3 = model._normalize_signal_shape(x3)
    assert y3.shape == (2, 1, 64)

    with pytest.raises(ValueError, match="Unsupported signal shape"):
        model._normalize_signal_shape(torch.randn(2, 3, 4, 5))


def test_build_interval_targets_shape_and_presence_flags() -> None:
    model = _make_model(interval_size=16)

    # B=1, T=32 => N=2 intervals.
    # interval 0 has P in [1..3], QRS in [5..7], T absent
    # interval 1 has T in [20..24], others absent
    mask = (torch.randn(1, 32).abs() * 0).long()
    mask[0, 1:4] = 1
    mask[0, 5:8] = 2
    mask[0, 20:25] = 3

    target = model._build_interval_targets(mask, n_intervals=2)
    assert target.shape == (1, 2, 3, 3)

    # confidence slots
    conf = target[0, :, :, 0]
    # interval 0: P/QRS present, T absent
    assert conf[0, 0].item() == 1.0
    assert conf[0, 1].item() == 1.0
    assert conf[0, 2].item() == 0.0
    # interval 1: T present only
    assert conf[1, 0].item() == 0.0
    assert conf[1, 1].item() == 0.0
    assert conf[1, 2].item() == 1.0


def test_forward_with_mask_output_shapes_and_range() -> None:
    torch.manual_seed(7)
    model = _make_model(interval_size=16)
    model.eval()

    signal = torch.randn(2, 1, 64)
    mask = (torch.randn(2, 64).abs() * 4).long()

    out = model(signal=signal, mask=mask)

    assert {"loss", "y_prob", "y_true", "logit", "cl_loss", "sel_loss"} <= set(
        out.keys()
    )
    assert out["loss"].ndim == 0
    assert out["cl_loss"].ndim == 0
    assert out["sel_loss"].ndim == 0

    # T=64, interval_size=16 => N=4
    assert out["logit"].shape == (2, 4, 3, 3)
    assert out["y_prob"].shape == (2, 4, 3, 3)
    assert out["y_true"].shape == (2, 4, 3, 3)

    # sigmoid output in [0,1]
    assert out["y_prob"].min().item() >= 0.0
    assert out["y_prob"].max().item() <= 1.0


def test_forward_without_mask_returns_zero_losses_and_targets() -> None:
    model = _make_model(interval_size=16)
    signal = torch.randn(2, 64)  # [B, T] also supported

    out = model(signal=signal)

    assert out["y_prob"].shape == (2, 4, 3, 3)
    assert out["y_true"].abs().sum().item() == pytest.approx(0.0)
    assert out["loss"].item() == pytest.approx(0.0)
    assert out["cl_loss"].item() == pytest.approx(0.0)
    assert out["sel_loss"].item() == pytest.approx(0.0)


def test_backward_computes_finite_gradients() -> None:
    torch.manual_seed(11)
    model = _make_model(interval_size=16)
    model.train()

    signal = torch.randn(2, 1, 64)
    mask = (torch.randn(2, 64).abs() * 4).long()

    out = model(signal=signal, mask=mask)
    out["loss"].backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), (
        "Expected at least one parameter to receive gradient"
    )
    assert all(
        (((g == g) & (g.abs() < float("inf"))).all().item())
        for g in grads
        if g is not None
    )


def test_forward_from_embedding_is_alias_of_forward() -> None:
    model = _make_model(interval_size=16)
    signal = torch.randn(2, 1, 64)
    mask = (torch.randn(2, 64).abs() * 4).long()

    out1 = model(signal=signal, mask=mask)
    out2 = model.forward_from_embedding(signal=signal, mask=mask)

    assert out1["y_prob"].shape == out2["y_prob"].shape
    assert out1["logit"].shape == out2["logit"].shape
