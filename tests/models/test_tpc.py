"""
Unit tests for the Temporal Pointwise Convolution (TPC) model.

This module contains fast synthetic tests for validating the PyHealth TPC
model implementation. The tests are designed to run quickly and verify core
model behavior without requiring real datasets.

Overview:
    The test suite checks:

    1. Output tensor shapes in both sequence and last-step prediction modes.
    2. Forward-pass correctness across major ablation variants.
    3. Configuration validation for invalid branch settings.
    4. Gradient propagation through the model during backpropagation.

    These tests provide lightweight coverage of the TPC model's core API and
    expected tensor behavior.

Key Coverage:
    - Sequence output mode
    - Last-step output mode
    - Temporal-only, pointwise-only, shared-temporal, and no-skip variants
    - Invalid configuration handling
    - Backward pass / gradient computation

Inputs:
    Synthetic tensors generated in-memory:
        - x_values: [B, T, F]
        - x_decay: [B, T, F]
        - static: [B, S]

Outputs:
    - PyTest assertions validating tensor shapes, finite outputs, raised
      exceptions, and gradient flow

Implementation Notes:
    - Uses only small synthetic tensors for speed and reproducibility.
    - Does not depend on eICU, MIMIC-IV, or any external files.
    - Intended to satisfy the project requirement for fast model tests using
      pseudo data.

Example:
    >>> python -m pytest tests/models/test_tpc.py -q

This test module is part of the TPC replication pipeline and validates the
core model implementation in ``pyhealth.models.tpc``.
"""
import pytest
import torch

from pyhealth.models.tpc import TPC


def make_inputs(
    batch_size: int = 2,
    seq_len: int = 6,
    num_features: int = 5,
    static_dim: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create small synthetic inputs for TPC model tests.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_features: Number of time-series features.
        static_dim: Static feature dimension.

    Returns:
        Tuple of ``(x_values, x_decay, static)`` tensors.
    """
    x_values = torch.randn(batch_size, seq_len, num_features)
    x_decay = torch.rand(batch_size, seq_len, num_features)
    static = torch.randn(batch_size, static_dim)
    return x_values, x_decay, static


def test_tpc_sequence_output_shape() -> None:
    """Test that sequence mode returns ``[B, T]`` output."""
    x_values, x_decay, static = make_inputs()

    model = TPC(
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
        return_sequence=True,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=static)
    assert y.shape == (2, 6)


def test_tpc_last_step_output_shape() -> None:
    """Test that last-step mode returns ``[B]`` output."""
    x_values, x_decay, static = make_inputs()

    model = TPC(
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
        return_sequence=False,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=static)
    assert y.shape == (2,)


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"use_temporal": True, "use_pointwise": True},
        {"use_temporal": True, "use_pointwise": False},
        {"use_temporal": False, "use_pointwise": True},
        {"use_temporal": True, "use_pointwise": True, "shared_temporal": True},
        {"use_temporal": True, "use_pointwise": True, "use_skip_connections": False},
    ],
)
def test_tpc_ablation_variants_forward(model_kwargs: dict) -> None:
    """Test forward pass across ablation variants.

    Args:
        model_kwargs: Variant-specific keyword arguments.
    """
    x_values, x_decay, static = make_inputs()

    model = TPC(
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
        return_sequence=True,
        **model_kwargs,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=static)
    assert y.shape == (2, 6)
    assert torch.isfinite(y).all()


def test_tpc_requires_at_least_one_branch() -> None:
    """Test that the model rejects configuration with both branches disabled."""
    with pytest.raises(ValueError):
        TPC(
            input_dim=5,
            static_dim=3,
            temporal_channels=4,
            pointwise_channels=4,
            num_layers=2,
            kernel_size=3,
            fc_dim=8,
            use_temporal=False,
            use_pointwise=False,
        )


def test_tpc_backward_pass() -> None:
    """Test that gradients propagate through the model."""
    x_values, x_decay, static = make_inputs()

    model = TPC(
        input_dim=5,
        static_dim=3,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=8,
        return_sequence=True,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=static)
    loss = y.mean()
    loss.backward()

    has_grad = any(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    assert has_grad
