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
