import torch
from pyhealth.models.tpc import TPC


def test_tpc_forward_last_step():
    B, T, F, S = 2, 6, 5, 2

    x_values = torch.randn(B, T, F)
    x_decay = torch.rand(B, T, F)
    x_static = torch.randn(B, S)

    model = TPC(
        input_dim=F,
        static_dim=S,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.1,
        return_sequence=False,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=x_static)

    assert y.shape == (B,)
    assert torch.isfinite(y).all()


def test_tpc_forward_sequence():
    B, T, F, S = 2, 6, 5, 2

    x_values = torch.randn(B, T, F)
    x_decay = torch.rand(B, T, F)
    x_static = torch.randn(B, S)

    model = TPC(
        input_dim=F,
        static_dim=S,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.1,
        return_sequence=True,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=x_static)

    assert y.shape == (B, T)
    assert torch.isfinite(y).all()


def test_tpc_raises_on_mismatched_shapes():
    B, T, F, S = 2, 6, 5, 2

    x_values = torch.randn(B, T, F)
    x_decay = torch.rand(B, T, F + 1)  # intentional mismatch
    x_static = torch.randn(B, S)

    model = TPC(
        input_dim=F,
        static_dim=S,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.1,
        return_sequence=False,
    )

    try:
        _ = model(x_values=x_values, x_decay=x_decay, static=x_static)
        assert False, "Expected ValueError for mismatched x_values/x_decay shapes"
    except ValueError as e:
        assert "same shape" in str(e)


def test_tpc_forward_sequence_variable_length_batch():
    B1, B2, F, S = 1, 1, 5, 2
    T1, T2 = 10, 6

    x_values_1 = torch.randn(B1, T1, F)
    x_decay_1 = torch.rand(B1, T1, F)
    x_static_1 = torch.randn(B1, S)

    x_values_2 = torch.randn(B2, T2, F)
    x_decay_2 = torch.rand(B2, T2, F)
    x_static_2 = torch.randn(B2, S)

    pad_len = T1 - T2
    x_values_2 = torch.cat([x_values_2, torch.zeros(B2, pad_len, F)], dim=1)
    x_decay_2 = torch.cat([x_decay_2, torch.zeros(B2, pad_len, F)], dim=1)

    x_values = torch.cat([x_values_1, x_values_2], dim=0)
    x_decay = torch.cat([x_decay_1, x_decay_2], dim=0)
    x_static = torch.cat([x_static_1, x_static_2], dim=0)

    model = TPC(
        input_dim=F,
        static_dim=S,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.1,
        return_sequence=True,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=x_static)

    assert y.shape == (2, T1)
    assert torch.isfinite(y).all()
