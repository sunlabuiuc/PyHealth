import torch

from pyhealth.models.tpc import TPC


def build_model():
    return TPC(
        input_dim=3,
        static_dim=2,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.0,
    )


def test_tpc_instantiation():
    model = build_model()
    assert model is not None


def test_tpc_forward_shape():
    model = build_model()

    x = torch.randn(2, 5, 3)      # [B, T, F]
    s = torch.randn(2, 2)         # [B, S]

    y = model(x, s)

    assert y.shape == (2,)


def test_tpc_forward_no_static():
    model = TPC(
        input_dim=3,
        static_dim=0,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.0,
    )

    x = torch.randn(2, 5, 3)
    y = model(x, None)

    assert y.shape == (2,)


def test_tpc_backward():
    model = build_model()

    x = torch.randn(2, 5, 3)
    s = torch.randn(2, 2)
    target = torch.tensor([4.0, 7.0])

    y = model(x, s)
    loss = torch.mean((y - target) ** 2)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_tpc_output_finite_and_positive():
    model = build_model()

    x = torch.randn(4, 6, 3)
    s = torch.randn(4, 2)

    y = model(x, s)

    assert torch.isfinite(y).all()
    assert torch.all(y >= 0)
