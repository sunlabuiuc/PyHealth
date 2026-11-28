"""Small deterministic selective SSM tests."""

import torch

from pyhealth.models.ehrmamba import SelectiveSSMBlock


def test_forward_shape_and_gradient():
    model = SelectiveSSMBlock(8, state_dim=3, conv_kernel=2)
    x = torch.randn(2, 4, 8, requires_grad=True)
    output = model(x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_left_padding_invariance():
    model = SelectiveSSMBlock(8, state_dim=3, conv_kernel=2)
    model.eval()
    x = torch.randn(1, 3, 8)
    padded = torch.cat([torch.zeros(1, 2, 8), x], dim=1)
    with torch.no_grad():
        expected = model(x)
        actual = model(padded, torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool))
    torch.testing.assert_close(actual[:, 2:], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[:, :2], torch.zeros(1, 2, 8))


def test_causality():
    model = SelectiveSSMBlock(8, state_dim=3, conv_kernel=2)
    model.eval()
    x = torch.randn(1, 4, 8)
    changed = x.clone()
    changed[:, 3] += 100
    with torch.no_grad():
        before = model(x)[:, :3]
        after = model(changed)[:, :3]
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
