import torch
from pyhealth.models.smart_adv import SmartAdversary


def test_smart_adversary_output_shape():
    model = SmartAdversary(n_sensitive=1)
    model.eval() 
    x = torch.randn(10, 1)
    output = model(x)
    assert output.shape == (10, 1)


def test_smart_adversary_varied_batch():
    model = SmartAdversary(n_sensitive=1)
    model.eval() 
    for batch_size in [1, 5, 32, 100]:
        x = torch.randn(batch_size, 1)
        output = model(x)
        assert output.shape == (batch_size, 1)


def test_smart_adversary_multiple_forward():
    model = SmartAdversary(n_sensitive=1)
    model.eval()
    for _ in range(5):
        x = torch.randn(8, 1)
        output = model(x)
        assert output.shape == (8, 1)


def test_smart_adversary_different_hidden_size():
    model = SmartAdversary(n_sensitive=1, n_hidden=64)
    model.eval()
    x = torch.randn(10, 1)
    output = model(x)
    assert output.shape == (10, 1)


def test_smart_adversary_different_dropout():
    model = SmartAdversary(n_sensitive=1, dropout_rate=0.5)
    model.eval()
    x = torch.randn(10, 1)
    output = model(x)
    assert output.shape == (10, 1)


def test_smart_adversary_gradients():
    model = SmartAdversary(n_sensitive=1)
    x = torch.randn(4, 1, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
