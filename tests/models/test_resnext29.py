import torch
from pyhealth.models import ResNeXt29


# ✅ PyHealth-compliant DummyDataset for unit testing
class DummyDataset:
    def __init__(self):
        self.task = "classification"
        self.input_schema = {"image": (torch.Tensor, (3, 32, 32))}
        self.output_schema = {"label": (int, 1)}
        self.output_size = 20  # Simulate CIFAR coarse-class task


def test_resnext29_forward_pass():
    dataset = DummyDataset()
    model = ResNeXt29(dataset=dataset, cardinality=8, depth=29, bottleneck_width=64)

    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    assert y.shape == (4, 20), "Output shape should be (batch_size, num_classes)"
    assert not torch.isnan(y).any(), "Output contains NaNs"


def test_resnext29_loss():
    dataset = DummyDataset()
    model = ResNeXt29(dataset=dataset, cardinality=8, depth=29, bottleneck_width=64)

    x = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 20, (4,))
    outputs = model(x)
    loss = model.compute_loss(outputs, labels)

    assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor"
    assert loss.item() > 0, "Loss should be positive"
