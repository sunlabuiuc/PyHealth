import torch
from pyhealth.models.resnet1d import ResNet1D


def test_resnet1d_output_shape():
    model = ResNet1D(input_channels=12, num_classes=30)
    dummy_input = torch.randn(2, 12, 250)  # (batch, channels, sequence_length)
    output = model(dummy_input)
    assert output.shape == (2, 30), f"Expected output shape (2, 30), got {output.shape}"
