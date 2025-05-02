# File: tests/models/test_trace_encoder.py

import torch
from pyhealth.models.trace_encoder import TRACEEncoder

def test_trace_encoder_shape():
    model = TRACEEncoder(input_channels=10, latent_dim=128)
    input_tensor = torch.randn(4, 10, 300)  # batch of 4 patients
    output = model(input_tensor)
    assert output.shape == (4, 128), f"Expected shape (4, 128), got {output.shape}"
