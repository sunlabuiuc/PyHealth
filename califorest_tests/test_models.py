import torch
import torch.nn as nn
from califorest_tests.utils import create_synthetic_ehr

"""
Model unit tests using tiny synthetic tensors.

These tests verify:
- Model instantiation
- Forward pass correctness
- Output shape validation
- Gradient computation during backpropagation
"""

class TinyModel(nn.Module):
    def __init__(self, in_features=8):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)


def test_model_instantiation():
    #Model can be created successfully
    model = TinyModel()
    assert model is not None


def test_forward_pass():
    #Forward pass returns outputs with correct batch size.
    X, y = create_synthetic_ehr()
    model = TinyModel()

    x_tensor = torch.tensor(X, dtype=torch.float32)
    output = model(x_tensor)

    assert output.shape[0] == X.shape[0]


def test_backward_pass():
    #Backward pass computes gradients successfully
    X, y = create_synthetic_ehr()
    model = TinyModel()

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

    criterion = nn.BCEWithLogitsLoss()

    output = model(x_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()

    assert model.fc.weight.grad is not None