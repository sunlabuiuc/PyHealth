"""Unit tests for the FeatureAttentionLSTM model."""

from typing import Dict

import torch

from pyhealth.models import FeatureAttentionLSTM


class MockDataset:
    """Lightweight mock dataset to satisfy BaseModel initialization."""

    def __init__(self, feature_dim: int = 16, output_dim: int = 2) -> None:
        self.input_info = {"ts_features": {"input_dim": feature_dim}}
        self.output_info = {"label": {"output_dim": output_dim}}
        self.input_schema = {"ts_features": "timeseries"}
        self.output_schema = {"label": "binary"}


def _create_synthetic_batch(
    batch_size: int = 4,
    time_steps: int = 12,
    feature_dim: int = 16,
    num_classes: int = 2,
    mode: str = "binary",
) -> Dict[str, torch.Tensor]:
    """Generate deterministic synthetic time-series tensors for testing."""
    torch.manual_seed(42)

    x = torch.randn(batch_size, time_steps, feature_dim)

    if mode == "binary":
        y = torch.randint(0, 2, (batch_size,)).long()
    else:
        y = torch.randint(0, num_classes, (batch_size,)).long()

    return {"ts_features": x, "label": y}


def test_feature_attention_lstm_instantiation() -> None:
    """Verify model initializes cleanly with valid parameters."""
    mock_ds = MockDataset(feature_dim=16, output_dim=2)
    model = FeatureAttentionLSTM(
        dataset=mock_ds,
        feature_keys=["ts_features"],
        label_key="label",
        mode="binary",
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
    )

    assert model.hidden_dim == 32
    assert model.attention_matrix.in_features == 16
    assert model.attention_matrix.out_features == 16


def test_feature_attention_lstm_forward_and_shapes() -> None:
    """Verify forward output keys, shapes, and attention normalization."""
    batch_size, time_steps, feature_dim = 4, 12, 16
    mock_ds = MockDataset(feature_dim=feature_dim, output_dim=2)

    model = FeatureAttentionLSTM(
        dataset=mock_ds,
        feature_keys=["ts_features"],
        label_key="label",
        mode="binary",
        hidden_dim=32,
    )
    model.eval()
    batch = _create_synthetic_batch(
        batch_size=batch_size,
        time_steps=time_steps,
        feature_dim=feature_dim,
        mode="binary",
    )

    with torch.no_grad():
        output = model(**batch)

    assert "loss" in output
    assert "y_hat" in output
    assert "y_true" in output
    assert "attention_weights" in output

    assert output["y_hat"].shape == (batch_size, 2)
    assert output["attention_weights"].shape == (
        batch_size,
        time_steps,
        feature_dim,
    )

    weight_sums = output["attention_weights"].sum(dim=-1)
    torch.testing.assert_close(
        weight_sums,
        torch.ones_like(weight_sums),
        rtol=1e-5,
        atol=1e-5,
    )


def test_feature_attention_lstm_gradients() -> None:
    """Verify loss backpropagates through LSTM and attention layers."""
    mock_ds = MockDataset(feature_dim=10, output_dim=3)
    model = FeatureAttentionLSTM(
        dataset=mock_ds,
        feature_keys=["ts_features"],
        label_key="label",
        mode="multiclass",
        hidden_dim=16,
    )
    model.train()

    batch = _create_synthetic_batch(
        batch_size=3,
        time_steps=8,
        feature_dim=10,
        num_classes=3,
        mode="multiclass",
    )

    output = model(**batch)
    loss = output["loss"]

    assert loss.requires_grad
    loss.backward()

    assert model.attention_matrix.weight.grad is not None
    assert torch.any(model.attention_matrix.weight.grad != 0)
    assert model.fc.weight.grad is not None
