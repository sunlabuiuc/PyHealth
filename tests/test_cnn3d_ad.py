# Authors: Paul Nguyen, Shayan Jaffar, William Lee
# Description: Pytest tests for cnn3d_ad.py. Tests cover:
#   - _make_norm helper function
#   - ConvBlock3D output shapes and value ranges
#   - CNN3DAD instantiation with various configs
#   - Forward pass output keys, types, and shapes
#   - Forward pass input shape flexibility
#   - Gradient computation through the model    

import pytest
import torch
import torch.nn as nn
import numpy as np

from pyhealth.datasets import create_sample_dataset

from pyhealth.models.cnn3d_ad import (
    CNN3DAD,    
    ConvBlock3D,
    _make_norm,
)

NUM_CLASSES = 3
BATCH = 2
SPATIAL = 96


def _make_dataset():
    samples = [
        {
            "patient_id": f"p{i}",
            "scan": np.random.randn(1, SPATIAL, SPATIAL, SPATIAL).astype("float32"),
            "age": np.array([70.0], dtype="float32"),
            "label": i % NUM_CLASSES,
        }
        for i in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"scan": "tensor", "age": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="test_adni",
    )


def _make_model(dataset, **kwargs) -> CNN3DAD:
    defaults = dict(
        scan_key="scan",
        age_key="age",
        label_key="label",
        widening_factor=1,
        num_blocks=4,
        age_encoding_dim=64,
    )
    defaults.update(kwargs)
    return CNN3DAD(dataset=dataset, **defaults).eval()


def _make_batch(model: CNN3DAD) -> dict:
    """Synthetic batch that bypasses the dataloader."""
    return {
        model.scan_key: torch.randn(BATCH, 1, SPATIAL, SPATIAL, SPATIAL),
        model.age_key: torch.tensor([[70.0], [75.0]]),
        model.label_key: torch.tensor([0, 2]),
    }



# 1. Fixtures
@pytest.fixture(scope="module")
def dataset():
    return _make_dataset()

@pytest.fixture(scope="module")
def model(dataset):
    return _make_model(dataset)

@pytest.fixture(scope="module")
def forward_out(model):
    batch = _make_batch(model)
    with torch.no_grad():
        return model(**batch)

# 2. _make_norm
def test_make_norm_instance_returns_correct_type():
    layer = _make_norm("instance", 16)
    assert isinstance(layer, nn.InstanceNorm3d)
    assert layer.affine is True


def test_make_norm_batch_returns_correct_type():
    layer = _make_norm("batch", 16)
    assert isinstance(layer, nn.BatchNorm3d)


def test_make_norm_invalid_raises():
    with pytest.raises(ValueError, match="norm_type must be"):
        _make_norm("group", 16)


# 3. ConvBlock3D
def test_conv_block_output_shape():
    # k=3 with default padding=1 preserves spatial dims
    block = ConvBlock3D(in_channels=1, out_channels=8, kernel_size=3, norm_type="instance")
    x = torch.randn(2, 1, 16, 16, 16)
    out = block(x)
    assert out.shape == (2, 8, 16, 16, 16)


def test_conv_block_output_is_non_negative():
    # ReLU at the end means all values >= 0
    block = ConvBlock3D(in_channels=1, out_channels=4, kernel_size=3, norm_type="instance")
    x = torch.randn(2, 1, 8, 8, 8)
    out = block(x)
    assert out.min().item() >= 0.0


# 4. CNN3DAD instantiation
def test_instantiation_sets_keys(model):
    assert model.scan_key == "scan"
    assert model.age_key == "age"
    assert model.label_key == "label"


def test_instantiation_backbone_length(model):
    # num_blocks=4 → 4 ConvBlock3D + 4 MaxPool3d = 8 children in Sequential
    assert len(model.backbone) == 8


def test_instantiation_age_encoding_enabled(model):
    assert model.use_age_encoding is True
    assert model.age_fc is not None
    assert hasattr(model, "age_pe")


def test_instantiation_age_encoding_disabled(dataset):
    m = _make_model(dataset, age_encoding_dim=0)
    assert m.use_age_encoding is False
    assert m.age_fc is None
    assert not hasattr(m, "age_pe")


def test_instantiation_batch_norm_variant(dataset):
    m = _make_model(dataset, norm_type="batch")
    first_block = m.backbone[0]
    assert isinstance(first_block.block[1], nn.BatchNorm3d)


def test_instantiation_widening_factor(dataset):
    m = _make_model(dataset, widening_factor=2)
    # _BLOCK_CHANNELS[0] * 2 = 4 * 2 = 8
    first_conv = m.backbone[0].block[0]
    assert first_conv.out_channels == 8


def test_age_pe_buffer_shape(model):
    assert model.age_pe.shape == (240, model.age_encoding_dim)

def test_age_mlp_layers(model):
    layer_types = [type(l) for l in model.age_fc]
    assert nn.Linear in layer_types
    assert nn.LayerNorm in layer_types

# 5. Forward pass — output keys, types, and shapes
def test_forward_returns_required_keys(forward_out):
    assert set(forward_out.keys()) == {"loss", "y_prob", "y_true", "logit"}


def test_forward_loss_is_scalar(forward_out):
    assert forward_out["loss"].shape == torch.Size([])


def test_forward_logit_shape(forward_out):
    assert forward_out["logit"].shape == (BATCH, NUM_CLASSES)


def test_forward_y_prob_shape(forward_out):
    assert forward_out["y_prob"].shape == (BATCH, NUM_CLASSES)


def test_forward_y_prob_sums_to_one(forward_out):
    sums = forward_out["y_prob"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


def test_forward_y_true_matches_labels(model):
    batch = _make_batch(model)
    with torch.no_grad():
        out = model(**batch)
    assert torch.equal(out["y_true"], batch[model.label_key])


# 6. Forward — input shape flexibility
def test_forward_accepts_4d_scan(model):
    # Scan without explicit channel dim: [B, D, H, W]
    batch = {
        model.scan_key: torch.randn(BATCH, SPATIAL, SPATIAL, SPATIAL),
        model.age_key: torch.tensor([[70.0], [75.0]]),
        model.label_key: torch.tensor([0, 1]),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out["logit"].shape == (BATCH, NUM_CLASSES)


def test_forward_accepts_1d_age(model):
    # Age as flat [B] rather than [B, 1]
    batch = {
        model.scan_key: torch.randn(BATCH, 1, SPATIAL, SPATIAL, SPATIAL),
        model.age_key: torch.tensor([70.0, 75.0]),
        model.label_key: torch.tensor([0, 1]),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out["logit"].shape == (BATCH, NUM_CLASSES)


def test_forward_no_age_encoding(dataset):
    m = _make_model(dataset, age_encoding_dim=0).eval()
    batch = _make_batch(m)
    with torch.no_grad():
        out = m(**batch)
    assert out["logit"].shape == (BATCH, NUM_CLASSES)

# 7. Gradient computation
def test_loss_backward_populates_classifier_gradients(dataset):
    m = _make_model(dataset).train()
    batch = _make_batch(m)
    out = m(**batch)
    out["loss"].backward()
    assert m.classifier.weight.grad is not None
    assert m.classifier.weight.grad.abs().sum().item() > 0


def test_gradients_flow_through_backbone(dataset):
    m = _make_model(dataset).train()
    batch = _make_batch(m)
    out = m(**batch)
    out["loss"].backward()
    first_conv = m.backbone[0].block[0]
    assert first_conv.weight.grad is not None
    assert first_conv.weight.grad.abs().sum().item() > 0


def test_gradients_flow_through_age_fc(dataset):
    m = _make_model(dataset).train()
    batch = _make_batch(m)
    out = m(**batch)
    out["loss"].backward()
    age_linear = m.age_fc[0]
    assert age_linear.weight.grad is not None
    assert age_linear.weight.grad.abs().sum().item() > 0
