"""Tests for CNN3DAD model for Alzheimer's disease classification.

This module contains comprehensive tests for:
- _make_norm helper function
- ConvBlock3D module
- CNN3DAD model instantiation and forward pass
- Gradient computation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.cnn3d_ad import CNN3DAD, ConvBlock3D, _make_norm


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def dataset():
    """Create synthetic ADNI-like dataset for testing."""
    samples = [
        {
            "patient_id": "p0",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([60.0]),
            "label": 0,  # CN
        },
        {
            "patient_id": "p1",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([72.0]),
            "label": 1,  # MCI
        },
        {
            "patient_id": "p2",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([68.0]),
            "label": 2,  # AD
        },
        {
            "patient_id": "p3",
            "scan": np.random.randn(1, 96, 96, 96).astype("float32"),
            "age": np.array([65.0]),
            "label": 0,  # CN
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"scan": "tensor", "age": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="adni_test",
    )


@pytest.fixture(scope="module")
def model(dataset):
    """Create CNN3DAD model with small widening factor for fast testing."""
    return CNN3DAD(dataset=dataset, widening_factor=1)


@pytest.fixture(scope="module")
def forward_out(model, dataset):
    """Run a single forward pass and cache the result."""
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
    with torch.no_grad():
        return model(**batch)


# =============================================================================
# Group 1: _make_norm tests
# =============================================================================


def test_make_norm_instance():
    """Test that _make_norm returns InstanceNorm3d for 'instance' type."""
    norm = _make_norm("instance", 32)
    assert isinstance(norm, nn.InstanceNorm3d)
    assert norm.num_features == 32


def test_make_norm_batch():
    """Test that _make_norm returns BatchNorm3d for 'batch' type."""
    norm = _make_norm("batch", 64)
    assert isinstance(norm, nn.BatchNorm3d)
    assert norm.num_features == 64


def test_make_norm_invalid():
    """Test that _make_norm raises ValueError for invalid type."""
    with pytest.raises(ValueError, match="Unknown norm_type"):
        _make_norm("invalid", 32)


# =============================================================================
# Group 2: ConvBlock3D tests
# =============================================================================


def test_convblock_output_shape():
    """Test ConvBlock3D output shape with 8x8x8 input."""
    block = ConvBlock3D(in_channels=1, out_channels=16, kernel_size=3)
    x = torch.randn(2, 1, 8, 8, 8)
    out = block(x)
    assert out.shape == (2, 16, 8, 8, 8)


def test_convblock_relu_activation():
    """Test that ConvBlock3D output is non-negative (ReLU applied)."""
    block = ConvBlock3D(in_channels=1, out_channels=16, kernel_size=3)
    x = torch.randn(2, 1, 8, 8, 8)
    out = block(x)
    assert (out >= 0).all(), "ReLU should ensure all outputs are non-negative"


# =============================================================================
# Group 3: CNN3DAD instantiation tests
# =============================================================================


def test_attributes_stored(model):
    """Test that model stores configuration attributes correctly."""
    assert model.scan_key == "scan"
    assert model.age_key == "age"
    assert model.label_key == "label"
    assert model.norm_type == "instance"


def test_backbone_length(model):
    """Test backbone has correct number of modules (4 conv + 4 pool = 8)."""
    assert len(model.backbone) == 8


def test_age_pe_buffer_shape(model):
    """Test age positional encoding buffer has correct shape."""
    assert hasattr(model, "age_pe")
    assert model.age_pe.shape == (240, 512)


def test_no_relu_in_age_mlp(model):
    """Test that age MLP has no ReLU activation (per reference)."""
    for module in model.age_fc.modules():
        assert not isinstance(module, nn.ReLU), "age_fc should not contain ReLU"


def test_age_disabled(dataset):
    """Test that age encoding can be disabled."""
    model_no_age = CNN3DAD(dataset=dataset, age_encoding_dim=0, widening_factor=1)
    assert model_no_age.age_fc is None
    assert model_no_age.age_dropout is None
    assert not model_no_age.use_age_encoding


def test_batch_norm_variant(dataset):
    """Test that batch normalization variant works."""
    model_bn = CNN3DAD(dataset=dataset, norm_type="batch", widening_factor=1)
    # Check first conv block uses BatchNorm3d
    first_block = model_bn.backbone[0]
    has_batch_norm = any(
        isinstance(m, nn.BatchNorm3d) for m in first_block.block.modules()
    )
    assert has_batch_norm, "First block should use BatchNorm3d"


def test_widening_factor(dataset):
    """Test that widening factor scales channel counts."""
    model_w2 = CNN3DAD(dataset=dataset, widening_factor=2)
    # First conv block output channels should be 4 * 2 = 8
    first_conv = model_w2.backbone[0].block[0]
    assert first_conv.out_channels == 8


# =============================================================================
# Group 4: Forward pass tests (reuse forward_out fixture)
# =============================================================================


def test_output_keys(forward_out):
    """Test that forward pass returns all required keys."""
    assert "loss" in forward_out
    assert "y_prob" in forward_out
    assert "y_true" in forward_out
    assert "logit" in forward_out


def test_loss_scalar(forward_out):
    """Test that loss is a scalar."""
    assert forward_out["loss"].shape == ()


def test_logit_shape(forward_out):
    """Test logit shape is (batch_size, num_classes)."""
    assert forward_out["logit"].shape == (4, 3)


def test_y_prob_shape(forward_out):
    """Test y_prob shape is (batch_size, num_classes)."""
    assert forward_out["y_prob"].shape == (4, 3)


def test_y_prob_sums_to_one(forward_out):
    """Test that softmax probabilities sum to 1."""
    sums = forward_out["y_prob"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_y_true_passthrough(forward_out):
    """Test that y_true matches input labels."""
    expected = torch.tensor([0, 1, 2, 0])
    assert torch.equal(forward_out["y_true"].cpu(), expected)


def test_classifier_two_layer(model):
    """Test that classifier has exactly 2 Linear layers."""
    linear_count = sum(1 for m in model.classifier if isinstance(m, nn.Linear))
    assert linear_count == 2


# =============================================================================
# Group 5: Gradient tests (fresh models for each)
# =============================================================================


def test_grad_classifier(dataset):
    """Test that gradients flow to classifier weights."""
    model = CNN3DAD(dataset=dataset, widening_factor=1)
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
    out = model(**batch)
    out["loss"].backward()

    # Check classifier first layer has gradients
    classifier_linear = model.classifier[0]
    assert classifier_linear.weight.grad is not None
    assert classifier_linear.weight.grad.abs().sum() > 0


def test_grad_backbone(dataset):
    """Test that gradients flow to backbone conv layers."""
    model = CNN3DAD(dataset=dataset, widening_factor=1)
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
    out = model(**batch)
    out["loss"].backward()

    # Check first backbone conv has gradients
    first_conv = model.backbone[0].block[0]
    assert first_conv.weight.grad is not None
    assert first_conv.weight.grad.abs().sum() > 0


def test_grad_age_mlp(dataset):
    """Test that gradients flow to age MLP."""
    model = CNN3DAD(dataset=dataset, widening_factor=1)
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
    out = model(**batch)
    out["loss"].backward()

    # Check age_fc first linear has gradients
    age_linear = model.age_fc[0]
    assert age_linear.weight.grad is not None
    assert age_linear.weight.grad.abs().sum() > 0
