"""Tests for TaskAugResNet, TaskAugPolicy, and _ResNet1D.

All tests use synthetic in-memory tensors — no dataset files required.
Signal length is set to 128 throughout so each test completes in milliseconds
on CPU (AdaptiveAvgPool1d handles any input length).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Short signal length so every test runs in milliseconds on CPU.
_T = 128


# ---------------------------------------------------------------------------
# Mock dataset helper
# ---------------------------------------------------------------------------

def _mock_dataset(output_size: int = 1) -> MagicMock:
    """Minimal SampleDataset mock for TaskAugResNet instantiation."""
    ds = MagicMock()
    ds.input_schema = {"ecg": "tensor"}
    ds.output_schema = {"label": "binary"}
    proc = MagicMock()
    proc.size.return_value = output_size
    ds.output_processors = {"label": proc}
    return ds


# ---------------------------------------------------------------------------
# _ResNet1D — instantiation, forward pass, output shapes, gradients
# ---------------------------------------------------------------------------

class TestResNet1D:
    def test_output_shape(self):
        from pyhealth.models.taskaug_resnet import _ResNet1D

        model = _ResNet1D(in_channels=12, num_classes=1)
        out = model(torch.randn(4, 12, _T))
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_variable_input_lengths(self):
        """AdaptiveAvgPool1d should handle any T >= 1."""
        from pyhealth.models.taskaug_resnet import _ResNet1D

        model = _ResNet1D(in_channels=12, num_classes=1)
        for length in (64, 128, 256):
            assert model(torch.randn(2, 12, length)).shape == (2, 1)

    def test_batch_size_one(self):
        from pyhealth.models.taskaug_resnet import _ResNet1D

        model = _ResNet1D(in_channels=12, num_classes=1)
        model.eval()
        assert model(torch.randn(1, 12, _T)).shape == (1, 1)

    def test_gradient_flow_through_backbone(self):
        """Gradients must flow from loss back to the input tensor."""
        from pyhealth.models.taskaug_resnet import _ResNet1D

        model = _ResNet1D(in_channels=12, num_classes=1)
        x = torch.randn(2, 12, _T, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_in_output(self):
        from pyhealth.models.taskaug_resnet import _ResNet1D

        model = _ResNet1D(in_channels=12, num_classes=1)
        out = model(torch.randn(4, 12, _T))
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# TaskAugPolicy — instantiation, forward pass, output shapes, gradients
# ---------------------------------------------------------------------------

class TestTaskAugPolicy:
    def test_output_shape_preserved(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy

        policy = TaskAugPolicy(num_stages=2)
        x = torch.randn(4, 12, _T)
        out = policy(x, torch.randint(0, 2, (4,)))
        assert out.shape == x.shape

    def test_gradient_flows_to_logits(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy

        policy = TaskAugPolicy(num_stages=2)
        x = torch.randn(4, 12, _T)
        policy(x, torch.randint(0, 2, (4,))).mean().backward()
        assert policy.logits.grad is not None
        assert not torch.isnan(policy.logits.grad).any()

    def test_gradient_flows_to_magnitudes(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy

        policy = TaskAugPolicy(num_stages=2)
        x = torch.randn(4, 12, _T)
        policy(x, torch.randint(0, 2, (4,))).mean().backward()
        assert policy.mag_neg.grad is not None
        assert policy.mag_pos.grad is not None

    def test_all_positive_labels(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy

        policy = TaskAugPolicy()
        x = torch.randn(3, 12, _T)
        out = policy(x, torch.ones(3, dtype=torch.long))
        assert out.shape == x.shape

    def test_all_negative_labels(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy

        policy = TaskAugPolicy()
        x = torch.randn(3, 12, _T)
        out = policy(x, torch.zeros(3, dtype=torch.long))
        assert out.shape == x.shape

    def test_single_stage(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy

        policy = TaskAugPolicy(num_stages=1)
        x = torch.randn(2, 12, _T)
        assert policy(x, torch.randint(0, 2, (2,))).shape == x.shape

    def test_learnable_parameter_count(self):
        from pyhealth.models.taskaug_resnet import TaskAugPolicy, _NUM_OPS

        policy = TaskAugPolicy(num_stages=2)
        n_params = sum(p.numel() for p in policy.parameters())
        # logits (2×N) + mag_neg (2×N) + mag_pos (2×N) = 3 × 2 × N_OPS
        assert n_params == 3 * 2 * _NUM_OPS


# ---------------------------------------------------------------------------
# TaskAugResNet — instantiation, forward pass, output shapes, gradients
# ---------------------------------------------------------------------------

class TestTaskAugResNet:
    def test_instantiation(self):
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        assert hasattr(model, "policy")
        assert hasattr(model, "backbone")

    def test_forward_no_label_output_keys(self):
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        model.eval()
        out = model(ecg=torch.randn(4, 12, _T))
        assert "logit" in out
        assert "y_prob" in out
        assert "loss" not in out

    def test_forward_output_shapes(self):
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        model.eval()
        out = model(ecg=torch.randn(4, 12, _T))
        assert out["logit"].shape == (4, 1)
        assert out["y_prob"].shape == (4, 1)

    def test_forward_with_label_output_keys(self):
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        model.train()
        out = model(ecg=torch.randn(4, 12, _T),
                    label=torch.randint(0, 2, (4,)))
        assert "loss" in out
        assert "y_true" in out
        assert out["loss"].ndim == 0   # scalar

    def test_y_prob_bounded(self):
        """y_prob must lie in [0, 1] (sigmoid output)."""
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        model.eval()
        out = model(ecg=torch.randn(8, 12, _T))
        assert (out["y_prob"] >= 0).all() and (out["y_prob"] <= 1).all()

    def test_no_augmentation_in_eval_mode(self):
        """Identical inputs must produce identical outputs in eval mode."""
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        model.eval()
        x = torch.randn(2, 12, _T)
        label = torch.zeros(2, dtype=torch.long)
        out1 = model(ecg=x.clone(), label=label)
        out2 = model(ecg=x.clone(), label=label)
        torch.testing.assert_close(out1["logit"], out2["logit"])

    def test_augmentation_stochastic_in_train_mode(self):
        """Two forward passes in train mode should produce different outputs."""
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        torch.manual_seed(0)
        model = TaskAugResNet(_mock_dataset())
        model.train()
        x = torch.randn(4, 12, _T)
        labels = torch.randint(0, 2, (4,))
        out1 = model(ecg=x.clone(), label=labels)
        out2 = model(ecg=x.clone(), label=labels)
        assert not torch.allclose(out1["logit"], out2["logit"])

    def test_gradient_computation_full_model(self):
        """Every trainable parameter must receive a gradient after backward."""
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        model.train()
        out = model(ecg=torch.randn(2, 12, _T),
                    label=torch.randint(0, 2, (2,)))
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_policy_backbone_param_groups_disjoint(self):
        from pyhealth.models.taskaug_resnet import TaskAugResNet

        model = TaskAugResNet(_mock_dataset())
        policy_params = set(model.policy_parameters())
        backbone_params = set(model.backbone_parameters())
        assert len(policy_params) > 0
        assert len(backbone_params) > 0
        assert policy_params.isdisjoint(backbone_params)
