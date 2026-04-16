"""
Tests for AdaptiveTransferModel, LSTMBackbone, and ResNetBackbone.

IMPORTANT: All tests use synthetic tensors only.
           No real data is loaded. Tests run in milliseconds.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset


# ── Synthetic Tensor Fixtures ──────────────────────────────────────────────────

def make_tensor_dataset(n: int = 20, n_classes: int = 19):
    """Make a small synthetic TensorDataset.

    Args:
        n: Number of samples.
        n_classes: Number of output classes.

    Returns:
        TensorDataset with X shape (n, 9, 125) and y shape (n,).
    """
    X = torch.randn(n, 9, 125)
    y = torch.randint(0, n_classes, (n,))
    return TensorDataset(X, y)


def make_source_datasets(n_sources: int = 4, n: int = 20):
    """Make dict of synthetic source domain datasets.

    Args:
        n_sources: Number of source domains.
        n: Samples per domain.

    Returns:
        Dict of sensor_id -> TensorDataset.
    """
    return {
        "s{}".format(i+1): make_tensor_dataset(n)
        for i in range(n_sources)
    }


# ── Backbone Tests ─────────────────────────────────────────────────────────────

class TestLSTMBackbone:
    """Tests for LSTMBackbone."""

    def test_output_shape(self):
        """Forward pass returns correct output shape."""
        from pyhealth.models.adaptive_transfer_model import LSTMBackbone
        model = LSTMBackbone(input_size=9, num_classes=19)
        x = torch.randn(4, 9, 125)
        out = model(x)
        assert out.shape == torch.Size([4, 19])

    def test_batch_size_1(self):
        """Works with batch size of 1."""
        from pyhealth.models.adaptive_transfer_model import LSTMBackbone
        model = LSTMBackbone()
        x = torch.randn(1, 9, 125)
        out = model(x)
        assert out.shape == torch.Size([1, 19])

    def test_gradients_flow(self):
        """Gradients flow through the model."""
        from pyhealth.models.adaptive_transfer_model import LSTMBackbone
        model = LSTMBackbone()
        x = torch.randn(4, 9, 125)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestResNetBackbone:
    """Tests for ResNetBackbone."""

    def test_output_shape(self):
        """Forward pass returns correct output shape."""
        from pyhealth.models.adaptive_transfer_model import ResNetBackbone
        model = ResNetBackbone(input_size=9, num_classes=19)
        x = torch.randn(4, 9, 125)
        out = model(x)
        assert out.shape == torch.Size([4, 19])

    def test_timestep_preserved(self):
        """Internal conv layers preserve timestep dimension."""
        from pyhealth.models.adaptive_transfer_model import ResNetBackbone
        model = ResNetBackbone()
        x = torch.randn(4, 9, 125)
        # Should not raise shape mismatch
        out = model(x)
        assert out.shape[0] == 4

    def test_gradients_flow(self):
        """Gradients flow through ResNet."""
        from pyhealth.models.adaptive_transfer_model import ResNetBackbone
        model = ResNetBackbone()
        x = torch.randn(4, 9, 125)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ── IPD Computation Tests ──────────────────────────────────────────────────────

class TestComputeIPD:
    """Tests for compute_ipd function."""

    def test_returns_float(self):
        """IPD computation returns a positive float."""
        from pyhealth.models.adaptive_transfer_model import compute_ipd
        src = np.random.randn(10, 9, 125).astype(np.float32)
        tgt = np.random.randn(10, 9, 125).astype(np.float32)
        ipd = compute_ipd(src, tgt, n_samples=20)
        assert isinstance(ipd, float)
        assert ipd >= 0.0

    def test_same_domain_low_ipd(self):
        """IPD of a domain with itself is lower than cross-domain IPD."""
        from pyhealth.models.adaptive_transfer_model import compute_ipd
        X = np.random.randn(20, 9, 125).astype(np.float32)
        Y = X + 5.0   # large shift = different domain
        ipd_same  = compute_ipd(X, X, n_samples=50)
        ipd_cross = compute_ipd(X, Y, n_samples=50)
        assert ipd_same < ipd_cross

    def test_euclidean_distance(self):
        """Euclidean distance metric runs without error."""
        from pyhealth.models.adaptive_transfer_model import compute_ipd
        src = np.random.randn(5, 9, 125).astype(np.float32)
        tgt = np.random.randn(5, 9, 125).astype(np.float32)
        ipd = compute_ipd(src, tgt, n_samples=10, distance="euclidean")
        assert ipd >= 0.0


# ── AdaptiveTransferModel Tests ────────────────────────────────────────────────

class TestAdaptiveTransferModel:
    """Tests for AdaptiveTransferModel."""

    def test_instantiation_lstm(self):
        """Model instantiates with LSTM backbone."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(backbone="lstm")
        assert model is not None

    def test_instantiation_resnet(self):
        """Model instantiates with ResNet backbone."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(backbone="resnet")
        assert model is not None

    def test_invalid_backbone(self):
        """ValueError raised for unknown backbone."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        with pytest.raises(ValueError, match="backbone"):
            AdaptiveTransferModel(backbone="transformer")

    def test_forward_pass(self):
        """Forward pass returns correct output shape."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(backbone="lstm")
        x = torch.randn(4, 9, 125)
        out = model(x)
        assert out.shape == torch.Size([4, 19])

    def test_compute_all_ipds(self):
        """compute_all_ipds returns IPD for each source domain."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(backbone="lstm")
        sources = make_source_datasets(n_sources=3, n=15)
        target  = make_tensor_dataset(n=15)
        ipds = model.compute_all_ipds(sources, target)
        assert len(ipds) == 3
        for sid, val in ipds.items():
            assert isinstance(val, float)
            assert val >= 0.0

    def test_fit_runs(self):
        """Full fit pipeline runs without errors."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(
            backbone="lstm",
            epochs_per_source=1,
            epochs_target=2,
        )
        sources = make_source_datasets(n_sources=2, n=20)
        target  = make_tensor_dataset(n=20)
        model.fit(sources, target)  # should not raise

    def test_evaluate_returns_float(self):
        """evaluate returns a float between 0 and 1."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(
            backbone="lstm",
            epochs_per_source=1,
            epochs_target=1,
        )
        sources = make_source_datasets(n_sources=2, n=20)
        target_train = make_tensor_dataset(n=20)
        target_test  = make_tensor_dataset(n=10)
        model.fit(sources, target_train)
        rcc = model.evaluate(target_test)
        assert isinstance(rcc, float)
        assert 0.0 <= rcc <= 1.0

    def test_predict_shape(self):
        """predict returns array of correct length."""
        from pyhealth.models.adaptive_transfer_model import AdaptiveTransferModel
        model = AdaptiveTransferModel(
            backbone="lstm",
            epochs_per_source=1,
            epochs_target=1,
        )
        sources = make_source_datasets(n_sources=2, n=20)
        target_train = make_tensor_dataset(n=20)
        target_test  = make_tensor_dataset(n=10)
        model.fit(sources, target_train)
        preds = model.predict(target_test)
        assert len(preds) == 10
        assert all(0 <= p <= 18 for p in preds)
