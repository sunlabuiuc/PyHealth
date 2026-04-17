"""Tests for CMAPriorEncoder.

Exercise the trainable encoder with synthetic tensor batches.
"""

import unittest
from types import SimpleNamespace

import torch


def _fake_sample_dataset_with_features(d: int = 4, n: int = 6):
    """Create a minimal SampleDataset-like object that has items."""
    samples = [
        {
            "features": torch.randn(d),
            "true_effect": torch.tensor([float(i)]),
        }
        for i in range(n)
    ]

    class FakeDS:
        input_schema = {"features": "tensor"}
        output_schema = {"true_effect": "regression"}

        def __getitem__(self, i):
            return samples[i]

        def __len__(self):
            return len(samples)

    return FakeDS()


class TestCMAPriorEncoder(unittest.TestCase):
    """Tests for CMAPriorEncoder."""

    def test_init_builds_expected_layers(self):
        """Encoder has encoder + head modules of expected shapes."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        model = CMAPriorEncoder(
            dataset=ds, hidden_dims=[16, 8], embed_dim=4,
        )
        self.assertEqual(model.input_dim, 4)
        self.assertEqual(model.embed_dim, 4)
        # Head should project embed -> 1 scalar
        self.assertEqual(model.head.out_features, 1)

    def test_init_validates_embed_dim(self):
        """embed_dim must be positive."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        with self.assertRaises(ValueError):
            CMAPriorEncoder(dataset=ds, embed_dim=0)

    def test_init_validates_dropout(self):
        """dropout must be in [0, 1)."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        with self.assertRaises(ValueError):
            CMAPriorEncoder(dataset=ds, dropout=1.0)

    def test_forward_shape(self):
        """Forward returns y_pred and embedding with correct shapes."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        model = CMAPriorEncoder(
            dataset=ds, hidden_dims=[8], embed_dim=3,
        )
        x = torch.randn(5, 4)
        out = model(features=x)
        self.assertEqual(out["y_pred"].shape, (5, 1))
        self.assertEqual(out["embedding"].shape, (5, 3))

    def test_loss_backprop(self):
        """Loss is computed and gradients flow."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        model = CMAPriorEncoder(dataset=ds, hidden_dims=[8], embed_dim=3)
        x = torch.randn(5, 4)
        y = torch.randn(5, 1)
        out = model(features=x, true_effect=y)
        self.assertIn("loss", out)
        out["loss"].backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_predict_prior_mean(self):
        """predict_prior_mean returns a 1D tensor."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        model = CMAPriorEncoder(dataset=ds, hidden_dims=[8], embed_dim=3)
        x = torch.randn(5, 4)
        m = model.predict_prior_mean(x)
        self.assertEqual(m.shape, (5,))

    def test_kernel_matrix_symmetry(self):
        """Kernel matrix is symmetric when X1 == X2."""
        from pyhealth.models.cma_prior_encoder import CMAPriorEncoder

        ds = _fake_sample_dataset_with_features(d=4)
        model = CMAPriorEncoder(dataset=ds, hidden_dims=[8], embed_dim=3)
        x = torch.randn(4, 4)
        k = model.predict_kernel_matrix(x)
        self.assertEqual(k.shape, (4, 4))
        self.assertTrue(torch.allclose(k, k.T, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
