"""Tests for MultiViewContrastive model.

Uses small synthetic tensors (2-5 samples) so all tests complete in under
one second.  No real datasets are downloaded or used.
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MultiViewContrastive


def _make_dataset(n_samples: int = 4, n_channels: int = 1, length: int = 178):
    """Create a tiny synthetic dataset for testing."""
    rng = np.random.RandomState(42)
    samples = []
    for i in range(n_samples):
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": i % 3,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="test_mvc",
    )


class TestMultiViewContrastiveInit(unittest.TestCase):
    """Test model initialization with various configurations."""

    def setUp(self):
        self.dataset = _make_dataset()

    def test_default_init(self):
        model = MultiViewContrastive(dataset=self.dataset)
        self.assertEqual(model.encoder_type, "transformer")
        self.assertEqual(model.view_type, "ALL")
        self.assertEqual(model.fusion_type, "attention")
        self.assertEqual(model.num_embedding, 64)

    def test_cnn_init(self):
        model = MultiViewContrastive(
            dataset=self.dataset, encoder_type="cnn"
        )
        self.assertEqual(model.encoder_type, "cnn")

    def test_gru_init(self):
        model = MultiViewContrastive(
            dataset=self.dataset, encoder_type="gru"
        )
        self.assertEqual(model.encoder_type, "gru")

    def test_invalid_encoder_type(self):
        with self.assertRaises(AssertionError):
            MultiViewContrastive(
                dataset=self.dataset, encoder_type="lstm"
            )

    def test_invalid_view_type(self):
        with self.assertRaises(AssertionError):
            MultiViewContrastive(
                dataset=self.dataset, view_type="XYZ"
            )


class TestMultiViewContrastiveForward(unittest.TestCase):
    """Test forward pass for all encoder x view x fusion combos."""

    def setUp(self):
        self.dataset = _make_dataset()

    def _run_forward(self, encoder_type, view_type, fusion_type):
        model = MultiViewContrastive(
            dataset=self.dataset,
            encoder_type=encoder_type,
            view_type=view_type,
            fusion_type=fusion_type,
            num_embedding=16,
            num_hidden=32,
            num_head=2,
            num_layers=1,
            dropout=0.0,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertEqual(ret["y_prob"].shape[0], 4)
        self.assertEqual(ret["loss"].dim(), 0)
        return ret

    def test_transformer_all_attention(self):
        self._run_forward("transformer", "ALL", "attention")

    def test_transformer_all_concat(self):
        self._run_forward("transformer", "ALL", "concat")

    def test_transformer_all_mean(self):
        self._run_forward("transformer", "ALL", "mean")

    def test_cnn_all_attention(self):
        self._run_forward("cnn", "ALL", "attention")

    def test_gru_all_attention(self):
        self._run_forward("gru", "ALL", "attention")

    def test_transformer_single_view_T(self):
        self._run_forward("transformer", "T", "concat")

    def test_transformer_single_view_D(self):
        self._run_forward("transformer", "D", "mean")

    def test_transformer_single_view_F(self):
        self._run_forward("transformer", "F", "concat")

    def test_transformer_dual_view_TD(self):
        self._run_forward("transformer", "TD", "attention")

    def test_transformer_dual_view_TF(self):
        self._run_forward("transformer", "TF", "concat")

    def test_transformer_dual_view_DF(self):
        self._run_forward("transformer", "DF", "mean")

    def test_cnn_single_view(self):
        self._run_forward("cnn", "T", "concat")

    def test_gru_dual_view(self):
        self._run_forward("gru", "TD", "attention")


class TestMultiViewContrastiveBackward(unittest.TestCase):
    """Test gradient computation."""

    def setUp(self):
        self.dataset = _make_dataset()

    def test_gradient_flow_transformer(self):
        model = MultiViewContrastive(
            dataset=self.dataset,
            encoder_type="transformer",
            num_embedding=16,
            num_hidden=32,
            num_head=2,
            num_layers=1,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        ret = model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_gradient_flow_cnn(self):
        model = MultiViewContrastive(
            dataset=self.dataset,
            encoder_type="cnn",
            num_embedding=16,
            num_hidden=32,
            num_layers=2,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        ret = model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_gradient_flow_gru(self):
        model = MultiViewContrastive(
            dataset=self.dataset,
            encoder_type="gru",
            num_embedding=16,
            num_hidden=32,
            num_layers=2,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        ret = model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in model.parameters()
        )
        self.assertTrue(has_grad)


class TestMultiViewContrastiveEmbed(unittest.TestCase):
    """Test embedding output and encode_views helper."""

    def setUp(self):
        self.dataset = _make_dataset()

    def test_embed_output(self):
        model = MultiViewContrastive(
            dataset=self.dataset,
            num_embedding=16,
            num_hidden=32,
            num_head=2,
            num_layers=1,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 4)
        self.assertEqual(ret["embed"].dim(), 2)

    def test_encode_views(self):
        model = MultiViewContrastive(
            dataset=self.dataset,
            num_embedding=16,
            num_hidden=32,
            num_head=2,
            num_layers=1,
        )
        x = torch.randn(2, 1, 178)
        with torch.no_grad():
            latents = model.encode_views(x)
        self.assertEqual(len(latents), 3)
        for v in ["t", "d", "f"]:
            self.assertIn(v, latents)
            self.assertEqual(latents[v].shape, (2, 32))


class TestComputeViews(unittest.TestCase):
    """Test static view computation."""

    def test_shapes(self):
        x = torch.randn(4, 1, 178)
        xt, dx, xf = MultiViewContrastive.compute_views(x)
        self.assertEqual(xt.shape, (4, 178, 1))
        self.assertEqual(dx.shape, (4, 178, 1))
        self.assertEqual(xf.shape, (4, 178, 1))

    def test_multichannel(self):
        x = torch.randn(2, 3, 206)
        xt, dx, xf = MultiViewContrastive.compute_views(x)
        self.assertEqual(xt.shape, (2, 206, 3))
        self.assertEqual(dx.shape, (2, 206, 3))
        self.assertEqual(xf.shape, (2, 206, 3))


if __name__ == "__main__":
    unittest.main()
