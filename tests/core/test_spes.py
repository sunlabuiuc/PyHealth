import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SPESResNet, SPESTransformer
from pyhealth.models.spes import MultiScaleResNet1D, SPESResponseEncoder


_N_STIM_CH = 8
_N_REC_CH = 10
_N_TIMESTEPS = 33


def _make_dataset(n_samples: int = 4, seed: int = 7):
    """Return a minimal synthetic SPES SampleDataset."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        x_stim = rng.normal(size=(2, _N_STIM_CH, _N_TIMESTEPS)).astype(np.float32)
        x_rec = rng.normal(size=(2, _N_REC_CH, _N_TIMESTEPS)).astype(np.float32)
        # Column 0 of mode 0 stores Euclidean distance; 0 means padded/invalid.
        x_stim[:, :, 0] = [10, 20, 0, 30, 40, 0, 50, 60]
        x_rec[:, :, 0] = [11, 0, 21, 31, 0, 41, 51, 61, 0, 71]
        samples.append({
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "X_stim": x_stim,
            "X_recording": x_rec,
            "electrode_lobes": np.array([i % 7], dtype=np.int64),
            "electrode_coords": rng.normal(size=3).astype(np.float32),
            "soz": i % 2,
        })
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "X_stim": "tensor",
            "X_recording": "tensor",
            "electrode_lobes": "tensor",
            "electrode_coords": "tensor",
        },
        output_schema={"soz": "binary"},
        dataset_name="test_spes",
    )


class TestMultiScaleResNet1D(unittest.TestCase):
    """Unit tests for the MultiScaleResNet1D backbone."""

    def test_output_dim_constant(self):
        """output_dim class attribute equals 256 * 3 = 768."""
        self.assertEqual(MultiScaleResNet1D.output_dim, 768)

    def test_output_shape_single_channel(self):
        """Single-channel input produces (batch, output_dim) embedding."""
        model = MultiScaleResNet1D(input_channel=1)
        with torch.no_grad():
            out = model(torch.randn(2, 1, 128))
        self.assertEqual(out.shape, (2, MultiScaleResNet1D.output_dim))

    def test_output_shape_two_channels(self):
        """Two-channel input produces (batch, output_dim) embedding."""
        model = MultiScaleResNet1D(input_channel=2)
        with torch.no_grad():
            out = model(torch.randn(3, 2, 64))
        self.assertEqual(out.shape, (3, MultiScaleResNet1D.output_dim))

    def test_short_signal_does_not_crash(self):
        """Very short signals are handled without error in eval mode."""
        model = MultiScaleResNet1D(input_channel=1)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 16))
        self.assertEqual(out.shape[1], MultiScaleResNet1D.output_dim)


class TestSPESResponseEncoder(unittest.TestCase):
    """Unit tests for the SPESResponseEncoder."""

    def _enc(self, **kwargs):
        defaults = dict(
            mean=True, std=True, embedding_dim=16, num_layers=1,
            dropout_rate=0.0, max_mlp_timesteps=16, expected_timesteps=32,
        )
        defaults.update(kwargs)
        return SPESResponseEncoder(**defaults)

    def _input(self, batch: int = 2):
        x = torch.randn(batch, 2, _N_STIM_CH, _N_TIMESTEPS)
        x[:, 0, :, 0] = torch.tensor([10, 20, 0, 30, 40, 0, 50, 60])
        return x

    def test_raises_if_neither_mean_nor_std(self):
        """At least one of mean/std must be enabled."""
        with self.assertRaises(ValueError):
            SPESResponseEncoder(mean=False, std=False)

    def test_output_shape_mean_and_std(self):
        enc = self._enc(mean=True, std=True)
        enc.eval()
        with torch.no_grad():
            out = enc(self._input())
        self.assertEqual(out.shape, (2, 16))

    def test_output_shape_mean_only(self):
        enc = self._enc(mean=True, std=False)
        enc.eval()
        with torch.no_grad():
            out = enc(self._input())
        self.assertEqual(out.shape, (2, 16))

    def test_conv_embedding_false(self):
        """MLP-only path (no ResNet) produces correct output shape."""
        enc = self._enc(conv_embedding=False, expected_timesteps=32)
        enc.eval()
        with torch.no_grad():
            out = enc(self._input())
        self.assertEqual(out.shape, (2, 16))

    def test_mlp_embedding_false(self):
        """ResNet-only path (no MLP prefix) produces correct output shape."""
        enc = self._enc(mlp_embedding=False)
        enc.eval()
        with torch.no_grad():
            out = enc(self._input())
        self.assertEqual(out.shape, (2, 16))

    def test_random_channels(self):
        """random_channels sub-sampling produces correct output shape."""
        enc = self._enc(random_channels=4)
        enc.eval()
        with torch.no_grad():
            out = enc(self._input())
        self.assertEqual(out.shape, (2, 16))

    def test_no_valid_channels_raises(self):
        """All-zero distance column (all padding) raises ValueError."""
        enc = self._enc(random_channels=4)
        x = torch.randn(1, 2, _N_STIM_CH, _N_TIMESTEPS)
        x[:, 0, :, 0] = 0
        with self.assertRaises(ValueError):
            enc(x)


class TestSPESResNet(unittest.TestCase):
    """Tests for the SPESResNet model."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4, shuffle=False)))

    def _model(self, **kwargs):
        defaults = dict(dataset=self.dataset, input_channels=6, noise_std=0.0)
        defaults.update(kwargs)
        return SPESResNet(**defaults)

    def test_invalid_input_type_raises(self):
        with self.assertRaises(ValueError):
            SPESResNet(dataset=self.dataset, input_type="invalid")

    def test_divergent_output_keys(self):
        """Forward pass returns loss, y_prob, y_true, and logit."""
        model = self._model(input_type="divergent")
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            with self.subTest(key=key):
                self.assertIn(key, ret)

    def test_divergent_output_shapes(self):
        model = self._model(input_type="divergent")
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch)
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["y_prob"].shape, (4, 1))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_backward_gradients_flow(self):
        model = self._model()
        ret = model(**self.batch)
        ret["loss"].backward()
        self.assertTrue(any(
            p.requires_grad and p.grad is not None for p in model.parameters()
        ))

    def test_embed_returned_when_requested(self):
        """embed=True adds an 'embed' key with shape (batch, output_dim)."""
        model = self._model()
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch, embed=True)
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (4, MultiScaleResNet1D.output_dim))

    def test_training_mode_runs(self):
        """Training mode (noise injection + channel dropout) does not crash."""
        model = self._model(noise_std=0.1)
        model.train()
        ret = model(**self.batch)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_all_padding_channels_raises(self):
        """Batch where all distance values are zero raises ValueError."""
        model = self._model()
        bad_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in self.batch.items()}
        bad_batch["X_stim"][:, 0, :, 0] = 0
        with self.assertRaises(ValueError):
            model(**bad_batch)


class TestSPESTransformer(unittest.TestCase):
    """Tests for the SPESTransformer model."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4, shuffle=False)))

    def _model(self, net_configs=None, **kwargs):
        defaults = dict(
            dataset=self.dataset,
            net_configs=net_configs or [{"type": "divergent", "mean": True, "std": True}],
            embedding_dim=16,
            num_layers=1,
            dropout_rate=0.0,
            noise_std=0.0,
            max_mlp_timesteps=16,
            expected_timesteps=32,
        )
        defaults.update(kwargs)
        return SPESTransformer(**defaults)

    def test_output_keys(self):
        """Forward pass returns loss, y_prob, y_true, and logit."""
        model = self._model()
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            with self.subTest(key=key):
                self.assertIn(key, ret)

    def test_output_shapes(self):
        model = self._model()
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch)
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["y_prob"].shape, (4, 1))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_backward_gradients_flow(self):
        model = self._model()
        ret = model(**self.batch)
        ret["loss"].backward()
        self.assertTrue(any(
            p.requires_grad and p.grad is not None for p in model.parameters()
        ))

    def test_multiple_net_configs(self):
        """Two encoders (convergent + divergent) concatenate without error."""
        model = self._model(net_configs=[
            {"type": "convergent", "mean": True, "std": True},
            {"type": "divergent", "mean": True, "std": False},
        ])
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch)
        self.assertEqual(ret["logit"].shape, (4, 1))

    def test_embed_returned_when_requested(self):
        """embed=True adds an 'embed' key with shape (batch, total_embedding_dim)."""
        model = self._model()
        model.eval()
        with torch.no_grad():
            ret = model(**self.batch, embed=True)
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (4, 16))

    def test_invalid_net_config_type_raises(self):
        """Unrecognised input type in net_configs raises ValueError."""
        model = self._model(net_configs=[{"type": "invalid", "mean": True, "std": True}])
        with self.assertRaises(ValueError):
            model(**self.batch)

    def test_training_mode_runs(self):
        """Training mode (noise injection + channel dropout) does not crash."""
        model = self._model(noise_std=0.1)
        model.train()
        ret = model(**self.batch)
        self.assertEqual(ret["loss"].dim(), 0)


if __name__ == "__main__":
    unittest.main()
