"""Fast unit tests for the TPC model. All data is small and synthetic."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.tpc import (
    MAX_LOS_DAYS,
    MIN_LOS_DAYS,
    TPC,
    TPCLayer,
    compute_decay_indicators,
    msle_loss,
)

STEPS, FEATS, STATIC, DIAG = 6, 3, 2, 5


def _make_dataset(n: int = 4):
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(n):
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "ts": torch.randn(STEPS, FEATS, generator=g).tolist(),
                "decay": torch.rand(STEPS, FEATS, generator=g).tolist(),
                "static": torch.randn(STATIC, generator=g).tolist(),
                "diag": (torch.rand(DIAG, generator=g) > 0.5).float().tolist(),
                "los": 1.0 + i,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "ts": "tensor",
            "decay": "tensor",
            "static": "tensor",
            "diag": "tensor",
        },
        output_schema={"los": "regression"},
        dataset_name="tpc_test",
    )


def _small_model(dataset, **kwargs):
    params = dict(
        value_key="ts",
        decay_key="decay",
        temporal_channels=3,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        final_fc_size=5,
        main_dropout=0.0,
        temporal_dropout=0.0,
    )
    params.update(kwargs)
    return TPC(dataset, **params)


class TestTPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def test_instantiation(self):
        model = _small_model(self.dataset)
        self.assertEqual(model.mode, "regression")
        self.assertEqual(len(model.layers), 2)

    def test_forward_shapes_and_keys(self):
        model = _small_model(self.dataset)
        out = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertEqual(out["y_true"].shape, (4, 1))
        self.assertEqual(out["loss"].dim(), 0)

    def test_forward_sequence_shape_and_bounds(self):
        model = _small_model(self.dataset)
        seq = model.forward_sequence(**self.batch)
        self.assertEqual(seq.shape, (4, STEPS))
        self.assertTrue(bool((seq >= MIN_LOS_DAYS).all()))
        self.assertTrue(bool((seq <= MAX_LOS_DAYS).all()))

    def test_with_static_and_diagnoses(self):
        model = _small_model(self.dataset, static_key="static", diag_key="diag")
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))

    def test_no_decay_ablation(self):
        model = _small_model(self.dataset, decay_key=None)
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))

    def test_mse_loss_option(self):
        model = _small_model(self.dataset, loss="mse")
        self.assertTrue(torch.isfinite(model(**self.batch)["loss"]))

    def test_gradients_flow(self):
        model = _small_model(self.dataset, static_key="static", diag_key="diag")
        model(**self.batch)["loss"].backward()
        # BaseModel keeps an empty `_dummy_param` only to track the device.
        grads = [
            p.grad
            for name, p in model.named_parameters()
            if p.requires_grad and name != "_dummy_param"
        ]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in grads))

    def test_causality(self):
        """Changing the last hour must not change earlier predictions."""
        model = _small_model(self.dataset).eval()
        batch = {k: v.clone() for k, v in self.batch.items() if torch.is_tensor(v)}
        base = model.forward_sequence(**batch)
        batch["ts"][:, -1, :] += 10.0
        changed = model.forward_sequence(**batch)
        self.assertTrue(torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5))

    def test_layer_feature_growth(self):
        layer = TPCLayer(2, num_features=3, temporal_channels=4, pointwise_channels=5)
        h = torch.randn(2, 3 + 5, 5, STEPS)  # R^2 = F + Z, C = Y + 1
        x, decay = torch.randn(2, 3, STEPS), torch.rand(2, 3, STEPS)
        h_next, point = layer(h, x, decay, None, [torch.randn(2, 5, STEPS)])
        self.assertEqual(h_next.shape, (2, 3 + 5 * 2, 5, STEPS))
        self.assertEqual(point.shape, (2, 5, STEPS))

    def test_wrong_mode_raises(self):
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "ts": torch.randn(STEPS, FEATS).tolist(),
                "decay": torch.rand(STEPS, FEATS).tolist(),
                "label": i % 2,
            }
            for i in range(2)
        ]
        ds = create_sample_dataset(
            samples=samples,
            input_schema={"ts": "tensor", "decay": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="tpc_bad_mode",
        )
        with self.assertRaises(ValueError):
            TPC(ds, value_key="ts", decay_key="decay")

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            TPC(self.dataset, value_key="nope")


class TestHelpers(unittest.TestCase):
    def test_decay_indicators(self):
        obs = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
        out = compute_decay_indicators(obs)
        self.assertEqual(out.shape, obs.shape)
        self.assertAlmostEqual(out[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(out[0, 2, 0].item(), 0.75**2, places=5)
        self.assertEqual(out[0, 0, 1].item(), 0.0)  # never observed yet
        self.assertAlmostEqual(out[0, 2, 1].item(), 1.0)

    def test_msle(self):
        y = torch.tensor([1.0, 2.0])
        self.assertAlmostEqual(msle_loss(y, y).item(), 0.0, places=6)
        self.assertGreater(msle_loss(y, y * 2).item(), 0.0)

    def test_msle_penalises_proportional_error(self):
        short = msle_loss(torch.tensor([3.0]), torch.tensor([1.0]))
        long = msle_loss(torch.tensor([32.0]), torch.tensor([30.0]))
        self.assertGreater(short.item(), long.item())


if __name__ == "__main__":
    unittest.main()
