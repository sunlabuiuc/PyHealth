"""Unit tests for JambaEHR hybrid Transformer-Mamba model.

Author: Joshua Steier

Tests cover:
    - Layer schedule generation
    - JambaLayer forward pass shapes
    - JambaEHR model init/forward/backward with various ratios
    - Edge cases: pure Transformer, pure Mamba, single-layer configs
    - Compatibility with PyHealth Trainer pipeline
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.jamba_ehr import (
    JambaEHR,
    JambaLayer,
    build_layer_schedule,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_sample_dataset():
    """Create a minimal PyHealth SampleDataset for testing."""
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "diagnoses": ["A", "B", "C"],
            "procedures": ["X", "Y"],
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v0",
            "diagnoses": ["D"],
            "procedures": ["Z", "Y"],
            "label": 0,
        },
    ]
    input_schema = {
        "diagnoses": "sequence",
        "procedures": "sequence",
    }
    output_schema = {"label": "binary"}
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test_jamba",
    )


def _make_batch(dataset):
    """Get a single batch from the sample dataset."""
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    return next(iter(loader))


# ------------------------------------------------------------------ #
# build_layer_schedule tests
# ------------------------------------------------------------------ #

class TestBuildLayerSchedule(unittest.TestCase):
    """Tests for the layer interleaving schedule builder."""

    def test_default_jamba_ratio(self):
        """1:7 ratio produces correct distribution."""
        schedule = build_layer_schedule(1, 7)
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule.count("transformer"), 1)
        self.assertEqual(schedule.count("mamba"), 7)

    def test_two_transformer_six_mamba(self):
        """2:6 ratio distributes Transformer layers evenly."""
        schedule = build_layer_schedule(2, 6)
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule.count("transformer"), 2)
        self.assertEqual(schedule.count("mamba"), 6)
        # Transformer layers should not be adjacent
        t_indices = [i for i, s in enumerate(schedule) if s == "transformer"]
        self.assertGreaterEqual(t_indices[1] - t_indices[0], 2)

    def test_pure_transformer(self):
        """All Transformer layers when num_mamba=0."""
        schedule = build_layer_schedule(4, 0)
        self.assertEqual(schedule, ["transformer"] * 4)

    def test_pure_mamba(self):
        """All Mamba layers when num_transformer=0."""
        schedule = build_layer_schedule(0, 4)
        self.assertEqual(schedule, ["mamba"] * 4)

    def test_empty(self):
        """Empty schedule when both counts are 0."""
        self.assertEqual(build_layer_schedule(0, 0), [])

    def test_even_split(self):
        """Equal number of both layer types."""
        schedule = build_layer_schedule(4, 4)
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule.count("transformer"), 4)
        self.assertEqual(schedule.count("mamba"), 4)

    def test_single_each(self):
        """Minimal case: one of each."""
        schedule = build_layer_schedule(1, 1)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule.count("transformer"), 1)
        self.assertEqual(schedule.count("mamba"), 1)


# ------------------------------------------------------------------ #
# JambaLayer tests
# ------------------------------------------------------------------ #

class TestJambaLayer(unittest.TestCase):
    """Tests for the hybrid encoder layer stack."""

    def test_output_shapes(self):
        """Output tensors have correct shapes."""
        layer = JambaLayer(
            feature_size=32,
            num_transformer_layers=1,
            num_mamba_layers=3,
            heads=2,
        )
        x = torch.randn(4, 10, 32)
        emb, cls_emb = layer(x)
        self.assertEqual(emb.shape, (4, 10, 32))
        self.assertEqual(cls_emb.shape, (4, 32))

    def test_with_mask(self):
        """Forward pass works with a padding mask."""
        layer = JambaLayer(
            feature_size=64,
            num_transformer_layers=1,
            num_mamba_layers=2,
            heads=4,
        )
        x = torch.randn(3, 8, 64)
        mask = torch.ones(3, 8)
        mask[0, 5:] = 0  # pad last 3 positions for first sample
        emb, cls_emb = layer(x, mask)
        self.assertEqual(emb.shape, (3, 8, 64))
        self.assertEqual(cls_emb.shape, (3, 64))

    def test_pure_transformer_layer(self):
        """JambaLayer with 0 Mamba layers is pure Transformer."""
        layer = JambaLayer(
            feature_size=32,
            num_transformer_layers=3,
            num_mamba_layers=0,
            heads=2,
        )
        self.assertEqual(len(layer.layers), 3)
        self.assertTrue(all(s == "transformer" for s in layer.schedule))
        x = torch.randn(2, 5, 32)
        emb, cls_emb = layer(x)
        self.assertEqual(emb.shape, (2, 5, 32))

    def test_pure_mamba_layer(self):
        """JambaLayer with 0 Transformer layers is pure Mamba."""
        layer = JambaLayer(
            feature_size=32,
            num_transformer_layers=0,
            num_mamba_layers=3,
        )
        self.assertEqual(len(layer.layers), 3)
        self.assertTrue(all(s == "mamba" for s in layer.schedule))
        x = torch.randn(2, 5, 32)
        emb, cls_emb = layer(x)
        self.assertEqual(emb.shape, (2, 5, 32))

    def test_gradient_flow(self):
        """Gradients flow through all layer types."""
        layer = JambaLayer(
            feature_size=32,
            num_transformer_layers=1,
            num_mamba_layers=2,
            heads=2,
        )
        x = torch.randn(2, 5, 32, requires_grad=True)
        emb, cls_emb = layer(x)
        cls_emb.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)


# ------------------------------------------------------------------ #
# JambaEHR model tests
# ------------------------------------------------------------------ #

class TestJambaEHR(unittest.TestCase):
    """Tests for the full JambaEHR model."""

    @classmethod
    def setUpClass(cls):
        """Create shared dataset and batch for all tests."""
        cls.dataset = _make_sample_dataset()
        cls.batch = _make_batch(cls.dataset)

    def _make_model(self, **kwargs):
        """Helper to create JambaEHR with default test params."""
        defaults = dict(
            dataset=self.dataset,
            embedding_dim=32,
            num_transformer_layers=1,
            num_mamba_layers=2,
            heads=2,
        )
        defaults.update(kwargs)
        return JambaEHR(**defaults)

    def test_forward_keys(self):
        """Output dict contains expected keys."""
        model = self._make_model()
        out = model(**self.batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)

    def test_backward(self):
        """Loss backward pass succeeds."""
        model = self._make_model()
        out = model(**self.batch)
        out["loss"].backward()

    def test_output_shapes(self):
        """Logit and probability shapes match label size."""
        model = self._make_model()
        out = model(**self.batch)
        batch_size = self.batch["label"].shape[0]
        self.assertEqual(out["logit"].shape[0], batch_size)
        self.assertEqual(out["y_prob"].shape[0], batch_size)

    def test_embed_flag(self):
        """embed=True returns the patient embedding."""
        model = self._make_model(
            num_transformer_layers=1,
            num_mamba_layers=1,
        )
        batch = dict(self.batch)
        batch["embed"] = True
        out = model(**batch)
        self.assertIn("embed", out)
        batch_size = self.batch["label"].shape[0]
        num_features = len(self.dataset.input_schema)
        self.assertEqual(out["embed"].shape, (batch_size, num_features * 32))

    def test_pure_transformer_config(self):
        """Model works with 0 Mamba layers (pure Transformer)."""
        model = self._make_model(
            num_transformer_layers=3,
            num_mamba_layers=0,
        )
        out = model(**self.batch)
        out["loss"].backward()

    def test_pure_mamba_config(self):
        """Model works with 0 Transformer layers (pure Mamba)."""
        model = self._make_model(
            num_transformer_layers=0,
            num_mamba_layers=3,
        )
        out = model(**self.batch)
        out["loss"].backward()

    def test_jamba_default_ratio(self):
        """Default 2:6 ratio (Jamba-like) works end to end."""
        model = self._make_model(
            embedding_dim=64,
            num_transformer_layers=2,
            num_mamba_layers=6,
            heads=4,
        )
        out = model(**self.batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_single_feature_key(self):
        """Model works with a single feature key."""
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "codes": ["A", "B"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v0",
                "codes": ["C"],
                "label": 0,
            },
        ]
        ds = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_single",
        )
        model = JambaEHR(
            dataset=ds,
            embedding_dim=32,
            num_transformer_layers=1,
            num_mamba_layers=1,
            heads=2,
        )
        loader = get_dataloader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_custom_hyperparameters(self):
        """Test JambaEHR with custom hyperparameters."""
        model = self._make_model(
            embedding_dim=64,
            num_transformer_layers=3,
            num_mamba_layers=5,
            heads=8,
            dropout=0.5,
        )
        out = model(**self.batch)
        self.assertIn("loss", out)

    def test_model_initialization(self):
        """Test that the JambaEHR model initializes correctly."""
        model = self._make_model()
        self.assertIsInstance(model, JambaEHR)
        self.assertEqual(model.label_key, "label")


if __name__ == "__main__":
    unittest.main()