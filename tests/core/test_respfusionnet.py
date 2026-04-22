"""
Unit tests for :class:`pyhealth.models.RespFusionNet`.

These tests use purely synthetic in-memory samples — no audio files, no
real ICBHI data — so the full suite runs in well under a second. The
goal is to cover the public contract of the model:

1. Instantiation (defaults + custom hyperparams + invalid configs).
2. Forward pass over a batched dataloader (output keys + shapes).
3. Backward pass produces gradients.
4. Modality ablation (audio-only, metadata-only, both).

Author:
    Andrew Zhao (andrew.zhao@aeroseal.com)
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RespFusionNet


_AUDIO_DIM = 32
_METADATA_DIM = 7
_OUTPUT_SCHEMA = {"label": "binary"}


def _make_samples(n: int = 4):
    """Build ``n`` deterministic synthetic samples with both modalities."""
    samples = []
    for i in range(n):
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "signal": [float((i + j) % 5) for j in range(_AUDIO_DIM)],
                "metadata": [
                    0.25 + 0.05 * i,  # normalized age
                    1.0,              # age-present
                    1.0 if i % 2 == 0 else 0.0,  # sex_M
                    0.0 if i % 2 == 0 else 1.0,  # sex_F
                    0.40,             # normalized BMI
                    1.0,              # BMI-present
                    0.20,             # normalized duration
                ],
                "label": i % 2,
            }
        )
    return samples


def _make_dataset(include_metadata: bool = True):
    """Build an in-memory SampleDataset for the given modality config."""
    samples = _make_samples()
    if include_metadata:
        input_schema = {"signal": "tensor", "metadata": "tensor"}
    else:
        # Drop metadata from samples too; TensorProcessor would otherwise
        # try to fit on a key it was not configured for.
        input_schema = {"signal": "tensor"}
        samples = [
            {k: v for k, v in s.items() if k != "metadata"} for s in samples
        ]
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=_OUTPUT_SCHEMA,
        dataset_name="test-respfusionnet",
    )


class TestRespFusionNetInit(unittest.TestCase):
    """Constructor behavior for RespFusionNet."""

    def setUp(self) -> None:
        self.dataset = _make_dataset(include_metadata=True)

    def test_default_instantiation(self):
        model = RespFusionNet(dataset=self.dataset)
        self.assertIsInstance(model, RespFusionNet)
        self.assertTrue(model.use_audio)
        self.assertTrue(model.use_metadata)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.audio_dim, _AUDIO_DIM)
        self.assertEqual(model.metadata_dim, _METADATA_DIM)
        # Both encoders registered when both modalities are enabled.
        self.assertIn("audio", model.encoders)
        self.assertIn("metadata", model.encoders)

    def test_custom_hyperparameters(self):
        model = RespFusionNet(
            dataset=self.dataset, hidden_dim=64, dropout=0.25,
        )
        self.assertEqual(model.hidden_dim, 64)
        self.assertAlmostEqual(model.dropout_p, 0.25)
        # Final classifier should map fusion_dim (= 2 * hidden_dim) to 1
        # for the binary label processor.
        final_linear = model.classifier[-1]
        self.assertIsInstance(final_linear, torch.nn.Linear)
        self.assertEqual(final_linear.in_features, 2 * 64)
        self.assertEqual(final_linear.out_features, 1)

    def test_both_modalities_disabled_raises(self):
        with self.assertRaises(ValueError):
            RespFusionNet(
                dataset=self.dataset, use_audio=False, use_metadata=False,
            )

    def test_missing_metadata_key_raises(self):
        dataset_no_meta = _make_dataset(include_metadata=False)
        with self.assertRaises(ValueError):
            # use_metadata=True but no metadata key in input_schema.
            RespFusionNet(dataset=dataset_no_meta, use_metadata=True)


class TestRespFusionNetForward(unittest.TestCase):
    """Forward / backward pass with both modalities enabled."""

    def setUp(self) -> None:
        self.dataset = _make_dataset(include_metadata=True)
        self.model = RespFusionNet(dataset=self.dataset, hidden_dim=16)
        self.loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False,
        )

    def test_forward_output_keys(self):
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        for key in ("logit", "y_prob", "loss", "y_true"):
            self.assertIn(key, out)

    def test_forward_shapes(self):
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        # Binary task => output_size=1. Batch size=2.
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertEqual(out["loss"].dim(), 0)
        # y_prob sits in [0, 1] after the sigmoid inside prepare_y_prob.
        self.assertTrue(torch.all(out["y_prob"] >= 0.0))
        self.assertTrue(torch.all(out["y_prob"] <= 1.0))

    def test_backward_produces_gradients(self):
        batch = next(iter(self.loader))
        out = self.model(**batch)
        out["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        self.assertTrue(
            has_grad,
            "No parameters received non-zero gradients after backward.",
        )


class TestRespFusionNetAblation(unittest.TestCase):
    """Audio-only / metadata-only variants still train and predict."""

    def setUp(self) -> None:
        self.dataset = _make_dataset(include_metadata=True)
        self.loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False,
        )

    def _assert_forward_ok(self, model: RespFusionNet) -> None:
        batch = next(iter(self.loader))
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))
        out["loss"].backward()

    def test_audio_only(self):
        model = RespFusionNet(
            dataset=self.dataset, hidden_dim=16,
            use_audio=True, use_metadata=False,
        )
        self.assertIn("audio", model.encoders)
        self.assertNotIn("metadata", model.encoders)
        self.assertEqual(model.metadata_dim, 0)
        # fusion_dim should equal hidden_dim when only one branch is on.
        self.assertEqual(model.classifier[-1].in_features, 16)
        self._assert_forward_ok(model)

    def test_metadata_only(self):
        model = RespFusionNet(
            dataset=self.dataset, hidden_dim=16,
            use_audio=False, use_metadata=True,
        )
        self.assertIn("metadata", model.encoders)
        self.assertNotIn("audio", model.encoders)
        self.assertEqual(model.audio_dim, 0)
        self.assertEqual(model.classifier[-1].in_features, 16)
        self._assert_forward_ok(model)

    def test_both_modalities(self):
        model = RespFusionNet(
            dataset=self.dataset, hidden_dim=16,
            use_audio=True, use_metadata=True,
        )
        self.assertEqual(model.classifier[-1].in_features, 32)
        self._assert_forward_ok(model)


if __name__ == "__main__":
    unittest.main()
