"""Tests for the Wav2Sleep model.

Uses small synthetic signals so every test completes in milliseconds.
No real PSG datasets are required.
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Wav2Sleep


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_samples(
    n: int = 4,
    n_channels: int = 1,
    length: int = 256,
    n_classes: int = 4,
    seed: int = 0,
):
    """Return a list of synthetic single-modality samples (ECG-like)."""
    rng = np.random.RandomState(seed)
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "ecg": rng.randn(n_channels, length).astype(np.float32),
            "label": i % n_classes,
        }
        for i in range(n)
    ]


def _make_multimodal_samples(
    n: int = 4,
    length_ecg: int = 256,
    length_ppg: int = 256,
    n_classes: int = 4,
    seed: int = 0,
):
    """Return multi-modality samples with ecg + ppg."""
    rng = np.random.RandomState(seed)
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "ecg": rng.randn(1, length_ecg).astype(np.float32),
            "ppg": rng.randn(1, length_ppg).astype(np.float32),
            "label": i % n_classes,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestWav2SleepSingleModality(unittest.TestCase):
    """Tests for Wav2Sleep with a single ECG modality."""

    def setUp(self):
        samples = _make_samples(n=4, n_channels=1, length=256, n_classes=4)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_single",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_initialization(self):
        """Model initialises with correct feature and label keys."""
        self.assertIsInstance(self.model, Wav2Sleep)
        self.assertIn("ecg", self.model.feature_keys)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("label", self.model.label_keys)
        self.assertEqual(self.model.feature_dim, 128)

    def test_forward_output_keys(self):
        """Forward pass returns the four required keys."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out, f"Missing key: {key}")

    def test_output_shapes(self):
        """Output tensors have the expected shapes."""
        batch = next(iter(self.loader))
        bs = batch["ecg"].shape[0]
        n_classes = 4
        with torch.no_grad():
            out = self.model(**batch)
        self.assertEqual(out["y_prob"].shape, (bs, n_classes))
        self.assertEqual(out["logit"].shape, (bs, n_classes))
        self.assertEqual(out["y_true"].shape, (bs,))
        self.assertEqual(out["loss"].dim(), 0)

    def test_y_prob_sums_to_one(self):
        """Softmax probabilities sum to 1 for each sample."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        sums = out["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_backward(self):
        """Loss backward produces gradients on model parameters."""
        batch = next(iter(self.loader))
        out = self.model(**batch)
        out["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameter received a gradient.")

    def test_embed_flag(self):
        """embed=True adds an 'embed' key with correct shape."""
        batch = next(iter(self.loader))
        batch["embed"] = True
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("embed", out)
        bs = batch["ecg"].shape[0]
        self.assertEqual(out["embed"].shape, (bs, self.model.feature_dim))


class TestWav2SleepMultiModality(unittest.TestCase):
    """Tests for Wav2Sleep with two input modalities (ECG + PPG)."""

    def setUp(self):
        samples = _make_multimodal_samples(
            n=4, length_ecg=256, length_ppg=256, n_classes=4
        )
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor", "ppg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_multi",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_feature_keys(self):
        """Both modality keys are registered."""
        self.assertIn("ecg", self.model.feature_keys)
        self.assertIn("ppg", self.model.feature_keys)
        self.assertEqual(len(self.model.feature_keys), 2)

    def test_forward_multimodal(self):
        """Forward pass succeeds with both modalities present."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

    def test_forward_single_modality_at_test_time(self):
        """Model degrades gracefully when only one modality is provided."""
        batch = next(iter(self.loader))
        # Simulate ECG-only inference by removing PPG from the batch
        ecg_only_batch = {k: v for k, v in batch.items() if k != "ppg"}
        with torch.no_grad():
            out = self.model(**ecg_only_batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)

    def test_output_shapes_multimodal(self):
        """Output shapes are correct with two modalities."""
        batch = next(iter(self.loader))
        bs = batch["ecg"].shape[0]
        n_classes = 4
        with torch.no_grad():
            out = self.model(**batch)
        self.assertEqual(out["y_prob"].shape, (bs, n_classes))
        self.assertEqual(out["logit"].shape, (bs, n_classes))


class TestWav2SleepHyperparameters(unittest.TestCase):
    """Tests for non-default hyperparameter configurations."""

    def setUp(self):
        samples = _make_samples(n=4, n_channels=1, length=256, n_classes=4)
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"ecg": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_hp",
        )
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_custom_feature_dim(self):
        """Custom feature_dim changes the embedding size."""
        model = Wav2Sleep(
            dataset=self.dataset,
            feature_dim=64,
            n_attention_heads=4,
        )
        self.assertEqual(model.feature_dim, 64)
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)
        self.assertIn("loss", out)

    def test_deeper_transformer(self):
        """More transformer layers still produces valid output."""
        model = Wav2Sleep(
            dataset=self.dataset,
            n_transformer_layers=4,
        )
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = model(**batch)
        self.assertIn("loss", out)

    def test_higher_dropout(self):
        """High dropout (0.5) during training does not crash."""
        model = Wav2Sleep(dataset=self.dataset, dropout=0.5)
        model.train()
        batch = next(iter(self.loader))
        out = model(**batch)
        out["loss"].backward()
        self.assertIn("loss", out)


class TestWav2SleepMultivariate(unittest.TestCase):
    """Tests for multi-channel (multivariate) input signals."""

    def setUp(self):
        rng = np.random.RandomState(42)
        # 2-channel respiratory belt (ABD + THX style)
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "resp": rng.randn(2, 256).astype(np.float32),
                "label": i % 4,
            }
            for i in range(4)
        ]
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"resp": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_wav2sleep_mv",
        )
        self.model = Wav2Sleep(dataset=self.dataset)
        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_multivariate_forward(self):
        """Model handles 2-channel input correctly."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
