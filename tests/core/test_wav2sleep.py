"""Unit tests for :class:`~pyhealth.models.wav2sleep.Wav2Sleep` on synthetic data.

Builds a small on-disk :class:`~pyhealth.datasets.sample_dataset.SampleDataset`,
instantiates a reduced ``Wav2Sleep``, and checks forward outputs, shapes, and
gradients.
"""

from __future__ import annotations

import gc
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any

import litdata
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.sample_dataset import SampleBuilder, SampleDataset
from pyhealth.models.wav2sleep import SIGNAL_TO_SAMPLES_PER_EPOCH, Wav2Sleep

#: Samples per epoch for ECG in tests (must match encoder expectations).
SP = SIGNAL_TO_SAMPLES_PER_EPOCH["ECG"]


def _one_epoch_wave_list() -> list[float]:
    """Return one random epoch of waveform samples as a Python float list.

    Returns:
        List of length :data:`SP` for use in synthetic sample dicts.
    """
    return torch.randn(SP, dtype=torch.float32).tolist()


def _synthetic_samples(*, n_patients: int = 2) -> list[dict[str, Any]]:
    """Build a minimal list of patient samples for :class:`SampleBuilder`.

    Args:
        n_patients: Number of synthetic patients (clamped to ``1..5`` by the
            caller).

    Returns:
        One dict per patient with ids, ``ecg`` / ``ppg`` wave lists, and a
        ``sleep_stage`` label.
    """
    assert 1 <= n_patients <= 5
    out: list[dict[str, Any]] = []
    for i in range(n_patients):
        out.append(
            {
                "patient_id": f"synthetic_patient_{i}",
                "visit_id": f"synthetic_visit_{i}",
                "ecg": _one_epoch_wave_list(),
                "ppg": _one_epoch_wave_list(),
                "sleep_stage": i % 5,
            }
        )
    return out


def _build_tiny_model(dataset: SampleDataset) -> Wav2Sleep:
    """Construct a small ``Wav2Sleep`` for fast tests.

    Args:
        dataset: Fitted dataset from :meth:`TestWav2SleepSynthetic.setUpClass`.

    Returns:
        Model with 32-dim embeddings and fusion/temporal depth reduced.
    """
    return Wav2Sleep(
        dataset=dataset,
        embedding_dim=32,
        hidden_dim=32,
        num_classes=5,
        num_fusion_heads=2,
        num_fusion_layers=1,
        num_temporal_layers=2,
        temporal_kernel_size=3,
        dropout=0.0,
        use_paper_faithful=True,
    )


class TestWav2SleepSynthetic(unittest.TestCase):
    """End-to-end checks against a shared temporary :class:`SampleDataset`.

    Attributes:
        _tmpdir: Temporary directory holding the litdata-optimized dataset.
        dataset: Open :class:`SampleDataset` built in :meth:`setUpClass`.
    """

    _tmpdir: tempfile.TemporaryDirectory[str]
    dataset: SampleDataset

    @classmethod
    def setUpClass(cls) -> None:
        """Create schema, materialize chunks, and open the dataset."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        try:
            root = Path(cls._tmpdir.name) / "ds"
            root.mkdir(parents=True, exist_ok=False)

            samples = _synthetic_samples(n_patients=2)
            builder = SampleBuilder(
                input_schema={"ecg": "tensor", "ppg": "tensor"},
                output_schema={"sleep_stage": "multiclass"},
            )
            builder.fit(samples)
            builder.save(str(root / "schema.pkl"))
            litdata.optimize(
                fn=builder.transform,
                inputs=[{"sample": pickle.dumps(s)} for s in samples],
                output_dir=str(root),
                chunk_bytes="64MB",
                num_workers=0,
            )
            cls.dataset = SampleDataset(path=str(root))
        except Exception:
            cls._tmpdir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        """Drain the dataloader, close the dataset, and remove temp files."""
        if hasattr(cls, "dataset"):
            loader = get_dataloader(cls.dataset, batch_size=2, shuffle=False)
            for _ in loader:
                pass
            cls.dataset.close()
            del cls.dataset
        gc.collect()
        cls._tmpdir.cleanup()

    def _one_batch(self) -> dict[str, Any]:
        """Return the first batch from the test dataloader.

        Returns:
            Batch dict compatible with :meth:`Wav2Sleep.forward`.
        """
        return next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))

    def test_wav2sleep_synthetic_instantiation(self) -> None:
        """Model exposes expected hyperparameters and label key."""
        model = _build_tiny_model(self.dataset)
        self.assertIsInstance(model, Wav2Sleep)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.label_key, "sleep_stage")

    def test_wav2sleep_synthetic_forward_pass(self) -> None:
        """Forward run returns loss and logits without storing grads."""
        model = _build_tiny_model(self.dataset)
        batch = self._one_batch()
        with torch.no_grad():
            out = model(**batch)
        self.assertIn("loss", out)
        self.assertIn("logit", out)

    def test_wav2sleep_synthetic_output_shapes(self) -> None:
        """Loss is scalar; logits and probabilities match batch and class count."""
        model = _build_tiny_model(self.dataset)
        batch = self._one_batch()
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(out["loss"].shape, ())
        self.assertEqual(tuple(out["logit"].shape), (2, 5))
        self.assertEqual(tuple(out["y_prob"].shape), (2, 5))
        self.assertEqual(tuple(out["y_true"].shape), (2,))
        self.assertTrue(torch.isfinite(out["loss"]).all().item())

    def test_wav2sleep_synthetic_gradient_computation(self) -> None:
        """At least one trainable parameter receives a non-zero gradient."""
        model = _build_tiny_model(self.dataset)
        batch = self._one_batch()
        out = model(**batch)
        out["loss"].backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
                if p.requires_grad
            )
        )


if __name__ == "__main__":
    unittest.main()
