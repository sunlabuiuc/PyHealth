"""Unit tests for :class:`~pyhealth.models.wav2sleep.Wav2Sleep` on synthetic data.

Raw synthetic patient dicts are written as ``pickle`` files under a
:class:`tempfile.TemporaryDirectory`, then read back so the test exercises real
disk I/O for **sample payloads**. A fitted ``schema.pkl`` lives alongside them.
:class:`~pyhealth.datasets.sample_dataset.InMemorySampleDataset` is built from
the **loaded** dicts so forwards stay fast without ``litdata.optimize`` (which
is what :class:`~pyhealth.datasets.sample_dataset.SampleDataset` normally needs
for chunk files). The temp tree is always deleted in
:meth:`TestWav2SleepSynthetic.tearDownClass`.
"""

from __future__ import annotations

import gc
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.sample_dataset import (
    InMemorySampleDataset,
    SampleBuilder,
    SampleDataset,
)
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


def _write_raw_samples_to_dir(samples: list[dict[str, Any]], raw_dir: Path) -> None:
    """Pickle each raw sample dict under ``raw_dir`` (``sample_000.pkl``, ...)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        path = raw_dir / f"sample_{i:03d}.pkl"
        with path.open("wb") as f:
            pickle.dump(sample, f)


def _load_raw_samples_from_dir(raw_dir: Path, n: int) -> list[dict[str, Any]]:
    """Load ``n`` pickled sample dicts from ``raw_dir`` in index order."""
    out: list[dict[str, Any]] = []
    for i in range(n):
        path = raw_dir / f"sample_{i:03d}.pkl"
        with path.open("rb") as f:
            out.append(pickle.load(f))
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
    """End-to-end checks: raw samples and schema on disk; dataset in RAM."""

    _tmpdir: tempfile.TemporaryDirectory[str]
    _n_samples: int
    dataset: InMemorySampleDataset

    @classmethod
    def setUpClass(cls) -> None:
        """Materialize raw samples + schema under a temp dir, then load RAM."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        try:
            root = Path(cls._tmpdir.name) / "sample_dataset_stub"
            root.mkdir(parents=True, exist_ok=False)
            raw_dir = root / "raw_samples"

            samples = _synthetic_samples(n_patients=2)
            cls._n_samples = len(samples)
            _write_raw_samples_to_dir(samples, raw_dir)
            loaded = _load_raw_samples_from_dir(raw_dir, cls._n_samples)

            builder = SampleBuilder(
                input_schema={"ecg": "tensor", "ppg": "tensor"},
                output_schema={"sleep_stage": "multiclass"},
            )
            builder.fit(loaded)
            schema_path = root / "schema.pkl"
            builder.save(str(schema_path))
            assert schema_path.is_file()

            cls.dataset = InMemorySampleDataset(
                samples=loaded,
                input_schema={"ecg": "tensor", "ppg": "tensor"},
                output_schema={"sleep_stage": "multiclass"},
            )
        except Exception:
            cls._tmpdir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        """Close the dataset and delete the temporary directory tree."""
        if hasattr(cls, "dataset"):
            cls.dataset.close()
            del cls.dataset
        gc.collect()
        if hasattr(cls, "_tmpdir"):
            cls._tmpdir.cleanup()

    def _one_batch(self) -> dict[str, Any]:
        """Return the first batch from the test dataloader.

        Returns:
            Batch dict compatible with :meth:`Wav2Sleep.forward`.
        """
        return next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))

    def test_wav2sleep_synthetic_schema_in_temp_dir(self) -> None:
        """Fitted schema was written under the class temp directory."""
        root = Path(self._tmpdir.name)
        self.assertTrue((root / "sample_dataset_stub" / "schema.pkl").is_file())

    def test_wav2sleep_synthetic_raw_pickles_in_temp_dir(self) -> None:
        """One pickle file per synthetic patient under ``raw_samples``."""
        raw_dir = Path(self._tmpdir.name) / "sample_dataset_stub" / "raw_samples"
        for i in range(self._n_samples):
            self.assertTrue((raw_dir / f"sample_{i:03d}.pkl").is_file())

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
