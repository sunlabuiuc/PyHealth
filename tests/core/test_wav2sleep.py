"""Fast synthetic Wav2Sleep tests: temp dir + disk dataset, instantiation, forward, shapes, grads."""

from __future__ import annotations

import pickle
import tempfile
import unittest
import gc
from pathlib import Path

import litdata
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.sample_dataset import SampleBuilder, SampleDataset
from pyhealth.models.wav2sleep import SIGNAL_TO_SAMPLES_PER_EPOCH, Wav2Sleep

SP = SIGNAL_TO_SAMPLES_PER_EPOCH["ECG"]


def _one_epoch_wave_list() -> list[float]:
    return torch.randn(SP, dtype=torch.float32).tolist()


def _synthetic_samples(*, n_patients: int = 2) -> list[dict]:
    assert 1 <= n_patients <= 5
    out = []
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
    """All tests share a disk-backed SampleDataset under a single TemporaryDirectory."""

    _tmpdir: tempfile.TemporaryDirectory
    dataset: SampleDataset

    @classmethod
    def setUpClass(cls) -> None:
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
        if hasattr(cls, "dataset"):
            loader = get_dataloader(cls.dataset, batch_size=2, shuffle=False)
            for _ in loader:
                pass
            cls.dataset.close()
            del cls.dataset
        gc.collect()
        cls._tmpdir.cleanup()

    def _one_batch(self):
        return next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))

    def test_wav2sleep_synthetic_instantiation(self) -> None:
        m = _build_tiny_model(self.dataset)
        self.assertIsInstance(m, Wav2Sleep)
        self.assertEqual(m.num_classes, 5)
        self.assertEqual(m.embedding_dim, 32)
        self.assertEqual(m.hidden_dim, 32)
        self.assertEqual(m.label_key, "sleep_stage")

    def test_wav2sleep_synthetic_forward_pass(self) -> None:
        m = _build_tiny_model(self.dataset)
        batch = self._one_batch()
        with torch.no_grad():
            out = m(**batch)
        self.assertIn("loss", out)
        self.assertIn("logit", out)

    def test_wav2sleep_synthetic_output_shapes(self) -> None:
        m = _build_tiny_model(self.dataset)
        batch = self._one_batch()
        with torch.no_grad():
            out = m(**batch)
        self.assertEqual(out["loss"].shape, ())
        self.assertEqual(tuple(out["logit"].shape), (2, 5))
        self.assertEqual(tuple(out["y_prob"].shape), (2, 5))
        self.assertEqual(tuple(out["y_true"].shape), (2,))
        self.assertTrue(torch.isfinite(out["loss"]).all().item())

    def test_wav2sleep_synthetic_gradient_computation(self) -> None:
        m = _build_tiny_model(self.dataset)
        batch = self._one_batch()
        out = m(**batch)
        out["loss"].backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in m.parameters()
                if p.requires_grad
            )
        )


if __name__ == "__main__":
    unittest.main()