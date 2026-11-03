"""EHRMamba unified mode over a tiny two-field temporal dataset."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EHRMamba
from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel


def _dataset():
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "codes": ([0.0, 2.0], ["A", "B"]),
            "labs": ([0.0, 2.0], [[1.0, 2.0], [3.0, 4.0]]),
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "codes": ([0.0], ["C"]),
            "labs": ([0.0], [[5.0, 6.0]]),
            "label": 0,
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"codes": "stagenet", "labs": "stagenet_tensor"},
        output_schema={"label": "binary"},
        dataset_name="test_ehrmamba_unified",
    )


class TestEHRMambaUnified(unittest.TestCase):
    def test_unified_forward_backward_loss_finite(self):
        dataset = _dataset()
        unified = UnifiedMultimodalEmbeddingModel(
            dataset.input_processors, embedding_dim=8
        )
        model = EHRMamba(
            dataset=dataset, embedding_dim=8, num_layers=1,
            unified_embedding=unified,
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertTrue(
            any(
                p.grad is not None
                for p in model.parameters()
                if p.requires_grad
            )
        )

    def test_per_field_mode_unchanged(self):
        dataset = _dataset()
        model = EHRMamba(dataset=dataset, embedding_dim=8, num_layers=1)
        self.assertFalse(model._use_unified)
        batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
