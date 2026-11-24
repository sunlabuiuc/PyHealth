"""BottleneckTransformer per-field and unified modes on a tiny two-field dataset."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BottleneckTransformer
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
        dataset_name="test_bottleneck_transformer",
    )


def _batch(dataset):
    return next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))


class TestBottleneckTransformer(unittest.TestCase):
    def test_per_field_forward_backward_finite(self):
        dataset = _dataset()
        model = BottleneckTransformer(
            dataset=dataset,
            embedding_dim=8,
            bottlenecks_n=2,
            fusion_startidx=1,
            num_layers=2,
            heads=2,
            dropout=0.0,
        )
        self.assertFalse(model._use_unified)
        out = model(**_batch(dataset))
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertTrue(
            any(p.grad is not None for p in model.parameters() if p.requires_grad)
        )

    def test_unified_forward_backward_finite(self):
        dataset = _dataset()
        unified = UnifiedMultimodalEmbeddingModel(
            dataset.input_processors, embedding_dim=8
        )
        model = BottleneckTransformer(
            dataset=dataset,
            embedding_dim=8,
            bottlenecks_n=2,
            fusion_startidx=1,
            num_layers=2,
            heads=2,
            dropout=0.0,
            unified_embedding=unified,
        )
        self.assertTrue(model._use_unified)
        out = model(**_batch(dataset))
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["y_prob"]).all())
        out["loss"].backward()
        self.assertTrue(
            any(p.grad is not None for p in model.parameters() if p.requires_grad)
        )


if __name__ == "__main__":
    unittest.main()
