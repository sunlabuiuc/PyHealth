"""Unified MLP threads collate pad_mask like the other heads."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


class TestP2MlpPadMask(unittest.TestCase):
    def test_mlp_threads_pad_mask_into_unified_inputs(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding
        from pyhealth.models.mlp import MLP
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        batch = [
            {"labs": (torch.tensor([6.0, 12.0, 24.0]), torch.ones(3, 2))},
            {"labs": (torch.tensor([6.0]), torch.ones(1, 2))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        self.assertIn(f"labs{PAD_MASK_SUFFIX}", collated)

        host = SimpleNamespace(
            feature_keys=["labs"],
            device="cpu",
            dataset=SimpleNamespace(
                input_processors={"labs": StageNetTensorProcessor()}
            ),
        )
        inputs = MLP._build_unified_inputs(host, collated)
        self.assertEqual(
            inputs["labs"]["pad_mask"].tolist(),
            [[True, True, True], [True, False, False]],
        )

    def test_unified_forward_ignores_padded_slots(self):
        from pyhealth.datasets import create_sample_dataset, get_dataloader
        from pyhealth.models.mlp import MLP
        from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel

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
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "stagenet", "labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_mlp_unified",
        )
        unified = UnifiedMultimodalEmbeddingModel(
            dataset.input_processors, embedding_dim=8
        )
        model = MLP(
            dataset=dataset, embedding_dim=8, hidden_dim=8,
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


if __name__ == "__main__":
    unittest.main()
