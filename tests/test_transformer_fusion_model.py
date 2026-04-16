import unittest

import torch

from pyhealth.datasets.sample_dataset import InMemorySampleDataset
from pyhealth.models import TransformerFusionModel


class TestTransformerFusionModel(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "feature_a": [1, 2, 3],
                "feature_b": [4, 5],
                "label": 1,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "feature_a": [2, 3, 4],
                "feature_b": [3, 2],
                "label": 0,
            },
        ]

        self.dataset = InMemorySampleDataset(
            samples=self.samples,
            input_schema={"feature_a": "sequence", "feature_b": "sequence"},
            output_schema={"label": "binary"},
        )

    def _build_batch(self) -> dict[str, torch.Tensor]:
        batch = {}
        for key in self.dataset.input_schema:
            tensors = [self.dataset[i][key] for i in range(len(self.dataset))]
            batch[key] = torch.stack(tensors, dim=0)

        label_tensors = [self.dataset[i]["label"] for i in range(len(self.dataset))]
        batch["label"] = torch.stack(label_tensors, dim=0)
        return batch

    def test_forward_pass_returns_expected_outputs(self):
        model = TransformerFusionModel(
            dataset=self.dataset,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.1,
        )

        batch = self._build_batch()
        output = model(**batch)

        self.assertSetEqual(
            set(output.keys()),
            {"loss", "y_prob", "y_true", "logit"},
        )
        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertEqual(output["y_prob"].shape, (2, 1))
        self.assertEqual(output["y_true"].shape, (2, 1))
        self.assertTrue(torch.isfinite(output["loss"]))

        output["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_forward_with_modality_token(self):
        model = TransformerFusionModel(
            dataset=self.dataset,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            use_modality_token=True,
        )

        batch = self._build_batch()
        output = model(**batch)

        self.assertEqual(output["logit"].shape, (2, 1))
        self.assertEqual(output["y_prob"].shape, (2, 1))
        self.assertTrue(torch.isfinite(output["loss"]))
