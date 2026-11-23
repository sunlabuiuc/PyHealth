"""Unit tests for CHET model."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CHET, compute_cooccurrence_adj


class TestCHET(unittest.TestCase):
    """Comprehensive unit tests for CHET."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"], ["c3"]],
                "procedures": [["p1"], ["p2"]],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c2"], ["c3", "c4"]],
                "procedures": [["p2"], ["p1"]],
                "label": 0,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "conditions": [["c1", "c4"], ["c2"]],
                "procedures": [["p1"]],
                "label": 1,
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }

    def test_compute_cooccurrence_adj(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        adj = compute_cooccurrence_adj(dataset, "conditions", self_loop=True)
        self.assertEqual(adj.dim(), 2)
        self.assertEqual(adj.shape[0], adj.shape[1])
        # Row 0 and Col 0 must be zero (padding row)
        self.assertTrue((adj[0] == 0.0).all())
        self.assertTrue((adj[:, 0] == 0.0).all())
        # All entries must be non-negative and finite
        self.assertTrue((adj >= 0.0).all())
        self.assertTrue(torch.isfinite(adj).all())
        # Connected rows must be row-normalized (sum to 1.0)
        row_sums = adj.sum(dim=-1)
        active_rows = row_sums > 0
        self.assertTrue(
            torch.allclose(
                row_sums[active_rows], torch.ones_like(row_sums[active_rows])
            )
        )

    def test_multiple_feature_keys_warning(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        with self.assertWarns(UserWarning):
            CHET(
                dataset=dataset,
                feature_keys=["conditions", "procedures"],
                code_size=16,
                graph_size=16,
                hidden_dim=32,
            )

    def test_invalid_args(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            CHET(dataset, code_size=0)
        with self.assertRaises(ValueError):
            CHET(dataset, graph_size=-1)
        with self.assertRaises(ValueError):
            CHET(dataset, hidden_dim=0)
        with self.assertRaises(ValueError):
            CHET(dataset, attention_size=-5)
        with self.assertRaises(ValueError):
            CHET(dataset, adj=torch.zeros((5, 5)))  # wrong shape
        with self.assertRaises(TypeError):
            CHET(dataset, adj="invalid_adj")

    def test_binary_classification(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        self.assertIn("embed", out)

        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["embed"].shape, (2, 32))
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["logit"]).all())
        self.assertTrue(torch.isfinite(out["embed"]).all())

    def test_multiclass_classification(self):
        samples_mc = [dict(s, label=i % 3) for i, s in enumerate(self.samples)]
        dataset = create_sample_dataset(
            samples=samples_mc,
            input_schema=self.input_schema,
            output_schema={"label": "multiclass"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=3)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertEqual(out["logit"].shape, (3, 3))
        self.assertEqual(out["y_prob"].shape, (3, 3))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_multilabel_classification(self):
        samples_ml = [
            dict(
                s,
                label=(
                    ["l0", "l1"]
                    if i == 0
                    else (["l1", "l2"] if i == 1 else ["l0", "l2"])
                ),
            )
            for i, s in enumerate(self.samples)
        ]
        dataset = create_sample_dataset(
            samples=samples_ml,
            input_schema=self.input_schema,
            output_schema={"label": "multilabel"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertEqual(out["logit"].shape, (2, 3))
        self.assertEqual(out["y_prob"].shape, (2, 3))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_inference_without_labels(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        del batch["label"]
        out = model(**batch)

        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)
        self.assertIn("logit", out)
        self.assertIn("embed", out)
        self.assertIn("y_prob", out)
        self.assertTrue(torch.isfinite(out["logit"]).all())

    def test_empty_patient(self):
        """Verifies model handles completely padded empty patient without NaNs."""
        dataset = create_sample_dataset(
            samples=self.samples[:2],
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
            dropout=0.0,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        # Zero all codes for patient 1
        batch["conditions"][1].zero_()

        out = model(**batch)
        self.assertFalse(torch.isnan(out["logit"]).any())
        self.assertFalse(torch.isnan(out["embed"]).any())
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue((out["embed"][1] == 0.0).all())

        out["loss"].backward()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_2d_flat_sequence(self):
        samples_2d = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": ["c1", "c2"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": ["c2", "c3"],
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples_2d,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertTrue(torch.isfinite(out["logit"]).all())
        self.assertEqual(out["logit"].shape, (2, 1))

    def test_tuple_input(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        cond = batch["conditions"]
        mask = cond != 0
        batch["conditions"] = (cond, mask)
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertTrue(torch.isfinite(out["logit"]).all())

    def test_custom_adjacency_matrix(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        num_codes = len(dataset.input_processors["conditions"].code_vocab)
        custom_adj = np.eye(num_codes, dtype=np.float32)
        custom_adj[0, 0] = 0.0

        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
            adj=custom_adj,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["logit"]).all())

    def test_unsupported_high_dim_feature(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        with self.assertRaises(ValueError):
            model(conditions=torch.zeros((2, 3, 4, 5), dtype=torch.long))

    def test_gradient_flow(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        out["loss"].backward()

        self.assertIsNotNone(model.c_embeddings.grad)
        self.assertIsNotNone(model.n_embeddings.grad)
        self.assertIsNotNone(model.u_embeddings.grad)
        self.assertTrue(torch.isfinite(model.c_embeddings.grad).all())
        self.assertTrue(torch.isfinite(model.fc.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
