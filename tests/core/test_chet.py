"""Unit tests for CHET model."""

import unittest
import warnings

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    CHET,
    SingleHeadAttentionLayer,
    compute_cooccurrence_adj,
)


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

    def test_compute_cooccurrence_adj_values(self):
        """Verifies exact numerical adjacency values on a toy dataset."""
        toy_samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"]],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c1", "c2"]],
                "label": 0,
            },
        ]
        toy_ds = create_sample_dataset(
            samples=toy_samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
        )
        vocab = toy_ds.input_processors["conditions"].code_vocab
        idx_c1 = vocab["c1"]
        idx_c2 = vocab["c2"]

        # With self-loop: degree of each code is 2 (self + co-occurrence)
        adj_loop = compute_cooccurrence_adj(toy_ds, "conditions", self_loop=True)
        self.assertEqual(adj_loop[0].sum().item(), 0.0)
        self.assertEqual(adj_loop[:, 0].sum().item(), 0.0)
        self.assertAlmostEqual(adj_loop[idx_c1, idx_c1].item(), 0.5)
        self.assertAlmostEqual(adj_loop[idx_c1, idx_c2].item(), 0.5)
        self.assertAlmostEqual(adj_loop[idx_c2, idx_c1].item(), 0.5)
        self.assertAlmostEqual(adj_loop[idx_c2, idx_c2].item(), 0.5)

        # Without self-loop: degree is 1 (only co-occurrence)
        adj_no_loop = compute_cooccurrence_adj(toy_ds, "conditions", self_loop=False)
        self.assertAlmostEqual(adj_no_loop[idx_c1, idx_c1].item(), 0.0)
        self.assertAlmostEqual(adj_no_loop[idx_c1, idx_c2].item(), 1.0)
        self.assertAlmostEqual(adj_no_loop[idx_c2, idx_c1].item(), 1.0)
        self.assertAlmostEqual(adj_no_loop[idx_c2, idx_c2].item(), 0.0)

    def test_compute_cooccurrence_adj_tuple_inputs(self):
        """Verifies compute_cooccurrence_adj handles (time, codes) and (None, codes)."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        adj_expected = compute_cooccurrence_adj(dataset, "conditions")

        class TupleDataset:
            def __init__(self, base_ds, mode):
                self.base_ds = base_ds
                self.input_processors = base_ds.input_processors
                self.output_processors = base_ds.output_processors
                self.mode = mode

            def __len__(self):
                return len(self.base_ds)

            def __iter__(self):
                for s in self.base_ds:
                    item = dict(s)
                    val = item["conditions"]
                    if self.mode == "time_first":
                        item["conditions"] = (torch.tensor([0.5, 1.2]), val)
                    elif self.mode == "none_first":
                        item["conditions"] = (None, val)
                    elif self.mode == "mask_second":
                        item["conditions"] = (val, torch.ones_like(val))
                    yield item

        adj_time = compute_cooccurrence_adj(
            TupleDataset(dataset, "time_first"), "conditions"
        )
        self.assertTrue(torch.allclose(adj_expected, adj_time))

        adj_none = compute_cooccurrence_adj(
            TupleDataset(dataset, "none_first"), "conditions"
        )
        self.assertTrue(torch.allclose(adj_expected, adj_none))

        adj_mask = compute_cooccurrence_adj(
            TupleDataset(dataset, "mask_second"), "conditions"
        )
        self.assertTrue(torch.allclose(adj_expected, adj_mask))

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

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            CHET(
                dataset=dataset,
                code_size=16,
                graph_size=16,
                hidden_dim=32,
            )
            user_warnings = [w for w in recorded if issubclass(w.category, UserWarning)]
            self.assertEqual(len(user_warnings), 0)

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

    def test_multi_output_dataset_with_label_key(self):
        """Verifies CHET supports datasets with multiple output label keys."""
        samples_multi = [
            dict(s, mortality=i % 2, readmission=(i + 1) % 2)
            for i, s in enumerate(self.samples)
        ]
        dataset = create_sample_dataset(
            samples=samples_multi,
            input_schema=self.input_schema,
            output_schema={"mortality": "binary", "readmission": "binary"},
        )
        model = CHET(
            dataset=dataset,
            label_key="readmission",
            code_size=16,
            graph_size=16,
            hidden_dim=32,
        )
        self.assertEqual(model.label_key, "readmission")
        self.assertEqual(
            model.label_processor, dataset.output_processors["readmission"]
        )

        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["logit"].shape, (2, 1))

        with self.assertRaises(KeyError):
            CHET(dataset=dataset, label_key="non_existent_label")

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

        # 1. (codes, mask)
        mask = cond != 0
        batch["conditions"] = (cond, mask)
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertTrue(torch.isfinite(out["logit"]).all())

        # 2. (time, codes)
        times = torch.rand(cond.shape[:2], dtype=torch.float32)
        batch["conditions"] = (times, cond)
        out2 = model(**batch)
        self.assertIn("logit", out2)
        self.assertTrue(torch.isfinite(out2["logit"]).all())

        # 3. (None, codes)
        batch["conditions"] = (None, cond)
        out3 = model(**batch)
        self.assertIn("logit", out3)
        self.assertTrue(torch.isfinite(out3["logit"]).all())

    def test_transition_branches_m1_m2_m3(self):
        """Verifies M1 (recurrent), M2 (neighbor), and M3 (novel) transition logic."""
        samples_trans = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1"], ["c1", "c2", "c3"]],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c1"], ["c2"]],  # only M2 at visit 1
                "label": 0,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "conditions": [["c1"], ["c3"]],  # only M3 at visit 1
                "label": 1,
            },
            {
                "patient_id": "p3",
                "visit_id": "v3",
                "conditions": [["c1"], ["c1"]],  # only M1 at visit 1
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples_trans,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
        )
        vocab = dataset.input_processors["conditions"].code_vocab
        num_codes = len(vocab)
        idx_c1 = vocab["c1"]
        idx_c2 = vocab["c2"]

        # Construct adjacency where c1 and c2 are neighbors, c3 has no edges
        adj = torch.zeros((num_codes, num_codes), dtype=torch.float32)
        adj[idx_c1, idx_c1] = 0.5
        adj[idx_c1, idx_c2] = 0.5
        adj[idx_c2, idx_c1] = 0.5
        adj[idx_c2, idx_c2] = 0.5

        model = CHET(
            dataset=dataset,
            code_size=16,
            graph_size=16,
            hidden_dim=32,
            adj=adj,
        )

        # Track GRU and transition attention invocations across visits
        gru_counts: list[int] = []
        attn_counts: list[int] = []

        hook_gru = model.gru.register_forward_hook(
            lambda module, input, output: gru_counts.append(input[0].shape[0])
        )
        hook_attn = model.transition_attn.register_forward_hook(
            lambda module, input, output: attn_counts.append(input[0].shape[0])
        )

        try:
            batches = list(get_dataloader(dataset, batch_size=1, shuffle=False))

            # Test Patient 3: ["c1"] -> ["c1"] (only recurring M1 at visit 1)
            gru_counts.clear()
            attn_counts.clear()
            model(**batches[3])
            # Visit 0 calls GRU (1 code), Visit 1 calls GRU (1 recurring code)
            self.assertEqual(gru_counts, [1, 1])
            # Transition attention does not fire because neither M2 nor M3 exists
            self.assertEqual(len(attn_counts), 0)

            # Test Patient 1: ["c1"] -> ["c2"] (only neighbor transition M2 at visit 1)
            gru_counts.clear()
            attn_counts.clear()
            model(**batches[1])
            # Visit 0 calls GRU (1 code), Visit 1 does not call recurrent GRU
            self.assertEqual(gru_counts, [1])
            # Transition attention fires exactly once at visit 1 for 1 code (M2)
            self.assertEqual(attn_counts, [1])

            # Test Patient 2: ["c1"] -> ["c3"] (only novel condition M3 at visit 1)
            gru_counts.clear()
            attn_counts.clear()
            model(**batches[2])
            # Visit 0 calls GRU (1 code), Visit 1 does not call recurrent GRU
            self.assertEqual(gru_counts, [1])
            # Transition attention fires exactly once at visit 1 for 1 code (M3)
            self.assertEqual(attn_counts, [1])

            # Test Patient 0: ["c1"] -> ["c1", "c2", "c3"] (all M1, M2, M3 fire)
            gru_counts.clear()
            attn_counts.clear()
            model(**batches[0])
            # Visit 0 calls GRU (1 code), Visit 1 calls GRU (1 recurring code)
            self.assertEqual(gru_counts, [1, 1])
            # Transition attention fires for 2 codes (neighbor M2 and novel M3)
            self.assertEqual(attn_counts, [2])
        finally:
            hook_gru.remove()
            hook_attn.remove()

        # Full batch forward
        loader = get_dataloader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertEqual(out["logit"].shape, (4, 1))
        self.assertTrue(torch.isfinite(out["logit"]).all())
        self.assertTrue(torch.isfinite(out["embed"]).all())
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_single_head_attention_differing_sizes(self):
        """Tests SingleHeadAttentionLayer with differing query, key, and value dims."""
        # 1. With explicit value_input_size differing from query_size, key_size
        layer = SingleHeadAttentionLayer(
            query_size=12,
            key_size=16,
            value_size=32,
            attention_size=8,
            value_input_size=20,
        )
        q = torch.randn(3, 12)
        k = torch.randn(5, 16)
        v = torch.randn(5, 20)
        out = layer(q, k, v)
        self.assertEqual(out.shape, (3, 32))
        self.assertTrue(torch.isfinite(out).all())

        # 2. With default value_input_size=None (v features match value_size)
        layer_default = SingleHeadAttentionLayer(
            query_size=14,
            key_size=18,
            value_size=28,
            attention_size=10,
        )
        q2 = torch.randn(2, 14)
        k2 = torch.randn(4, 18)
        v2 = torch.randn(4, 28)
        out2 = layer_default(q2, k2, v2)
        self.assertEqual(out2.shape, (2, 28))
        self.assertTrue(torch.isfinite(out2).all())

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
