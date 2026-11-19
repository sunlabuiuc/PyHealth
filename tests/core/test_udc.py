import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import UDC, DiscreteClueEncoder, ResidualVectorQuantizer


class TestResidualVectorQuantizer(unittest.TestCase):
    """Unit tests for ResidualVectorQuantizer."""

    def test_init_valid(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=3, codebook_size=64, embedding_dim=32, commitment_cost=0.5
        )
        self.assertEqual(len(rvq.codebooks), 3)
        self.assertEqual(rvq.codebook_size, 64)
        self.assertEqual(rvq.embedding_dim, 32)
        self.assertEqual(rvq.commitment_cost, 0.5)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            ResidualVectorQuantizer(n_codebooks=0)
        with self.assertRaises(ValueError):
            ResidualVectorQuantizer(codebook_size=-1)
        with self.assertRaises(ValueError):
            ResidualVectorQuantizer(embedding_dim=0)
        with self.assertRaises(ValueError):
            ResidualVectorQuantizer(commitment_cost=-0.1)

    def test_forward_shapes(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=2, codebook_size=32, embedding_dim=16
        )
        # 2D input (B, D)
        x_2d = torch.randn(4, 16)
        q_2d, loss_2d, idx_2d = rvq(x_2d)
        self.assertEqual(q_2d.shape, (4, 16))
        self.assertEqual(idx_2d.shape, (4, 2))
        self.assertTrue(loss_2d.item() >= 0.0)

        # 3D input (B, T, D)
        x_3d = torch.randn(2, 5, 16)
        q_3d, loss_3d, idx_3d = rvq(x_3d)
        self.assertEqual(q_3d.shape, (2, 5, 16))
        self.assertEqual(idx_3d.shape, (2, 5, 2))
        self.assertTrue(loss_3d.item() >= 0.0)

    def test_straight_through_gradients(self):
        rvq = ResidualVectorQuantizer(
            n_codebooks=2, codebook_size=16, embedding_dim=8
        )
        x = torch.randn(3, 8, requires_grad=True)
        q, loss, _ = rvq(x)
        total_loss = q.sum() + loss
        total_loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestDiscreteClueEncoder(unittest.TestCase):
    """Unit tests for DiscreteClueEncoder."""

    def test_projection_matching_and_differing_dim(self):
        # Matching dim
        enc_same = DiscreteClueEncoder(
            input_dim=16, hidden_dim=16, n_codebooks=2, codebook_size=32
        )
        self.assertIsInstance(enc_same.proj, torch.nn.Identity)

        # Differing dim
        enc_diff = DiscreteClueEncoder(
            input_dim=32, hidden_dim=16, n_codebooks=2, codebook_size=32
        )
        self.assertIsInstance(enc_diff.proj, torch.nn.Linear)

    def test_forward_and_backward(self):
        enc = DiscreteClueEncoder(
            input_dim=24, hidden_dim=16, n_codebooks=2, codebook_size=32
        )
        x = torch.randn(2, 4, 24, requires_grad=True)
        fused, vq_loss, indices = enc(x)
        self.assertEqual(fused.shape, (2, 4, 16))
        self.assertEqual(indices.shape, (2, 4, 2))
        self.assertTrue(vq_loss.item() >= 0.0)

        (fused.sum() + vq_loss).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestUDC(unittest.TestCase):
    """Unit tests for the full UDC model."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["c1", "c2"], ["c3"]],
                "procedures": [["p1"], ["p2", "p3"]],
                "drugs": ["d1", "d2"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["c2"], ["c4", "c5"]],
                "procedures": [["p3"], ["p1"]],
                "drugs": ["d3"],
                "label": 0,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": [["c1", "c4"]],
                "procedures": [["p2"]],
                "drugs": ["d1"],
                "label": 1,
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
            "drugs": "sequence",
        }

    def test_invalid_args(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            UDC(dataset, intra_visit_pooling="invalid")
        with self.assertRaises(ValueError):
            UDC(dataset, temporal_model="unknown")
        with self.assertRaises(ValueError):
            UDC(dataset, embedding_dim=0)
        with self.assertRaises(ValueError):
            UDC(dataset, hidden_dim=-5)
        with self.assertRaises(ValueError):
            UDC(dataset, n_codebooks=0)
        with self.assertRaises(ValueError):
            UDC(dataset, codebook_size=0)
        with self.assertRaises(ValueError):
            UDC(dataset, vq_loss_weight=-1.0)

    def test_binary_classification(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=16,
            codebook_size=32,
            n_codebooks=2,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        out = model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        self.assertIn("embed", out)
        self.assertIn("vq_loss", out)

        self.assertTrue((out["y_prob"] >= 0.0).all())
        self.assertTrue((out["y_prob"] <= 1.0).all())
        self.assertEqual(out["logit"].shape[-1], 1)

        # Backward pass
        out["loss"].backward()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_multiclass_classification(self):
        samples_mc = [
            dict(s, label=i % 3) for i, s in enumerate(self.samples)
        ]
        dataset = create_sample_dataset(
            samples=samples_mc,
            input_schema=self.input_schema,
            output_schema={"label": "multiclass"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=16,
            codebook_size=32,
            temporal_model="gru",
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        out = model(**batch)
        self.assertEqual(out["logit"].shape[-1], 3)
        prob_sum = out["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5))

        out["loss"].backward()

    def test_multilabel_classification(self):
        samples_ml = [
            dict(s, label=[f"l{i}", f"l{i+1}"])
            for i, s in enumerate(self.samples)
        ]
        dataset = create_sample_dataset(
            samples=samples_ml,
            input_schema=self.input_schema,
            output_schema={"label": "multilabel"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=16,
            codebook_size=32,
            temporal_model="lstm",
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        out = model(**batch)
        self.assertTrue((out["y_prob"] >= 0.0).all())
        self.assertTrue((out["y_prob"] <= 1.0).all())

        out["loss"].backward()

    def test_transformer_temporal_model(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=32,
            hidden_dim=32,
            temporal_model="transformer",
            num_layers=1,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("loss", out)
        out["loss"].backward()

    def test_intra_visit_pooling_modes(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        for pool_mode in ("sum", "mean", "max", "attention"):
            model = UDC(
                dataset=dataset,
                embedding_dim=16,
                hidden_dim=16,
                intra_visit_pooling=pool_mode,
            )
            out = model(**batch)
            self.assertIn("loss", out)
            self.assertTrue(torch.isfinite(out["loss"]))
            self.assertIn("embed", out)
            self.assertEqual(out["embed"].shape, (2, 16))
            self.assertTrue(torch.isfinite(out["embed"]).all())

    def test_inference_without_labels(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(dataset=dataset, embedding_dim=16, hidden_dim=16)
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        del batch["label"]

        model.eval()
        with torch.no_grad():
            out = model(**batch)

        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)
        self.assertIn("y_prob", out)
        self.assertIn("logit", out)
        self.assertIn("embed", out)
        self.assertIn("vq_indices", out)

    def test_tuple_input_handling(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(dataset=dataset, embedding_dim=16, hidden_dim=16)
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        # Pass conditions and label as (tensor, mask) tuple
        cond_tensor = batch["conditions"]
        batch["conditions"] = (cond_tensor, cond_tensor != 0)
        batch["label"] = (batch["label"], torch.ones_like(batch["label"]))

        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_optimizer_step(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(dataset=dataset, embedding_dim=16, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        initial_weight = model.fc.weight.clone()
        optimizer.zero_grad()
        out = model(**batch)
        out["loss"].backward()
        optimizer.step()

        self.assertFalse(torch.equal(initial_weight, model.fc.weight))

    def test_unequal_visit_counts_across_features(self):
        """Tests that features with unequal visit counts pad masks correctly."""
        unequal_samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"]],  # 1 visit
                "procedures": [["p1"], ["p2"], ["p3"]],  # 3 visits
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c2"], ["c3"]],  # 2 visits
                "procedures": [["p2"], ["p1"]],  # 2 visits
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=unequal_samples,
            input_schema={
                "conditions": "nested_sequence",
                "procedures": "nested_sequence",
            },
            output_schema={"label": "binary"},
        )
        model = UDC(dataset=dataset, embedding_dim=16, hidden_dim=16)
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_max_pooling_with_padded_visits(self):
        """Verifies max pooling zeroes out empty visits and does not retain -1e9."""
        samples_with_empty = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"], ["c3"]],  # 2 visits
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c2"], ["c4"], ["c5"], ["c1"]],  # 4 visits
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples_with_empty,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=16,
            intra_visit_pooling="max",
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        # For p0, visits 2 and 3 are padding; verify embedding does not have -1e9
        self.assertFalse(torch.isnan(out["embed"]).any())
        self.assertTrue(torch.all(out["embed"].abs() < 50.0))

    def test_padding_invariance(self):
        """Verifies a patient's prediction and embedding do not change when padded."""
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1"]],  # 1 valid visit
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c2"], ["c3"], ["c4"]],  # 3 valid visits
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
        )
        for temp_model in ("gru", "transformer"):
            torch.manual_seed(42)
            model = UDC(
                dataset=dataset,
                embedding_dim=16,
                hidden_dim=16,
                temporal_model=temp_model,
                dropout=0.0,
            )
            model.eval()

            loader = get_dataloader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))

            with torch.no_grad():
                out_batch = model(**batch)
                batch_single = {
                    k: v[0:1, :1] if k == "conditions" else v[0:1]
                    for k, v in batch.items()
                }
                out_single = model(**batch_single)

            self.assertTrue(
                torch.allclose(
                    out_batch["logit"][0], out_single["logit"][0], atol=1e-5
                ),
                f"Logit padding variance on {temp_model}",
            )
            self.assertTrue(
                torch.allclose(
                    out_batch["embed"][0], out_single["embed"][0], atol=1e-5
                ),
                f"Embed padding variance on {temp_model}",
            )

    def test_empty_patient_with_transformer(self):
        """Verifies transformer handles all-pad patient without producing NaNs."""
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"]],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c3"]],
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=16,
            temporal_model="transformer",
            dropout=0.0,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["conditions"][1].zero_()

        out = model(**batch)
        self.assertFalse(torch.isnan(out["logit"]).any())
        self.assertFalse(torch.isnan(out["embed"]).any())
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_unsupported_high_dim_feature(self):
        """Verifies ValueError is raised for features with more than 4 dimensions."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=16,
        )
        original_embed = model.embedding_model.forward
        model.embedding_model.forward = lambda kw: {
            "conditions": torch.zeros((2, 3, 4, 5, 16))
        }
        with self.assertRaises(ValueError):
            model(conditions=torch.zeros((2, 3, 4, 5)))
        model.embedding_model.forward = original_embed

    def test_time_value_tuple_inputs(self):
        """Verifies (time, value) tuple inputs compute mask from embeddings."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = UDC(dataset=dataset, embedding_dim=16, hidden_dim=16)
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        cond_val = batch["conditions"]
        fake_time = torch.zeros_like(cond_val, dtype=torch.float32)
        batch["conditions"] = (fake_time, cond_val)
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["logit"]).all())


if __name__ == "__main__":
    unittest.main()
