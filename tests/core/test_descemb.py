import unittest

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    DescEmb,
    DescEmbCodeEncoder,
    simple_subword_tokenize,
    simple_word_tokenize,
)


class TestDescEmbCodeEncoder(unittest.TestCase):
    """Unit tests for DescEmbCodeEncoder."""

    def test_tokenize_helper(self):
        tokens = simple_subword_tokenize("Type 2 diabetes mellitus, uncomplicated")
        self.assertEqual(
            tokens, ["type", "2", "diabetes", "mellitus", "uncomplicated"]
        )
        word_tokens = simple_word_tokenize(
            "Type 2 diabetes mellitus, uncomplicated"
        )
        self.assertEqual(tokens, word_tokens)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            DescEmbCodeEncoder(vocab_size=0)
        with self.assertRaises(ValueError):
            DescEmbCodeEncoder(subword_dim=-1)
        with self.assertRaises(ValueError):
            DescEmbCodeEncoder(hidden_dim=0)
        with self.assertRaises(ValueError):
            DescEmbCodeEncoder(output_dim=-10)

    def test_forward_shapes(self):
        encoder = DescEmbCodeEncoder(
            vocab_size=50, subword_dim=16, hidden_dim=16, output_dim=32
        )
        # 2D input (num_codes, max_len)
        ids_2d = torch.randint(1, 40, (5, 8))
        embs_2d = encoder(ids_2d)
        self.assertEqual(embs_2d.shape, (5, 32))

        # 3D input (batch, seq_len, max_len)
        ids_3d = torch.randint(1, 40, (2, 4, 8))
        embs_3d = encoder(ids_3d)
        self.assertEqual(embs_3d.shape, (2, 4, 32))

    def test_gradient_flow(self):
        encoder = DescEmbCodeEncoder(
            vocab_size=50, subword_dim=16, hidden_dim=16, output_dim=32
        )
        ids = torch.randint(1, 40, (4, 6))
        out = encoder(ids)
        loss = out.sum()
        loss.backward()

        for param in encoder.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())


class TestDescEmb(unittest.TestCase):
    """Unit tests for the full DescEmb model."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"], ["c3"]],
                "procedures": [["p1"], ["p2", "p3"]],
                "drugs": ["d1", "d2"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c2"], ["c4", "c5"]],
                "procedures": [["p3"], ["p1"]],
                "drugs": ["d3"],
                "label": 0,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
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
            DescEmb(dataset, intra_visit_pooling="invalid")
        with self.assertRaises(ValueError):
            DescEmb(dataset, temporal_model="unknown")
        with self.assertRaises(ValueError):
            DescEmb(dataset, embedding_dim=0)
        with self.assertRaises(ValueError):
            DescEmb(dataset, text_hidden_dim=-1)
        with self.assertRaises(ValueError):
            DescEmb(dataset, temporal_hidden_dim=0)

    def test_binary_classification(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        out = model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        self.assertIn("embed", out)

        self.assertTrue((out["y_prob"] >= 0.0).all())
        self.assertTrue((out["y_prob"] <= 1.0).all())
        self.assertEqual(out["logit"].shape[-1], 1)

        out["loss"].backward()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_multiclass_classification(self):
        samples_mc = [dict(s, label=i % 3) for i, s in enumerate(self.samples)]
        dataset = create_sample_dataset(
            samples=samples_mc,
            input_schema=self.input_schema,
            output_schema={"label": "multiclass"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
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
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
            temporal_model="lstm",
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        out = model(**batch)
        self.assertTrue((out["y_prob"] >= 0.0).all())
        self.assertTrue((out["y_prob"] <= 1.0).all())

        out["loss"].backward()

    def test_description_initialization(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        code_descriptions = {
            "c1": "essential primary hypertension",
            "c2": "type 2 diabetes mellitus without complications",
            "c3": "congestive heart failure",
            "p1": "coronary artery bypass graft",
        }
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        emb_layer = model.embedding_model.embedding_layers["conditions"]
        before_weight = emb_layer.weight.clone()
        model.initialize_with_descriptions(code_descriptions)
        after_weight = emb_layer.weight.clone()

        # Embedding weights must change upon description initialization
        self.assertFalse(torch.equal(before_weight, after_weight))
        # Padding row index 0 must remain zero
        self.assertTrue(torch.all(after_weight[0] == 0.0))

        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_intra_visit_pooling_modes(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        for pool_mode in ("sum", "mean", "max", "attention"):
            model = DescEmb(
                dataset=dataset,
                embedding_dim=16,
                text_hidden_dim=16,
                temporal_hidden_dim=16,
                intra_visit_pooling=pool_mode,
            )
            out = model(**batch)
            self.assertIn("loss", out)
            self.assertTrue(torch.isfinite(out["loss"]))

    def test_transformer_temporal_model(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=32,
            text_hidden_dim=16,
            temporal_hidden_dim=32,
            temporal_model="transformer",
            num_layers=1,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("loss", out)
        out["loss"].backward()

    def test_value_conditioning_and_tuple_unpacking(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        cond_tensor = batch["conditions"]
        # Tuple with continuous float values matching feat shape[:-1]
        float_values = torch.randn_like(cond_tensor, dtype=torch.float32)
        batch["conditions"] = (cond_tensor, float_values)
        batch["label"] = (batch["label"], torch.ones_like(batch["label"]))

        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        # Verify value conditioning layers receive gradients
        self.assertIsNotNone(model.val_proj.weight.grad)
        self.assertIsNotNone(model.post_val_proj.weight.grad)

    def test_padding_stays_masked_after_description_initialization(self):
        """Verifies row 0 is kept zero and padding stays masked with
        code_descriptions.
        """
        desc_map = {
            "c1": "essential hypertension",
            "c2": "type 2 diabetes mellitus",
            "c3": "hyperlipidemia",
        }
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
            code_descriptions=desc_map,
        )
        emb_layer = model.embedding_model.embedding_layers["conditions"]
        # Padding row index 0 must be strictly zero
        self.assertTrue(
            torch.all(emb_layer.weight.data[0] == 0.0),
            "Padding row at index 0 must remain zero after description initialization.",
        )

        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_time_value_tuple_inputs(self):
        """Verifies (time, value) tuple inputs forward correctly without crash."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        cond_val = batch["conditions"]
        fake_time = torch.zeros_like(cond_val, dtype=torch.float32)
        # Pass (time, value)
        batch["conditions"] = (fake_time, cond_val)
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (2, 1))

    def test_inference_without_labels(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
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

    def test_optimizer_step(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        initial_weight = model.fc.weight.clone()
        optimizer.zero_grad()
        out = model(**batch)
        out["loss"].backward()
        optimizer.step()

        self.assertFalse(torch.equal(initial_weight, model.fc.weight))

    def test_padding_invariance(self):
        """Verifies predictions do not change when padding is added."""
        dataset = create_sample_dataset(
            samples=self.samples[:2],
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        loader_single = get_dataloader(dataset[:1], batch_size=1)
        batch_single = next(iter(loader_single))
        val_single = torch.randn_like(
            batch_single["conditions"], dtype=torch.float32
        )
        batch_single["conditions"] = (
            batch_single["conditions"],
            val_single,
        )

        for temporal_model in ("gru", "lstm", "transformer"):
            for pool_mode in ("sum", "mean", "max", "attention"):
                model = DescEmb(
                    dataset=dataset,
                    embedding_dim=16,
                    text_hidden_dim=16,
                    temporal_hidden_dim=16,
                    temporal_model=temporal_model,
                    intra_visit_pooling=pool_mode,
                )
                model.eval()

                cond_raw = batch_single["conditions"][0]
                val_raw = batch_single["conditions"][1]
                cond_padded = F.pad(cond_raw, (0, 2, 0, 1), value=0)
                val_padded = F.pad(val_raw, (0, 2, 0, 1), value=0.0)

                proc_raw = batch_single["procedures"]
                proc_padded = F.pad(proc_raw, (0, 2, 0, 1), value=0)

                drugs_raw = batch_single["drugs"]
                drugs_padded = F.pad(drugs_raw, (0, 1), value=0)

                batch_padded = {
                    "conditions": (cond_padded, val_padded),
                    "procedures": proc_padded,
                    "drugs": drugs_padded,
                }

                with torch.no_grad():
                    out_single = model(**batch_single)
                    out_padded = model(**batch_padded)

                diff_logit = (
                    (out_single["logit"] - out_padded["logit"])
                    .abs()
                    .max()
                    .item()
                )
                diff_embed = (
                    (out_single["embed"] - out_padded["embed"])
                    .abs()
                    .max()
                    .item()
                )
                self.assertLess(
                    diff_logit,
                    1e-4,
                    f"Logit changed by {diff_logit} with padding "
                    f"(model={temporal_model}, pool={pool_mode})",
                )
                self.assertLess(
                    diff_embed,
                    1e-4,
                    f"Embed changed by {diff_embed} with padding "
                    f"(model={temporal_model}, pool={pool_mode})",
                )

    def test_value_mask_tuple_input(self):
        """Verifies (value, mask) tuple input unpacks correctly."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        cond = batch["conditions"]
        mask = cond != 0
        batch["conditions"] = (cond, mask)
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertTrue(torch.isfinite(out["logit"]).all())

    def test_empty_patient_with_transformer(self):
        """Verifies transformer handles all-pad patient without producing NaNs."""
        dataset = create_sample_dataset(
            samples=self.samples[:2],
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
            temporal_model="transformer",
            dropout=0.0,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        for key in ("conditions", "procedures", "drugs"):
            if key in batch:
                if isinstance(batch[key], tuple):
                    for elem in batch[key]:
                        elem[1].zero_()
                else:
                    batch[key][1].zero_()

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
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        original_embed = model.embedding_model.forward
        model.embedding_model.forward = lambda kw: {
            "conditions": torch.zeros((2, 3, 4, 5, 16))
        }
        with self.assertRaises(ValueError):
            model(conditions=torch.zeros((2, 3, 4, 5)))
        model.embedding_model.forward = original_embed

    def test_pretrained_text_encoder_arg(self):
        """Verifies pretrained_text_encoder and text_vocab are used and frozen."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        custom_encoder = DescEmbCodeEncoder(
            vocab_size=100,
            subword_dim=16,
            hidden_dim=16,
            output_dim=16,
            dropout=0.0,
        )
        custom_vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "essential": 2,
            "hypertension": 3,
        }
        descriptions = {"c1": "essential hypertension"}
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
            pretrained_text_encoder=custom_encoder,
            code_descriptions=descriptions,
            text_vocab=custom_vocab,
        )
        self.assertIs(model.code_text_encoder, custom_encoder)
        for param in model.code_text_encoder.parameters():
            self.assertFalse(param.requires_grad)

        # Verify custom vocab correctly mapped "c1" tokens to [2, 3]
        # and initialized the embedding layer matching custom_encoder output
        c1_idx = dataset.input_processors["conditions"].code_vocab.get("c1")
        self.assertIsNotNone(c1_idx)
        c1_emb = model.embedding_model.embedding_layers["conditions"].weight[
            c1_idx
        ]
        with torch.no_grad():
            expected_ids = torch.tensor([[2, 3] + [0] * 14])
            expected_emb = custom_encoder(expected_ids)[0]
        self.assertTrue(torch.allclose(c1_emb, expected_emb, atol=1e-5))

    def test_cross_feature_length_alignment(self):
        """Verifies forward pads and aligns features with unequal visit lengths."""
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["c1", "c2"], ["c3"]],
                "procedures": [["p1"], ["p2"], ["p3"]],
                "drugs": ["d1", "d2", "d3", "d4"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["c2"]],
                "procedures": [["p3"], ["p1"]],
                "drugs": ["d2"],
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions": "nested_sequence",
                "procedures": "nested_sequence",
                "drugs": "sequence",
            },
            output_schema={"label": "binary"},
        )
        model = DescEmb(
            dataset=dataset,
            embedding_dim=16,
            text_hidden_dim=16,
            temporal_hidden_dim=16,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertTrue(torch.isfinite(out["logit"]).all())
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["embed"]).all())
        self.assertEqual(out["embed"].shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
