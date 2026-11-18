# Contributor: Ibrahim Rabiu (GitHub: @alhibb, alhibb01@gmail.com)
# Tests for BEHRT (Li et al., Scientific Reports 2020)

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.behrt import BEHRT, BEHRTEmbedding, BEHRTLayer


class TestBEHRT(unittest.TestCase):
    """Test suite for the BEHRT model and its layers."""

    def setUp(self):
        """Builds a test dataset with sequence features."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["c1", "c2", "c3"],
                "procedures": ["p1", "p2"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["c2", "c4"],
                "procedures": ["p2", "p3"],
                "label": 0,
            },
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_behrt",
        )
        self.model = BEHRT(
            dataset=self.dataset,
            embedding_dim=64,
            heads=2,
            num_layers=1,
        )

    def test_model_initialization(self):
        """Verifies BEHRT initializes with expected properties."""
        self.assertIsInstance(self.model, BEHRT)
        self.assertEqual(self.model.embedding_dim, 64)
        self.assertEqual(self.model.heads, 2)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(len(self.model.code_feature_keys), 2)
        self.assertEqual(self.model.pooling, "cls")

    def test_invalid_pooling_raises(self):
        """Ensures invalid pooling modes raise ValueError."""
        with self.assertRaises(ValueError):
            BEHRT(dataset=self.dataset, pooling="unsupported_pool")

    def test_empty_code_features_raises(self):
        """Ensures initializing without code features raises ValueError."""
        dummy_ds = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "age": [50], "label": 1},
                {"patient_id": "p1", "age": [60], "label": 0},
            ],
            input_schema={"age": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_empty",
        )
        with self.assertRaises(ValueError):
            BEHRT(dataset=dummy_ds)

    def test_embedding_layer_standalone(self):
        """Verifies BEHRTEmbedding combines codes, age, segments, and positions."""
        layer = BEHRTEmbedding(
            embedding_dim=32,
            max_age=100,
            max_segments=16,
            max_position=64,
        )
        code_embs = torch.randn(2, 5, 32)
        age_ids = torch.tensor([[45, 45, 45, 46, 46], [60, 60, 61, 61, 61]])
        segment_ids = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 2]])
        position_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

        out = layer(code_embs, age_ids=age_ids, segment_ids=segment_ids)
        self.assertEqual(out.shape, (2, 5, 32))

        # Test with 1D age / segment
        out_1d = layer(
            code_embs,
            age_ids=torch.tensor([30, 50]),
            segment_ids=torch.tensor([0, 1]),
            position_ids=position_ids,
        )
        self.assertEqual(out_1d.shape, (2, 5, 32))

    def test_mismatched_temporal_lengths_padded_and_truncated(self):
        """Verifies age and segment inputs with mismatched lengths are aligned."""
        layer = BEHRTEmbedding(embedding_dim=32, max_age=100, max_segments=30)
        code_embs = torch.randn(2, 6, 32)  # target seq_len is 6

        # age is shorter (length 3), segment is longer (length 9)
        short_age = torch.tensor([[40, 41, 42], [55, 55, 56]])
        long_segment = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4], [0, 1, 1, 2, 2, 3, 3, 4, 4]]
        )

        out = layer(code_embs, age_ids=short_age, segment_ids=long_segment)
        self.assertEqual(out.shape, (2, 6, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_behrt_layer_standalone(self):
        """Verifies BEHRTLayer encoder stack and attention masking."""
        layer = BEHRTLayer(hidden_size=32, heads=2, num_layers=2)
        x = torch.randn(2, 6, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        out = layer(x, mask=mask)
        self.assertEqual(out.shape, (2, 6, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_padding_mask_correctness(self):
        """Ensures fully masked tokens do not cause NaN attention outputs."""
        layer = BEHRTLayer(hidden_size=32, heads=2, num_layers=1)
        x = torch.randn(2, 4, 32)
        # Sequence 0 has tokens, sequence 1 is all padding
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
        out = layer(x, mask=mask)
        self.assertEqual(out.shape, (2, 4, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_binary_classification_forward(self):
        """Tests forward pass on binary classification dataset."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch)

        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        self.assertIn("y_true", out)

        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertTrue(torch.all(out["y_prob"] >= 0.0))
        self.assertTrue(torch.all(out["y_prob"] <= 1.0))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_multiclass_classification_forward(self):
        """Tests forward pass on multiclass dataset."""
        mc_samples = [
            {"patient_id": "p0", "conditions": ["c1", "c2"], "label": 0},
            {"patient_id": "p1", "conditions": ["c2", "c3"], "label": 1},
            {"patient_id": "p2", "conditions": ["c1", "c3"], "label": 2},
        ]
        mc_dataset = create_sample_dataset(
            samples=mc_samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_mc",
        )
        mc_model = BEHRT(
            dataset=mc_dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
        )
        loader = get_dataloader(mc_dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        out = mc_model(**batch)

        self.assertEqual(out["logit"].shape, (3, 3))
        self.assertEqual(out["y_prob"].shape, (3, 3))
        sums = out["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_multilabel_classification_forward(self):
        """Tests forward pass on multilabel dataset."""
        ml_samples = [
            {"patient_id": "p0", "conditions": ["c1"], "label": [0, 1]},
            {"patient_id": "p1", "conditions": ["c2"], "label": [1]},
        ]
        ml_dataset = create_sample_dataset(
            samples=ml_samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="test_ml",
        )
        ml_model = BEHRT(
            dataset=ml_dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
        )
        loader = get_dataloader(ml_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = ml_model(**batch)

        self.assertEqual(out["y_prob"].shape[0], 2)
        self.assertTrue(torch.all(out["y_prob"] >= 0.0))
        self.assertTrue(torch.all(out["y_prob"] <= 1.0))

    def test_with_explicit_age_and_segment(self):
        """Tests BEHRT when age and segment tensors are explicitly passed."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        batch["age"] = torch.tensor([[55, 55, 56], [70, 70, 71]])
        batch["segment"] = torch.tensor([[0, 0, 1], [0, 1, 1]])

        out = self.model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertIn("loss", out)

    def test_protect_age_and_segment_in_dataset_schema(self):
        """Ensures age and segment in input_schema are not treated as code tokens."""
        ds_with_age = create_sample_dataset(
            samples=[
                {
                    "patient_id": "p0",
                    "conditions": ["c1", "c2"],
                    "age": [45, 45],
                    "label": 1,
                },
                {
                    "patient_id": "p1",
                    "conditions": ["c2", "c3"],
                    "age": [60, 60],
                    "label": 0,
                },
            ],
            input_schema={"conditions": "sequence", "age": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_protected",
        )
        model = BEHRT(dataset=ds_with_age, embedding_dim=32, heads=1, num_layers=1)
        self.assertEqual(model.code_feature_keys, ["conditions"])
        self.assertEqual(model.fc.in_features, 32)

        loader = get_dataloader(ds_with_age, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))

    def test_tuple_valued_features(self):
        """Tests that features returned as tuples (value, mask) forward correctly."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        # Wrap conditions into (value, mask) tuple format
        cond_val = batch["conditions"]
        cond_mask = torch.ones_like(cond_val, dtype=torch.bool)
        batch["conditions"] = (cond_val, cond_mask)

        out = self.model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertIn("loss", out)

    def test_mean_pooling_strategy(self):
        """Tests mean pooling across valid tokens and verifies pad invariance."""
        mean_model = BEHRT(
            dataset=self.dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
            pooling="mean",
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = mean_model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))

        # Padded tokens with mask=False do not affect mean pooling of valid tokens
        mask = torch.tensor(
            [[True, True, False, False], [True, True, False, False]], dtype=torch.bool
        )
        layer = BEHRTLayer(hidden_size=16, heads=2, num_layers=1)
        layer.eval()
        x = torch.zeros(2, 4, 16)
        x[:, 0:2, :] = 1.0
        x[1, 2:, :] = 42.0  # arbitrary pad values in sequence 1

        with torch.no_grad():
            enc = layer(x, mask=mask)
        mask_exp = mask.unsqueeze(-1).float()
        pooled = (enc * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
        self.assertTrue(torch.allclose(pooled[0], pooled[1], atol=1e-5))

    def test_odd_embedding_dim(self):
        """Verifies odd embedding_dim support in embedding layer and model."""
        odd_emb = BEHRTEmbedding(embedding_dim=63)
        code_embs = torch.randn(2, 5, 63)
        out = odd_emb(code_embs)
        self.assertEqual(out.shape, (2, 5, 63))

        odd_model = BEHRT(
            dataset=self.dataset,
            embedding_dim=63,
            heads=1,
            num_layers=1,
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        model_out = odd_model(**batch)
        self.assertEqual(model_out["logit"].shape, (2, 1))

        with self.assertRaises(ValueError):
            BEHRT(dataset=self.dataset, embedding_dim=63, heads=2)

    def test_forward_from_embedding_direct(self):
        """Tests direct call to forward_from_embedding with pre-embedded features."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        embedded = self.model.embedding_model(
            {"conditions": batch["conditions"], "procedures": batch["procedures"]}
        )
        direct_kwargs = {
            "conditions": (embedded["conditions"],),
            "procedures": (embedded["procedures"],),
            "label": batch["label"],
            "age": torch.tensor([[45, 45, 46], [60, 60, 61]]),
        }
        out = self.model.forward_from_embedding(**direct_kwargs)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertIn("loss", out)

    def test_embed_flag_and_backward(self):
        """Verifies patient embedding output and gradient flow."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch, embed=True)

        self.assertIn("embed", out)
        self.assertEqual(
            out["embed"].shape, (2, 64 * len(self.model.code_feature_keys))
        )

        loss = out["loss"]
        loss.backward()

        self.assertIsNotNone(self.model.fc.weight.grad)
        self.assertIsNotNone(self.model.pooler[0].weight.grad)
        for name, param in self.model.behrt_layers.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Layer parameter {name} has no gradient",
                )


if __name__ == "__main__":
    unittest.main()
