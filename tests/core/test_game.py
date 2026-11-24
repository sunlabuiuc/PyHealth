"""Unit tests for GAME model and embedding initializers."""

import os
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    GAME,
    GAMEEmbedding,
    init_embedding_with_game,
    load_game_embeddings,
    normalize_game_code,
)


class TestGAME(unittest.TestCase):
    """Comprehensive test suite for GAME."""

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

    def test_normalize_game_code(self):
        self.assertEqual(normalize_game_code("ICD10CM:I10"), "i10")
        self.assertEqual(normalize_game_code("CCS:1"), "1")
        self.assertEqual(normalize_game_code(" LOINC:15069-8 "), "15069-8")
        self.assertEqual(normalize_game_code("RxNorm:12345"), "12345")
        self.assertEqual(normalize_game_code("401.9"), "401.9")
        self.assertEqual(normalize_game_code(""), "")
        self.assertEqual(normalize_game_code("   "), "")

    def test_load_game_embeddings_dict_and_tensor(self):
        # Dict of tensors
        raw_dict = {"c1": torch.ones(8), "c2": torch.full((8,), 2.0)}
        embs = load_game_embeddings(raw_dict, embedding_dim=8)
        self.assertIn("c1", embs)
        self.assertIn("c2", embs)
        self.assertEqual(embs["c1"].shape, torch.Size([8]))

        # Dict of lists
        raw_list_dict = {"c1": [1.0] * 8, "c2": [2.0] * 8}
        embs_list = load_game_embeddings(raw_list_dict, embedding_dim=8)
        self.assertTrue(torch.allclose(embs_list["c1"], torch.ones(8)))

        # 2D tensor
        tensor_embs = torch.randn(5, 8)
        t_dict = load_game_embeddings(tensor_embs, embedding_dim=8)
        self.assertEqual(len(t_dict), 5)
        self.assertEqual(t_dict["0"].shape, torch.Size([8]))

    def test_load_game_embeddings_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # .pt file
            pt_path = os.path.join(tmpdir, "embs.pt")
            torch.save({"c1": torch.ones(8), "c2": torch.full((8,), 2.0)}, pt_path)
            loaded_pt = load_game_embeddings(pt_path, embedding_dim=8)
            self.assertEqual(len(loaded_pt), 2)
            self.assertIn("c1", loaded_pt)

            # .npy file
            npy_path = os.path.join(tmpdir, "embs.npy")
            np.save(npy_path, np.ones((4, 8), dtype=np.float32))
            loaded_npy = load_game_embeddings(npy_path, embedding_dim=8)
            self.assertEqual(len(loaded_npy), 4)

            # .npz file with codes
            npz_path = os.path.join(tmpdir, "embs.npz")
            np.savez(
                npz_path,
                embeddings=np.ones((2, 8), dtype=np.float32),
                codes=np.array(["c1", "c2"]),
            )
            loaded_npz = load_game_embeddings(npz_path, embedding_dim=8)
            self.assertIn("c1", loaded_npz)

            # .csv / text file
            txt_path = os.path.join(tmpdir, "embs.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("c1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n")
                f.write("c2 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0\n")
            loaded_txt = load_game_embeddings(txt_path, embedding_dim=8)
            self.assertIn("c1", loaded_txt)
            self.assertTrue(torch.allclose(loaded_txt["c1"], torch.ones(8)))

    def test_init_embedding_with_game(self):
        vocab = {"<pad>": 0, "c1": 1, "c2": 2, "c3": 3}
        embs = {"c1": torch.ones(8), "c2": torch.full((8,), 3.0)}
        emb = nn.Embedding(len(vocab), 8, padding_idx=0)

        # Before init, c1 is random
        matched = init_embedding_with_game(
            embedding=emb,
            code_vocab=vocab,
            game_embeddings=embs,
            freeze=True,
            normalize=True,
        )
        self.assertEqual(matched, 2)
        # Padding row is strictly 0
        self.assertTrue((emb.weight[0] == 0.0).all())
        # Weights are frozen
        self.assertFalse(emb.weight.requires_grad)
        # Matched rows are normalized
        self.assertAlmostEqual(emb.weight[1].norm(p=2).item(), 1.0, places=5)
        self.assertAlmostEqual(emb.weight[2].norm(p=2).item(), 1.0, places=5)

    def test_game_embedding_module(self):
        vocab = {"<pad>": 0, "c1": 1, "c2": 2}
        layer = GAMEEmbedding(
            vocab_size=len(vocab),
            embedding_dim=8,
            code_vocab=vocab,
            game_embeddings={"c1": torch.ones(8)},
            projection_dim=16,
        )
        # 1D input
        x1 = torch.tensor([1, 2, 0])
        out1 = layer(x1)
        self.assertEqual(out1.shape, (3, 16))

        # 2D input
        x2 = torch.tensor([[1, 2], [0, 1]])
        out2 = layer(x2)
        self.assertEqual(out2.shape, (2, 2, 16))

        # 3D input
        x3 = torch.tensor([[[1, 2], [0, 0]], [[2, 1], [1, 0]]])
        out3 = layer(x3)
        self.assertEqual(out3.shape, (2, 2, 2, 16))

    def test_binary_classification(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        game_embs = {
            "c1": torch.ones(16),
            "c2": torch.full((16,), 2.0),
            "p1": torch.full((16,), 0.5),
        }
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
            game_embeddings=game_embs,
            temporal_model="GRU",
            pooling_mode="attention",
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_multiclass_classification(self):
        samples_mc = [dict(s, label=i % 3) for i, s in enumerate(self.samples)]
        dataset = create_sample_dataset(
            samples=samples_mc,
            input_schema=self.input_schema,
            output_schema={"label": "multiclass"},
        )
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
            temporal_model="LSTM",
            pooling_mode="mean",
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
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
            temporal_model="Transformer",
            pooling_mode="max",
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertEqual(out["logit"].shape, (2, 3))
        self.assertEqual(out["y_prob"].shape, (2, 3))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_empty_patient(self):
        """Verifies completely padded patient visits do not cause NaNs."""
        samples_with_empty = list(self.samples) + [
            {
                "patient_id": "p_empty",
                "visit_id": "v_empty",
                "conditions": [[]],
                "procedures": [[]],
                "label": 0,
            }
        ]
        dataset = create_sample_dataset(
            samples=samples_with_empty,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
            pooling_mode="last",
        )
        loader = get_dataloader(dataset, batch_size=4)
        batch = next(iter(loader))
        out = model(**batch)

        self.assertTrue(torch.isfinite(out["embed"]).all())
        self.assertTrue(torch.isfinite(out["logit"]).all())
        self.assertTrue(torch.isfinite(out["y_prob"]).all())

    def test_gradient_flow(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
            freeze_embeddings=False,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        out = model(**batch)
        loss = out["loss"]
        loss.backward()

        # Check gradients exist
        cond_weight = model.game_embeddings["conditions"].embedding.weight
        self.assertIsNotNone(cond_weight.grad)
        self.assertTrue((cond_weight.grad[0] == 0.0).all())

    def test_tuple_input(self):
        """Tests tuple inputs (time, value) and (value, mask)."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
        )
        loader = get_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        cond = batch["conditions"]

        # 1. (value, mask)
        mask = cond != 0
        batch["conditions"] = (cond, mask)
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))

        # 2. (time, value)
        times = torch.rand(cond.shape[:2], dtype=torch.float32)
        batch["conditions"] = (times, cond)
        out2 = model(**batch)
        self.assertEqual(out2["logit"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out2["logit"]).all())

    def test_transformer_empty_patient_and_nhead_divisibility(self):
        """Verifies Transformer with d_model=200 and empty patient produces no NaNs."""
        dataset = create_sample_dataset(
            samples=self.samples[:2],
            input_schema={
                "conditions": "nested_sequence",
                "procedures": "nested_sequence",
            },
            output_schema={"label": "binary"},
        )
        # embedding_dim=100 with 2 features gives total_input_dim=200 (divisible by 8)
        model = GAME(
            dataset=dataset,
            embedding_dim=100,
            hidden_dim=64,
            temporal_model="Transformer",
            pooling_mode="attention",
        )
        num_heads = model.temporal_model.layers[0].self_attn.num_heads
        self.assertEqual(200 % num_heads, 0)

        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        # Zero out patient 1 completely
        batch["conditions"][1].zero_()
        batch["procedures"][1].zero_()

        out = model(**batch)
        self.assertFalse(torch.isnan(out["logit"]).any())
        self.assertFalse(torch.isnan(out["embed"]).any())
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue((out["embed"][1] == 0.0).all())

        # Test explicit invalid nhead raises ValueError
        with self.assertRaises(ValueError):
            GAME(
                dataset=dataset,
                embedding_dim=100,
                temporal_model="Transformer",
                nhead=3,  # 200 is not divisible by 3
            )

    def test_code_mapping(self):
        """Verifies custom code_mapping translates dataset tokens to GAME codes."""
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "binary"},
        )
        code_mapping = {"c1": "ICD10CM:E11", "c2": "ICD10CM:I10"}
        game_embs = {"e11": [1.0] * 16, "i10": [2.0] * 16}
        model = GAME(
            dataset=dataset,
            embedding_dim=16,
            hidden_dim=32,
            game_embeddings=game_embs,
            code_mapping=code_mapping,
        )
        emb_layer = model.game_embeddings["conditions"].embedding
        vocab = dataset.input_processors["conditions"].code_vocab
        idx_c1 = vocab["c1"]
        self.assertTrue(torch.allclose(emb_layer.weight[idx_c1], torch.ones(16)))

    def test_pt_file_loading_weights_only(self):
        """Verifies loading GAME embeddings from .pt file with weights_only=True."""
        embs_dict = {"e11": torch.randn(16), "i10": torch.randn(16)}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(embs_dict, tmp_path)
            loaded = load_game_embeddings(tmp_path, embedding_dim=16)
            self.assertIn("e11", loaded)
            self.assertEqual(loaded["e11"].shape, (16,))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_args(self):
        dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            GAME(dataset, embedding_dim=0)
        with self.assertRaises(ValueError):
            GAME(dataset, hidden_dim=-1)
        with self.assertRaises(ValueError):
            GAME(dataset, temporal_model="INVALID")
        with self.assertRaises(ValueError):
            GAME(dataset, pooling_mode="INVALID")
        with self.assertRaises(ValueError):
            GAMEEmbedding(vocab_size=0, embedding_dim=16)
        with self.assertRaises(ValueError):
            GAMEEmbedding(vocab_size=10, embedding_dim=0)
