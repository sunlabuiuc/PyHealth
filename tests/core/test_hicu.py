"""Unit tests for the HiCu model (pyhealth.models.hicu).

Authors: Huyen Nguyen (hnguyen654), Aaron Tran (atran403)
Course: CSE 6250 Big Data for Health, Georgia Tech, Fall 2026

All tests use a handful of synthetic notes and tiny layer sizes, so each
runs in milliseconds and no real or protected data is needed.
"""

import unittest

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import HiCu
from pyhealth.models.hicu import (
    AsymmetricLoss,
    icd9_code_path,
    train_poincare_embeddings,
)

SAMPLES = [
    {
        "patient_id": "p0",
        "text": "Hypertension, on amlodipine. HTN stable.",
        "icd_codes": ["401.9"],
    },
    {
        "patient_id": "p1",
        "text": "Type II diabetes on glyburide; htn.",
        "icd_codes": ["250.00", "401.9"],
    },
    {
        "patient_id": "p2",
        "text": "Coronary bypass graft surgery (CABG).",
        "icd_codes": ["36.15", "414.01"],
    },
    {
        "patient_id": "p3",
        "text": "Diabetes, insulin dependent.",
        "icd_codes": ["250.01"],
    },
]

# tiny sizes keep every test in milliseconds
SMALL = {
    "embedding_dim": 8,
    "filter_sizes": (3, 5),
    "num_filter_maps": 4,
    "hyperbolic_dim": 4,
    "poincare_epochs": 1,
    "min_word_freq": 1,
}


class TestHiCu(unittest.TestCase):
    """Tests for the HiCu model and its helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = create_sample_dataset(
            samples=SAMPLES,
            input_schema={"text": "text"},
            output_schema={"icd_codes": "multilabel"},
            dataset_name="hicu_test",
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    @classmethod
    def tearDownClass(cls) -> None:
        # create_sample_dataset writes to a temporary directory: remove it
        cls.dataset.close()

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = HiCu(self.dataset, **SMALL)

    # ------------------------------------------------------------- helpers

    def test_icd9_code_path(self) -> None:
        """Codes map to 5-level chapter/block/code paths."""
        self.assertEqual(
            icd9_code_path("401.9"),
            ["390-459", "401-405", "401", "401.9", "401.9"],
        )
        self.assertEqual(
            icd9_code_path("38.93"), ["35-39", "38-38", "38", "38.9", "38.93"]
        )
        self.assertEqual(icd9_code_path("V45.81")[:2], ["V01-V91", "V40-V49"])
        self.assertEqual(icd9_code_path("E849.7")[:2], ["E000-E999", "E846-E849"])
        # MIMIC-III style codes without a dot share the dotted parents
        self.assertEqual(icd9_code_path("4019")[3], "401.9")
        self.assertEqual(icd9_code_path("4019")[-1], "4019")
        # unknown codes go to an OTHER branch instead of failing
        self.assertEqual(icd9_code_path("xyz")[0], "OTHER")

    def test_poincare_embeddings_in_ball(self) -> None:
        """Poincare embeddings stay strictly inside the unit ball."""
        edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
        emb = train_poincare_embeddings(edges, 5, dim=3, epochs=5, batch_size=2)
        self.assertEqual(emb.shape, (5, 3))
        self.assertTrue(torch.all(emb.norm(dim=1) < 1))

    def test_asl_reduces_to_bce(self) -> None:
        """With no focusing and no clipping, ASL equals BCE."""
        logits = torch.randn(3, 5)
        targets = (torch.rand(3, 5) > 0.5).float()
        asl = AsymmetricLoss(0.0, 0.0, 0.0, reduction="mean")
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        self.assertTrue(torch.allclose(asl(logits, targets), bce, atol=1e-6))

    def test_asl_downweights_easy_negatives(self) -> None:
        """A confident correct negative costs less under ASL than BCE."""
        logits = torch.tensor([[-3.0]])
        targets = torch.zeros(1, 1)
        asl = AsymmetricLoss(gamma_neg=1.0, gamma_pos=0.0, clip=0.05)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
        self.assertLess(asl(logits, targets).item(), bce.item())

    # -------------------------------------------------------- construction

    def test_initialization(self) -> None:
        """The model reads the schema and builds one decoder per level."""
        model = self.model
        self.assertEqual(model.feature_key, "text")
        self.assertEqual(model.label_key, "icd_codes")
        self.assertEqual(model.num_levels, 5)
        self.assertEqual(model.current_level, 0)
        # leaf level is the label processor's codes, in its order
        vocab = self.dataset.output_processors["icd_codes"].label_vocab
        self.assertEqual(model.level_nodes[-1], sorted(vocab, key=vocab.get))
        # 3 chapters: 240-279, 35-39 (procedures), 390-459
        self.assertEqual(len(model.level_nodes[0]), 3)
        self.assertEqual(len(model.queries), 5)
        self.assertEqual(len(model.hyperbolic_fc), 5)
        self.assertIn("diabetes", model.word2id)

    def test_baseline_has_one_level(self) -> None:
        """use_curriculum=False gives plain MultiResCNN on the codes."""
        model = HiCu(self.dataset, use_curriculum=False, **SMALL)
        self.assertEqual(model.num_levels, 1)
        self.assertIsNone(model.hyperbolic)
        out = model(**self.batch)
        self.assertEqual(out["logit"].shape, (4, 5))

    def test_invalid_arguments(self) -> None:
        """Bad options raise ValueError."""
        with self.assertRaises(ValueError):
            HiCu(self.dataset, hyperbolic="mult", **SMALL)
        with self.assertRaises(ValueError):
            HiCu(self.dataset, loss="focal", **SMALL)
        with self.assertRaises(ValueError):
            HiCu(self.dataset, **{**SMALL, "filter_sizes": (4,)})
        with self.assertRaises(ValueError):
            self.model.set_level(5)

    def test_custom_code_paths(self) -> None:
        """Any same-depth hierarchy can replace the ICD-9 tree."""
        codes = ["250.00", "250.01", "36.15", "401.9", "414.01"]
        paths = {c: ["A" if c < "3" else "B", c] for c in codes}
        model = HiCu(self.dataset, code_paths=paths, **SMALL)
        self.assertEqual(model.num_levels, 2)
        self.assertEqual(model.level_nodes[0], ["A", "B"])
        with self.assertRaises(ValueError):
            HiCu(self.dataset, code_paths={"401.9": ["B", "401.9"]}, **SMALL)
        with self.assertRaises(ValueError):
            HiCu(self.dataset, code_paths={**paths, "401.9": ["401.9"]}, **SMALL)

    # ------------------------------------------------------------- forward

    def test_forward_shapes_every_level(self) -> None:
        """Each level predicts its own nodes, with targets mapped up the tree."""
        for level in range(self.model.num_levels):
            self.model.set_level(level)
            with torch.no_grad():
                out = self.model(**self.batch)
            n_nodes = len(self.model.level_nodes[level])
            self.assertEqual(out["logit"].shape, (4, n_nodes))
            self.assertEqual(out["y_prob"].shape, (4, n_nodes))
            self.assertEqual(out["y_true"].shape, (4, n_nodes))
            self.assertEqual(out["loss"].dim(), 0)
            self.assertTrue(torch.all((out["y_prob"] >= 0) & (out["y_prob"] <= 1)))

    def test_level_targets(self) -> None:
        """A node is positive when any code below it is positive."""
        y_leaf = self.batch["icd_codes"]
        y_top = self.model.level_targets(y_leaf, 0)
        chapters = self.model.level_nodes[0]
        # sample 1 has 250.00 (240-279) and 401.9 (390-459)
        expected = [1.0 if c in ("240-279", "390-459") else 0.0 for c in chapters]
        self.assertEqual(y_top[1].tolist(), expected)
        self.assertTrue(torch.equal(self.model.level_targets(y_leaf, 4), y_leaf))

    def test_forward_without_labels(self) -> None:
        """Inference without labels returns predictions only."""
        out = self.model(text=self.batch["text"])
        self.assertNotIn("loss", out)
        self.assertIn("y_prob", out)

    def test_backward(self) -> None:
        """Gradients reach the encoder and the current level's decoder only."""
        self.model.set_level(2)
        self.model(**self.batch)["loss"].backward()
        self.assertIsNotNone(self.model.encoder.embedding.weight.grad)
        self.assertIsNotNone(self.model.queries[2].grad)
        self.assertIsNotNone(self.model.hyperbolic_fc[2].weight.grad)
        self.assertIsNone(self.model.queries[3].grad)

    def test_knowledge_transfer(self) -> None:
        """Moving down one level copies each parent's query to its children."""
        model = self.model
        for level in range(1, model.num_levels):
            model.set_level(level)
            parent = getattr(model, f"_parent_{level}")
            self.assertTrue(
                torch.equal(model.queries[level], model.queries[level - 1][parent])
            )
            self.assertTrue(
                torch.equal(model.weights[level], model.weights[level - 1][parent])
            )

    def test_set_level_without_transfer(self) -> None:
        """transfer=False keeps the level's own (random) initialization."""
        before = self.model.queries[1].detach().clone()
        self.model.set_level(1, transfer=False)
        self.assertTrue(torch.equal(self.model.queries[1], before))

    def test_hyperbolic_modes(self) -> None:
        """HiCu-C (concatenation) and no correction both run."""
        for mode, n_fc in (("cat", 5), (None, 0)):
            model = HiCu(self.dataset, hyperbolic=mode, **SMALL)
            self.assertEqual(len(model.hyperbolic_fc), n_fc)
            model.set_level(1)
            self.assertEqual(model(**self.batch)["logit"].shape[0], 4)

    def test_bce_loss_option(self) -> None:
        """loss='bce' uses binary cross-entropy at every level."""
        model = HiCu(self.dataset, loss="bce", **SMALL)
        out = model(**self.batch)
        bce = F.binary_cross_entropy_with_logits(out["logit"], out["y_true"])
        self.assertTrue(torch.allclose(out["loss"], bce))

    def test_text_encoding(self) -> None:
        """Text is tokenized, unknown words map to 1, length is capped."""
        self.assertEqual(
            HiCu.tokenize("HTN, on 5mg Amlodipine!"), ["htn", "on", "mg", "amlodipine"]
        )
        model = HiCu(self.dataset, max_length=3, **SMALL)
        ids = model._encode_text(["diabetes zzzz on insulin today", ""])
        self.assertEqual(ids.shape, (2, 3))
        self.assertEqual(ids[0, 1].item(), 1)  # unknown word
        self.assertEqual(ids[1].tolist(), [0, 0, 0])  # empty note is padding

    def test_embed_output(self) -> None:
        """embed=True also returns a pooled representation."""
        out = self.model(embed=True, **self.batch)
        self.assertEqual(out["embed"].shape, (4, self.model.encoder.output_dim))


if __name__ == "__main__":
    unittest.main()
