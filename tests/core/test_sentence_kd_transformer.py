"""Unit tests for ``pyhealth.models.SentenceKDTransformer``.

All tests use either small synthetic tensors or a tiny synthetic text
dataset paired with ``prajjwal1/bert-tiny`` (17 MB). No real medical
datasets are touched. HuggingFace caches are redirected into a
per-class temporary directory that is cleaned up in :meth:`tearDownClass`.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from typing import List

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SentenceKDTransformer
from pyhealth.models.sentence_kd_transformer import supervised_contrastive_loss


_TINY_BERT = "prajjwal1/bert-tiny"


def _make_text_dataset():
    """Tiny 3-way synthetic sentence-classification dataset.

    Returns a fitted ``SampleDataset`` whose label vocab is deterministic
    (sorted strings → ``{"abnormal": 0, "normal": 1, "uncertain": 2}``).
    """
    samples = [
        {"patient_id": "p0", "sentence": "no acute findings", "label": "normal"},
        {"patient_id": "p1", "sentence": "large pleural effusion", "label": "abnormal"},
        {"patient_id": "p2", "sentence": "possible early consolidation",
         "label": "uncertain"},
        {"patient_id": "p3", "sentence": "lungs are clear", "label": "normal"},
        {"patient_id": "p4", "sentence": "opacity in right lower lobe",
         "label": "abnormal"},
        {"patient_id": "p5", "sentence": "may represent atelectasis",
         "label": "uncertain"},
        {"patient_id": "p6", "sentence": "no evidence of pneumothorax",
         "label": "normal"},
        {"patient_id": "p7", "sentence": "right upper lobe mass",
         "label": "abnormal"},
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"sentence": "text"},
        output_schema={"label": "multiclass"},
        dataset_name="sentence-kd-test",
    )


class TestSupervisedContrastiveLoss(unittest.TestCase):
    """Tests for the module-level ``supervised_contrastive_loss`` helper."""

    def test_basic_positive_and_has_gradient(self):
        torch.manual_seed(0)
        features = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        loss = supervised_contrastive_loss(features, labels, temperature=0.07)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_degenerate_no_positive_pairs_returns_zero_with_grad(self):
        features = torch.randn(3, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        loss = supervised_contrastive_loss(features, labels)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_identical_features_same_class_yield_log_n_minus_one(self):
        # When all embeddings collapse to the same unit vector and share a
        # class label, every anchor treats its (N-1) siblings as positives
        # with identical similarity. The loss then reduces analytically to
        # ``log(N-1)`` -- this is the contrastive "floor" for fully
        # collapsed representations.
        import math

        z = torch.randn(1, 8)
        features = z.repeat(4, 1).clone().requires_grad_(True)
        labels = torch.tensor([0, 0, 0, 0])
        loss = supervised_contrastive_loss(features, labels, temperature=0.1)
        self.assertAlmostEqual(loss.item(), math.log(3), places=4)

    def test_invalid_shapes_raise(self):
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(torch.randn(4, 2, 3), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(torch.randn(4, 8), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(
                torch.randn(4, 8), torch.tensor([0, 1, 0, 1]), temperature=0
            )


class TestSentenceKDTransformer(unittest.TestCase):
    """Model-level tests for ``SentenceKDTransformer`` with bert-tiny."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp(prefix="skd_hf_cache_")
        cls._prev_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = cls._tmp
        cls.dataset = _make_text_dataset()
        cls.model = SentenceKDTransformer(
            dataset=cls.dataset,
            model_name=_TINY_BERT,
            lam=1.0,
            temperature=0.07,
            max_length=16,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._prev_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = cls._prev_hf_home
        shutil.rmtree(cls._tmp, ignore_errors=True)

    # ---- instantiation -------------------------------------------------
    def test_instantiation_and_inferred_shapes(self):
        model = self.model
        self.assertEqual(model.feature_key, "sentence")
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.get_output_size(), 3)
        self.assertEqual(model.fc.out_features, 3)
        self.assertEqual(model.mode, "multiclass")

    def test_constructor_validates_inputs(self):
        with self.assertRaises(ValueError):
            SentenceKDTransformer(
                dataset=self.dataset, model_name=_TINY_BERT, lam=-0.5
            )
        with self.assertRaises(ValueError):
            SentenceKDTransformer(
                dataset=self.dataset, model_name=_TINY_BERT, temperature=0.0
            )
        with self.assertRaises(ValueError):
            SentenceKDTransformer(
                dataset=self.dataset, model_name=_TINY_BERT, doc_agg="bogus"
            )

    # ---- forward --------------------------------------------------------
    def test_forward_output_keys_and_shapes(self):
        loader = get_dataloader(self.dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (8, 3))
        self.assertEqual(out["y_prob"].shape, (8, 3))
        self.assertEqual(out["y_true"].shape, (8,))
        self.assertEqual(out["loss"].dim(), 0)
        probs = out["y_prob"]
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5))

    def test_forward_with_embed_flag(self):
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("embed", out)
        self.assertEqual(out["embed"].shape[0], 4)
        self.assertEqual(out["embed"].shape[1], self.model.model.config.hidden_size)

    def test_forward_backward_updates_parameters(self):
        loader = get_dataloader(self.dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch)
        out["loss"].backward()
        head_grad = self.model.fc.weight.grad
        self.assertIsNotNone(head_grad)
        self.assertTrue(torch.isfinite(head_grad).all())
        self.assertGreater(head_grad.abs().sum().item(), 0.0)

    def test_lambda_zero_loss_matches_cross_entropy(self):
        model = SentenceKDTransformer(
            dataset=self.dataset, model_name=_TINY_BERT, lam=0.0, max_length=16
        )
        loader = get_dataloader(self.dataset, batch_size=6, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(**batch)
            ce = F.cross_entropy(out["logit"], out["y_true"])
        self.assertTrue(torch.allclose(out["loss"], ce, atol=1e-6))

    # ---- document-level inference --------------------------------------
    def test_document_predict_shapes_and_ranges(self):
        sentences: List[str] = [
            "no acute findings",
            "patchy opacity in the lung base",
            "possible infection",
            "heart size normal",
            "large effusion",
        ]
        doc = self.model.document_predict(sentences)
        self.assertIn("pa", doc)
        self.assertIn("pn", doc)
        self.assertIn("per_sentence_probs", doc)
        self.assertIn("abnormal_index", doc)
        self.assertEqual(doc["per_sentence_probs"].shape, (5, 3))
        self.assertGreaterEqual(doc["pa"], 0.0)
        self.assertLessEqual(doc["pa"], 1.0)
        self.assertAlmostEqual(doc["pa"] + doc["pn"], 1.0, places=5)
        # Label vocab sorts strings → "abnormal" should resolve to 0.
        self.assertEqual(doc["abnormal_index"], 0)

    def test_document_predict_agg_modes(self):
        sentences = [
            "no acute findings",
            "large effusion",
            "possible infection",
        ]
        mode_to_pa = {}
        for mode in ("max", "topk_mean", "attn"):
            d = self.model.document_predict(sentences, doc_agg=mode)
            self.assertEqual(d["per_sentence_probs"].shape, (3, 3))
            mode_to_pa[mode] = d["pa"]
        # "max" is always at least as large as any weighted average.
        self.assertGreaterEqual(mode_to_pa["max"], mode_to_pa["topk_mean"] - 1e-6)
        self.assertGreaterEqual(mode_to_pa["max"], mode_to_pa["attn"] - 1e-6)

    def test_document_predict_max_matches_paper_eq4(self):
        # Paper Eq. 4: p_a(doc) = max_j p_abnormal(sentence_j).
        sentences = ["no acute findings", "large effusion"]
        doc = self.model.document_predict(sentences, doc_agg="max")
        abn_idx = doc["abnormal_index"]
        expected = float(doc["per_sentence_probs"][:, abn_idx].max().item())
        self.assertAlmostEqual(doc["pa"], expected, places=5)

    def test_document_predict_empty_raises(self):
        with self.assertRaises(ValueError):
            self.model.document_predict([])

    def test_document_predict_preserves_training_mode(self):
        self.model.train()
        _ = self.model.document_predict(["one", "two"])
        self.assertTrue(self.model.training)
        self.model.eval()
        _ = self.model.document_predict(["one"])
        self.assertFalse(self.model.training)


class TestSentenceKDTransformerTiming(unittest.TestCase):
    """Ensure all documented tests in this file are fast enough for CI.

    Slow tests are penalized by the DL4H rubric. bert-tiny on CPU forward +
    backward for batch size 8 must stay well under 1 second.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _make_text_dataset()
        cls.model = SentenceKDTransformer(
            dataset=cls.dataset, model_name=_TINY_BERT, lam=1.0, max_length=16
        )

    def test_forward_backward_under_one_second(self):
        loader = get_dataloader(self.dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        # Warm-up pass to prime tokenizer caches and layer fusion.
        _ = self.model(**batch)
        start = time.perf_counter()
        out = self.model(**batch)
        out["loss"].backward()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"forward+backward took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
