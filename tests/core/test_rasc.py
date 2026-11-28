"""Tests for RASCCrossEncoder.

Author: Wenchang Qi (NetID: qi14)

Uses a randomly initialised two-layer BERT (about 6k parameters) built in a
temporary directory and four in-memory (title, candidate) pairs, so no
network access or model download. The task-side tests (text inputs, hybrid
retrieval, task-to-model pipeline) are in
test_value_set_pair_classification.py.
"""

import contextlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RASCCrossEncoder

WORDS = [
    "type",
    "1",
    "2",
    "diabetes",
    "mellitus",
    "hypertension",
    "essential",
    "drugs",
    "kidney",
    "disorder",
    "of",
    "name",
    "gestational",
    "without",
    "complications",
    "e11",
    "e10",
    "i10",
    "9",
    "65",
    ".",
]
PAIRS = [
    ("type 2 diabetes", "type 2 diabetes mellitus without complications", 0.9, 1),
    ("type 2 diabetes", "gestational diabetes mellitus", 0.7, 0),
    ("hypertension", "essential hypertension", 0.8, 1),
    ("hypertension", "type 1 diabetes mellitus", 0.2, 0),
]

_SHARED: dict = {}


def write_tiny_bert(path: Path) -> str:
    """Saves a random two-layer BERT pair classifier and its tokenizer."""
    path.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(str(path / "vocab.txt")).save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=1,
    )
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(path)
    return str(path)


def make_dataset(label="binary", fusion=True):
    """Four (title, candidate) pairs, with or without the fusion inputs."""
    labels = [1, 0, 1, 0] if label != "multiclass" else ["a", "b", "c", "a"]
    samples = []
    for i, (title, candidate, score, _) in enumerate(PAIRS):
        sample = {
            "patient_id": f"vs{i // 2}",
            "title_text": title,
            "candidate_text": candidate,
            "label": labels[i],
        }
        if fusion:
            sample["score"] = [score]
            sample["code_system"] = [1.0, 0.0, 0.0]
        samples.append(sample)
    schema = {"title_text": "text", "candidate_text": "text"}
    if fusion:
        schema.update(score="tensor", code_system="tensor")
    return create_sample_dataset(
        samples=samples,
        input_schema=schema,
        output_schema={"label": label},
        dataset_name="test_rasc",
        in_memory=True,
    )


def setUpModule() -> None:
    """Builds the tiny encoder and the two in-memory datasets once."""
    tmp = Path(tempfile.mkdtemp())
    _SHARED.update(
        tmp=tmp,
        bert=write_tiny_bert(tmp / "bert"),
        fused=make_dataset(),
        text_only=make_dataset(fusion=False),
    )


def tearDownModule() -> None:
    """Closes the datasets and removes the temporary directory."""
    for name in ("fused", "text_only"):
        _SHARED[name].close()
    shutil.rmtree(_SHARED["tmp"], ignore_errors=True)


class RASCFixture(unittest.TestCase):
    """Exposes the module fixture and builds seeded models."""

    def model(self, dataset: str = "fused", **kwargs) -> RASCCrossEncoder:
        # The tiny encoder reads 32 positions, below the default of 128.
        kwargs.setdefault("max_length", 16)
        torch.manual_seed(0)
        model = RASCCrossEncoder(_SHARED[dataset], model_name=_SHARED["bert"], **kwargs)
        return model.eval()

    def batch(self, dataset: str = "fused") -> dict:
        return next(iter(get_dataloader(_SHARED[dataset], batch_size=4)))


class TestRASCCrossEncoderInit(RASCFixture):
    def test_defaults(self):
        model = self.model()
        self.assertEqual(model.mode, "binary")
        self.assertEqual(model.max_length, 16)
        self.assertEqual(model.fusion_keys, ["score", "code_system"])
        self.assertEqual(model.fusion_dim, 4)
        self.assertEqual(model.model.config.num_labels, 1)

    def test_fusion_starts_at_zero(self):
        self.assertEqual(float(self.model().fusion.weight.abs().sum()), 0.0)

    def test_fusion_can_be_turned_off_or_narrowed(self):
        self.assertIsNone(self.model(fusion_keys=()).fusion)
        self.assertIsNone(self.model("text_only").fusion)
        self.assertEqual(self.model(fusion_keys=["score"]).fusion_dim, 1)

    def test_dropout_reaches_the_encoder(self):
        config = self.model(dropout=0.25).model.config
        self.assertEqual(config.hidden_dropout_prob, 0.25)
        self.assertEqual(config.attention_probs_dropout_prob, 0.25)

    def test_gradient_checkpointing(self):
        self.assertFalse(self.model().model.is_gradient_checkpointing)
        model = self.model(gradient_checkpointing=True)
        self.assertTrue(model.model.is_gradient_checkpointing)

    def test_revision_reaches_every_download(self):
        # A local path ignores the revision, so the real loaders still run.
        loaders = (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
        with contextlib.ExitStack() as stack:
            spies = [
                stack.enter_context(
                    mock.patch.object(
                        loader, "from_pretrained", wraps=loader.from_pretrained
                    )
                )
                for loader in loaders
            ]
            model = self.model(revision="main")
        self.assertEqual(model.revision, "main")
        for loader, spy in zip(loaders, spies):
            self.assertEqual(spy.call_args.kwargs["revision"], "main", msg=loader)
        self.assertIsNone(self.model().revision)

    def test_rejects_bad_arguments(self):
        for kwargs in (
            {"max_length": 4},
            {"max_length": 33},  # the tiny encoder reads 32 positions
            {"title_key": "missing"},
            {"candidate_key": "score"},  # not a text input
            {"fusion_keys": ["title_text"]},  # a text input
            {"fusion_keys": ["missing"]},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                self.model(**kwargs)

    def test_rejects_non_binary_label(self):
        dataset = make_dataset(label="multiclass")
        try:
            with self.assertRaisesRegex(ValueError, "binary"):
                RASCCrossEncoder(dataset, model_name=_SHARED["bert"], max_length=16)
        finally:
            dataset.close()


class TestRASCCrossEncoderForward(RASCFixture):
    def test_output_shapes(self):
        out = self.model(max_length=16)(**self.batch())
        for key in ("logit", "y_prob", "y_true"):
            self.assertEqual(out[key].shape, (4, 1))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())

    def test_without_label_returns_no_loss(self):
        batch = self.batch()
        del batch["label"]
        out = self.model(max_length=16)(**batch)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (4, 1))

    def test_gradients_reach_encoder_head_and_fusion(self):
        model = self.model(max_length=16).train()
        model(**self.batch())["loss"].backward()
        embeddings = model.model.bert.embeddings.word_embeddings.weight.grad
        self.assertGreater(float(embeddings.abs().sum()), 0)
        self.assertGreater(float(model.model.classifier.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.fusion.weight.grad.abs().sum()), 0)

    def test_reads_the_two_texts_as_one_pair(self):
        model = self.model(max_length=16)
        tokenize = model.tokenizer
        with mock.patch.object(model, "tokenizer", wraps=tokenize) as spy:
            model.encode_pairs(["hypertension"], ["essential hypertension"])
        (titles, candidates), kwargs = spy.call_args
        self.assertEqual(titles, ["hypertension"])
        self.assertEqual(candidates, ["essential hypertension"])
        self.assertEqual(kwargs["max_length"], 16)
        ids = tokenize("hypertension", "essential hypertension")
        self.assertEqual(ids["token_type_ids"], [0, 0, 0, 1, 1, 1])
        self.assertEqual(ids["input_ids"].count(tokenize.sep_token_id), 2)

    def test_segment_order_matters(self):
        model = self.model(max_length=16)
        titles = [p[0] for p in PAIRS]
        candidates = [p[1] for p in PAIRS]
        with torch.no_grad():
            forward = model.encode_pairs(titles, candidates)
            swapped = model.encode_pairs(candidates, titles)
        self.assertFalse(torch.allclose(forward, swapped))

    def test_truncates_to_max_length(self):
        model = self.model(max_length=8)
        long = "diabetes " * 30
        with torch.no_grad():
            a = model.encode_pairs(["hypertension"], [long + "kidney"])
            b = model.encode_pairs(["hypertension"], [long + "drugs"])
        torch.testing.assert_close(a, b)

    def test_pair_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.model(max_length=16).encode_pairs(["a", "b"], ["c"])

    def test_untrained_fusion_equals_text_only(self):
        batch = self.batch()
        with torch.no_grad():
            fused = self.model(max_length=16)(**batch)["logit"]
            text = self.model(max_length=16, fusion_keys=())(**batch)["logit"]
        torch.testing.assert_close(fused, text)

    def test_fusion_adds_a_linear_term(self):
        model = self.model(max_length=16, fusion_keys=["score"])
        batch = self.batch()
        with torch.no_grad():
            before = model(**batch)["logit"]
            model.fusion.weight.fill_(2.0)
            after = model(**batch)["logit"]
        scores = torch.tensor([[p[2]] for p in PAIRS])
        torch.testing.assert_close(after - before, 2.0 * scores, atol=1e-5, rtol=0)

    def test_loss_uses_the_label_pos_weight(self):
        dataset = make_dataset(label=("binary", {"pos_weight": 3.0}))
        try:
            torch.manual_seed(0)
            model = RASCCrossEncoder(
                dataset, model_name=_SHARED["bert"], max_length=16
            ).eval()
            out = model(**next(iter(get_dataloader(dataset, batch_size=4))))
        finally:
            dataset.close()
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            out["logit"], out["y_true"], pos_weight=torch.tensor([3.0])
        )
        torch.testing.assert_close(out["loss"], expected)

    def test_one_optimizer_step_lowers_the_loss(self):
        model = self.model(max_length=16)
        batch = self.batch()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        first = model(**batch)["loss"]
        first.backward()
        optimizer.step()
        with torch.no_grad():
            self.assertLess(float(model(**batch)["loss"]), float(first))


class TestRASCCrossEncoderMixedPrecision(unittest.TestCase):
    """mixed_precision is a no-op on CPU and keeps 32-bit outputs."""

    def test_cpu_scores_match_full_precision(self):
        dataset = make_dataset()
        torch.manual_seed(0)
        plain = RASCCrossEncoder(dataset, model_name=_SHARED["bert"], max_length=16)
        mixed = RASCCrossEncoder(
            dataset, model_name=_SHARED["bert"], max_length=16, mixed_precision=True
        )
        mixed.load_state_dict(plain.state_dict())
        plain.eval()
        mixed.eval()
        with torch.no_grad():
            a = plain.encode_pairs(["alpha beta"], ["gamma"])
            b = mixed.encode_pairs(["alpha beta"], ["gamma"])
        self.assertEqual(b.dtype, torch.float32)
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
