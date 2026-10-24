"""Unit tests for TransformersModel: input length, long-text windows, pooling.

Uses tiny randomly initialised Hugging Face test models (about 90k
parameters) so the tests run in seconds.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import TransformersModel, transformers_model

BERT = "hf-internal-testing/tiny-random-bert"  # has a pooler
DISTILBERT = "hf-internal-testing/tiny-random-DistilBertModel"  # no pooler

SHORT = "chest pain"
LONG = " ".join(f"word{i}" for i in range(200))  # several 16-token windows


def _make_dataset(label_schema="binary"):
    labels = {"binary": [1, 0, 1], "multiclass": ["a", "b", "c"]}[label_schema]
    samples = [
        {"patient_id": f"p{i}", "text": text, "label": label}
        for i, (text, label) in enumerate(zip([SHORT, LONG, "fever"], labels))
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"text": "text"},
        output_schema={"label": label_schema},
        dataset_name="test_transformers",
        in_memory=True,
    )


class TestTransformersModelBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def model(self, **kwargs):
        torch.manual_seed(0)
        model = TransformersModel(self.dataset, model_name=BERT, **kwargs)
        return model.eval()

    def windows(self, model, text):
        """Number of windows the tokenizer makes for one text."""
        batch = model.tokenizer(
            [text],
            truncation=True,
            max_length=model.max_length,
            return_overflowing_tokens=True,
        )
        return len(batch["input_ids"])


class TestMaxLength(TestTransformersModelBase):
    def test_default_is_256(self):
        self.assertEqual(self.model().max_length, 256)

    def test_none_is_the_encoder_limit(self):
        self.assertEqual(self.model(max_length=None).max_length, 512)

    def test_default_fits_a_shorter_encoder(self):
        with mock.patch.object(TransformersModel, "_encoder_max_length",
                               return_value=128):
            self.assertEqual(self.model().max_length, 128)
            with self.assertRaises(ValueError):
                self.model(max_length=200)

    def test_encoder_limit_respects_the_tokenizer(self):
        model = self.model()
        model.tokenizer.model_max_length = 100  # shorter than the 512 positions
        self.assertEqual(model._encoder_max_length(), 100)
        model.tokenizer.model_max_length = int(1e30)  # no limit configured
        self.assertEqual(model._encoder_max_length(), 512)

    def test_custom_max_length(self):
        self.assertEqual(self.model(max_length=128).max_length, 128)

    def test_rejects_lengths_the_encoder_cannot_read(self):
        with self.assertRaises(ValueError):
            self.model(max_length=1024)
        with self.assertRaises(ValueError):
            self.model(max_length=2)

    def test_truncation_reads_only_the_first_window(self):
        model = self.model(max_length=16)
        cut = LONG.split()
        # Only the first 16 tokens are read, so changing later words is ignored.
        changed = " ".join(cut[:1] + ["different"] * (len(cut) - 1))
        with torch.no_grad():
            a, b = model.encode([LONG, changed])
        self.assertFalse(torch.allclose(a, b))
        tail = " ".join(cut[:50] + ["different"] * (len(cut) - 50))
        with torch.no_grad():
            a, b = model.encode([LONG, tail])
        torch.testing.assert_close(a, b)


class TestChunking(TestTransformersModelBase):
    def test_rejects_bad_max_chunks(self):
        with self.assertRaises(ValueError):
            self.model(max_chunks=0)

    def test_windows_need_a_fast_tokenizer(self):
        load = transformers_model.AutoTokenizer.from_pretrained
        with mock.patch.object(
            transformers_model.AutoTokenizer, "from_pretrained",
            side_effect=lambda name, **kwargs: load(name, use_fast=False),
        ):
            self.assertFalse(self.model().tokenizer.is_fast)  # one window: fine
            with self.assertRaisesRegex(ValueError, "fast tokenizer"):
                self.model(max_chunks=None)

    def test_long_text_is_split_into_windows(self):
        model = self.model(max_length=16, max_chunks=None)
        self.assertGreater(self.windows(model, LONG), 1)
        self.assertEqual(self.windows(model, SHORT), 1)

    def encoded_rows(self, model, texts):
        """Rows (windows) the encoder receives when encoding `texts`."""
        rows = []
        hook = model.model.register_forward_pre_hook(
            lambda module, args, kwargs: rows.append(kwargs["input_ids"].shape[0]),
            with_kwargs=True,
        )
        try:
            with torch.no_grad():
                model.encode(texts)
        finally:
            hook.remove()
        return rows[0]

    def test_every_window_is_encoded(self):
        model = self.model(max_length=16, max_chunks=None)
        expected = self.windows(model, LONG) + self.windows(model, SHORT)
        self.assertEqual(self.encoded_rows(model, [LONG, SHORT]), expected)

    def test_text_vector_is_max_over_window_vectors(self):
        model = self.model(max_length=16, max_chunks=None, pooling="cls")
        batch = model.tokenizer(
            [LONG],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,
            return_overflowing_tokens=True,
        )
        batch.pop("overflow_to_sample_mapping")
        with torch.no_grad():
            per_window = model.model(**batch).last_hidden_state[:, 0]
            pooled = model.encode([LONG])
        torch.testing.assert_close(pooled[0], per_window.max(dim=0).values)

    def test_texts_in_a_batch_do_not_mix(self):
        model = self.model(max_length=16, max_chunks=None)
        with torch.no_grad():
            together = model.encode([SHORT, LONG, "fever"])
            alone = torch.cat([model.encode([t]) for t in (SHORT, LONG, "fever")])
        self.assertEqual(together.shape, (3, model.model.config.hidden_size))
        torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-4)

    def test_max_chunks_caps_the_windows(self):
        model = self.model(max_length=16, max_chunks=2)
        self.assertEqual(self.encoded_rows(model, [LONG, SHORT]), 3)
        words = LONG.split()
        # Words after the second window are never read.
        tail = " ".join(words[:-20] + ["different"] * 20)
        with torch.no_grad():
            a, b = model.encode([LONG, tail])
        torch.testing.assert_close(a, b)


class TestPooling(TestTransformersModelBase):
    def test_rejects_unknown_pooling(self):
        with self.assertRaises(ValueError):
            self.model(pooling="max")

    def test_mean_pooling_ignores_padding(self):
        model = self.model(pooling="mean")
        with torch.no_grad():
            alone = model.encode([SHORT])
            padded = model.encode([SHORT, LONG])[:1]
        torch.testing.assert_close(alone, padded, atol=1e-5, rtol=1e-4)

    def test_auto_pooling_matches_the_previous_implementation(self):
        # Earlier releases tokenized with max_length=256 and used pooler_output.
        model = self.model()
        texts = [SHORT, LONG]
        batch = model.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        with torch.no_grad():
            expected = model.model(**batch).pooler_output
            torch.testing.assert_close(model.encode(texts), expected)

    def test_encoder_without_pooler(self):
        torch.manual_seed(0)
        model = TransformersModel(self.dataset, model_name=DISTILBERT).eval()
        with torch.no_grad():
            out = model(text=[SHORT, LONG], label=torch.tensor([[1.0], [0.0]]))
        self.assertEqual(out["y_prob"].shape, (2, 1))


class TestDropout(TestTransformersModelBase):
    def assert_dropout(self, model, settings, p):
        for name in settings:
            self.assertEqual(getattr(model.model.config, name), p)
        layers = [m for m in model.model.modules() if isinstance(m, nn.Dropout)]
        self.assertTrue(layers)
        self.assertEqual({m.p for m in layers}, {p})

    def test_bert_dropout_settings(self):
        model = self.model(dropout=0.3)
        self.assert_dropout(
            model, ("hidden_dropout_prob", "attention_probs_dropout_prob"), 0.3
        )
        self.assertEqual(model.dropout.p, 0.3)

    def test_distilbert_dropout_settings(self):
        model = TransformersModel(self.dataset, model_name=DISTILBERT, dropout=0.3)
        self.assert_dropout(model, ("dropout", "attention_dropout"), 0.3)


class TestForward(TestTransformersModelBase):
    def test_binary_forward_and_backward(self):
        model = self.model(max_length=16, max_chunks=None)
        model.train()
        out = model(text=[SHORT, LONG], label=torch.tensor([[1.0], [0.0]]))
        for key in ("loss", "y_prob", "y_true"):
            self.assertIn(key, out)
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertIsNotNone(model.fc.weight.grad)

    def test_multiclass_forward(self):
        dataset = _make_dataset("multiclass")
        try:
            torch.manual_seed(0)
            model = TransformersModel(
                dataset, model_name=BERT, max_length=16, max_chunks=None
            )
            out = model(text=[SHORT, LONG], label=torch.tensor([0, 2]))
            self.assertEqual(out["y_prob"].shape, (2, 3))
            torch.testing.assert_close(out["y_prob"].sum(dim=1), torch.ones(2))
        finally:
            dataset.close()


if __name__ == "__main__":
    unittest.main()
