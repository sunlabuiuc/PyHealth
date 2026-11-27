"""Unit tests for QueryFocusedEHRModel.

Uses a small deterministic tokenizer and encoder to test the
model without downloading ClinicalBERT or clinical datasets.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from pyhealth.models.query_focused_ehr import QueryFocusedEHRModel

class TinyTokenizer:
    """Deterministic tokenizer compatible with model inputs."""

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ):
        tokens = []

        for text in texts:
            words = text.lower().split()

            # Reserve zero for padding.
            ids = [
                (sum(word.encode("utf-8")) % 127) + 1
                for word in words
            ]

            ids = ids[:max_length] or [1]
            tokens.append(ids)

        length = max(len(ids) for ids in tokens)

        padded = [
            ids + [0] * (length - len(ids))
            for ids in tokens
        ]

        mask = [
            [1] * len(ids) + [0] * (length - len(ids))
            for ids in tokens
        ]

        return {
            "input_ids": torch.tensor(
                padded, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                mask, dtype=torch.long
            ),
        }


class TinyEncoder(nn.Module):
    """Small trainable stand-in for ClinicalBERT."""

    def __init__(self, hidden_size=16):
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=hidden_size
        )

        self.embedding = nn.Embedding(
            num_embeddings=256,
            embedding_dim=hidden_size,
            padding_idx=0,
        )

        self.projection = nn.Linear(
            hidden_size, hidden_size
        )

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            embeddings = embeddings * mask

            pooled = embeddings.sum(dim=1) / (
                mask.sum(dim=1).clamp(min=1)
            )
        else:
            pooled = embeddings.mean(dim=1)

        # Create a contextual first-token representation.
        contextual = self.projection(pooled)

        hidden_states = embeddings.clone()
        hidden_states[:, 0, :] = contextual

        return SimpleNamespace(
            last_hidden_state=hidden_states
        )


class TestQueryFocusedEHRModel(unittest.TestCase):
    """Tests forward passes, learning, and sentence attention."""

    def setUp(self):
        torch.manual_seed(42)

        self.sentences = [
            [
                "History of DVT.",
                "Reports chest pain.",
                "Recent immobilization.",
            ],
            [
                "Patient reports fever.",
                "Cough for three days.",
            ],
        ]

        self.queries = [
            "Pulmonary embolism",
            "Pneumonia",
        ]

        self.labels = torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        )

    def make_model(
        self, freeze_encoder=True, cache_embeddings=False,
        cache_max_size=50000,
    ):
        return QueryFocusedEHRModel(
            dataset=None,
            hidden_dim=16,
            freeze_encoder=freeze_encoder,
            cache_embeddings=cache_embeddings,
            cache_max_size=cache_max_size,
            max_token_length=32,
            encoder_batch_size=2,
            dropout=0.0,
            encoder=TinyEncoder(hidden_size=16),
            tokenizer=TinyTokenizer(),
        )

    def test_forward_output_shapes(self):
        model = self.make_model()

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
            label=self.labels,
        )

        self.assertEqual(
            tuple(outputs["logit"].shape),
            (2, 1),
        )
        self.assertEqual(
            tuple(outputs["y_prob"].shape),
            (2, 1),
        )
        self.assertEqual(
            tuple(outputs["attention"].shape),
            (2, 3),
        )
        self.assertEqual(
            tuple(outputs["sentence_mask"].shape),
            (2, 3),
        )
        self.assertEqual(
            outputs["loss"].ndim,
            0,
        )

    def test_probabilities_are_valid(self):
        model = self.make_model()

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
        )

        probabilities = outputs["y_prob"]

        self.assertTrue(
            torch.isfinite(probabilities).all().item()
        )
        self.assertTrue(
            (probabilities >= 0).all().item()
        )
        self.assertTrue(
            (probabilities <= 1).all().item()
        )

    def test_attention_normalization(self):
        model = self.make_model()

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
        )

        attention = outputs["attention"]

        sums = attention.sum(dim=1)

        torch.testing.assert_close(
            sums,
            torch.ones_like(sums),
        )

        # The second patient has only two sentences.
        self.assertEqual(
            attention[1, 2].item(),
            0.0,
        )

    def test_binary_cross_entropy_loss(self):
        model = self.make_model()

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
            label=self.labels,
        )

        expected_loss = nn.BCEWithLogitsLoss()(
            outputs["logit"],
            self.labels.reshape(-1, 1),
        )

        torch.testing.assert_close(
            outputs["loss"],
            expected_loss,
        )

        self.assertEqual(
            tuple(outputs["y_true"].shape),
            (2, 1),
        )

    def test_frozen_encoder_gradients(self):
        model = self.make_model(
            freeze_encoder=True
        )

        model.train()

        self.assertFalse(model.encoder.training)

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
            label=self.labels,
        )

        outputs["loss"].backward()

        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.encoder.parameters()
            )
        )

        self.assertIsNotNone(
            model.sentence_projection.weight.grad
        )
        self.assertIsNotNone(
            model.query_projection.weight.grad
        )
        self.assertIsNotNone(
            model.classifier[0].weight.grad
        )

    def test_trainable_encoder_gradients(self):
        model = self.make_model(
            freeze_encoder=False
        )

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
            label=self.labels,
        )

        outputs["loss"].backward()

        encoder_grads = [
            parameter.grad
            for parameter in model.encoder.parameters()
        ]

        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all().item()
                and gradient.abs().sum().item() > 0
                for gradient in encoder_grads
            )
        )

    def test_inference_without_labels(self):
        model = self.make_model()

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
        )

        self.assertNotIn("loss", outputs)
        self.assertNotIn("y_true", outputs)
        self.assertIn("y_prob", outputs)
        self.assertIn("attention", outputs)

    def test_top_k_sentence_ranking(self):
        model = self.make_model()

        summaries = model.extract_top_k(
            sentences=self.sentences,
            query=self.queries,
            k=2,
        )

        self.assertEqual(len(summaries), 2)

        for index, summary in enumerate(summaries):
            self.assertEqual(len(summary), 2)

            scores = [
                item["score"]
                for item in summary
            ]

            self.assertEqual(
                scores,
                sorted(scores, reverse=True),
            )

            for item in summary:
                self.assertIn(
                    item["sentence"],
                    self.sentences[index],
                )
                self.assertGreaterEqual(
                    item["score"],
                    0.0,
                )

    def test_single_example(self):
        model = self.make_model()

        outputs = model(
            sentences=self.sentences[0],
            query=self.queries[0],
            label=torch.tensor([1.0]),
        )

        self.assertEqual(
            tuple(outputs["logit"].shape),
            (1, 1),
        )

        self.assertEqual(
            tuple(outputs["attention"].shape),
            (1, 3),
        )

    def test_invalid_inputs(self):
        model = self.make_model()

        invalid_inputs = [
            {
                "sentences": [],
                "query": [],
            },
            {
                "sentences": [[]],
                "query": ["Pulmonary embolism"],
            },
            {
                "sentences": [
                    ["Clinical sentence."]
                ],
                "query": [
                    "Pulmonary embolism",
                    "Pneumonia",
                ],
            },
            {
                "sentences": [
                    ["Clinical sentence.", ""]
                ],
                "query": ["Pulmonary embolism"],
            },
        ]

        for inputs in invalid_inputs:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    model(**inputs)

        with self.assertRaises(ValueError):
            model.extract_top_k(
                self.sentences,
                self.queries,
                k=0,
            )

    def test_training_step(self):
        """One optimizer step should update trainable weights."""
        model = self.make_model(
            freeze_encoder=False
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

        before = (
            model.classifier[0].weight
            .detach().clone()
        )

        outputs = model(
            sentences=self.sentences,
            query=self.queries,
            label=self.labels,
        )

        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()

        after = (
            model.classifier[0].weight
            .detach().clone()
        )

        self.assertFalse(
            torch.equal(before, after)
        )


    def test_cached_uncached_equivalence(self):
        """A populated frozen-encoder cache must not change predictions."""
        model = self.make_model(cache_embeddings=True)
        model.eval()

        inputs = {"sentences": self.sentences, "query": self.queries}
        with torch.no_grad():
            first = model(**inputs)
            second = model(**inputs)
            model.clear_embedding_cache()
            model.cache_embeddings = False
            uncached = model(**inputs)

        for key in ("logit", "y_prob", "attention"):
            torch.testing.assert_close(first[key], second[key])
            torch.testing.assert_close(first[key], uncached[key])

    def test_cache_hits_and_unique_texts(self):
        """Only unique inputs are encoded; later calls reuse them."""
        model = self.make_model(cache_embeddings=True)
        calls = []
        handle = model.encoder.register_forward_hook(
            lambda module, inputs, output: calls.append(1)
        )
        try:
            texts = ["Repeated sentence.", "Repeated sentence.",
                     "Another sentence."]
            first = model._encode_texts(texts)
            count_after_first = len(calls)
            self.assertEqual(count_after_first, 1)  # 2 unique texts
            self.assertEqual(model.embedding_cache_info()["size"], 2)
            self.assertEqual(model.embedding_cache_info()["misses"], 2)

            second = model._encode_texts(texts)
            self.assertEqual(len(calls), count_after_first)
            self.assertEqual(model.embedding_cache_info()["hits"], 2)
            torch.testing.assert_close(first, second)
            torch.testing.assert_close(first[0], first[1])
        finally:
            handle.remove()

    def test_cache_eviction_lru(self):
        """The least recently accessed embedding is evicted."""
        model = self.make_model(cache_embeddings=True, cache_max_size=2)
        model._encode_texts(["Alpha.", "Beta."])
        model._encode_texts(["Alpha."])  # Refresh Alpha.
        model._encode_texts(["Gamma."])  # Evict Beta.

        info = model.embedding_cache_info()
        self.assertEqual(info["size"], 2)
        self.assertEqual(info["max_size"], 2)
        self.assertEqual(set(model._embedding_cache), {"Alpha.", "Gamma."})

    def test_clear_embedding_cache(self):
        """Clearing removes entries and resets hit/miss counters."""
        model = self.make_model(cache_embeddings=True)
        model._encode_texts(["Clinical text."])
        model._encode_texts(["Clinical text."])
        self.assertGreater(model.embedding_cache_info()["hits"], 0)

        model.clear_embedding_cache()
        info = model.embedding_cache_info()
        self.assertEqual(info["size"], 0)
        self.assertEqual(info["hits"], 0)
        self.assertEqual(info["misses"], 0)

    def test_cache_gradient_flow(self):
        """Cached encoder outputs must not detach trainable projections."""
        model = self.make_model(cache_embeddings=True)
        model.train()
        inputs = {
            "sentences": self.sentences,
            "query": self.queries,
            "label": self.labels,
        }
        model(**inputs)  # Fill cache first.
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)  # Use cached embeddings.
        outputs["loss"].backward()

        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        for layer in (model.sentence_projection, model.query_projection,
                      model.classifier[0]):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all().item())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_cache_rejected_when_finetuning(self):
        """Fine-tuning and detached embedding caching cannot be combined."""
        with self.assertRaises(ValueError):
            self.make_model(freeze_encoder=False, cache_embeddings=True)

    def test_cache_invalidated_by_model_conversion(self):
        """Module conversions must clear potentially stale embeddings."""
        model = self.make_model(cache_embeddings=True)
        model._encode_texts(["Clinical text."])
        self.assertEqual(model.embedding_cache_info()["size"], 1)
        model.to("cpu")
        self.assertEqual(model.embedding_cache_info()["size"], 0)

    def test_cache_invalidated_by_checkpoint_load(self):
        """Restoring encoder weights must discard prior embeddings."""
        model = self.make_model(cache_embeddings=True)
        model._encode_texts(["Clinical text."])
        self.assertEqual(model.embedding_cache_info()["size"], 1)
        state = model.state_dict()
        model.load_state_dict(state)
        self.assertEqual(model.embedding_cache_info()["size"], 0)

if __name__ == "__main__":
    unittest.main()
