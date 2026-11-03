"""Frozen [CLS] cache keyed on real tokens, with an on/off switch."""

from __future__ import annotations

import builtins
import unittest
from unittest import mock

import torch
import torch.nn as nn

from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
from pyhealth.processors.stagenet_processor import StageNetTensorProcessor


def _numeric_model(**kwargs) -> UnifiedMultimodalEmbeddingModel:
    proc = StageNetTensorProcessor()
    proc.fit([{"labs": ([0.0], [[1.0] * 10])}], "labs")
    kwargs.setdefault("cache_frozen_text", True)
    return UnifiedMultimodalEmbeddingModel(
        {"labs": proc}, embedding_dim=8, freeze_text_encoder=True, **kwargs
    )


class CountingEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.lin = nn.Linear(1, 8)
        self.config = type("C", (), {"hidden_size": 8})()

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        b, l = input_ids.shape
        scale = input_ids[:, :1, None].float()
        h = torch.ones(b, l, 8) * scale
        return type("O", (), {"last_hidden_state": h})()


class TestFrozenTextCache(unittest.TestCase):
    def _cached(self, **kwargs):
        model = _numeric_model(**kwargs)
        enc = CountingEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")
        return model, enc

    def test_cache_keys_ignore_padding_tokens(self):
        model, enc = self._cached()
        ids_a = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 9, 9]])
        mask_a = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        h1 = model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        h2 = model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        self.assertTrue(torch.equal(h1, h2))
        # Same real tokens under a wider pad width still hits.
        ids_b = torch.tensor([[1, 2, 3, 7, 7, 7]])
        mask_b = torch.tensor([[1, 1, 1, 0, 0, 0]])
        h3 = model._encode_text_cls("notes", enc, ids_b, mask_b)
        self.assertEqual(enc.calls, 1)
        self.assertEqual(h1.shape[0], 2)
        self.assertEqual(h3.shape[0], 1)
        self.assertTrue(torch.equal(h3[0], h1[0]))

    def test_switch_off_runs_encoder_every_forward(self):
        model, enc = self._cached(cache_frozen_text=False)
        ids_a = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 9, 9]])
        mask_a = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        model._encode_text_cls("notes", enc, ids_a, mask_a)
        model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 2)
        self.assertEqual(model._frozen_text_cache, {})

    def test_unfrozen_field_never_caches(self):
        model = _numeric_model()
        enc = CountingEnc()
        model.encoders["notes"] = enc
        # Deliberately NOT added to _frozen_text_fields: trainable encoder.
        ids_a = torch.tensor([[1, 2, 3]])
        h1 = model._encode_text_cls("notes", enc, ids_a, None)
        h2 = model._encode_text_cls("notes", enc, ids_a, None)
        self.assertEqual(enc.calls, 2)
        self.assertTrue(torch.equal(h1, h2))

    def test_empty_rows_return_zeros_without_encoding(self):
        model, enc = self._cached()
        out = model._encode_text_cls(
            "notes", enc, torch.zeros(0, 4, dtype=torch.long), None
        )
        self.assertEqual(out.shape, (0, 8))
        self.assertEqual(enc.calls, 0)

    def test_hash_collision_does_not_return_wrong_embedding(self):
        # Keys use Python hash(); a collision must not silently return the
        # wrong [CLS]. The key carries the real-token length as a cheap
        # guard, so two sequences with the same hash but different lengths
        # still miss. Simulate a total collision by forcing hash() to 42.
        model, enc = self._cached()
        ids1 = torch.tensor([[1, 2, 3]])
        mask1 = torch.tensor([[1, 1, 1]])
        ids2 = torch.tensor([[9, 8]])
        mask2 = torch.tensor([[1, 1]])
        with mock.patch.object(builtins, "hash", return_value=42):
            h1 = model._encode_text_cls("notes", enc, ids1, mask1)
            h2 = model._encode_text_cls("notes", enc, ids2, mask2)
        # Different first tokens -> different [CLS]; a collision would
        # have returned h1 for both rows.
        self.assertFalse(torch.equal(h1, h2))
        self.assertEqual(
            sum(len(c) for c in model._frozen_text_cache.values()), 2
        )


if __name__ == "__main__":
    unittest.main()
