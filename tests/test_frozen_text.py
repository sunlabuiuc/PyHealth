"""Proofs for frozen-text encoder behaviour in UnifiedMultimodalEmbeddingModel."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
from pyhealth.processors.stagenet_processor import StageNetTensorProcessor


def _numeric_model(**kwargs) -> UnifiedMultimodalEmbeddingModel:
    proc = StageNetTensorProcessor()
    proc.fit([{"labs": ([0.0], [[1.0] * 10])}], "labs")
    return UnifiedMultimodalEmbeddingModel(
        {"labs": proc}, embedding_dim=8, freeze_text_encoder=True, **kwargs
    )


class TinyEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.9)
        self.lin = nn.Linear(1, 8)
        self.config = type("C", (), {"hidden_size": 8})()

    def forward(self, input_ids, attention_mask=None):
        b, l = input_ids.shape
        h = self.drop(torch.ones(b, l, 8))
        return type("O", (), {"last_hidden_state": h})()


class TestFrozenEncoderEval(unittest.TestCase):
    def test_train_keeps_frozen_text_encoder_in_eval(self):
        model = _numeric_model()
        enc = TinyEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(enc.training)
        model.eval()
        self.assertFalse(enc.training)
        model.train()
        self.assertFalse(enc.training)


class CountingEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.lin = nn.Linear(1, 8)
        self.config = type("C", (), {"hidden_size": 8})()

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        b, l = input_ids.shape
        scale = input_ids[:, :1].float()
        h = torch.ones(b, l, 8) * scale
        return type("O", (), {"last_hidden_state": h})()


class TestFrozenTextCache(unittest.TestCase):
    def test_cache_keys_ignore_padding_tokens(self):
        model = _numeric_model(cache_frozen_text=True)
        enc = CountingEnc()
        model.encoders["notes"] = enc
        model._frozen_text_fields.add("notes")

        ids_a = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 9, 9]])
        mask_a = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        h1 = model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        h2 = model._encode_text_cls("notes", enc, ids_a, mask_a)
        self.assertEqual(enc.calls, 1)
        ids_b = torch.tensor([[1, 2, 3, 7, 7, 7]])
        mask_b = torch.tensor([[1, 1, 1, 0, 0, 0]])
        h3 = model._encode_text_cls("notes", enc, ids_b, mask_b)
        self.assertEqual(enc.calls, 1)
        self.assertEqual(h1.shape[0], 2)
        self.assertEqual(h3.shape[0], 1)


if __name__ == "__main__":
    unittest.main()

