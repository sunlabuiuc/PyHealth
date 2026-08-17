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


if __name__ == "__main__":
    unittest.main()
