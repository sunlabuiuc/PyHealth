"""Zero-slot text/image batches must not hit the ambiguous -1 reshape."""

import unittest

import torch
import torch.nn as nn
from types import SimpleNamespace

from pyhealth.processors import TimeImageProcessor, TupleTimeTextProcessor
from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel


class TinyEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 8})()

    def forward(self, input_ids, attention_mask=None):
        return type(
            "O", (), {"last_hidden_state": torch.zeros(input_ids.shape[0], 4, 8)}
        )()


class TestEmptyBatchReshape(unittest.TestCase):
    def test_empty_note_slots_return_zero_length(self):
        proc = TupleTimeTextProcessor()
        pre = SimpleNamespace(
            transformer=TinyEnc(), fc=nn.Identity(), embedding_dim=8
        )
        model = UnifiedMultimodalEmbeddingModel(
            {"notes": proc}, embedding_dim=8, field_embeddings={"notes": pre}
        )
        out = model(
            {
                "notes": {
                    "value": torch.zeros(1, 0, 4, dtype=torch.long),
                    "time": torch.zeros(1, 0),
                }
            }
        )
        self.assertEqual(out["sequence"].shape, (1, 0, 8))

    def test_empty_image_slots_return_zero_length(self):
        proc = TimeImageProcessor(image_size=16)
        model = UnifiedMultimodalEmbeddingModel(
            {"cxr": proc}, embedding_dim=8, image_size=16, patch_size=8
        )
        out = model(
            {
                "cxr": {
                    "value": torch.zeros(1, 0, 3, 16, 16),
                    "time": torch.zeros(1, 0),
                }
            }
        )
        self.assertEqual(out["sequence"].shape, (1, 0, 8))


if __name__ == "__main__":
    unittest.main()
