"""Field-embedding reuse and patch image encoder for the unified model."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from pyhealth.processors import (
    StageNetProcessor,
    StageNetTensorProcessor,
    TimeImageProcessor,
    TupleTimeTextProcessor,
)
from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel


def _code_processor():
    proc = StageNetProcessor()
    proc.fit([{"codes": ([0.0], ["A", "B"])}], "codes")
    return proc


class TestFieldEmbeddingReuse(unittest.TestCase):
    def test_code_encoder_reuses_prebuilt_layer(self):
        proc = _code_processor()
        layer = nn.Embedding(proc.value_dim(), 8, padding_idx=0)
        pre = SimpleNamespace(
            embedding_layers=nn.ModuleDict({"codes": layer}), embedding_dim=8
        )
        model = UnifiedMultimodalEmbeddingModel(
            {"codes": proc}, embedding_dim=8, field_embeddings={"codes": pre}
        )
        self.assertIs(model.encoders["codes"], layer)

    def test_code_reuse_projects_when_dims_differ(self):
        proc = _code_processor()
        layer = nn.Embedding(proc.value_dim(), 4, padding_idx=0)
        pre = SimpleNamespace(
            embedding_layers=nn.ModuleDict({"codes": layer}), embedding_dim=4
        )
        model = UnifiedMultimodalEmbeddingModel(
            {"codes": proc}, embedding_dim=8, field_embeddings={"codes": pre}
        )
        out = model(
            {
                "codes": {
                    "value": torch.zeros(1, 2, dtype=torch.long),
                    "time": torch.zeros(1, 2),
                }
            }
        )
        self.assertEqual(out["sequence"].shape, (1, 2, 8))

    def test_vision_reuse_wraps_backbone_with_mean_pool(self):
        proc = TimeImageProcessor(image_size=16)
        backbone = nn.Identity()
        pre = SimpleNamespace(
            embedding_layers=nn.ModuleDict({"cxr": backbone}), embedding_dim=8
        )
        model = UnifiedMultimodalEmbeddingModel(
            {"cxr": proc}, embedding_dim=8, field_embeddings={"cxr": pre}
        )
        encoder = model.encoders["cxr"]
        self.assertIsInstance(encoder, nn.Sequential)
        self.assertIs(encoder[0], backbone)

    def test_text_freeze_marks_field_frozen(self):
        proc = TupleTimeTextProcessor()
        transformer = nn.Linear(4, 4)
        pre = SimpleNamespace(transformer=transformer, fc=nn.Linear(4, 8))
        model = UnifiedMultimodalEmbeddingModel(
            {"notes": proc},
            embedding_dim=8,
            field_embeddings={"notes": pre},
            freeze_text_encoder=True,
        )
        self.assertIn("notes", model._frozen_text_fields)
        self.assertTrue(
            all(not p.requires_grad for p in transformer.parameters())
        )

    def test_default_image_encoder_is_patch_plus_mean_pool(self):
        proc = TimeImageProcessor(image_size=16)
        model = UnifiedMultimodalEmbeddingModel(
            {"cxr": proc}, embedding_dim=8, image_size=16, patch_size=8
        )
        out = model(
            {
                "cxr": {
                    "value": torch.zeros(1, 2, 3, 16, 16),
                    "time": torch.zeros(1, 2),
                }
            }
        )
        self.assertEqual(out["sequence"].shape, (1, 2, 8))

    def test_unknown_image_pool_raises(self):
        proc = TimeImageProcessor(image_size=16)
        with self.assertRaises(NotImplementedError):
            UnifiedMultimodalEmbeddingModel(
                {"cxr": proc}, embedding_dim=8, image_pool="max"
            )


if __name__ == "__main__":
    unittest.main()
