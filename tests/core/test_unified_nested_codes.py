"""Nested CODE flattening and content-embedding output for the unified model."""

import unittest
import warnings

import torch

from pyhealth.processors import StageNetProcessor
from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel


class TestNestedCodes(unittest.TestCase):
    def _model(self):
        proc = StageNetProcessor()
        proc.fit([{"codes": ([0.0], [["A", "B"]])}], "codes")
        return UnifiedMultimodalEmbeddingModel({"codes": proc}, embedding_dim=8)

    def test_nested_codes_flatten_with_time_expansion(self):
        model = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model(
                {
                    "codes": {
                        "value": torch.zeros(1, 2, 3, dtype=torch.long),
                        "time": torch.tensor([[0.0, 5.0]]),
                    }
                }
            )
        self.assertEqual(out["sequence"].shape, (1, 6, 8))
        self.assertEqual(out["time"].shape, (1, 6))
        # Each visit time repeats along its codes after the time sort.
        self.assertTrue(
            torch.equal(
                out["time"], torch.tensor([[0.0, 0.0, 0.0, 5.0, 5.0, 5.0]])
            )
        )

    def test_warns_once_about_flattening(self):
        model = self._model()
        inputs = {
            "codes": {
                "value": torch.zeros(1, 1, 2, dtype=torch.long),
                "time": torch.zeros(1, 1),
            }
        }
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            model(inputs)
            model(inputs)
        flatten = [w for w in seen if "Flattening" in str(w.message)]
        self.assertEqual(len(flatten), 1)

    def test_token_emb_is_content_before_time_and_type(self):
        model = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model(
                {
                    "codes": {
                        "value": torch.zeros(1, 2, dtype=torch.long),
                        "time": torch.zeros(1, 2),
                    }
                }
            )
        self.assertEqual(out["token_emb"].shape, out["sequence"].shape)
        # token_emb differs from sequence: time/type terms are added on top.
        self.assertFalse(torch.equal(out["token_emb"], out["sequence"]))

    def test_unsupported_code_rank_raises(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model(
                {
                    "codes": {
                        "value": torch.zeros(1, 1, 1, 1, dtype=torch.long),
                        "time": torch.zeros(1, 1),
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
