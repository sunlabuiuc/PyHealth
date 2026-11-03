"""Content norm, mask zeroing, and numeric standardizers for the unified model."""

import unittest

import torch
import torch.nn as nn

from pyhealth.processors import StageNetTensorProcessor
from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel


class _Double(nn.Module):
    """Test standardizer: records its inputs, doubles unobserved-safe values."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value, observed):
        self.calls.append((value.shape, observed.dtype))
        return value * 2.0


def _labs_processor():
    proc = StageNetTensorProcessor()
    proc.fit([{"labs": ([0.0], [[1.0, 2.0]])}], "labs")
    return proc


class TestContentAssembly(unittest.TestCase):
    def test_padded_slots_are_zeroed(self):
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": _labs_processor()}, embedding_dim=8
        )
        out = model(
            {
                "labs": {
                    "value": torch.zeros(1, 2, 2),
                    "time": torch.zeros(1, 2),
                    "mask": torch.tensor([[1.0, 0.0]]),
                }
            }
        )
        self.assertTrue(torch.equal(out["mask"], torch.tensor([[1.0, 0.0]])))
        self.assertTrue(bool((out["sequence"][0, 1] == 0).all()))

    def test_normalize_content_bounds_large_magnitudes(self):
        proc = _labs_processor()
        loud = UnifiedMultimodalEmbeddingModel(
            {"labs": proc}, embedding_dim=8, normalize_content=True
        )
        quiet = UnifiedMultimodalEmbeddingModel(
            {"labs": proc}, embedding_dim=8, normalize_content=False
        )
        quiet.load_state_dict(loud.state_dict())
        big = torch.full((1, 2, 2), 1000.0)
        times = torch.zeros(1, 2)
        seq_norm = loud({"labs": {"value": big, "time": times}})["sequence"]
        seq_raw = quiet({"labs": {"value": big, "time": times}})["sequence"]
        self.assertLess(
            seq_norm.abs().max().item(), seq_raw.abs().max().item() / 100
        )

    def test_standardizer_runs_before_projection(self):
        std = _Double()
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": _labs_processor()},
            embedding_dim=8,
            numeric_standardizers={"labs": std},
        )
        out = model(
            {
                "labs": {
                    "value": torch.ones(1, 2, 2),
                    "time": torch.zeros(1, 2),
                },
                "labs_mask": {
                    "value": torch.ones(1, 2, 2),
                    "time": torch.zeros(1, 2),
                },
            }
        )
        # The mask sibling is not a modality: only the 2 lab events surface.
        self.assertEqual(out["sequence"].shape, (1, 2, 8))
        self.assertEqual(std.calls, [((1, 2, 2), torch.bool)])

    def test_standardizer_without_sibling_raises(self):
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": _labs_processor()},
            embedding_dim=8,
            numeric_standardizers={"labs": _Double()},
        )
        with self.assertRaises(ValueError):
            model(
                {"labs": {"value": torch.ones(1, 1, 2), "time": torch.zeros(1, 1)}}
            )

    def test_standardizer_shape_mismatch_raises(self):
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": _labs_processor()},
            embedding_dim=8,
            numeric_standardizers={"labs": _Double()},
        )
        with self.assertRaises(ValueError):
            model(
                {
                    "labs": {"value": torch.ones(1, 2, 2), "time": torch.zeros(1, 2)},
                    "labs_mask": {
                        "value": torch.ones(1, 3, 2),
                        "time": torch.zeros(1, 3),
                    },
                }
            )


    def test_mask_field_without_standardizer_stays_a_modality(self):
        model = UnifiedMultimodalEmbeddingModel(
            {"labs": _labs_processor(), "labs_mask": _labs_processor()},
            embedding_dim=8,
        )
        out = model(
            {
                "labs": {
                    "value": torch.ones(1, 2, 2),
                    "time": torch.zeros(1, 2),
                },
                "labs_mask": {
                    "value": torch.zeros(1, 1, 2),
                    "time": torch.ones(1, 1),
                },
            }
        )
        # No standardizer registered for "labs": the "labs_mask" field is not
        # consumed by anyone, so both fields encode (2 + 1 events).
        self.assertEqual(out["sequence"].shape, (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
