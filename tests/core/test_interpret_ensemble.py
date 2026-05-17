"""
Test suite for Ensemble interpreter implementation.
"""
import unittest

import torch
import torch.nn as nn

from pyhealth.models import BaseModel
from pyhealth.interpret.methods.base_ensemble import BaseInterpreterEnsemble
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter

# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestEnsembleFlattenValues(unittest.TestCase):
    """Tests for the _flatten_values method."""

    def test_flatten_single_feature_1d(self):
        """Test flattening a single 1D feature."""
        values = {
            "feature": torch.randn(2, 3),  # (B=2, M=3)
        }
        flattened = BaseInterpreterEnsemble._flatten_attributions(values)

        self.assertEqual(flattened.shape, torch.Size([2, 3]))
        torch.testing.assert_close(flattened, values["feature"])

    def test_flatten_single_feature_2d(self):
        """Test flattening a single 2D feature."""
        values = {
            "feature": torch.randn(2, 4, 5),  # (B=2, 4, 5) -> (B=2, 20)
        }
        flattened = BaseInterpreterEnsemble._flatten_attributions(values)

        self.assertEqual(flattened.shape, torch.Size([2, 20]))
        expected = values["feature"].reshape(2, -1)
        torch.testing.assert_close(flattened, expected)

    def test_flatten_multiple_features(self):
        """Test flattening multiple features with different shapes."""
        values = {
            "feature_a": torch.randn(3, 5),      # (B=3, 5) -> (B=3, 5)
            "feature_b": torch.randn(3, 2, 3),   # (B=3, 2, 3) -> (B=3, 6)
            "feature_c": torch.randn(3, 4),      # (B=3, 4) -> (B=3, 4)
        }
        flattened = BaseInterpreterEnsemble._flatten_attributions(values)

        # Total flattened size = 5 + 6 + 4 = 15
        self.assertEqual(flattened.shape, torch.Size([3, 15]))

    def test_flatten_preserves_batch_size(self):
        """Test that flattened output has correct batch size."""
        for batch_size in [1, 2, 5, 16]:
            values = {
                "feature_a": torch.randn(batch_size, 3),
                "feature_b": torch.randn(batch_size, 2, 4),
            }
            flattened = BaseInterpreterEnsemble._flatten_attributions(values)

            self.assertEqual(flattened.shape[0], batch_size)

    def test_flatten_consistency_with_sorted_keys(self):
        """Test that flattening is consistent with sorted key ordering."""
        values = {
            "zebra": torch.randn(2, 3),
            "apple": torch.randn(2, 4),
            "banana": torch.randn(2, 2),
        }

        flattened = BaseInterpreterEnsemble._flatten_attributions(values)

        # Should be ordered alphabetically: apple (4), banana (2), zebra (3)
        self.assertEqual(flattened.shape, torch.Size([2, 9]))

        # Verify order by checking slices
        apple_slice = flattened[:, :4]
        banana_slice = flattened[:, 4:6]
        zebra_slice = flattened[:, 6:9]

        torch.testing.assert_close(apple_slice, values["apple"].reshape(2, -1))
        torch.testing.assert_close(banana_slice, values["banana"].reshape(2, -1))
        torch.testing.assert_close(zebra_slice, values["zebra"].reshape(2, -1))


class TestEnsembleUnflattenValues(unittest.TestCase):
    """Tests for the _unflatten_values method."""

    def test_unflatten_single_feature_1d(self):
        """Test unflattening a single 1D feature."""
        shapes = {"feature": torch.Size([2, 3])}
        flattened = torch.randn(2, 3)

        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        self.assertIn("feature", unflattened)
        self.assertEqual(unflattened["feature"].shape, torch.Size([2, 3]))
        torch.testing.assert_close(unflattened["feature"], flattened)

    def test_unflatten_single_feature_2d(self):
        """Test unflattening a single 2D feature."""
        original_shape = torch.Size([2, 4, 5])
        shapes = {"feature": original_shape}
        flattened = torch.randn(2, 20)

        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        self.assertEqual(unflattened["feature"].shape, original_shape)

    def test_unflatten_multiple_features(self):
        """Test unflattening multiple features."""
        shapes = {
            "feature_a": torch.Size([3, 5]),
            "feature_b": torch.Size([3, 2, 3]),
            "feature_c": torch.Size([3, 4]),
        }
        flattened = torch.randn(3, 15)

        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        self.assertEqual(len(unflattened), 3)
        self.assertEqual(unflattened["feature_a"].shape, torch.Size([3, 5]))
        self.assertEqual(unflattened["feature_b"].shape, torch.Size([3, 2, 3]))
        self.assertEqual(unflattened["feature_c"].shape, torch.Size([3, 4]))

    def test_unflatten_preserves_batch_size(self):
        """Test that unflattening preserves batch dimension."""
        for batch_size in [1, 2, 5, 16]:
            shapes = {
                "feature_a": torch.Size([batch_size, 3]),
                "feature_b": torch.Size([batch_size, 2, 4]),
            }
            flattened = torch.randn(batch_size, 11)

            unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

            self.assertEqual(unflattened["feature_a"].shape[0], batch_size)
            self.assertEqual(unflattened["feature_b"].shape[0], batch_size)


class TestEnsembleRoundtrip(unittest.TestCase):
    """Tests for flatten/unflatten roundtrip consistency."""

    def test_roundtrip_single_feature(self):
        """Test that flatten->unflatten recovers original single feature."""
        original = {
            "feature": torch.randn(4, 6),
        }

        shapes = {k: v.shape for k, v in original.items()}
        flattened = BaseInterpreterEnsemble._flatten_attributions(original)
        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        torch.testing.assert_close(unflattened["feature"], original["feature"])

    def test_roundtrip_multiple_features(self):
        """Test that flatten->unflatten recovers original with multiple features."""
        original = {
            "feature_a": torch.randn(5, 3),
            "feature_b": torch.randn(5, 2, 4),
            "feature_c": torch.randn(5, 7),
        }

        shapes = {k: v.shape for k, v in original.items()}
        flattened = BaseInterpreterEnsemble._flatten_attributions(original)
        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        for key in original:
            torch.testing.assert_close(unflattened[key], original[key])

    def test_roundtrip_high_dimensional(self):
        """Test roundtrip with high-dimensional tensors."""
        original = {
            "feature_a": torch.randn(2, 3, 4, 5),
            "feature_b": torch.randn(2, 2, 3),
        }

        shapes = {k: v.shape for k, v in original.items()}
        flattened = BaseInterpreterEnsemble._flatten_attributions(original)
        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        for key in original:
            torch.testing.assert_close(unflattened[key], original[key])

    def test_roundtrip_maintains_device(self):
        """Test that roundtrip maintains tensor device."""
        original = {
            "feature_a": torch.randn(2, 3),
            "feature_b": torch.randn(2, 4, 5),
        }

        shapes = {k: v.shape for k, v in original.items()}
        flattened = BaseInterpreterEnsemble._flatten_attributions(original)
        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        for key in original:
            self.assertEqual(unflattened[key].device, original[key].device)

    def test_roundtrip_with_gradients(self):
        """Test that roundtrip works with gradient-tracking tensors."""
        original = {
            "feature_a": torch.randn(2, 3, requires_grad=True),
            "feature_b": torch.randn(2, 4, 5, requires_grad=True),
        }

        shapes = {k: v.shape for k, v in original.items()}
        flattened = BaseInterpreterEnsemble._flatten_attributions(original)
        unflattened = BaseInterpreterEnsemble._unflatten_attributions(flattened, shapes)

        for key in original:
            torch.testing.assert_close(unflattened[key].detach(), original[key].detach())


class TestCompetitiveRankingNormalize(unittest.TestCase):
    """Tests for competitive_ranking_noramlize static method."""

    def test_all_distinct_values(self):
        """When all values are distinct, ranks should be 0..M-1 (normalized)."""
        # (B=1, I=1, M=5) with distinct values
        x = torch.tensor([[[3.0, 1.0, 4.0, 1.5, 2.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)

        # Sorted ascending: 1.0(idx1)=0, 1.5(idx3)=1, 2.0(idx4)=2, 3.0(idx0)=3, 4.0(idx2)=4
        # Normalized by (M-1)=4
        expected = torch.tensor([[[3/4, 0/4, 4/4, 1/4, 2/4]]])
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Output shape must match input shape."""
        for B, I, M in [(2, 3, 5), (1, 1, 10), (4, 2, 7)]:
            x = torch.randn(B, I, M)
            result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
            self.assertEqual(result.shape, x.shape)

    def test_output_range(self):
        """All output values must lie in [0, 1]."""
        x = torch.randn(4, 3, 20)
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

    def test_tied_scores_get_same_rank(self):
        """Tied scores must receive the same (minimum) rank — '1224' ranking."""
        # [1, 2, 2, 4] -> ranks [0, 1, 1, 3], normalized by 3
        x = torch.tensor([[[1.0, 2.0, 2.0, 4.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        expected = torch.tensor([[[0/3, 1/3, 1/3, 3/3]]])
        torch.testing.assert_close(result, expected)

    def test_all_tied(self):
        """When every item has the same score, all ranks should be 0."""
        x = torch.tensor([[[5.0, 5.0, 5.0, 5.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        expected = torch.zeros_like(x)
        torch.testing.assert_close(result, expected)

    def test_single_item(self):
        """M=1 edge case: should return zeros."""
        x = torch.tensor([[[7.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        expected = torch.zeros_like(x)
        torch.testing.assert_close(result, expected)

    def test_two_items_distinct(self):
        """M=2, distinct values."""
        x = torch.tensor([[[3.0, 1.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        # 1.0 -> rank 0, 3.0 -> rank 1; normalized by 1
        expected = torch.tensor([[[1.0, 0.0]]])
        torch.testing.assert_close(result, expected)

    def test_two_items_tied(self):
        """M=2, tied values."""
        x = torch.tensor([[[3.0, 3.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        expected = torch.zeros_like(x)
        torch.testing.assert_close(result, expected)

    def test_multiple_tie_groups(self):
        """Multiple distinct tie groups: [1,1,3,3,5]."""
        x = torch.tensor([[[1.0, 1.0, 3.0, 3.0, 5.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        # ranks: [0, 0, 2, 2, 4], normalized by 4
        expected = torch.tensor([[[0/4, 0/4, 2/4, 2/4, 4/4]]])
        torch.testing.assert_close(result, expected)

    def test_batch_and_expert_independence(self):
        """Each (batch, expert) slice must be ranked independently."""
        x = torch.tensor([
            [[3.0, 1.0, 2.0],   # batch 0, expert 0
             [1.0, 3.0, 2.0]],  # batch 0, expert 1
            [[2.0, 2.0, 1.0],   # batch 1, expert 0
             [5.0, 5.0, 5.0]],  # batch 1, expert 1
        ])  # (B=2, I=2, M=3)

        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)

        # batch0, expert0: [3,1,2] -> ranks [2,0,1] / 2
        torch.testing.assert_close(result[0, 0], torch.tensor([2/2, 0/2, 1/2]))
        # batch0, expert1: [1,3,2] -> ranks [0,2,1] / 2
        torch.testing.assert_close(result[0, 1], torch.tensor([0/2, 2/2, 1/2]))
        # batch1, expert0: [2,2,1] -> ranks [1,1,0] / 2
        torch.testing.assert_close(result[1, 0], torch.tensor([1/2, 1/2, 0/2]))
        # batch1, expert1: [5,5,5] -> ranks [0,0,0] / 2
        torch.testing.assert_close(result[1, 1], torch.tensor([0.0, 0.0, 0.0]))

    def test_negative_values(self):
        """Negative values should be ranked correctly."""
        x = torch.tensor([[[-3.0, -1.0, -2.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        # Ascending: -3(0), -2(1), -1(2); normalized by 2
        expected = torch.tensor([[[0/2, 2/2, 1/2]]])
        torch.testing.assert_close(result, expected)

    def test_already_sorted_ascending(self):
        """Input already sorted ascending."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        expected = torch.tensor([[[0/4, 1/4, 2/4, 3/4, 4/4]]])
        torch.testing.assert_close(result, expected)

    def test_sorted_descending(self):
        """Input sorted descending."""
        x = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        expected = torch.tensor([[[4/4, 3/4, 2/4, 1/4, 0/4]]])
        torch.testing.assert_close(result, expected)

    def test_competitive_not_dense_ranking(self):
        """Verify it's competitive (1224) NOT dense (1223) ranking.

        For [10, 20, 20, 40]: competitive ranks are [0,1,1,3], not [0,1,1,2].
        """
        x = torch.tensor([[[10.0, 20.0, 20.0, 40.0]]])
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        # Competitive: [0, 1, 1, 3] / 3
        expected = torch.tensor([[[0/3, 1/3, 1/3, 3/3]]])
        torch.testing.assert_close(result, expected)
        # Dense would give [0, 1, 1, 2] / 3 — assert that's NOT what we get
        dense = torch.tensor([[[0/3, 1/3, 1/3, 2/3]]])
        self.assertFalse(torch.allclose(result, dense))

    def test_preserves_device(self):
        """Output should be on the same device as input."""
        x = torch.randn(2, 3, 5)
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        self.assertEqual(result.device, x.device)

    def test_dtype_float32(self):
        """Works with float32 input."""
        x = torch.randn(2, 2, 6, dtype=torch.float32)
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        self.assertEqual(result.dtype, torch.float32)

    def test_dtype_float64(self):
        """Works with float64 input."""
        x = torch.randn(2, 2, 6, dtype=torch.float64)
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        self.assertEqual(result.dtype, torch.float64)

    def test_large_tensor(self):
        """Smoke test on a larger tensor."""
        x = torch.randn(8, 5, 100)
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

    def test_max_is_one_min_is_zero_when_all_distinct(self):
        """When all items are distinct, min rank = 0, max rank = 1."""
        x = torch.randn(3, 2, 10)
        result = BaseInterpreterEnsemble._competitive_ranking_normalize(x)
        for b in range(3):
            for i in range(2):
                self.assertAlmostEqual(result[b, i].min().item(), 0.0, places=6)
                self.assertAlmostEqual(result[b, i].max().item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
