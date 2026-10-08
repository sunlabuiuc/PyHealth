"""Tests that RNNLayer and ConCare handle zero-length sequences without crashing.

When code_mapping collapses vocabularies, some patients may have all codes
map to <unk>, producing all-zero embeddings and all-zero masks. These tests
verify that the layers handle this edge case gracefully instead of crashing
with IndexError (pack_padded_sequence) or ZeroDivisionError (covariance).
"""

import unittest
import torch

from pyhealth.models.rnn import RNNLayer
from pyhealth.models.concare import ConCareLayer, MultiHeadedAttention


class TestRNNLayerZeroLengthGuard(unittest.TestCase):
    """RNNLayer should not crash when mask contains all-zero rows."""

    def setUp(self):
        torch.manual_seed(42)
        self.input_dim = 4
        self.hidden_dim = 4
        self.batch_size = 2
        self.seq_len = 5

    def test_gru_all_zero_mask_single_sample(self):
        """GRU should handle a batch where one sample has an all-zero mask."""
        layer = RNNLayer(self.input_dim, self.hidden_dim, rnn_type="GRU")
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.int)
        mask[1, :] = 0

        outputs, last_outputs = layer(x, mask)

        self.assertEqual(outputs.shape[0], self.batch_size)
        self.assertEqual(last_outputs.shape, (self.batch_size, self.hidden_dim))

    def test_gru_all_zero_mask_entire_batch(self):
        """GRU should handle a batch where ALL samples have all-zero masks."""
        layer = RNNLayer(self.input_dim, self.hidden_dim, rnn_type="GRU")
        x = torch.zeros(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.int)

        outputs, last_outputs = layer(x, mask)

        self.assertEqual(last_outputs.shape, (self.batch_size, self.hidden_dim))

    def test_lstm_all_zero_mask(self):
        """LSTM should handle all-zero masks the same as GRU."""
        layer = RNNLayer(self.input_dim, self.hidden_dim, rnn_type="LSTM")
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.int)
        mask[0, :] = 0

        outputs, last_outputs = layer(x, mask)

        self.assertEqual(last_outputs.shape, (self.batch_size, self.hidden_dim))

    def test_normal_mask_unchanged(self):
        """Normal (non-zero) masks should produce the same results as before."""
        layer = RNNLayer(self.input_dim, self.hidden_dim, rnn_type="GRU")
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.int)
        mask[0, :3] = 1
        mask[1, :5] = 1

        outputs, last_outputs = layer(x, mask)

        self.assertEqual(last_outputs.shape, (self.batch_size, self.hidden_dim))


class TestConCareCovarianceGuard(unittest.TestCase):
    """ConCare covariance should not divide by zero on single-element inputs."""

    def setUp(self):
        torch.manual_seed(42)

    def test_cov_single_feature(self):
        """Covariance with x.size(1)==1 should not raise ZeroDivisionError."""
        attn = MultiHeadedAttention(1, 4, 0.0)
        m = torch.randn(2, 1)

        cov = attn.cov(m)

        self.assertEqual(cov.shape, (2, 2))

    def test_cov_normal_input(self):
        """Covariance with normal inputs should still work correctly."""
        attn = MultiHeadedAttention(1, 4, 0.0)
        m = torch.randn(2, 4)

        cov = attn.cov(m)

        self.assertEqual(cov.shape, (2, 2))


class TestConCareLayerZeroMask(unittest.TestCase):
    """ConCareLayer should handle all-zero masks without crashing."""

    def setUp(self):
        torch.manual_seed(42)

    def test_all_zero_mask_single_sample(self):
        """ConCareLayer should not crash when one sample has an all-zero mask."""
        input_dim = 4
        hidden_dim = 4
        batch_size = 2
        seq_len = 5

        layer = ConCareLayer(input_dim=input_dim, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.int)
        mask[1, :] = 0

        out, decov_loss = layer(x, mask=mask)

        self.assertEqual(out.shape, (batch_size, hidden_dim))


if __name__ == "__main__":
    unittest.main()
