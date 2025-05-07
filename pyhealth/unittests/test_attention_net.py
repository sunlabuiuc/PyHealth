"""
Nikhil Sarin (nsarin2)
DL4H Final Project Contribution

Test case for AttentionNet model.
This script demonstrates how to use the AttentionNet model for a readmission prediction task.
"""

import unittest
import os
import torch
import numpy as np

# Import the AttentionNet model - adjust the import path as needed
from pyhealth.models.attention_net import AttentionNet, SelfAttentionLayer


class TestAttentionNet(unittest.TestCase):
    """Test case for AttentionNet model."""

    def test_attention_layer(self):
        """Tests the self-attention layer independently."""
        # Parameters
        batch_size = 5
        seq_len = 10
        input_dim = 16
        hidden_dim = 32
        
        # Create random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Initialize attention layer
        attention_layer = SelfAttentionLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=0.1
        )
        
        # Forward pass
        output = attention_layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))


if __name__ == "__main__":
    unittest.main()