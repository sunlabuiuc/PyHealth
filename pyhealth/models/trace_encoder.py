# File: pyhealth/models/trace_encoder.py
# The following model has been added as part of UIUC MCS-DS 598
# deep learning for healthcare.
# netid: dfelker2 UID: 658368229
"""
TRACEEncoder model based on the architecture and methodology presented in:

Horn, Max, et al. "Learning Unsupervised Representations for ICU Time Series." 
Proceedings of the Conference on Health, Inference, and Learning (CHIL). 2020.

Citation:
Horn, Max, et al. "Learning Unsupervised Representations for ICU Time Series." 
Proceedings of the 1st Conference on Health, Inference, and Learning. PMLR, 2020, pp. 100–110.

This PyHealth-compatible encoder is adapted for modular training and evaluation.
Original TRACE codebase: https://github.com/mhorn/TRACE
"""

import torch
import torch.nn as nn
from pyhealth.models import BaseModel

class TRACEEncoder(BaseModel):
    """
    PyHealth-compatible implementation of the TRACE-style encoder for multivariate time-series data.
    
    This encoder uses 1D convolutions and pooling to compress multichannel time-series data into a
    fixed-length latent embedding vector. Designed to support ICU data with shape 
    (batch_size, num_channels, timesteps).

    Parameters:
        input_channels (int): Number of input signal channels (e.g., 10 for ICU vitals).
        latent_dim (int): Dimension of the output embedding vector.
    """

    def __init__(self, input_channels: int = 10, latent_dim: int = 128):
        super().__init__(None, None)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce latent embedding.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, timesteps).

        Returns:
            Tensor: Latent embedding of shape (batch_size, latent_dim).
        """
        features = self.encoder(x)
        features = features.squeeze(-1)
        embedding = self.fc(features)
        return embedding
