# ------------------------------------------------------------------------------
# Author:      Subin Pradeep & Utkarsh Prasad
# NetID:       subinpp2 & uprasad3
# Description: CRNN for EEG Seizure Detection (Conv → GRU → FC)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn


class SeizureCRNN(nn.Module):
    """
    A simple convolutional‑recurrent network for binary seizure detection.
    """

    def __init__(
        self,
        in_channels: int = 19,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super(SeizureCRNN, self).__init__()

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels=19, time_steps)
        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # Conv → (batch, 64, T')
        x = self.encoder(x)

        # Repack for RNN: (batch, T', features)
        x = x.permute(0, 2, 1)

        # GRU → (batch, T', hidden*2)
        out, _ = self.gru(x)

        # Take final time step
        final = out[:, -1, :]

        # Classify
        logits = self.classifier(final)
        return logits
