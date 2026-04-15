"""
Author:Elizabeth Binkina
Paper: Feature Robustness in Non-stationary Health Records:
Caveats to Deployable Model Performance in Common Clinical Machine Learning Tasks
Paper link: https://proceedings.mlr.press/v106/nestor19a.html

Description:
Custom TemporalFusionMLP model for admission-level in-hospital mortality prediction.
This model fuses sparse clinical code features with an optional temporal feature
(admission year) using a small multilayer perceptron.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class TemporalFusionMLP(BaseModel):
    """Multilayer perceptron for temporal mortality prediction.

    Args:
        input_dim: Dimensionality of the input feature vector.
        hidden_dim: Hidden layer size.
        dropout: Dropout probability.
        **kwargs: Additional keyword arguments forwarded to BaseModel.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass.

        Args:
            x: Tensor of shape [batch_size, input_dim].

        Returns:
            Tensor of shape [batch_size] containing logits.
        """
        return self.net(x).squeeze(1)
