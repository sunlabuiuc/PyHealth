"""
Author: Elizabeth Binkina
Paper: Feature Robustness in Non-stationary Health Records:
Caveats to Deployable Model Performance in Common Clinical Machine Learning Tasks
Paper link: https://proceedings.mlr.press/v106/nestor19a.html
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class TemporalFusionMLP(BaseModel):
    """Multilayer perceptron for temporal mortality prediction."""

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        mode: str = "binary",
    ) -> None:
        super().__init__(
            dataset=dataset
        )
        
        self.feature_keys=feature_keys
        self.label_key = label_key
        self.mode = mode

        self.input_dim = 4

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _build_feature_tensor(self, **kwargs) -> torch.Tensor:
        conditions = kwargs["conditions"]
        procedures = kwargs["procedures"]
        drugs = kwargs["drugs"]
        admission_year = kwargs["admission_year"]

        cond_counts = torch.tensor([len(x) for x in conditions], dtype=torch.float32, device=self.device)
        proc_counts = torch.tensor([len(x) for x in procedures], dtype=torch.float32, device=self.device)
        drug_counts = torch.tensor([len(x) for x in drugs], dtype=torch.float32, device=self.device)
        year_values = torch.tensor(
            [x[0] if isinstance(x, list) else float(x) for x in admission_year],
            dtype=torch.float32,
            device=self.device,
        )

        return torch.stack([cond_counts, proc_counts, drug_counts, year_values], dim=1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = self._build_feature_tensor(**kwargs)
        logit = self.net(x).squeeze(-1)

        # convert labels manually
        y_true = torch.tensor(
            kwargs[self.label_key],
            dtype=torch.float32,
            device=logit.device,
        )

        # compute loss manually
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logit, y_true)

        # compute probabilities manually
        y_prob = torch.sigmoid(logit)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }