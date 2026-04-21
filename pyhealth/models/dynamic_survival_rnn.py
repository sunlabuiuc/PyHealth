from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from pyhealth.models import BaseModel


def sigmoid_inv(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class DynamicSurvivalRNN(BaseModel):
    """A simple discrete-time survival RNN for early event prediction."""

    def __init__(
        self,
        dataset,
        feature_key: str = "x",
        label_key: str = "event_within_h",
        hazard_label_key: str = "hazard_y",
        hazard_mask_key: str = "hazard_mask",
        hidden_dim: int = 64,
        horizon: int = 168,
        rnn_type: str = "GRU",
        dropout: float = 0.0,
        bias_init_prevalence: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(dataset=dataset)
        self.feature_key = feature_key
        self.label_key = label_key
        self.hazard_label_key = hazard_label_key
        self.hazard_mask_key = hazard_mask_key
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.rnn_type = rnn_type
        self.dropout = dropout

        sample = self.dataset[0]
        x0 = torch.as_tensor(sample[self.feature_key], dtype=torch.float32)
        if x0.ndim != 2:
            raise ValueError(
                f"Expected {self.feature_key} to have shape (time, features), "
                f"got {tuple(x0.shape)}"
            )
        input_dim = int(x0.shape[-1])

        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, horizon)

        if bias_init_prevalence is not None:
            self.initialize_bias_from_prevalence(bias_init_prevalence)

    def initialize_bias_from_prevalence(self, prevalence: np.ndarray) -> None:
        if prevalence.shape[0] != self.horizon:
            raise ValueError(
                f"Expected prevalence shape ({self.horizon},), got {prevalence.shape}"
            )
        with torch.no_grad():
            self.head.bias.copy_(
                torch.tensor(sigmoid_inv(prevalence), dtype=torch.float32)
            )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = kwargs[self.feature_key].to(self.device).float()
        hazard_y = kwargs[self.hazard_label_key].to(self.device).float()
        hazard_mask = kwargs[self.hazard_mask_key].to(self.device).float()
        y_true = kwargs[self.label_key].to(self.device).float().view(-1, 1)

        outputs = self.rnn(x)
        if self.rnn_type == "LSTM":
            _, (h_n, _) = outputs
        else:
            _, h_n = outputs

        last_hidden = self.dropout_layer(h_n[-1])
        hazard_logit = self.head(last_hidden)
        hazard = torch.sigmoid(hazard_logit)
        cdf = 1.0 - torch.cumprod(1.0 - hazard, dim=1)
        y_prob = cdf[:, -1].unsqueeze(-1)

        bce = nn.functional.binary_cross_entropy_with_logits(
            hazard_logit,
            hazard_y,
            reduction="none",
        )
        loss = (bce * hazard_mask).sum() / hazard_mask.sum().clamp_min(1.0)

        logit = torch.logit(y_prob.clamp(1e-6, 1.0 - 1e-6))

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
            "hazard": hazard,
            "cdf": cdf,
        }
