"""Deep Cox Mixtures (DCM) model for survival regression.

Reference:
    Nagpal, C., Yadlowsky, S., Rostamzadeh, N., & Heller, K. (2021).
    Deep Cox Mixtures for Survival Regression.
    Proceedings of the 6th Machine Learning for Healthcare Conference,
    PMLR 149:674-708.
    https://proceedings.mlr.press/v149/nagpal21a.html
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class DeepCoxMixturesLayer(nn.Module):
    """Core neural network layer for the Deep Cox Mixtures model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_mixtures: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_mixtures = num_mixtures

        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.Tanh())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.gate = nn.Linear(hidden_dim, num_mixtures)
        self.hazard_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_mixtures)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the DCM layer."""
        h = self.encoder(x)
        gate_probs = F.softmax(self.gate(h), dim=-1)
        log_hazards = torch.cat([head(h) for head in self.hazard_heads], dim=-1)
        return gate_probs, log_hazards


class DeepCoxMixtures(BaseModel):
    """Deep Cox Mixtures (DCM) model for survival regression (Nagpal et al. 2021).
 
    Instead of one global Cox model, DCM learns K latent patient subgroups and
    fits a separate hazard per subgroup. A gating network assigns each patient
    soft mixture weights, and the final risk score is their weighted combination."""

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_mixtures: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.num_layers = num_layers
        self.dropout = dropout

        assert len(self.label_keys) == 1, (
            "DeepCoxMixtures expects exactly one label key."
        )
        self.label_key = self.label_keys[0]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        total_input_dim = embedding_dim * len(self.feature_keys)
        self.dcm_layer = DeepCoxMixturesLayer(
            input_dim=total_input_dim,
            hidden_dim=hidden_dim,
            num_mixtures=num_mixtures,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _embed_features(self, kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed all input features and concatenate into a single vector."""
        feature_vecs: List[torch.Tensor] = []

        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]

            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            processor = self.dataset.input_processors[feature_key]
            schema = processor.schema()

            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' has no 'value' in its schema."
                )

            value = value.to(self.device)

            if value.dim() == 3:
                b, s, inner = value.shape
                value = value.view(b, s * inner)
                if mask is not None and mask.dim() == 3:
                    mask = mask.view(b, s * inner)

            if mask is not None:
                mask = mask.to(self.device)
                embedded = self.embedding_model(
                    {feature_key: value}, masks={feature_key: mask}
                )[feature_key]
            else:
                embedded = self.embedding_model({feature_key: value})[feature_key]

            if embedded.dim() == 3:
                embedded = embedded.mean(dim=1)

            feature_vecs.append(embedded)

        return torch.cat(feature_vecs, dim=-1)

    @staticmethod
    def _cox_mixture_loss(
        gate_probs: torch.Tensor,
        log_hazards: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture-weighted MSE loss over implied log-hazard targets."""
        eps = 1e-7
        y_true = y_true.view(-1).clamp(eps, 1.0 - eps)
        log_h_target = torch.log(-torch.log(y_true))
        sq_err = (log_hazards - log_h_target.unsqueeze(1)) ** 2
        return (gate_probs * sq_err).sum(dim=-1).mean()

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for the Deep Cox Mixtures model."""
        patient_emb = self._embed_features(kwargs)
        gate_probs, log_hazards = self.dcm_layer(patient_emb)

        logit = (gate_probs * log_hazards).sum(dim=-1, keepdim=True)
        y_prob = torch.exp(-torch.exp(logit)).clamp(1e-7, 1.0 - 1e-7)

        results: Dict[str, torch.Tensor] = {"logit": logit, "y_prob": y_prob}

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self._cox_mixture_loss(gate_probs, log_hazards, y_true)
            results["y_true"] = y_true.view(-1, 1)

        return results