"""Personalized discrete-time survival MoE (Morrill et al., ML4H).

Paper: https://proceedings.mlr.press/v297/morrill26a.html
Reference: https://github.com/ToddMorrill/survival-moe.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class PersonalizedSurvivalMoE(BaseModel):
    """Mix patient-specific MTLR probability mass functions in log space.

    Args:
        dataset: Optional PyHealth dataset with dense, preprocessed features.
        input_dim: Number of continuous input features.
        num_time_bins: Number of categories, including the final overflow bin.
        hidden_dim: Encoder width.
        num_experts: Number of personalized experts.
        dropout: Dropout before each encoder linear layer.
        num_hidden_layers: Number of encoder layers.
        categorical_cardinalities: Vocabulary sizes for optional categorical inputs.
        embedding_dim: Width of each categorical embedding.

    The encoder uses disjoint expert value subspaces. Routing
    temperature is learned from 2.0; load-balancing regularization weight is 0.01.
    ``hidden_dim`` must be divisible by ``num_experts``.

    Labels use zero-based categories. An event contributes -log p[category];
    censoring contributes -log sum(p[category:]), matching the reference's
    inclusive-bin convention. The final category represents later/no event.
    ``survival[:, j]`` is P(category > j), for finite categories only.
    Fit preprocessing and time boundaries on training patients only.

    Survival is not a standard classification mode: use a custom training loop
    and censoring-aware metrics, as in the accompanying SUPPORT2 example.

    Examples:
        >>> model = PersonalizedSurvivalMoE(None, input_dim=3, num_time_bins=5)
        >>> model(features=torch.zeros(2, 3))["survival"].shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        dataset: SampleDataset | None,
        input_dim: int,
        num_time_bins: int,
        hidden_dim: int = 128,
        num_experts: int = 8,
        dropout: float = 0.0,
        num_hidden_layers: int = 1,
        categorical_cardinalities: tuple[int, ...] = (),
        embedding_dim: int = 4,
    ) -> None:
        super().__init__(dataset)
        for name, value in {
            "input_dim": input_dim,
            "num_time_bins": num_time_bins,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "num_hidden_layers": num_hidden_layers,
            "embedding_dim": embedding_dim,
            **{f"cardinality_{i}": n for i, n in enumerate(categorical_cardinalities)},
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_time_bins < 2:
            raise ValueError("num_time_bins must include a finite and overflow bin")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if hidden_dim % num_experts:
            raise ValueError("hidden_dim must be divisible by num_experts")
        self.mode = None
        self.input_dim, self.num_time_bins = input_dim, num_time_bins
        self.num_experts = num_experts
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, embedding_dim) for n in categorical_cardinalities]
        )
        encoder_dim = input_dim + len(self.embeddings) * embedding_dim
        layers = []
        for i in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(encoder_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.Parameter(torch.empty(num_experts, hidden_dim))
        nn.init.kaiming_uniform_(self.keys, a=math.sqrt(5))
        self.value = nn.Linear(hidden_dim, hidden_dim)
        width = hidden_dim // num_experts
        self.experts = nn.ModuleList(
            [nn.Linear(width, num_time_bins) for _ in range(num_experts)]
        )
        self.raw_temperature = nn.Parameter(
            torch.tensor(2.0 + math.log(-math.expm1(-2.0)))
        )

    @staticmethod
    def negative_log_likelihood(
        log_pmf: Tensor, time_bin: Tensor, event: Tensor
    ) -> Tensor:
        """Return mean event/censoring NLL with inclusive censor categories.

        Args:
            log_pmf: Normalized log masses of shape (batch, categories).
            time_bin: Integer categories, shape (batch,) or (batch, 1).
            event: Binary observed-event flags, same number of rows.

        Returns:
            Scalar mean negative log likelihood.

        Raises:
            ValueError: For invalid shapes, categories or event flags.
        """
        if log_pmf.ndim != 2 or not log_pmf.shape[0]:
            raise ValueError("log_pmf must be a nonempty matrix")
        n, bins = log_pmf.shape
        time_bin = torch.as_tensor(time_bin, device=log_pmf.device)
        event = torch.as_tensor(event, device=log_pmf.device)
        if time_bin.shape not in [(n,), (n, 1)] or event.shape not in [(n,), (n, 1)]:
            raise ValueError("labels must have shape (batch,) or (batch, 1)")
        time_bin, event = time_bin.reshape(-1), event.reshape(-1)
        if not torch.isfinite(time_bin).all() or not torch.all(
            time_bin == time_bin.long()
        ):
            raise ValueError("time_bin must contain finite integers")
        if torch.any((time_bin < 0) | (time_bin >= bins)):
            raise ValueError("time_bin is outside the modeled categories")
        if not torch.all((event == 0) | (event == 1)):
            raise ValueError("event must contain only 0 or 1")
        idx = time_bin.long().unsqueeze(1)
        log_tail = log_pmf.flip(-1).logcumsumexp(-1).flip(-1)
        log_likelihood = torch.where(
            event.bool(),
            log_pmf.gather(1, idx).squeeze(1),
            log_tail.gather(1, idx).squeeze(1),
        )
        return -log_likelihood.mean()

    def forward(self, **kwargs: Any) -> dict[str, Tensor]:
        """Predict PMFs, survival and routing weights; optionally compute loss.

        Args:
            **kwargs: Dense features (batch, input_dim), optional paired
                time_bin/event labels, integer categories (batch, number of
                embeddings), and ignored sample metadata.

        Returns:
            logit (log mixture PMF), pmf, survival/y_prob, expert_pmf,
            mixture_weights, and, when labeled, loss, nll and y_true.
        """
        x = torch.as_tensor(
            kwargs["features"], device=self.device, dtype=self.query.weight.dtype
        )
        if x.ndim != 2 or x.shape[1] != self.input_dim or not x.shape[0]:
            raise ValueError("features must have shape (nonempty batch, input_dim)")
        if not torch.isfinite(x).all():
            raise ValueError("features must be finite; impute missing values first")
        if self.embeddings:
            categories = torch.as_tensor(kwargs["categories"], device=self.device)
            if categories.shape != (len(x), len(self.embeddings)):
                raise ValueError("categories must have one column per embedding")
            if not torch.isfinite(categories).all() or not torch.all(
                categories == categories.long()
            ):
                raise ValueError("categories must contain finite integers")
            embedded = []
            for i, embedding in enumerate(self.embeddings):
                col = categories[:, i].long()
                if torch.any((col < 0) | (col >= embedding.num_embeddings)):
                    raise ValueError("category is outside its vocabulary")
                embedded.append(embedding(col))
            x = torch.cat([x, *embedded], dim=-1)
        h = self.encoder(x)
        router_logits = self.query(h) @ self.keys.T
        temp = F.softplus(self.raw_temperature).clamp_min(torch.finfo(h.dtype).eps)
        log_weights = F.log_softmax(router_logits / temp, dim=-1)
        values = self.value(h)
        chunks = values.chunk(self.num_experts, dim=-1)
        increments = torch.stack([head(v) for head, v in zip(self.experts, chunks)], 1)
        scores = increments.flip(-1).cumsum(-1).flip(-1)
        log_experts = F.log_softmax(scores, dim=-1)
        log_pmf = torch.logsumexp(log_weights.unsqueeze(-1) + log_experts, dim=1)
        survival = log_pmf.flip(-1).logcumsumexp(-1).flip(-1)[:, 1:].exp().clamp(max=1)
        result = {
            "logit": log_pmf,
            "pmf": log_pmf.exp(),
            "survival": survival,
            "y_prob": survival,
            "expert_pmf": log_experts.exp(),
            "mixture_weights": log_weights.exp(),
        }
        labeled = [key in kwargs for key in ("time_bin", "event")]
        if any(labeled) and not all(labeled):
            raise ValueError("time_bin and event must be provided together")
        if all(labeled):
            time = torch.as_tensor(kwargs["time_bin"], device=self.device)
            event = torch.as_tensor(kwargs["event"], device=self.device)
            nll = self.negative_log_likelihood(log_pmf, time, event)
            # Balance untempered routing probabilities, as in the reference.
            avg = router_logits.softmax(-1).mean(0)
            balance = self.num_experts * avg.square().sum()
            result.update(
                nll=nll,
                loss=nll + 0.01 * balance,
                y_true=torch.stack([time.reshape(-1), event.reshape(-1)], -1),
            )
        return result
