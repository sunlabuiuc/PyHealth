"""Discrete-time survival mixture-of-experts models for SUPPORT2.

Author: Shuxian Xu (NetID: sx30).
Paper: `Let the Experts Speak: Improving Survival Prediction & Calibration
via Mixture-of-Experts Heads <https://proceedings.mlr.press/v297/morrill26a.html>`_.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel

ExpertType = Literal["personalized", "fixed", "mtlr"]


class SurvivalMoE(nn.Module):
    """Mixture-of-experts model for discrete-time survival prediction.

    Predicts a probability mass function over event-time bins and supports
    right-censored observations through an MTLR-style likelihood.

    Args:
        input_dim: Number of input features.
        num_time_bins: Number of discrete event-time bins; must be at least 2.
        hidden_dim: Width of each backbone layer.
        num_layers: Number of linear/ReLU backbone layers.
        num_experts: Number of experts for ``personalized`` and ``fixed`` models.
        expert_type: ``"personalized"``, ``"fixed"``, or the single-head
            ``"mtlr"`` baseline.
        top_k: Optional number of experts retained per sample. ``None`` uses
            dense routing.
        load_balance_weight: Weight of the routing load-balancing penalty.

    Example:
        >>> import torch
        >>> model = SurvivalMoE(input_dim=4, num_time_bins=3, hidden_dim=8,
        ...                     num_experts=2)
        >>> output = model(torch.zeros(2, 4))
        >>> tuple(output["pmf"].shape)
        (2, 3)
    """

    def __init__(
        self,
        input_dim: int,
        num_time_bins: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_experts: int = 8,
        expert_type: ExpertType = "personalized",
        top_k: int | None = None,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_time_bins < 2:
            raise ValueError("input_dim must be positive and num_time_bins must be >= 2")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if expert_type not in {"personalized", "fixed", "mtlr"}:
            raise ValueError(f"Unsupported expert_type: {expert_type}")
        if expert_type != "mtlr" and num_experts < 2:
            raise ValueError("MoE models require at least two experts")
        if expert_type == "personalized" and hidden_dim % num_experts:
            raise ValueError("hidden_dim must be divisible by num_experts")
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if load_balance_weight < 0:
            raise ValueError("load_balance_weight cannot be negative")

        self.expert_type = expert_type
        self.num_time_bins = num_time_bins
        self.num_experts = 1 if expert_type == "mtlr" else num_experts
        self.top_k = None if expert_type == "mtlr" else top_k
        self.load_balance_weight = load_balance_weight
        backbone_layers: list[nn.Module] = []
        layer_input_dim = input_dim
        for _ in range(num_layers):
            backbone_layers.extend(
                [nn.Linear(layer_input_dim, hidden_dim), nn.ReLU()]
            )
            layer_input_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)

        if expert_type == "personalized":
            chunk_size = hidden_dim // num_experts
            self.router_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)
            self.expert_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.expert_heads = nn.ModuleList(
                nn.Linear(chunk_size, num_time_bins)
                for _ in range(num_experts)
            )
            self.log_temperature = nn.Parameter(torch.tensor(2.0).log())
        elif expert_type == "fixed":
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)
            self.expert_logits = nn.Parameter(
                torch.empty(num_experts, num_time_bins)
            )
            nn.init.normal_(self.expert_logits, mean=0.0, std=0.02)
            self.log_temperature = nn.Parameter(torch.tensor(2.0).log())
        else:
            self.mtlr_head = nn.Linear(hidden_dim, num_time_bins)

    def _routing_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.expert_type == "personalized":
            router_hidden = self.router_projection(hidden)
            logits = self.router(router_hidden)
        else:
            logits = self.router(hidden)

        temperature = self.log_temperature.exp().clamp(min=0.05, max=20.0)
        logits = logits / temperature
        dense_routing = torch.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < self.num_experts:
            selected = torch.topk(logits, self.top_k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=selected, value=True
            )
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            sparse_routing = torch.softmax(logits, dim=-1)
            return sparse_routing + dense_routing - dense_routing.detach()
        return dense_routing

    def _expert_distributions(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.expert_type == "personalized":
            expert_hidden = self.expert_projection(hidden)
            chunks = expert_hidden.chunk(self.num_experts, dim=-1)
            logits = torch.stack(
                [
                    head(chunk)
                    for head, chunk in zip(self.expert_heads, chunks)
                ],
                dim=1,
            )
        elif self.expert_type == "fixed":
            logits = self.expert_logits.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        else:
            increment_logits = self.mtlr_head(hidden)
            logits = torch.flip(
                torch.cumsum(
                    torch.flip(increment_logits, dims=(-1,)),
                    dim=-1,
                ),
                dims=(-1,),
            ).unsqueeze(1)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        duration_bin: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict event-time probabilities and, optionally, training loss.

        Args:
            features: Dense feature matrix with shape ``(batch, input_dim)``.
            duration_bin: Optional integer bin indices with shape ``(batch,)``.
                Required together with ``event`` to compute the loss.
            event: Optional observed-event indicators with shape ``(batch,)``;
                ``True`` denotes an observed event and ``False`` denotes
                right-censoring.

        Returns:
            A dictionary containing ``pmf`` and ``survival`` with shape
            ``(batch, num_time_bins)``, and ``routing`` with shape
            ``(batch, num_experts)`` (one column for MTLR). If both target
            tensors are supplied, it also contains scalar ``loss``, ``nll``,
            and ``load_balance`` tensors.

        Raises:
            ValueError: If only one target is supplied, batch dimensions differ,
                or a duration-bin index is outside the configured time grid.
        """
        hidden = self.backbone(features.float())
        if self.expert_type == "mtlr":
            routing = torch.ones(
                (features.shape[0], 1),
                device=features.device,
                dtype=hidden.dtype,
            )
        else:
            routing = self._routing_weights(hidden)

        expert_pmf = self._expert_distributions(hidden)
        pmf = torch.sum(routing.unsqueeze(-1) * expert_pmf, dim=1)
        survival = torch.flip(
            torch.cumsum(torch.flip(pmf, dims=(-1,)), dim=-1),
            dims=(-1,),
        )
        output = {
            "pmf": pmf,
            "survival": survival,
            "routing": routing,
        }

        if duration_bin is not None or event is not None:
            if duration_bin is None or event is None:
                raise ValueError("duration_bin and event must be provided together")
            if duration_bin.ndim != 1 or event.ndim != 1:
                raise ValueError("duration_bin and event must be one-dimensional")
            if duration_bin.shape[0] != features.shape[0] or event.shape[0] != features.shape[0]:
                raise ValueError("features and survival targets must have equal batch sizes")
            if torch.any(duration_bin < 0) or torch.any(duration_bin >= self.num_time_bins):
                raise ValueError("duration_bin contains an index outside the time grid")

            event = event.bool()
            event_mass = pmf.gather(1, duration_bin.long().unsqueeze(1)).squeeze(1)
            survival_mass = survival.gather(
                1, duration_bin.long().unsqueeze(1)
            ).squeeze(1)
            likelihood = torch.where(event, event_mass, survival_mass)
            nll = -torch.log(likelihood.clamp_min(1e-8)).mean()

            if self.expert_type == "mtlr":
                load_balance = nll.new_zeros(())
            else:
                mean_routing = routing.mean(dim=0)
                load_balance = self.num_experts * mean_routing.square().sum()
            loss = nll + self.load_balance_weight * load_balance
            output.update(
                {
                    "loss": loss,
                    "nll": nll,
                    "load_balance": load_balance,
                }
            )
        return output


class SurvivalMoEModel(BaseModel):
    """PyHealth BaseModel adapter for dense features and discrete survival labels.

    The dataset must define a dense tensor feature and tensor outputs for the
    duration-bin and event labels. Fit the time grid on the training split and
    pass it to the model; do not fit preprocessing or the grid on held-out data.

    Args:
        dataset: PyHealth sample dataset defining the feature and label keys.
        input_dim: Number of dense input features.
        num_time_bins: Number of bins in the training-fitted time grid.
        hidden_dim: Width of each backbone layer.
        num_layers: Number of linear/ReLU backbone layers.
        num_experts: Number of experts for ``personalized`` and ``fixed``.
        expert_type: ``"personalized"``, ``"fixed"``, or ``"mtlr"``.
        top_k: Optional number of experts retained per sample; ``None`` uses
            dense routing.
        load_balance_weight: Weight of the load-balancing loss term.
        feature_key: Dataset input key. If omitted, the dataset must have one
            feature key.
        duration_key: Dataset output key containing integer time-bin indices.
        event_key: Dataset output key containing event indicators (1 observed,
            0 right-censored).
        time_grid: Optional strictly increasing training-fitted bin values.

    The forward output contains ``y_prob`` (survival probabilities by bin),
    ``routing``, and, when labels are present, ``y_true`` with duration-bin and
    event columns. The core model also returns ``pmf``, ``survival``, and the
    loss components during training.

    Example:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(
        ...     samples=[{"features": [0.0, 1.0], "duration_bin": 0, "event": 1}],
        ...     input_schema={"features": "tensor"},
        ...     output_schema={
        ...         "duration_bin": ("tensor", {"dtype": torch.int64}),
        ...         "event": ("tensor", {"dtype": torch.int64}),
        ...     },
        ...     in_memory=True,
        ... )
        >>> model = SurvivalMoEModel(
        ...     dataset=dataset, input_dim=2, num_time_bins=2, hidden_dim=4,
        ...     num_experts=2, time_grid=[1.0, 2.0],
        ... )
        >>> output = model(
        ...     features=torch.tensor([[0.0, 1.0]]),
        ...     duration_bin=torch.tensor([0]),
        ...     event=torch.tensor([1]),
        ... )
        >>> tuple(output["y_prob"].shape)
        (1, 2)

    Raises:
        ValueError: If the dataset schema is incompatible or the time grid is
            not finite, strictly increasing, and the configured length.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_dim: int,
        num_time_bins: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_experts: int = 8,
        expert_type: ExpertType = "personalized",
        top_k: int | None = None,
        load_balance_weight: float = 0.01,
        feature_key: str | None = None,
        duration_key: str = "duration_bin",
        event_key: str = "event",
        time_grid: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__(dataset)
        if not self.feature_keys:
            raise ValueError("The PyHealth dataset must define a feature input.")
        if feature_key is None:
            if len(self.feature_keys) != 1:
                raise ValueError(
                    "SurvivalMoEModel requires exactly one dense input; "
                    "set feature_key explicitly if the dataset has multiple inputs."
                )
            feature_key = self.feature_keys[0]
        if feature_key not in self.feature_keys:
            raise ValueError(f"Unknown feature_key {feature_key!r}.")
        for label_key in (duration_key, event_key):
            if label_key not in self.label_keys:
                raise ValueError(
                    f"Dataset output schema must include {label_key!r}."
                )

        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        if time_grid is None:
            grid = None
        else:
            grid = torch.as_tensor(time_grid, dtype=torch.float32)
            if grid.ndim != 1 or grid.numel() != num_time_bins:
                raise ValueError("time_grid must contain one value per time bin.")
            if not torch.isfinite(grid).all() or torch.any(torch.diff(grid) <= 0):
                raise ValueError("time_grid values must be finite and strictly increasing.")
        self.register_buffer("time_grid", grid)
        self.core = SurvivalMoE(
            input_dim=input_dim,
            num_time_bins=num_time_bins,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_experts=num_experts,
            expert_type=expert_type,
            top_k=top_k,
            load_balance_weight=load_balance_weight,
        )

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Run a PyHealth batch through the survival model.

        Args:
            **kwargs: Collated dataset batch containing the configured feature
                key and, optionally, both configured survival-label keys.

        Returns:
            Model outputs including ``y_prob`` and ``routing``. When labels are
            provided, also returns ``y_true`` and the training loss.

        Raises:
            TypeError: If the dense feature or survival labels are not tensors.
            ValueError: If required batch keys or paired labels are missing.
        """
        if self.feature_key not in kwargs:
            raise ValueError(f"Missing feature batch {self.feature_key!r}.")
        raw_features = kwargs[self.feature_key]
        features = raw_features[0] if isinstance(raw_features, tuple) else raw_features
        if not isinstance(features, torch.Tensor):
            raise TypeError("The dense feature batch must be a torch.Tensor.")
        features = features.to(self.device, dtype=torch.float32)

        duration = kwargs.get(self.duration_key)
        event = kwargs.get(self.event_key)
        if (duration is None) != (event is None):
            raise ValueError("Duration-bin and event labels must be supplied together.")
        if duration is None:
            output = self.core(features)
            output["y_prob"] = output["survival"]
            return output
        if not isinstance(duration, torch.Tensor) or not isinstance(event, torch.Tensor):
            raise TypeError("Survival labels must be torch.Tensor batches.")
        duration_batch = duration.to(self.device).reshape(-1).long()
        event_batch = event.to(self.device).reshape(-1)
        output = self.core(features, duration_batch, event_batch)
        output["y_prob"] = output["survival"]
        output["y_true"] = torch.stack(
            (duration_batch, event_batch.long()),
            dim=1,
        )
        return output
