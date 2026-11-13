"""Neural Fine-Gray model for continuous-time competing risks.

This module implements the cumulative-incidence model proposed by Jeanselme
et al. (CHIL 2023).  The implementation follows the published equations and
PyHealth's model interface; it does not copy the authors' training API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor

__all__ = ["NeuralFineGray", "PositiveLinear"]

ActivationName = Literal["relu", "tanh", "softplus"]
WeightTransform = Literal["square", "softplus", "exp"]
CDFMode = Literal["nfg", "sumonet"]


def _activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError("activation must be 'relu', 'tanh', or 'softplus'")


def _value_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Extract the primary tensor emitted by a PyHealth processor."""
    return value[0] if isinstance(value, tuple) else value


def _infer_input_dim(dataset: SampleDataset, feature_key: str) -> int:
    try:
        value = dataset[0][feature_key]
    except Exception as exc:  # pragma: no cover - defensive streaming fallback
        raise ValueError(
            "Could not infer input_dim from the dataset; pass it explicitly."
        ) from exc
    tensor = torch.as_tensor(_value_tensor(value))
    if tensor.ndim != 1:
        raise ValueError("NeuralFineGray expects a flat numeric feature vector")
    return int(tensor.shape[0])


class PositiveLinear(nn.Module):
    """Linear layer whose effective weights are strictly non-negative.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to learn an unconstrained bias.
        weight_transform: Positive transform applied to the raw weights.
            ``"square"`` matches the paper. ``"softplus"`` and ``"exp"``
            support the monotonicity ablation proposed with this contribution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_transform: WeightTransform = "square",
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if weight_transform not in {"square", "softplus", "exp"}:
            raise ValueError("weight_transform must be 'square', 'softplus', or 'exp'")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_transform = weight_transform
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def weight(self) -> Tensor:
        """Return the transformed non-negative weight matrix."""
        if self.weight_transform == "square":
            return self.raw_weight.square()
        if self.weight_transform == "softplus":
            return nn.functional.softplus(self.raw_weight)
        return self.raw_weight.exp()

    def reset_parameters(self) -> None:
        """Initialize the effective weights with Xavier magnitudes."""
        target = torch.empty_like(self.raw_weight)
        nn.init.xavier_uniform_(target)
        target = target.abs().clamp_min(1e-4)
        with torch.no_grad():
            if self.weight_transform == "square":
                self.raw_weight.copy_(target.sqrt())
            elif self.weight_transform == "softplus":
                self.raw_weight.copy_(torch.log(torch.expm1(target)))
            else:
                self.raw_weight.copy_(target.log())
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the positive linear transformation."""
        return nn.functional.linear(inputs, self.weight, self.bias)


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: ActivationName,
    dropout: float = 0.0,
    activate_output: bool = False,
) -> nn.Sequential:
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    widths = [input_dim, *hidden_dims, output_dim]
    if any(width <= 0 for width in widths):
        raise ValueError("all network dimensions must be positive")
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(pairwise(widths)):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2 or activate_output:
            layers.append(_activation(activation))
            if dropout and index < len(widths) - 2:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _MonotonicNetwork(nn.Module):
    """Smooth positive network that is non-decreasing in every input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["tanh", "softplus"],
        weight_transform: WeightTransform,
    ) -> None:
        super().__init__()
        if activation not in {"tanh", "softplus"}:
            raise ValueError("monotonic_activation must be 'tanh' or 'softplus'")
        widths = [input_dim, *hidden_dims, 1]
        if any(width <= 0 for width in widths):
            raise ValueError("all monotonic network dimensions must be positive")
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(pairwise(widths)):
            layers.append(
                PositiveLinear(
                    left,
                    right,
                    bias=True,
                    weight_transform=weight_transform,
                )
            )
            layers.append(
                nn.Softplus()
                if index == len(widths) - 2
                else _activation(activation)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).squeeze(-1)


class _NeuralFineGrayNetwork(nn.Module):
    """Embedding, balancing, and cause-specific monotonic networks."""

    def __init__(
        self,
        input_dim: int,
        num_risks: int,
        hidden_dims: Sequence[int],
        monotonic_hidden_dims: Sequence[int],
        activation: ActivationName,
        monotonic_activation: Literal["tanh", "softplus"],
        dropout: float,
        weight_transform: WeightTransform,
    ) -> None:
        super().__init__()
        self.embedding = _mlp(
            input_dim,
            hidden_dims,
            input_dim,
            activation,
            dropout,
            activate_output=True,
        )
        self.balance = _mlp(
            input_dim,
            hidden_dims,
            num_risks,
            activation,
            activate_output=True,
        )
        self.outcomes = nn.ModuleList(
            [
                _MonotonicNetwork(
                    input_dim + 1,
                    monotonic_hidden_dims,
                    monotonic_activation,
                    weight_transform,
                )
                for _ in range(num_risks)
            ]
        )

    def encode(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.embedding(features)
        balance_logits = self.balance(embedding)
        log_balance = torch.log_softmax(balance_logits, dim=-1)
        return embedding, balance_logits, log_balance

    def cumulative_hazard(
        self,
        embedding: Tensor,
        time: Tensor,
        cdf_mode: CDFMode,
    ) -> Tensor:
        inputs = torch.cat([embedding, time.unsqueeze(-1)], dim=-1)
        monotonic = torch.stack([network(inputs) for network in self.outcomes], dim=-1)
        return time.unsqueeze(-1) * monotonic if cdf_mode == "nfg" else monotonic


class NeuralFineGray(BaseModel):
    """Monotonic neural cumulative-incidence model for competing risks.

    The model factorizes the cumulative incidence of cause ``r`` as

    ``F_r(t | x) = B_r(E(x)) * (1 - exp(-H_r(t, E(x))))``.

    In the paper configuration, ``H_r(t, x) = t * M_r(t, x)``. ``B`` is a
    softmax balancing network, so the CIFs cannot sum above one, and each
    ``M_r`` is a smooth positive monotonic network. Event densities are
    obtained by automatic differentiation, allowing exact likelihood training
    without time discretization or numerical quadrature.

    Args:
        dataset: PyHealth sample dataset with dense features, duration, and
            integer cause labels. Cause 0 denotes censoring; causes 1 through
            ``num_risks`` denote observed competing events.
        num_risks: Number of observed event causes.
        feature_key: Input field containing a flat numeric feature vector.
        duration_key: Output field containing a strictly positive duration.
        event_key: Output field containing the integer event cause.
        input_dim: Feature width. Inferred from the first sample if omitted.
        hidden_dims: Hidden widths used by the embedding and balance networks.
        monotonic_hidden_dims: Hidden widths in every cause-specific network.
        activation: Embedding and balancing activation.
        monotonic_activation: Smooth monotonic activation used in ``M_r``.
        dropout: Dropout in the embedding network.
        weight_transform: Non-negative weight parameterization. ``"square"``
            matches the paper; the other options are useful for ablations.
        cdf_mode: ``"nfg"`` uses ``t * M_r``. ``"sumonet"`` removes the
            time multiplier, reproducing the parameterization ablation in
            which the conditional CDF need not start at zero.
        time_grid: Positive times at which ``forward`` returns CIF curves.
        time_scale: Positive training-time scale. Durations and prediction
            times are divided by this value inside the network; the likelihood
            derivative remains with respect to the original time unit. The
            paper uses the maximum training duration.
        min_density: Lower numerical bound for event densities.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "features": [0.0, 1.0],
        ...      "duration": [1.0], "event": [1]},
        ...     {"patient_id": "p1", "features": [1.0, 0.0],
        ...      "duration": [2.0], "event": [0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = NeuralFineGray(dataset, num_risks=2, hidden_dims=(8,))
        >>> model.predict_cif(torch.tensor([[0.0, 1.0]]), torch.tensor([1.0])).shape
        torch.Size([1, 2, 1])

    References:
        Jeanselme, V., Yoon, C. H., Tom, B., and Barrett, J. Neural Fine-Gray:
        Monotonic Neural Networks for Competing Risks. CHIL, 2023.
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Return no standard label processor; the model owns its likelihood."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        num_risks: int,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        hidden_dims: Sequence[int] = (50, 50, 50),
        monotonic_hidden_dims: Sequence[int] = (50, 50, 50),
        activation: ActivationName = "tanh",
        monotonic_activation: Literal["tanh", "softplus"] = "tanh",
        dropout: float = 0.25,
        weight_transform: WeightTransform = "square",
        cdf_mode: CDFMode = "nfg",
        time_grid: Tensor | None = None,
        time_scale: float = 1.0,
        min_density: float = 1e-10,
    ) -> None:
        super().__init__(dataset)
        if num_risks <= 0:
            raise ValueError("num_risks must be positive")
        if cdf_mode not in {"nfg", "sumonet"}:
            raise ValueError("cdf_mode must be 'nfg' or 'sumonet'")
        if not math.isfinite(min_density) or min_density <= 0:
            raise ValueError("min_density must be positive and finite")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive and finite")
        if feature_key not in dataset.input_schema:
            raise ValueError(f"feature_key {feature_key!r} is not in input_schema")
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in output_schema")
        if input_dim is None:
            input_dim = _infer_input_dim(dataset, feature_key)
        self.input_dim = int(input_dim)
        self.num_risks = int(num_risks)
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.cdf_mode = cdf_mode
        self.time_scale = float(time_scale)
        self.min_density = float(min_density)
        self.network = _NeuralFineGrayNetwork(
            self.input_dim,
            self.num_risks,
            tuple(hidden_dims),
            tuple(monotonic_hidden_dims),
            activation,
            monotonic_activation,
            dropout,
            weight_transform,
        )
        if time_grid is None:
            time_grid = torch.linspace(0.01, 10.0, 100)
        time_grid = torch.as_tensor(time_grid, dtype=torch.float32).reshape(-1)
        if (
            time_grid.numel() == 0
            or not torch.isfinite(time_grid).all()
            or torch.any(time_grid <= 0)
        ):
            raise ValueError("time_grid must contain finite positive values")
        self.register_buffer("time_grid", time_grid)

    def _prepare_batch(
        self,
        kwargs: dict[str, Tensor | tuple[Tensor, ...]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        missing = [
            key
            for key in (self.feature_key, self.duration_key, self.event_key)
            if key not in kwargs
        ]
        if missing:
            raise KeyError(f"Missing required batch fields: {missing}")
        features = torch.as_tensor(
            _value_tensor(kwargs[self.feature_key]),
            dtype=torch.float32,
            device=self.device,
        )
        duration = torch.as_tensor(
            _value_tensor(kwargs[self.duration_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        raw_event = torch.as_tensor(
            _value_tensor(kwargs[self.event_key]),
            device=self.device,
        ).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must have shape [batch, input_dim]; "
                f"received {tuple(features.shape)}"
            )
        if features.shape[0] != duration.shape[0] or duration.shape != raw_event.shape:
            raise ValueError("features, duration, and event must share a batch size")
        if features.shape[0] == 0 or not torch.isfinite(features).all():
            raise ValueError("features must be nonempty and finite")
        if not torch.isfinite(duration).all() or torch.any(duration <= 0):
            raise ValueError("duration must contain finite positive values")
        if raw_event.is_floating_point() and not torch.equal(
            raw_event, raw_event.round()
        ):
            raise ValueError("event must contain integer cause labels")
        event = raw_event.to(torch.long)
        if torch.any(event < 0) or torch.any(event > self.num_risks):
            raise ValueError(
                f"event must be between 0 and {self.num_risks}, inclusive"
            )
        return features, duration, event

    def _likelihood(
        self,
        features: Tensor,
        duration: Tensor,
        event: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Likelihood evaluation needs first derivatives with respect to time;
        # training additionally needs their parameter gradients.
        with torch.enable_grad():
            time = duration.detach().clone().requires_grad_(True)
            embedding, balance_logits, log_balance = self.network.encode(features)
            scaled_time = time / self.time_scale
            hazard = self.network.cumulative_hazard(
                embedding, scaled_time, self.cdf_mode
            )
            derivatives = []
            for risk in range(self.num_risks):
                derivative = torch.autograd.grad(
                    hazard[:, risk].sum(),
                    time,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]
                derivatives.append(derivative)
            hazard_derivative = torch.stack(derivatives, dim=-1).clamp_min(
                self.min_density
            )

            log_survival_by_cause = log_balance - hazard
            censored_log_likelihood = torch.logsumexp(
                log_survival_by_cause, dim=-1
            )
            cause_index = (event - 1).clamp_min(0).unsqueeze(-1)
            event_log_likelihood = (
                log_survival_by_cause + hazard_derivative.log()
            ).gather(1, cause_index).squeeze(1)
            per_sample = torch.where(
                event == 0,
                censored_log_likelihood,
                event_log_likelihood,
            )
            loss = -per_sample.mean()
            cif = log_balance.exp() * (-torch.expm1(-hazard))
        return loss, cif, balance_logits, hazard_derivative

    def forward(
        self,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Compute exact competing-risk likelihood and CIF curves."""
        features, duration, event = self._prepare_batch(kwargs)
        loss, cif_observed, balance_logits, density_factor = self._likelihood(
            features, duration, event
        )
        with torch.no_grad():
            cif_grid = self._predict_cif_impl(features, self.time_grid)
        survival_grid = 1.0 - cif_grid.sum(dim=1)
        return {
            "loss": loss,
            "logit": balance_logits,
            "y_prob": cif_grid,
            "y_true": torch.stack([duration, event.to(duration.dtype)], dim=-1),
            "cif": cif_grid,
            "cif_at_observed_time": cif_observed,
            "survival": survival_grid,
            "balance": torch.softmax(balance_logits, dim=-1),
            "hazard_derivative": density_factor,
        }

    def _predict_cif_impl(self, features: Tensor, times: Tensor) -> Tensor:
        times = torch.as_tensor(times, dtype=features.dtype, device=features.device)
        times = times.reshape(-1)
        if (
            times.numel() == 0
            or not torch.isfinite(times).all()
            or torch.any(times < 0)
        ):
            raise ValueError("times must contain finite non-negative values")
        embedding, _, log_balance = self.network.encode(features)
        batch_size, num_times = features.shape[0], times.numel()
        expanded_embedding = (
            embedding.unsqueeze(1)
            .expand(batch_size, num_times, self.input_dim)
            .reshape(batch_size * num_times, self.input_dim)
        )
        expanded_time = (
            times.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            / self.time_scale
        )
        hazard = self.network.cumulative_hazard(
            expanded_embedding, expanded_time, self.cdf_mode
        ).reshape(batch_size, num_times, self.num_risks)
        conditional_cdf = -torch.expm1(-hazard)
        cif = log_balance.exp().unsqueeze(1) * conditional_cdf
        return cif.permute(0, 2, 1)

    @torch.no_grad()
    def predict_cif(self, features: Tensor, times: Tensor) -> Tensor:
        """Predict cause-specific cumulative incidence curves.

        Args:
            features: Dense input with shape ``[batch, input_dim]``.
            times: One-dimensional non-negative time grid.

        Returns:
            CIF tensor with shape ``[batch, num_risks, num_times]``.
        """
        features = torch.as_tensor(
            features, dtype=torch.float32, device=self.device
        )
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape [batch, input_dim]")
        previous_modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self._predict_cif_impl(features, times)
        finally:
            for module, training in previous_modes:
                module.training = training

    def forward_from_embedding(
        self,
        embeddings: Tensor,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run ``forward`` with dense features supplied directly."""
        kwargs[self.feature_key] = embeddings
        return self.forward(**kwargs)
