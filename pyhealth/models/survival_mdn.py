# Contributor: Neil Hajela (nhajela2@illinois.edu)
"""Survival Mixture Density Network for right-censored survival data.

This module implements Survival MDN from Han, Goldstein, and Ranganath,
"Survival Mixture Density Networks" (MLHC 2022). The implementation is
written from the paper's equations and validated against the released
SUPPORT benchmark behavior; it is not copied from the authors' repository.

Paper: https://proceedings.mlr.press/v182/han22a.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

__all__ = ["SurvivalMDN"]

TimeTransform = Literal["softplus", "exp"]
Reduction = Literal["mean", "sum", "none"]


@dataclass
class _MDNParameters:
    """Latent Gaussian-mixture parameters for one batch."""

    log_weights: Tensor
    means: Tensor
    scales: Tensor
    raw_weight_logits: Tensor
    raw_scales: Tensor

    @property
    def weights(self) -> Tensor:
        """Return normalized mixture weights."""
        return self.log_weights.exp()


class _PositiveTimeTransform(nn.Module):
    """Invertible map from unconstrained latent time to positive time."""

    def inverse(self, time: Tensor) -> Tensor:
        """Map positive observed time back to latent time."""
        raise NotImplementedError

    def log_abs_det_inverse_jacobian(self, time: Tensor) -> Tensor:
        """Return log absolute derivative of the inverse map."""
        raise NotImplementedError


class _SoftplusTimeTransform(_PositiveTimeTransform):
    """Positive-time map ``time = softplus(latent_time)``."""

    def __init__(self) -> None:
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, latent_time: Tensor) -> Tensor:
        """Map latent time to positive time."""
        return self.softplus(latent_time)

    def inverse(self, time: Tensor) -> Tensor:
        """Apply a numerically stable inverse softplus."""
        _validate_positive_time(time)
        return time + torch.log(-torch.expm1(-time))

    def log_abs_det_inverse_jacobian(self, time: Tensor) -> Tensor:
        """Return the inverse-softplus log-Jacobian."""
        _validate_positive_time(time)
        return -torch.log(-torch.expm1(-time))


class _ExpTimeTransform(_PositiveTimeTransform):
    """Positive-time map ``time = exp(latent_time)``."""

    def forward(self, latent_time: Tensor) -> Tensor:
        """Map latent time to positive time."""
        return torch.exp(latent_time)

    def inverse(self, time: Tensor) -> Tensor:
        """Map positive time to log-time."""
        _validate_positive_time(time)
        return torch.log(time)

    def log_abs_det_inverse_jacobian(self, time: Tensor) -> Tensor:
        """Return the inverse-exponential log-Jacobian."""
        _validate_positive_time(time)
        return -torch.log(time)


def _validate_positive_time(time: Tensor) -> None:
    """Validate nonempty finite strictly positive time values."""
    if time.numel() == 0 or not torch.isfinite(time).all() or torch.any(time <= 0):
        raise ValueError(
            "duration/time grid must contain finite strictly positive values"
        )


def _make_time_transform(name: TimeTransform) -> _PositiveTimeTransform:
    if name == "softplus":
        return _SoftplusTimeTransform()
    if name == "exp":
        return _ExpTimeTransform()
    raise ValueError("time_transform must be 'softplus' or 'exp'")


def _value_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Extract the primary value emitted by a PyHealth processor."""
    if isinstance(value, tuple):
        return value[0]
    return value


def _infer_input_dim(dataset: SampleDataset, feature_key: str) -> int:
    """Infer a dense vector width from one processed sample."""
    try:
        sample = dataset[0]
    except Exception as exc:  # pragma: no cover - defensive streaming fallback
        raise ValueError(
            "Could not infer input_dim from the dataset; pass input_dim explicitly."
        ) from exc
    if feature_key not in sample:
        raise ValueError(f"feature_key {feature_key!r} is not present in the dataset")
    value = sample[feature_key]
    if isinstance(value, tuple):
        value = value[0]
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(
            "SurvivalMDN expects each sample's feature field to be a 1D dense "
            "numeric vector; pass input_dim explicitly only if this is intended."
        )
    return int(tensor.shape[0])


class _SurvivalMDNDistribution:
    """Positive-time distribution induced by a Gaussian mixture."""

    _LOG_2PI = 1.8378770664093453

    def __init__(
        self,
        params: _MDNParameters,
        time_transform: _PositiveTimeTransform,
    ) -> None:
        self.params = params
        self.time_transform = time_transform

    def _standardized_latent(self, time: Tensor) -> Tensor:
        latent = self.time_transform.inverse(time)
        return (latent.unsqueeze(-1) - self.params.means) / self.params.scales

    def log_prob(self, time: Tensor, include_jacobian: bool = True) -> Tensor:
        """Return ``log f(time | x)`` for each sample."""
        latent = self.time_transform.inverse(time).unsqueeze(-1)
        z = (latent - self.params.means) / self.params.scales
        component_log_pdf = (
            -0.5 * z.square() - torch.log(self.params.scales) - 0.5 * self._LOG_2PI
        )
        result = torch.logsumexp(
            self.params.log_weights + component_log_pdf,
            dim=-1,
        )
        if include_jacobian:
            result = result + self.time_transform.log_abs_det_inverse_jacobian(time)
        return result

    def log_cdf(self, time: Tensor) -> Tensor:
        """Return the stable log CDF."""
        z = self._standardized_latent(time)
        component_log_cdf = torch.special.log_ndtr(z)
        return torch.logsumexp(
            self.params.log_weights + component_log_cdf,
            dim=-1,
        )

    def log_survival(self, time: Tensor) -> Tensor:
        """Return stable ``log S(time | x)`` without ``1 - CDF`` cancellation."""
        z = self._standardized_latent(time)
        component_log_survival = torch.special.log_ndtr(-z)
        return torch.logsumexp(
            self.params.log_weights + component_log_survival,
            dim=-1,
        )

    def survival(self, time: Tensor) -> Tensor:
        """Return survival probability ``S(time | x)``."""
        return self.log_survival(time).exp().clamp(0.0, 1.0)


class _SurvivalMDNNetwork(nn.Module):
    """Reference Survival-MDN network used by :class:`SurvivalMDN`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_components: int,
        num_hidden_layers: int,
        min_scale: float,
        time_transform: TimeTransform,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be positive")

        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(width, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.PReLU(),
                ]
            )
            width = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.weight_head = nn.Linear(hidden_dim, num_components)
        self.mean_head = nn.Linear(hidden_dim, num_components)
        self.scale_head = nn.Linear(hidden_dim, num_components)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.scale_softplus = nn.Softplus()
        self.time_transform = _make_time_transform(time_transform)
        self.min_scale = float(min_scale)
        self.num_components = int(num_components)

    def initialize_reference_heads(self) -> None:
        """Initialize mixture heads to the reference SUPPORT configuration.

        The public reference checkpoint starts with equal mixture weights,
        component means spread over roughly [-3, 3], and common positive-scale
        logits. Hidden-layer parameters retain PyTorch's standard initialization.
        """
        with torch.no_grad():
            self.weight_head.weight.zero_()
            self.mean_head.weight.zero_()
            self.scale_head.weight.zero_()
            self.weight_head.bias.zero_()
            self.mean_head.bias.copy_(
                torch.linspace(
                    -3.0,
                    3.0,
                    self.num_components,
                    device=self.mean_head.bias.device,
                    dtype=self.mean_head.bias.dtype,
                )
            )
            self.scale_head.bias.fill_(0.5)

    def mixture_parameters(self, features: Tensor) -> _MDNParameters:
        """Compute latent Gaussian-mixture parameters."""
        hidden = self.feature_net(features)
        weight_logits = self.weight_head(hidden)
        means = self.mean_head(hidden)
        raw_scales = self.scale_head(hidden)
        scales = self.scale_softplus(raw_scales).clamp_min(self.min_scale)
        return _MDNParameters(
            log_weights=self.log_softmax(weight_logits),
            means=means,
            scales=scales,
            raw_weight_logits=weight_logits,
            raw_scales=raw_scales,
        )

    def censored_nll(
        self,
        params: _MDNParameters,
        duration: Tensor,
        event: Tensor,
        reduction: Reduction = "mean",
        include_jacobian: bool = True,
    ) -> Tensor:
        """Compute right-censored negative log-likelihood."""
        duration = duration.reshape(-1)
        event = event.reshape(-1).to(dtype=params.means.dtype)
        if params.means.shape[0] != duration.shape[0]:
            raise ValueError("features and duration must have the same batch size")
        if params.means.shape[0] != event.shape[0]:
            raise ValueError("features and event must have the same batch size")
        if torch.any((event != 0) & (event != 1)):
            raise ValueError("event must contain only 0/1 indicators")

        distribution = _SurvivalMDNDistribution(params, self.time_transform)
        log_pdf = distribution.log_prob(
            duration,
            include_jacobian=include_jacobian,
        )
        log_survival = distribution.log_survival(duration)
        per_sample = -torch.where(event.bool(), log_pdf, log_survival)
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def survival_at(
        self,
        params: _MDNParameters,
        times: Tensor,
    ) -> Tensor:
        """Return ``S(t | x)`` with shape ``[batch, num_times]``."""
        times = times.reshape(-1).to(
            device=params.means.device,
            dtype=params.means.dtype,
        )
        batch_size = params.means.shape[0]
        expanded = _MDNParameters(
            log_weights=params.log_weights.unsqueeze(1),
            means=params.means.unsqueeze(1),
            scales=params.scales.unsqueeze(1),
            raw_weight_logits=params.raw_weight_logits.unsqueeze(1),
            raw_scales=params.raw_scales.unsqueeze(1),
        )
        distribution = _SurvivalMDNDistribution(
            expanded,
            self.time_transform,
        )
        grid = times.unsqueeze(0).expand(batch_size, -1)
        return distribution.survival(grid)


class SurvivalMDN(BaseModel):
    """Mixture-density network for right-censored continuous survival.

    Survival MDN models a Gaussian mixture in unconstrained latent time and
    maps that distribution to positive event time. Training maximizes the
    usual right-censored likelihood: observed events contribute a density
    term and censored records contribute a survival-probability term.

    Args:
        dataset: Fitted PyHealth ``SampleDataset``. The model expects one dense
            numeric feature vector and two tensor outputs: duration and event.
        feature_key: Input field containing the dense feature vector.
        duration_key: Output field containing strictly positive durations.
        event_key: Output field containing ``1`` for an observed event and
            ``0`` for right censoring.
        input_dim: Width of the dense input vector. If omitted, the model
            infers it from the first processed sample.
        hidden_dim: Width of each shared hidden layer.
        num_components: Number of Gaussian mixture components.
        num_hidden_layers: Number of Linear-BatchNorm-PReLU hidden blocks.
        time_transform: Positive-time transformation. ``"softplus"`` is the
            paper baseline; ``"exp"`` is supported for ablation studies.
        min_scale: Minimum Gaussian standard deviation for numerical safety.
        time_grid: Times at which ``y_prob`` returns survival probabilities.
            If omitted, a 128-point grid from 0.001 to 5.6 is used.
        include_jacobian: Whether event-density likelihoods include the
            transformation Jacobian. Keep this ``True`` when comparing
            different time transformations.
        reference_initialization: If ``True``, initialize the three mixture
            heads to the reference SUPPORT configuration used in the project
            reproduction.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "features": [0.0, 1.0, -0.5],
        ...         "duration": [1.2],
        ...         "event": [1.0],
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "features": [1.0, 0.2, 0.4],
        ...         "duration": [2.0],
        ...         "event": [0.0],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = SurvivalMDN(dataset, hidden_dim=8, num_components=3)
        >>> model.num_components
        3

    References:
        Han, X., Goldstein, M., and Ranganath, R. Survival Mixture Density
        Networks. Proceedings of Machine Learning for Healthcare, 2022.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        hidden_dim: int = 32,
        num_components: int = 10,
        num_hidden_layers: int = 3,
        time_transform: TimeTransform = "softplus",
        min_scale: float = 1e-8,
        time_grid: Tensor | None = None,
        include_jacobian: bool = True,
        reference_initialization: bool = True,
    ) -> None:
        super().__init__(dataset)
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.include_jacobian = bool(include_jacobian)
        self.num_components = int(num_components)
        self.time_transform = time_transform
        self.mode = None

        if feature_key not in dataset.input_schema:
            raise ValueError(
                f"feature_key {feature_key!r} is not in dataset.input_schema"
            )
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in dataset.output_schema")

        if input_dim is None:
            input_dim = _infer_input_dim(dataset, feature_key)
        self.input_dim = int(input_dim)
        self.network = _SurvivalMDNNetwork(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_components=num_components,
            num_hidden_layers=num_hidden_layers,
            min_scale=min_scale,
            time_transform=time_transform,
        )
        if reference_initialization:
            self.network.initialize_reference_heads()
        if time_grid is None:
            time_grid = torch.linspace(0.001, 5.6, 128)
        time_grid = torch.as_tensor(time_grid, dtype=torch.float32)
        _validate_positive_time(time_grid)
        self.register_buffer("time_grid", time_grid.reshape(-1))

    def forward(
        self,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run one Survival-MDN forward pass.

        Args:
            **kwargs: Batch fields produced by the PyHealth dataloader. The
                batch must include ``feature_key``, ``duration_key``, and
                ``event_key``.

        Returns:
            Dictionary containing ``loss``, ``logit``, ``y_prob``, ``y_true``,
            and the fitted mixture weights, means, and scales. ``y_prob`` is
            the survival curve evaluated on ``self.time_grid``.
        """
        missing = [
            key
            for key in (self.feature_key, self.duration_key, self.event_key)
            if key not in kwargs
        ]
        if missing:
            raise KeyError(f"Missing required batch fields: {missing}")

        features = _value_tensor(kwargs[self.feature_key])
        duration = _value_tensor(kwargs[self.duration_key])
        event = _value_tensor(kwargs[self.event_key])
        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )
        duration = torch.as_tensor(
            duration,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        event = torch.as_tensor(
            event,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must have shape [batch, input_dim]; "
                f"received {tuple(features.shape)}"
            )

        if features.shape[0] == 0 or not torch.isfinite(features).all():
            raise ValueError("features must be nonempty and finite")

        # Compute the BatchNorm-containing feature network once per batch.
        params = self.network.mixture_parameters(features)
        loss = self.network.censored_nll(
            params,
            duration,
            event,
            include_jacobian=self.include_jacobian,
        )
        y_prob = self.network.survival_at(params, self.time_grid)
        y_true = torch.stack([duration, event], dim=-1)
        logit = torch.cat(
            [
                params.raw_weight_logits,
                params.means,
                params.raw_scales,
            ],
            dim=-1,
        )
        return {
            "loss": loss,
            "logit": logit,
            "y_prob": y_prob,
            "y_true": y_true,
            "mixture_weights": params.weights,
            "mixture_means": params.means,
            "mixture_scales": params.scales,
        }

    def forward_from_embedding(
        self,
        embeddings: Tensor,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run ``forward`` with dense features supplied directly."""
        kwargs[self.feature_key] = embeddings
        return self.forward(**kwargs)

    def predict_survival(
        self,
        features: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Predict survival probabilities on an arbitrary time grid.

        Args:
            features: Dense input tensor with shape ``[batch, input_dim]``.
            times: Strictly positive time points.

        Returns:
            Survival probabilities with shape ``[batch, num_times]``.
        """
        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )
        times = torch.as_tensor(
            times,
            dtype=torch.float32,
            device=self.device,
        )
        params = self.network.mixture_parameters(features)
        return self.network.survival_at(params, times)
