"""Deep Cox Mixtures for survival regression.

Contributor: <AUTHOR NAME> (NetID: liu323)

Paper: Chirag Nagpal, Steve Yadlowsky, Negar Rostamzadeh and Katherine Heller.
Deep Cox Mixtures for Survival Regression. Machine Learning for Healthcare
(MLHC), PMLR 149, 2021. https://proceedings.mlr.press/v149/nagpal21a.html

Deep Cox Mixtures (DCM) model the survival function of a patient as a mixture
of K Cox proportional-hazards models. A shared neural encoder feeds a gating
head, which gives the probability of each latent group, and a head with one
log hazard ratio per group. Each group has its own nonparametric (Breslow)
baseline hazard. Training uses the paper's approximate Monte Carlo EM: an
E-step computes posterior group responsibilities from the observed time and
censoring status, hard assignments are sampled from them, a gradient step
maximizes the gate log-likelihood plus one Cox partial likelihood per group,
and the per-group Breslow baselines are re-estimated on the full training set.

This is an independent PyTorch implementation written from the paper. The
authors' TensorFlow code (Apache-2.0) and the PyTorch DCM in auton-survival
(MIT) were consulted to check update order and defaults; no code was copied.
"""

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "tanh": nn.Tanh,
}
# Smallest per-group likelihood used in the E-step. The reference code floors
# probabilities at the same value so that one group with zero density (for
# example an event after that group's last event time) cannot produce NaNs.
_LIKELIHOOD_FLOOR = 1e-10


def _as_vector(value: Any, name: str, device: torch.device) -> Tensor:
    """Convert a label field of shape (n,) or (n, 1) to a float64 vector."""
    if isinstance(value, tuple):
        value = value[0]
    tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape (batch,) or (batch, 1)")
    return tensor


def _check_labels(duration: Tensor, event: Tensor) -> None:
    """Validate right-censored labels."""
    if duration.shape != event.shape:
        raise ValueError("duration and event must have the same number of rows")
    if not torch.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("duration must be finite and strictly positive")
    if not torch.all((event == 0) | (event == 1)):
        raise ValueError("event must contain only 0 (censored) or 1 (event)")


def cox_partial_log_likelihood(
    log_risk: Tensor, duration: Tensor, event: Tensor
) -> Tensor:
    """Cox partial log-likelihood with Breslow's handling of tied times.

    For each observed event ``i`` the risk set holds every patient with
    ``duration >= duration[i]``, so tied event times share one denominator.

    Args:
        log_risk: Log hazard ratios, shape ``(n,)``.
        duration: Event or censoring times, shape ``(n,)``.
        event: 1 for an observed event, 0 for censoring, shape ``(n,)``.

    Returns:
        The summed partial log-likelihood (a scalar, 0 if there are no
        events). It is differentiable with respect to ``log_risk``.

    Examples:
        >>> import torch
        >>> ll = cox_partial_log_likelihood(
        ...     torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]),
        ...     torch.tensor([1.0, 1.0, 0.0]),
        ... )
        >>> round(float(ll), 4)  # -(log 3 + log 2)
        -1.7918
    """
    if log_risk.numel() == 0:
        return log_risk.sum()
    order = torch.argsort(duration, descending=True, stable=True)
    sorted_time = duration[order]
    sorted_risk = log_risk[order]
    sorted_event = event[order].to(log_risk.dtype)
    log_cumulative = torch.logcumsumexp(sorted_risk, dim=0)
    # In descending time order the risk set of a tie group ends at its last
    # member, so every member of the group uses that position's denominator.
    _, inverse, counts = torch.unique_consecutive(
        sorted_time, return_inverse=True, return_counts=True
    )
    group_end = torch.cumsum(counts, dim=0) - 1
    log_denominator = log_cumulative[group_end[inverse]]
    return ((sorted_risk - log_denominator) * sorted_event).sum()


def breslow_cumulative_hazard(
    log_risk: Tensor, duration: Tensor, event: Tensor
) -> tuple[Tensor, Tensor]:
    """Breslow estimate of the cumulative baseline hazard.

    ``H0(tau_j) = sum_{tau_m <= tau_j} d_m / sum_{l: t_l >= tau_m} exp(f_l)``,
    where ``tau`` are the distinct event times and ``d`` their event counts.

    Args:
        log_risk: Log hazard ratios, shape ``(n,)``.
        duration: Event or censoring times, shape ``(n,)``.
        event: 1 for an observed event, 0 for censoring, shape ``(n,)``.

    Returns:
        ``(event_times, cumulative_hazard)``, both of shape ``(m,)`` with
        ``m`` distinct event times in increasing order. Both are empty when
        there is no event.

    Examples:
        >>> import torch
        >>> times, hazard = breslow_cumulative_hazard(
        ...     torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]),
        ...     torch.tensor([1.0, 1.0, 0.0]),
        ... )
        >>> times.tolist(), [round(h, 4) for h in hazard.tolist()]
        ([1.0, 2.0], [0.3333, 0.8333])
    """
    log_risk = log_risk.detach().to(torch.float64)
    duration = duration.detach().to(torch.float64)
    observed = event.detach() > 0.5
    if not observed.any():
        empty = duration.new_zeros(0)
        return empty, empty
    event_times, deaths = torch.unique(duration[observed], return_counts=True)
    order = torch.argsort(duration)
    sorted_time = duration[order]
    # Sum of exp(f) over everyone at or after each position, computed in log
    # space so that large hazard ratios cannot overflow.
    log_suffix = torch.logcumsumexp(log_risk[order].flip(0), dim=0).flip(0)
    first_at_risk = torch.searchsorted(sorted_time, event_times, right=False)
    increments = deaths.to(torch.float64) * torch.exp(-log_suffix[first_at_risk])
    return event_times, torch.cumsum(increments, dim=0)


class DeepCoxMixtures(BaseModel):
    """Deep Cox Mixtures (Nagpal et al., MLHC 2021) for right-censored data.

    The survival function of a patient with features ``x`` is

    ``S(t | x) = sum_k pi_k(x) * S_0k(t) ** exp(f_k(x))``,

    where ``pi(x) = softmax(g(phi(x)))`` are latent-group probabilities,
    ``f_k(phi(x))`` is group ``k``'s log hazard ratio, ``phi`` is an MLP
    encoder and ``S_0k`` is group ``k``'s Breslow baseline survival. Within a
    group the proportional hazards assumption holds; the mixture does not.

    Train with :meth:`fit`, which runs the paper's Monte Carlo EM: for every
    mini-batch it computes the posterior group responsibilities ``gamma``
    (E-step), samples hard assignments ``zeta ~ Categorical(gamma)``, takes a
    gradient step on ``sum gamma log pi + sum_k log PL_k`` with one Cox
    partial likelihood per sampled group, and then re-estimates every
    group's Breslow baseline on the full training set using hard assignments
    sampled from the full-data posterior. Partial likelihoods use mini-batch
    risk sets, as in the paper. The generic :class:`~pyhealth.trainer.Trainer`
    is not used because the baselines must be refit between gradient steps.

    Baseline survival between event times: each group's cumulative baseline
    hazard is interpolated linearly between its own distinct event times
    (a piecewise-constant hazard), starting from 0 at time 0, and held
    constant after the group's last event time. This gives survival curves in
    [0, 1] that never increase and nonnegative event densities for the
    E-step. The paper instead smooths the Breslow survival curve with a cubic
    spline, and the authors' later PyTorch release with a linear smoothing
    spline; neither is guaranteed monotone.

    Survival outputs have no label processor, so ``mode`` is ``None``.
    Features must be dense and finite: impute and scale them first, using
    training patients only.

    Args:
        dataset: Optional fitted ``SampleDataset`` with a dense feature vector
            and ``duration``/``event`` tensor outputs. Use ``None`` together
            with ``input_dim`` to build the model without a dataset.
        input_dim: Width of the feature vector. Inferred from the first
            sample of ``dataset`` when omitted.
        num_components: Number of latent groups ``K``. ``K = 1`` is a neural
            Cox model (DeepSurv-style) with a Breslow baseline.
        hidden_dims: Widths of the encoder's hidden layers. An empty
            sequence gives linear gate and risk functions of the raw
            features (the paper's "Cox Mixture"). Default ``(100, 100)``.
        activation: Encoder activation: ``"selu"`` (the authors' code),
            ``"relu"``, ``"relu6"`` or ``"tanh"``.
        use_bias: Whether the encoder layers have bias terms. The authors'
            code uses none. The risk head never has one, since a constant is
            absorbed by the baseline hazard.
        log_risk_bound: Log hazard ratios are clamped to
            ``[-log_risk_bound, log_risk_bound]`` to keep ``exp(f)`` finite,
            as in auton-survival (default 10). ``None`` disables the clamp.
        feature_key: Input field with the dense feature vector.
        duration_key: Output field with strictly positive times.
        event_key: Output field with 1 for an event and 0 for censoring.
        time_grid: Times at which ``forward`` returns ``y_prob`` (survival
            probabilities) once baselines are fitted. Optional.

    Examples:
        >>> import torch
        >>> from pyhealth.models import DeepCoxMixtures
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> x = torch.randn(40, 4)
        >>> time = torch.rand(40) * 10 + 0.1
        >>> event = (torch.rand(40) < 0.7).float()
        >>> model = DeepCoxMixtures(None, input_dim=4, num_components=2,
        ...                         hidden_dims=(8,))
        >>> history = model.fit((x, time, event), epochs=2, batch_size=16,
        ...                     seed=0)
        >>> model.predict_survival(x[:3], [1.0, 5.0]).shape
        torch.Size([3, 2])
        >>> bool(torch.allclose(model.predict_latent(x[:3]).sum(dim=1),
        ...                     torch.ones(3)))
        True
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """No label processors: survival outputs are scored by the model."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset | None = None,
        input_dim: int | None = None,
        num_components: int = 3,
        hidden_dims: Sequence[int] = (100, 100),
        activation: str = "selu",
        use_bias: bool = False,
        log_risk_bound: float | None = 10.0,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        time_grid: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__(dataset)
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        if dataset is not None:
            if feature_key not in dataset.input_schema:
                raise ValueError(
                    f"feature_key {feature_key!r} is not in dataset.input_schema"
                )
            for key in (duration_key, event_key):
                if key not in dataset.output_schema:
                    raise ValueError(
                        f"output key {key!r} is not in dataset.output_schema"
                    )
            if input_dim is None:
                input_dim = self._infer_input_dim(dataset, feature_key)
        if input_dim is None:
            raise ValueError("input_dim is required when dataset is None")
        for name, value in (
            ("input_dim", input_dim),
            ("num_components", num_components),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        hidden_dims = tuple(hidden_dims)
        if any(
            isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if log_risk_bound is not None and not (
            math.isfinite(log_risk_bound) and log_risk_bound > 0
        ):
            raise ValueError("log_risk_bound must be positive and finite, or None")

        self.input_dim = input_dim
        self.num_components = num_components
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.log_risk_bound = log_risk_bound

        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(width, hidden, bias=use_bias),
                _ACTIVATIONS[activation](),
            ]
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.gate = nn.Linear(width, num_components, bias=False)
        self.risk = nn.Linear(width, num_components, bias=False)

        if time_grid is None:
            grid = torch.zeros(0)
        else:
            grid = torch.as_tensor(time_grid, dtype=torch.float32).reshape(-1)
            if not torch.isfinite(grid).all() or (grid < 0).any():
                raise ValueError("time_grid must be finite and non-negative")
        # Not a persistent buffer: saved through get_extra_state so a model
        # built with another (or no) grid can load the checkpoint.
        self.register_buffer("time_grid", grid, persistent=False)
        # Baselines: per-group knots (0, then the group's event times, then
        # +inf padding) and the cumulative hazard there, shape (K, M).
        self._knots: Tensor | None = None
        self._cumulative_hazard: Tensor | None = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _infer_input_dim(dataset: SampleDataset, feature_key: str) -> int:
        """Read the feature width from the first processed sample."""
        value = dataset[0][feature_key]
        if isinstance(value, tuple):
            value = value[0]
        tensor = torch.as_tensor(value)
        if tensor.ndim != 1:
            raise ValueError(
                "DeepCoxMixtures expects a 1D dense feature vector per sample"
            )
        return int(tensor.shape[0])

    def _features(self, value: Any) -> Tensor:
        """Validate and move a feature matrix to the model's device."""
        if isinstance(value, tuple):
            value = value[0]
        x = torch.as_tensor(value, device=self.device, dtype=self.gate.weight.dtype)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape (batch, {self.input_dim}), "
                f"got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("features must be finite; impute missing values first")
        return x

    def _network(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return log gate probabilities and log hazard ratios, both (n, K)."""
        hidden = self.encoder(x)
        log_gate = F.log_softmax(self.gate(hidden), dim=-1)
        log_risk = self.risk(hidden)
        if self.log_risk_bound is not None:
            log_risk = log_risk.clamp(-self.log_risk_bound, self.log_risk_bound)
        return log_gate, log_risk

    @property
    def is_fitted(self) -> bool:
        """Whether the per-group baseline hazards have been estimated."""
        return self._knots is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Baseline hazards are not fitted; call fit() or fit_baselines()"
            )

    # -------------------------------------------------------- baseline state
    def get_extra_state(self) -> dict[str, Tensor | None]:
        """Store baselines and ``time_grid`` in ``state_dict``.

        Returns:
            The fitted baseline knots and cumulative hazards (``None`` before
            fitting) and the time grid.
        """
        return {
            "knots": self._knots,
            "cumulative_hazard": self._cumulative_hazard,
            "time_grid": self.time_grid,
        }

    def set_extra_state(self, state: Mapping[str, Tensor | None]) -> None:
        """Restore baselines saved by :meth:`get_extra_state`.

        Args:
            state: Output of :meth:`get_extra_state`.

        Raises:
            ValueError: If the saved baselines are incomplete or were fit for
                a different number of components.
        """
        knots, hazard = state.get("knots"), state.get("cumulative_hazard")
        if (knots is None) != (hazard is None):
            raise ValueError("Saved baseline state is incomplete")
        if knots is not None:
            knots = knots.to(device=self.device, dtype=torch.float64)
            hazard = hazard.to(device=self.device, dtype=torch.float64)
            if knots.shape != hazard.shape or knots.shape[0] != self.num_components:
                raise ValueError("Saved baselines do not match num_components")
        self._knots, self._cumulative_hazard = knots, hazard
        if state.get("time_grid") is not None:
            self.time_grid = state["time_grid"].to(self.device)

    def _apply(self, fn: Any, recurse: bool = True) -> "DeepCoxMixtures":
        """Move the baseline tensors along with the module (``.to()`` etc.)."""
        module = super()._apply(fn, recurse)
        if self._knots is not None:
            device = self._dummy_param.device
            self._knots = self._knots.to(device)
            self._cumulative_hazard = self._cumulative_hazard.to(device)
        return module

    def _baseline_at(self, times: Tensor) -> tuple[Tensor, Tensor]:
        """Interpolated cumulative baseline hazard and hazard rate.

        Args:
            times: Query times, shape ``(q,)``.

        Returns:
            ``(cumulative_hazard, hazard)``, each of shape ``(q, K)``.
        """
        knots, hazard = self._knots, self._cumulative_hazard
        k, m = knots.shape
        query = times.to(knots).reshape(1, -1).expand(k, -1).contiguous()
        upper = torch.searchsorted(knots, query, right=False).clamp(1, m - 1)
        lower = upper - 1
        t_lo, t_hi = knots.gather(1, lower), knots.gather(1, upper)
        h_lo, h_hi = hazard.gather(1, lower), hazard.gather(1, upper)
        inside = torch.isfinite(t_hi)
        width = torch.where(inside, t_hi - t_lo, torch.ones_like(t_lo))
        slope = torch.where(inside, (h_hi - h_lo) / width, torch.zeros_like(t_lo))
        fraction = ((query - t_lo) / width).clamp(0.0, 1.0)
        value = torch.where(inside, h_lo + fraction * (h_hi - h_lo), h_lo)
        return value.T, slope.T

    def _set_baselines(self, curves: list[tuple[Tensor, Tensor]]) -> None:
        """Pack per-group ``(event_times, cumulative_hazard)`` into (K, M)."""
        width = 2 + max(len(times) for times, _ in curves)
        knots = torch.full(
            (self.num_components, width),
            math.inf,
            dtype=torch.float64,
            device=self.device,
        )
        hazard = torch.zeros_like(knots)
        knots[:, 0] = 0.0
        for k, (times, cumulative) in enumerate(curves):
            m = len(times)
            knots[k, 1 : m + 1] = times
            hazard[k, 1 : m + 1] = cumulative
            hazard[k, m + 1 :] = cumulative[-1] if m else 0.0
        self._knots, self._cumulative_hazard = knots, hazard

    # ------------------------------------------------------------- EM pieces
    def _component_log_likelihood(
        self, log_risk: Tensor, duration: Tensor, event: Tensor
    ) -> Tensor:
        """Per-group log-likelihood of each observation, shape (n, K).

        Censored rows use ``log S_k(t | x)``; events use the log density
        ``f_k + log h_0k(t) + log S_k(t | x)``. Values are floored at
        ``log(1e-10)``.
        """
        cumulative, rate = self._baseline_at(duration)
        log_risk = log_risk.to(torch.float64)
        log_survival = -cumulative * torch.exp(log_risk)
        with torch.no_grad():
            log_rate = torch.log(rate)
        log_density = log_risk + log_rate + log_survival
        observed = event.to(torch.bool).unsqueeze(1)
        log_likelihood = torch.where(observed, log_density, log_survival)
        return log_likelihood.clamp_min(math.log(_LIKELIHOOD_FLOOR))

    def _log_posterior(
        self, log_gate: Tensor, log_risk: Tensor, duration: Tensor, event: Tensor
    ) -> Tensor:
        """E-step: log responsibilities ``log gamma``, shape (n, K).

        Before any baseline exists (the first mini-batch), the gate
        probabilities are used, as in the authors' code.
        """
        log_gate = log_gate.detach().to(torch.float64)
        if not self.is_fitted:
            return log_gate
        log_likelihood = self._component_log_likelihood(
            log_risk.detach(), duration, event
        )
        return torch.log_softmax(log_gate + log_likelihood, dim=-1)

    @staticmethod
    def _sample_assignment(
        log_posterior: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        """Draw ``zeta_i ~ Categorical(gamma_i)``; returns (n,) group ids."""
        probs = log_posterior.exp().float().cpu()
        draws = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        return draws.to(log_posterior.device)

    def _expected_complete_loss(
        self,
        log_gate: Tensor,
        log_risk: Tensor,
        duration: Tensor,
        event: Tensor,
        log_posterior: Tensor,
        assignment: Tensor,
    ) -> Tensor:
        """Negative Monte Carlo Q-function per observation (M-step loss)."""
        gate_term = (log_posterior.exp().to(log_gate) * log_gate).sum()
        partial = log_gate.new_zeros(())
        for k in range(self.num_components):
            members = assignment == k
            if members.any():
                partial = partial + cox_partial_log_likelihood(
                    log_risk[members, k], duration[members], event[members]
                )
        return -(gate_term + partial) / len(duration)

    # --------------------------------------------------------------- forward
    def forward(self, **kwargs: Any) -> dict[str, Tensor]:
        """Run the network and, when labels are given, the EM objective.

        Args:
            **kwargs: Batch fields. ``feature_key`` (batch, input_dim) is
                required. ``duration_key`` and ``event_key`` (batch,) or
                (batch, 1) are optional and must be given together.

        Returns:
            Dictionary with ``logit`` and ``log_risk`` (log hazard ratios,
            (batch, K)) and ``mixture_weights`` (gate probabilities,
            (batch, K)). Once baselines are fitted and ``time_grid`` is set,
            also ``survival`` and ``y_prob``: survival at ``time_grid``,
            (batch, T). With labels, also ``loss`` (the negative Monte Carlo
            EM objective per patient, using responsibilities from the current
            baselines), ``posterior`` (responsibilities, (batch, K)) and
            ``y_true`` ((batch, 2): duration, event).
        """
        if self.feature_key not in kwargs:
            raise ValueError(f"Missing required input {self.feature_key!r}")
        x = self._features(kwargs[self.feature_key])
        log_gate, log_risk = self._network(x)
        result = {
            "logit": log_risk,
            "log_risk": log_risk,
            "mixture_weights": log_gate.exp(),
        }
        if self.is_fitted and self.time_grid.numel():
            survival = self._mixture_survival(log_gate, log_risk, self.time_grid)
            result.update(survival=survival, y_prob=survival)
        labeled = [key in kwargs for key in (self.duration_key, self.event_key)]
        if any(labeled) and not all(labeled):
            raise ValueError(
                f"{self.duration_key} and {self.event_key} must be given together"
            )
        if all(labeled):
            duration = _as_vector(kwargs[self.duration_key], "duration", self.device)
            event = _as_vector(kwargs[self.event_key], "event", self.device)
            if len(duration) != len(x):
                raise ValueError("labels and features have different batch sizes")
            _check_labels(duration, event)
            log_posterior = self._log_posterior(log_gate, log_risk, duration, event)
            assignment = self._sample_assignment(log_posterior, None)
            loss = self._expected_complete_loss(
                log_gate, log_risk, duration, event, log_posterior, assignment
            )
            result.update(
                loss=loss,
                posterior=log_posterior.exp().to(log_gate),
                y_true=torch.stack([duration, event], dim=-1).to(log_gate),
            )
        return result

    def _mixture_survival(
        self, log_gate: Tensor, log_risk: Tensor, times: Tensor
    ) -> Tensor:
        """``sum_k pi_k exp(-H_0k(t) exp(f_k))`` for every row and time."""
        cumulative, _ = self._baseline_at(times)  # (T, K)
        risk = torch.exp(log_risk.to(torch.float64))  # (n, K)
        log_terms = log_gate.to(torch.float64).unsqueeze(1) - (
            cumulative.unsqueeze(0) * risk.unsqueeze(1)
        )  # (n, T, K)
        survival = torch.logsumexp(log_terms, dim=-1).exp().clamp(0.0, 1.0)
        return survival.to(log_gate.dtype)

    # -------------------------------------------------------------- training
    def _collect(self, data: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Turn ``(x, t, e)``, a mapping or a dataset of samples into tensors."""
        if isinstance(data, Mapping):
            parts = (
                data[self.feature_key],
                data[self.duration_key],
                data[self.event_key],
            )
        elif (
            isinstance(data, (tuple, list))
            and len(data) == 3
            and not (isinstance(data[0], Mapping))
        ):
            parts = tuple(data)
        else:
            samples = [data[i] for i in range(len(data))]
            if not samples:
                raise ValueError("data is empty")

            def stack(key: str) -> Tensor:
                values = []
                for sample in samples:
                    value = sample[key]
                    if isinstance(value, tuple):
                        value = value[0]
                    values.append(torch.as_tensor(value, dtype=torch.float64))
                return torch.stack(values)

            parts = (
                stack(self.feature_key),
                stack(self.duration_key),
                stack(self.event_key),
            )
        x = self._features(parts[0])
        duration = _as_vector(parts[1], "duration", self.device)
        event = _as_vector(parts[2], "event", self.device)
        if not len(x) == len(duration) == len(event):
            raise ValueError("features, duration and event lengths differ")
        if not len(x):
            raise ValueError("data is empty")
        _check_labels(duration, event)
        return x, duration, event

    @torch.no_grad()
    def fit_baselines(
        self,
        features: Tensor,
        duration: Tensor,
        event: Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        """Re-estimate every group's Breslow baseline on the given data.

        Hard assignments are sampled from the posterior under the current
        baselines (from the gate probabilities if none exist yet). A group
        with no assigned patient keeps its previous baseline; if it has none,
        it is fit on all patients with its own hazard ratios. A group whose
        patients have no event gets a zero hazard (survival 1).

        Args:
            features: Dense features, shape ``(n, input_dim)``.
            duration: Strictly positive times, shape ``(n,)``.
            event: 1 for an event, 0 for censoring, shape ``(n,)``.
            generator: Optional random generator for the sampled assignments.

        Examples:
            >>> import torch
            >>> model = DeepCoxMixtures(None, input_dim=2, num_components=2,
            ...                         hidden_dims=())
            >>> model.fit_baselines(torch.randn(6, 2), torch.arange(1., 7.),
            ...                     torch.ones(6))
            >>> model.is_fitted
            True
        """
        x = self._features(features)
        duration = _as_vector(duration, "duration", self.device)
        event = _as_vector(event, "event", self.device)
        _check_labels(duration, event)
        log_gate, log_risk = self._network(x)
        log_posterior = self._log_posterior(log_gate, log_risk, duration, event)
        assignment = self._sample_assignment(log_posterior, generator)
        previous = None
        if self.is_fitted:
            previous = (self._knots.clone(), self._cumulative_hazard.clone())
        curves = []
        for k in range(self.num_components):
            members = assignment == k
            if not members.any() and previous is not None:
                finite = torch.isfinite(previous[0][k])
                finite[0] = False  # drop the time-0 origin knot
                curves.append((previous[0][k][finite], previous[1][k][finite]))
                continue
            if not members.any():
                members = torch.ones_like(members)
            curves.append(
                breslow_cumulative_hazard(
                    log_risk[members, k], duration[members], event[members]
                )
            )
        self._set_baselines(curves)

    @torch.no_grad()
    def negative_log_likelihood(
        self, features: Tensor, duration: Tensor, event: Tensor
    ) -> float:
        """Mean negative log-likelihood of the fitted mixture.

        Uses the interpolated baselines, with each group's likelihood floored
        at ``1e-10`` (see the class docstring). :meth:`fit` uses it as the
        validation loss for early stopping.

        Args:
            features: Dense features, shape ``(n, input_dim)``.
            duration: Strictly positive times, shape ``(n,)``.
            event: 1 for an event, 0 for censoring, shape ``(n,)``.

        Returns:
            The mean negative log-likelihood per patient.

        Examples:
            >>> import torch
            >>> model = DeepCoxMixtures(None, input_dim=2, num_components=1,
            ...                         hidden_dims=())
            >>> x, t, e = torch.randn(6, 2), torch.arange(1., 7.), torch.ones(6)
            >>> model.fit_baselines(x, t, e)
            >>> model.negative_log_likelihood(x, t, e) > 0
            True
        """
        self._require_fitted()
        x = self._features(features)
        duration = _as_vector(duration, "duration", self.device)
        event = _as_vector(event, "event", self.device)
        _check_labels(duration, event)
        log_gate, log_risk = self._network(x)
        joint = log_gate.to(torch.float64) + self._component_log_likelihood(
            log_risk, duration, event
        )
        return float(-torch.logsumexp(joint, dim=-1).mean())

    def fit(
        self,
        train_data: Any,
        val_data: Any | None = None,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int | None = 2,
        baseline_update_every: int = 1,
        seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Train with the paper's Monte Carlo EM (Algorithm 1).

        Each epoch shuffles the training data. For every mini-batch: E-step
        responsibilities under the current baselines, hard assignments
        sampled from them, one Adam step on the negative Q-function, and
        (every ``baseline_update_every`` batches) Breslow baselines refit on
        the full training set. With ``val_data``, training stops after
        ``patience`` epochs without a lower validation negative
        log-likelihood, and the best weights and baselines are restored.

        Args:
            train_data: ``(features, duration, event)``, a mapping with the
                model's keys, or a dataset (for example a ``SampleDataset``)
                of samples with those keys. Features must be finite.
            val_data: Optional validation data in the same formats.
            epochs: Maximum number of epochs (the paper's code uses 50).
            batch_size: Mini-batch size (the paper uses 128).
            learning_rate: Adam learning rate (the paper uses 1e-3).
            weight_decay: Adam weight decay. Default 0, as in the paper.
            patience: Early-stopping patience in epochs. ``None`` or no
                ``val_data`` trains for all epochs.
            baseline_update_every: Refit the baselines after every this many
                mini-batches. The authors' code refits after every batch.
            seed: Seed for shuffling and the sampled assignments.

        Returns:
            History with ``train_loss`` (mean M-step loss per epoch) and,
            with validation data, ``val_nll`` per epoch.

        Examples:
            >>> import torch
            >>> model = DeepCoxMixtures(None, input_dim=3, num_components=2,
            ...                         hidden_dims=(4,))
            >>> data = (torch.randn(30, 3), torch.rand(30) + 0.1,
            ...         torch.ones(30))
            >>> sorted(model.fit(data, data, epochs=1, seed=0))
            ['train_loss', 'val_nll']
        """
        for name, value in (
            ("epochs", epochs),
            ("batch_size", batch_size),
            ("baseline_update_every", baseline_update_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if patience is not None and (
            isinstance(patience, bool) or not isinstance(patience, int) or patience < 1
        ):
            raise ValueError("patience must be a positive integer or None")
        x, duration, event = self._collect(train_data)
        validation = None if val_data is None else self._collect(val_data)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._knots = self._cumulative_hazard = None
        history: dict[str, list[float]] = {"train_loss": []}
        if validation is not None:
            history["val_nll"] = []
        best_loss, best_state, stale = math.inf, None, 0
        n = len(x)
        for _ in range(epochs):
            self.train()
            order = torch.randperm(n, generator=generator).to(self.device)
            total = 0.0
            for batch, start in enumerate(range(0, n, batch_size)):
                index = order[start : start + batch_size]
                xb, tb, eb = x[index], duration[index], event[index]
                log_gate, log_risk = self._network(xb)
                log_posterior = self._log_posterior(log_gate, log_risk, tb, eb)
                assignment = self._sample_assignment(log_posterior, generator)
                loss = self._expected_complete_loss(
                    log_gate, log_risk, tb, eb, log_posterior, assignment
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
                if batch % baseline_update_every == 0:
                    self.fit_baselines(x, duration, event, generator)
            history["train_loss"].append(total / n)
            self.eval()
            if validation is None:
                continue
            val_loss = self.negative_log_likelihood(*validation)
            history["val_nll"].append(val_loss)
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return history

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_survival(
        self, features: Tensor, times: Sequence[float] | Tensor
    ) -> Tensor:
        """Survival probabilities ``S(t | x)`` at the given times.

        Times after a group's last training event time use that group's
        last baseline value (no extrapolation).

        Args:
            features: Dense features, shape ``(n, input_dim)``.
            times: Non-negative times, shape ``(T,)``, in any order.

        Returns:
            Survival probabilities in [0, 1], shape ``(n, T)``.

        Examples:
            >>> import torch
            >>> model = DeepCoxMixtures(None, input_dim=2, num_components=2,
            ...                         hidden_dims=())
            >>> x, t, e = torch.randn(8, 2), torch.arange(1., 9.), torch.ones(8)
            >>> model.fit_baselines(x, t, e)
            >>> model.predict_survival(x, [2.0, 4.0]).shape
            torch.Size([8, 2])
        """
        self._require_fitted()
        x = self._features(features)
        times = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        times = times.reshape(-1)
        if not torch.isfinite(times).all() or (times < 0).any():
            raise ValueError("times must be finite and non-negative")
        log_gate, log_risk = self._network(x)
        return self._mixture_survival(log_gate, log_risk, times)

    @torch.no_grad()
    def predict_latent(self, features: Tensor) -> Tensor:
        """Latent-group probabilities ``P(Z = k | x)``.

        Args:
            features: Dense features, shape ``(n, input_dim)``.

        Returns:
            Probabilities of shape ``(n, K)``; each row sums to 1.

        Examples:
            >>> import torch
            >>> model = DeepCoxMixtures(None, input_dim=2, num_components=3,
            ...                         hidden_dims=(4,))
            >>> model.predict_latent(torch.zeros(2, 2)).shape
            torch.Size([2, 3])
        """
        log_gate, _ = self._network(self._features(features))
        return log_gate.exp()

    @torch.no_grad()
    def predict_log_risk(self, features: Tensor) -> Tensor:
        """Per-group log hazard ratios ``f_k(x)``.

        Args:
            features: Dense features, shape ``(n, input_dim)``.

        Returns:
            Log hazard ratios of shape ``(n, K)``.

        Examples:
            >>> import torch
            >>> model = DeepCoxMixtures(None, input_dim=2, num_components=3,
            ...                         hidden_dims=(4,))
            >>> model.predict_log_risk(torch.zeros(2, 2)).shape
            torch.Size([2, 3])
        """
        _, log_risk = self._network(self._features(features))
        return log_risk
