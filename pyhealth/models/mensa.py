# Author: Zhenshuo Xu (zx65).
# MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
# Likelihood Estimation, Lillelund et al. https://arxiv.org/abs/2409.06525
# Adapted from thecml/mensa (MIT License, commit 6b219cd),
# Copyright (c) 2024 Christian Lillelund. Reimplemented natively; no code
# from that repository is imported.
"""MENSA model, with its pure Weibull-mixture functions and losses.

The module-level functions are stateless: tensors in, tensors out.
:class:`MENSA` wraps them in a PyHealth model. States follow the
reference code's order: 0 is the event-free state (training only), and
1..K are the observed events (for Rotterdam, 1 = recurrence, 2 = death).

Each mixture component is a Weibull distribution with log shape ``k`` and
log rate ``b``, so shape ``rho = exp(k)`` and rate ``lam = exp(b)``::

    log S(t) = -(lam * t) ** rho
    log f(t) = k + b + (rho - 1) * (b + log t) + log S(t)

Components are mixed with weights ``softmax(gate / temperature)``.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor

TRAJECTORY_VARIANTS: tuple[str, ...] = ("paper", "reference")

# Numerical guards: log-parameters are capped before exponentiation and times
# are floored before taking logarithms.
MAX_LOG_PARAMETER = 20.0
MIN_TIME = 1e-12
# Cap on the survival exponent (e^b t)^(e^k), in log space. exp overflows
# float32 just above 88.7; beyond the cap survival is still 0 in float32, but
# the gradient stays finite instead of 0 * inf = NaN (see weibull_log_survival).
MAX_SURVIVAL_EXPONENT = 88.0


def _rate_and_shape(log_shape: Tensor, log_rate: Tensor) -> tuple[Tensor, Tensor]:
    """Return the capped log rate and the shape ``exp(min(k, 20))``."""
    capped_log_rate = log_rate.clamp(max=MAX_LOG_PARAMETER)
    shape = torch.exp(log_shape.clamp(max=MAX_LOG_PARAMETER))
    return capped_log_rate, shape


def weibull_log_survival(time: Tensor, log_shape: Tensor, log_rate: Tensor) -> Tensor:
    """Log survival of each Weibull component, ``-(exp(b) t) ** exp(k)``.

    Args:
        time: Non-negative times, broadcastable against the parameters
            (e.g. ``[B, 1]`` against ``[B, K]``).
        log_shape: Log shape ``k`` per component.
        log_rate: Log rate ``b`` per component.

    Returns:
        Log survival per component, with the broadcast shape. Exactly 0
        (survival 1) wherever ``time == 0``. The exponent is capped at
        ``MAX_SURVIVAL_EXPONENT``, so log survival never reaches ``-inf``:
        a component that would overflow, even one with negligible mixture
        weight or in a censored term the loss does not use, would otherwise
        get the gradient ``0 * inf = NaN`` and corrupt every weight. Values
        below the cap are unchanged.

    Examples:
        >>> weibull_log_survival(torch.tensor([2.0]), torch.zeros(1), torch.zeros(1))
        tensor([-2.])
    """
    capped_log_rate, shape = _rate_and_shape(log_shape, log_rate)
    log_time = torch.log(time.clamp(min=MIN_TIME))
    exponent = (shape * (capped_log_rate + log_time)).clamp(max=MAX_SURVIVAL_EXPONENT)
    log_survival = -torch.exp(exponent)
    return torch.where(time > 0, log_survival, torch.zeros_like(log_survival))


def weibull_log_density(time: Tensor, log_shape: Tensor, log_rate: Tensor) -> Tensor:
    """Log density of each Weibull component.

    ``log f = k + b + (exp(k) - 1) * (b + log t) + log S``. Times are
    floored at a tiny positive value, so a zero time gives a large finite
    value rather than an infinite one.

    Args:
        time: Non-negative times, broadcastable against the parameters.
        log_shape: Log shape ``k`` per component.
        log_rate: Log rate ``b`` per component.

    Returns:
        Log density per component, with the broadcast shape.

    Examples:
        >>> weibull_log_density(torch.tensor([2.0]), torch.zeros(1), torch.zeros(1))
        tensor([-2.])
    """
    _, shape = _rate_and_shape(log_shape, log_rate)
    clamped = time.clamp(min=MIN_TIME)
    log_hazard = log_shape + log_rate + (shape - 1) * (log_rate + clamped.log())
    return log_hazard + weibull_log_survival(clamped, log_shape, log_rate)


def mixture_log_weights(gate_logits: Tensor, temperature: float = 1000.0) -> Tensor:
    """Log mixture weights, ``log_softmax(gate / temperature)`` over components.

    Args:
        gate_logits: Unnormalized gate outputs, ``[..., K]``.
        temperature: Positive softmax temperature. The reference uses 1000,
            which keeps the weights close to uniform early in training.

    Returns:
        Log weights, ``[..., K]``, whose exponentials sum to 1.

    Raises:
        ValueError: If the temperature is not positive.

    Examples:
        >>> mixture_log_weights(torch.tensor([0.0, 0.0])).exp()
        tensor([0.5000, 0.5000])
    """
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    return torch.log_softmax(gate_logits / temperature, dim=-1)


def mixture_log_survival(
    time: Tensor,
    log_shape: Tensor,
    log_rate: Tensor,
    gate_logits: Tensor,
    temperature: float = 1000.0,
) -> Tensor:
    """Log survival of a Weibull mixture, ``logsumexp(log pi + log S_j)``.

    Args:
        time: Times with shape ``[...]``; a component axis is added.
        log_shape: Log shapes, ``[..., K]``.
        log_rate: Log rates, ``[..., K]``.
        gate_logits: Unnormalized gate outputs, ``[..., K]``.
        temperature: Softmax temperature for the gate.

    Returns:
        Mixture log survival with shape ``[...]``; exactly 0 where
        ``time == 0`` and never above 0. When every component is still
        near 1, float32 rounding in ``logsumexp`` can return a value one
        step above 0 (survival 1.0000001); capping removes it.

    Examples:
        Two equal unit-rate exponential components (shape 1, rate 1):

        >>> zeros = torch.zeros(2, 2)
        >>> mixture_log_survival(torch.tensor([0.0, 1.0]), zeros, zeros, zeros).exp()
        tensor([1.0000, 0.3679])
    """
    log_weights = mixture_log_weights(gate_logits, temperature)
    components = weibull_log_survival(time.unsqueeze(-1), log_shape, log_rate)
    log_survival = torch.logsumexp(log_weights + components, dim=-1).clamp(max=0.0)
    return torch.where(time > 0, log_survival, torch.zeros_like(log_survival))


def mixture_log_density(
    time: Tensor,
    log_shape: Tensor,
    log_rate: Tensor,
    gate_logits: Tensor,
    temperature: float = 1000.0,
) -> Tensor:
    """Log density of a Weibull mixture, ``logsumexp(log pi + log f_j)``.

    Args:
        time: Times with shape ``[...]``; a component axis is added.
        log_shape: Log shapes, ``[..., K]``.
        log_rate: Log rates, ``[..., K]``.
        gate_logits: Unnormalized gate outputs, ``[..., K]``.
        temperature: Softmax temperature for the gate.

    Returns:
        Mixture log density with shape ``[...]``.

    Examples:
        >>> zeros = torch.zeros(1, 2)
        >>> mixture_log_density(torch.tensor([1.0]), zeros, zeros, zeros).exp()
        tensor([0.3679])
    """
    log_weights = mixture_log_weights(gate_logits, temperature)
    components = weibull_log_density(time.unsqueeze(-1), log_shape, log_rate)
    return torch.logsumexp(log_weights + components, dim=-1)


def extend_labels(
    event_times: Tensor, event_indicators: Tensor
) -> tuple[Tensor, Tensor]:
    """Prepend the event-free state to per-event times and indicators.

    The event-free state's time is the patient's largest observed time,
    and its indicator is 1 only when no event was observed. The input
    times are copied unchanged into columns 1..K.

    Args:
        event_times: ``[B, K]`` times, one column per event.
        event_indicators: ``[B, K]`` 0/1 indicators in the same order.

    Returns:
        ``(times, indicators)``, each ``[B, K + 1]`` with state 0 first.

    Raises:
        ValueError: If the inputs are not matching 2-D tensors.

    Examples:
        >>> t, e = extend_labels(torch.tensor([[1200.0, 2100.0]]),
        ...                      torch.tensor([[0.0, 1.0]]))
        >>> t.tolist(), e.tolist()
        ([[2100.0, 1200.0, 2100.0]], [[0.0, 0.0, 1.0]])
    """
    if event_times.ndim != 2 or event_times.shape != event_indicators.shape:
        raise ValueError("event_times and event_indicators must both be [B, K]")
    event_free_time = event_times.max(dim=1, keepdim=True).values
    event_free = (event_indicators.sum(dim=1, keepdim=True) == 0).to(
        event_indicators.dtype
    )
    return (
        torch.cat([event_free_time, event_times], dim=1),
        torch.cat([event_free, event_indicators], dim=1),
    )


def compute_state_weights(train_indicators: Tensor) -> Tensor:
    """Inverse-frequency state weights, rescaled to sum to the state count.

    ``w_p = 1 / (n_p + 1e-8)`` where ``n_p`` is the number of training
    patients with state ``p`` observed, then ``w * P / sum(w)``. Compute
    these from the training split only.

    Args:
        train_indicators: ``[N, P]`` extended 0/1 indicators, as returned
            by :func:`extend_labels`.

    Returns:
        ``[P]`` weights summing to ``P``.

    Examples:
        >>> compute_state_weights(torch.tensor([[1.0, 0, 0], [0, 1, 1], [1, 0, 1]]))
        tensor([0.7500, 1.5000, 0.7500])
    """
    counts = train_indicators.sum(dim=0).to(torch.float32)
    weights = 1.0 / (counts + 1e-8)
    return weights / weights.sum() * counts.numel()


def survival_nll(
    log_density: Tensor,
    log_survival: Tensor,
    indicators: Tensor,
    state_weights: Tensor | None = None,
) -> Tensor:
    """Weighted censored negative log-likelihood over all states.

    ``-(1/N) sum_p w_p sum_i [d_ip log f_p(t_ip) + (1 - d_ip) log S_p(t_ip)]``.
    Each patient contributes the density for observed states and the
    survival for censored ones. The unused term is never multiplied in, so
    an infinite unused value cannot produce NaN.

    Args:
        log_density: ``[B, P]`` mixture log density at each state's time.
        log_survival: ``[B, P]`` mixture log survival at each state's time.
        indicators: ``[B, P]`` 0/1 observed indicators.
        state_weights: Optional ``[P]`` weights; all ones when omitted.

    Returns:
        Scalar loss averaged over the batch.

    Examples:
        One patient, state 0 observed (density 0.5) and state 1 censored
        (survival 0.4), so the loss is ``-(log 0.5 + log 0.4)``:

        >>> log_f = torch.tensor([[0.5, 0.2]]).log()
        >>> log_s = torch.tensor([[0.8, 0.4]]).log()
        >>> survival_nll(log_f, log_s, torch.tensor([[1.0, 0.0]]))
        tensor(1.6094)
    """
    observed = indicators == 1
    log_likelihood = torch.where(observed, log_density, log_survival)
    if state_weights is not None:
        log_likelihood = log_likelihood * state_weights
    return -log_likelihood.sum() / indicators.shape[0]


def trajectory_nll(
    log_survival_later: Tensor,
    indicators: Tensor,
    earlier: int = 1,
    later: int = 2,
    variant: str = "paper",
) -> Tensor:
    """Penalty for predicting the later event before the earlier one.

    Applies to patients who had both events. With ``variant="paper"``
    (Eq. 1 of the proposal), pass ``log S_later(t_earlier)``: the model
    should say the later event had not yet happened when the earlier one
    did. The sum is divided by the full batch size. With
    ``variant="reference"`` (the authors' code), pass
    ``log S_later(t_later)``; the loss is the mean over patients with both
    events, or 0 when there are none.

    Args:
        log_survival_later: ``[B]`` log survival of the later event at the
            time the variant requires.
        indicators: ``[B, P]`` extended 0/1 indicators.
        earlier: State index of the earlier event (1 = recurrence).
        later: State index of the later event (2 = death).
        variant: ``"paper"`` or ``"reference"``.

    Returns:
        Scalar penalty; 0 when no patient in the batch had both events.

    Raises:
        ValueError: If ``variant`` is unknown.

    Examples:
        Two patients; only the first had both events, with
        ``S_death(t_recurrence) = 0.9``, so the paper variant gives
        ``-log(0.9) / 2``:

        >>> indicators = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        >>> log_s_later = torch.tensor([0.9, 0.5]).log()
        >>> trajectory_nll(log_s_later, indicators)
        tensor(0.0527)
        >>> trajectory_nll(log_s_later, indicators, variant="reference")
        tensor(0.1054)
    """
    if variant not in TRAJECTORY_VARIANTS:
        raise ValueError(f"variant must be one of {list(TRAJECTORY_VARIANTS)}")
    both = (indicators[:, earlier] == 1) & (indicators[:, later] == 1)
    selected = torch.where(
        both, log_survival_later, torch.zeros_like(log_survival_later)
    )
    if variant == "paper":
        return -selected.sum() / indicators.shape[0]
    count = both.sum()
    return -selected.sum() / count.clamp(min=1)


def mensa_loss(
    log_density: Tensor,
    log_survival: Tensor,
    indicators: Tensor,
    log_survival_later: Tensor,
    state_weights: Tensor | None = None,
    trajectory_weight: float = 0.25,
    variant: str = "paper",
    earlier: int = 1,
    later: int = 2,
) -> Tensor:
    """Total MENSA loss, ``(1 - lambda) * L_surv + lambda * L_traj``.

    Args:
        log_density: ``[B, P]`` mixture log density at each state's time.
        log_survival: ``[B, P]`` mixture log survival at each state's time.
        indicators: ``[B, P]`` extended 0/1 indicators.
        log_survival_later: ``[B]`` input for :func:`trajectory_nll`.
        state_weights: Optional ``[P]`` weights from the training split.
        trajectory_weight: ``lambda`` in ``[0, 1]``; 0.25 in the paper's
            Rotterdam setting.
        variant: Trajectory variant, ``"paper"`` or ``"reference"``.
        earlier: State index of the earlier event.
        later: State index of the later event.

    Returns:
        Scalar loss.

    Raises:
        ValueError: If ``trajectory_weight`` is outside ``[0, 1]`` or
            ``variant`` is unknown.

    Examples:
        >>> indicators = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        >>> log_f = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]]).log()
        >>> log_s = torch.tensor([[0.9, 0.5, 0.6], [0.8, 0.4, 0.7]]).log()
        >>> log_s_later = torch.tensor([0.9, 0.5]).log()
        >>> mensa_loss(log_f, log_s, indicators, log_s_later)
        tensor(1.7766)
        >>> mensa_loss(log_f, log_s, indicators, log_s_later, trajectory_weight=0.0)
        tensor(2.3513)
    """
    if not 0.0 <= trajectory_weight <= 1.0:
        raise ValueError("trajectory_weight must be in [0, 1]")
    surv = survival_nll(log_density, log_survival, indicators, state_weights)
    traj = trajectory_nll(log_survival_later, indicators, earlier, later, variant)
    return (1.0 - trajectory_weight) * surv + trajectory_weight * traj


def _value_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Return the primary tensor emitted by a PyHealth processor."""
    return value[0] if isinstance(value, tuple) else value


class _MENSANetwork(nn.Module):
    """Shared trunk with a residual adapter and Weibull heads per state.

    Follows the reference architecture: ``Linear -> BatchNorm1d -> ReLU6 ->
    Dropout`` blocks, then for each state ``h_p = h + adapter_p(h)`` and
    linear shape, rate and gate heads. Shape and rate are
    ``SELU(head(h_p)) + base``, where the per-state base offsets start at -1.

    Args:
        input_dim: Number of input features.
        hidden_dims: Widths of the shared hidden layers.
        n_dists: Weibull components per state.
        n_states: Number of states, including the event-free state.
        dropout: Dropout probability after each shared layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        n_dists: int,
        n_states: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            blocks += [
                nn.Linear(width, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU6(),
                nn.Dropout(dropout),
            ]
            width = hidden
        self.embedding = nn.Sequential(*blocks)
        adapter_width = max(16, width // 2)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, adapter_width),
                nn.ReLU6(),
                nn.Linear(adapter_width, width),
            )
            for _ in range(n_states)
        )
        self.shape_heads = nn.ModuleList(
            nn.Linear(width, n_dists) for _ in range(n_states)
        )
        self.rate_heads = nn.ModuleList(
            nn.Linear(width, n_dists) for _ in range(n_states)
        )
        self.gate_heads = nn.ModuleList(
            nn.Linear(width, n_dists, bias=False) for _ in range(n_states)
        )
        self.base_log_shape = nn.Parameter(-torch.ones(n_states, n_dists))
        self.base_log_rate = nn.Parameter(-torch.ones(n_states, n_dists))
        self.activation = nn.SELU()

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Map standardized features to mixture parameters.

        Args:
            features: ``[B, F]`` standardized features.

        Returns:
            ``(log_shape, log_rate, gate_logits)``, each ``[B, P, K]``. Gate
            logits are raw; the mixture functions apply the temperature.
        """
        shared = self.embedding(features)
        log_shapes, log_rates, gates = [], [], []
        for p, adapter in enumerate(self.adapters):
            state = shared + adapter(shared)
            log_shapes.append(
                self.activation(self.shape_heads[p](state)) + self.base_log_shape[p]
            )
            log_rates.append(
                self.activation(self.rate_heads[p](state)) + self.base_log_rate[p]
            )
            gates.append(self.gate_heads[p](state))
        return (
            torch.stack(log_shapes, dim=1),
            torch.stack(log_rates, dim=1),
            torch.stack(gates, dim=1),
        )


class MENSA(BaseModel):
    """Multi-event survival network with a trajectory-based likelihood.

    MENSA (Lillelund et al.) predicts a survival curve for each of several
    events from one shared network. Each event, plus an event-free state used
    only in training, gets a mixture of Weibull distributions. Training
    minimizes ``(1 - lambda) * L_surv + lambda * L_traj``: a weighted
    censored likelihood over all states, plus a penalty when a patient who
    had both events is predicted to have the later event before the earlier
    one (:func:`mensa_loss`).

    Works with the ``"tensor"`` outputs of
    :class:`~pyhealth.tasks.MultiEventSurvivalRotterdam`: ``features [F]``,
    ``event_times [K]`` and ``event_indicators [K]``, in source time units.
    Like :class:`~pyhealth.models.SurvivalMDN`, it has no label processor, so
    ``mode`` is ``None`` and PyHealth's Trainer tracks only the loss. Compute
    survival metrics from :meth:`predict_survival` or ``y_prob``.

    Before training, call :meth:`fit_preprocessing` with the **training
    split only**. It fits the feature standardization, the state weights and
    the time grid used for ``y_prob``, and stores them as buffers, so they
    are saved and restored with the model's ``state_dict``.

    Defaults follow this reproduction's Rotterdam setting: one hidden layer
    of 32, dropout 0.25, three components and ``lambda = 0.25``, trained with
    batch size 32, learning rate 1e-3, weight decay 1e-3 and gradient
    clipping at norm 1.0, e.g.
    ``get_dataloader(train, batch_size=32, shuffle=True, drop_last=True)``
    and ``Trainer.train(...)`` with ``optimizer_params={"lr": 1e-3}``,
    ``weight_decay=1e-3`` and ``max_grad_norm=1.0``. ``drop_last`` matters
    because BatchNorm cannot train on a batch of one. The reference code's
    Rotterdam config (``configs/mensa/rotterdam.yaml`` in thecml/mensa,
    commit 6b219cd) instead uses ``hidden_dims=(16,)``,
    ``dropout=0.5`` and weight decay 1e-4, with the same components,
    ``lambda = 0.25``, batch size, learning rate and up to 1000 epochs; pass
    those to match it.

    Args:
        dataset: Sample dataset defining the schema; used to check keys and
            infer the number of features and events.
        feature_key: Input key of the dense feature vector.
        times_key: Output key of the per-event times.
        indicators_key: Output key of the per-event 0/1 indicators.
        hidden_dims: Widths of the shared hidden layers.
        n_dists: Weibull components per state.
        dropout: Dropout probability in the shared layers.
        trajectory_weight: ``lambda`` in ``[0, 1]``.
        trajectory_variant: ``"paper"`` (Eq. 1, ``S_later(t_earlier)``) or
            ``"reference"`` (the authors' code, ``S_later(t_later)``).
        trajectory: ``(earlier, later)`` event positions in ``event_times``,
            1-based; ``(1, 2)`` is recurrence before death.
        temperature: Softmax temperature of the mixture gates.
        num_time_bins: Training-time quantiles in the ``y_prob`` grid; the
            grid also starts at 0. The reference uses ``ceil(sqrt(N))``.

    Examples:
        >>> from pyhealth.datasets import RotterdamDataset, split_by_patient
        >>> from pyhealth.models import MENSA
        >>> samples = RotterdamDataset(root="../data").set_task()  # doctest: +SKIP
        >>> ratios = [0.7, 0.1, 0.2]
        >>> train, val, test = split_by_patient(samples, ratios)  # doctest: +SKIP
        >>> model = MENSA(samples)  # doctest: +SKIP
        >>> model.fit_preprocessing(train)  # doctest: +SKIP
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """None: the model scores its survival outputs with its own loss.

        Examples:
            >>> model.label_processors  # doctest: +SKIP
            {}
        """
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        times_key: str = "event_times",
        indicators_key: str = "event_indicators",
        hidden_dims: Sequence[int] = (32,),
        n_dists: int = 3,
        dropout: float = 0.25,
        trajectory_weight: float = 0.25,
        trajectory_variant: str = "paper",
        trajectory: tuple[int, int] = (1, 2),
        temperature: float = 1000.0,
        num_time_bins: int = 50,
    ) -> None:
        super().__init__(dataset)
        if feature_key not in dataset.input_schema:
            raise ValueError(f"feature_key {feature_key!r} is not an input key")
        for key in (times_key, indicators_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in the dataset")
        if not 0.0 <= trajectory_weight <= 1.0:
            raise ValueError("trajectory_weight must be in [0, 1]")
        if trajectory_variant not in TRAJECTORY_VARIANTS:
            raise ValueError(
                f"trajectory_variant must be one of {list(TRAJECTORY_VARIANTS)}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if n_dists < 1 or num_time_bins < 2 or not temperature > 0:
            raise ValueError(
                "n_dists must be >= 1, num_time_bins >= 2 and temperature > 0"
            )
        sample = dataset[0]
        features = torch.as_tensor(_value_tensor(sample[feature_key]))
        times = torch.as_tensor(_value_tensor(sample[times_key]))
        if features.ndim != 1 or times.ndim != 1:
            raise ValueError("features and event times must be 1-D per sample")
        self.feature_key = feature_key
        self.times_key = times_key
        self.indicators_key = indicators_key
        self.input_dim = int(features.shape[0])
        self.n_events = int(times.shape[0])
        self.n_states = self.n_events + 1
        earlier, later = trajectory
        if not 1 <= earlier <= self.n_events or not 1 <= later <= self.n_events:
            raise ValueError(f"trajectory must name events 1..{self.n_events}")
        if earlier == later:
            raise ValueError("trajectory needs two different events")
        self.trajectory = (int(earlier), int(later))
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.n_dists = int(n_dists)
        self.dropout = float(dropout)
        self.trajectory_weight = float(trajectory_weight)
        self.trajectory_variant = trajectory_variant
        self.temperature = float(temperature)
        self.network = _MENSANetwork(
            self.input_dim, self.hidden_dims, self.n_dists, self.n_states, dropout
        )
        # Fixed-size buffers, filled by fit_preprocessing and kept in state_dict.
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_scale", torch.ones(self.input_dim))
        self.register_buffer("state_weights", torch.ones(self.n_states))
        self.register_buffer("time_grid", torch.zeros(num_time_bins + 1))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_preprocessing(self, train_dataset: SampleDataset) -> None:
        """Fit standardization, state weights and the time grid on training data.

        Feature means and population standard deviations (constant features
        get scale 1), inverse-frequency state weights
        (:func:`compute_state_weights`), and a grid of 0 followed by
        ``num_time_bins`` quantiles of all training event times.

        Args:
            train_dataset: The training split only. Using validation or test
                patients here leaks their information into the model.

        Raises:
            ValueError: If the split is empty, has the wrong widths, or holds
                non-finite values or invalid labels.

        Examples:
            The grid is 0 followed by ``num_time_bins`` (50) quantiles:

            >>> model = MENSA(samples)  # doctest: +SKIP
            >>> model.fit_preprocessing(train)  # doctest: +SKIP
            >>> model.time_grid.shape  # doctest: +SKIP
            torch.Size([51])
        """
        features, times, indicators = [], [], []
        for batch in get_dataloader(train_dataset, batch_size=1024, shuffle=False):
            features.append(torch.as_tensor(_value_tensor(batch[self.feature_key])))
            times.append(torch.as_tensor(_value_tensor(batch[self.times_key])))
            indicators.append(
                torch.as_tensor(_value_tensor(batch[self.indicators_key]))
            )
        if not features:
            raise ValueError("train_dataset must contain at least one sample")
        x = torch.cat(features).double()
        t = torch.cat(times).double()
        e = torch.cat(indicators).double()
        if x.shape[1] != self.input_dim or t.shape[1] != self.n_events:
            raise ValueError("train_dataset widths do not match the model")
        self._check_labels(t, e)
        if not torch.isfinite(x).all():
            raise ValueError("training features must be finite")
        scale = x.std(dim=0, unbiased=False)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        levels = torch.linspace(0, 1, self.time_grid.numel() - 1, dtype=torch.float64)
        grid = torch.cat([torch.zeros(1, dtype=torch.float64), t.quantile(levels)])
        _, extended = extend_labels(t, e)
        self.feature_mean.copy_(x.mean(dim=0))
        self.feature_scale.copy_(scale)
        self.state_weights.copy_(compute_state_weights(extended))
        self.time_grid.copy_(grid)
        self.is_fitted.fill_(True)

    @staticmethod
    def _check_labels(times: Tensor, indicators: Tensor) -> None:
        """Reject non-finite or non-positive times and non-binary indicators."""
        if times.shape != indicators.shape:
            raise ValueError("event_times and event_indicators must match in shape")
        if not torch.isfinite(times).all() or (times <= 0).any():
            raise ValueError("event times must be finite and strictly positive")
        if not ((indicators == 0) | (indicators == 1)).all():
            raise ValueError("event indicators must be 0 or 1")

    def _mixture(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Standardize features and return per-state mixture parameters."""
        if not bool(self.is_fitted):
            raise RuntimeError(
                "Call fit_preprocessing(train_dataset) before using MENSA"
            )
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must be [batch, {self.input_dim}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] == 0 or not torch.isfinite(features).all():
            raise ValueError("features must be nonempty and finite")
        return self.network((features - self.feature_mean) / self.feature_scale)

    def _event_survival(
        self, log_shape: Tensor, log_rate: Tensor, gate: Tensor, times: Tensor
    ) -> Tensor:
        """Survival of each event (not the event-free state) at shared times."""
        batch, n_times = log_shape.shape[0], times.numel()
        grid = times.reshape(1, 1, n_times).expand(batch, self.n_events, n_times)
        log_survival = mixture_log_survival(
            grid,
            log_shape[:, 1:, None, :],
            log_rate[:, 1:, None, :],
            gate[:, 1:, None, :],
            self.temperature,
        )
        return log_survival.exp()

    def forward(self, **kwargs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Compute the loss and survival curves for one batch.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader. The feature
                key is required; with the times and indicators keys present,
                the loss and ``y_true`` are also returned.

        Returns:
            ``y_prob``: ``[B, K, T]`` survival of each event on
            :attr:`time_grid`. With labels, also ``loss`` (scalar) and
            ``y_true``: ``[B, 2K]`` as event times then indicators.

        Raises:
            RuntimeError: If :meth:`fit_preprocessing` has not been called.
            KeyError: If the feature key, or only one label key, is missing.
            ValueError: For malformed features or labels.

        Examples:
            >>> from pyhealth.datasets import get_dataloader
            >>> loader = get_dataloader(test, batch_size=64)  # doctest: +SKIP
            >>> batch = next(iter(loader))  # doctest: +SKIP
            >>> out = model(**batch)  # doctest: +SKIP
            >>> sorted(out), out["y_prob"].shape  # doctest: +SKIP
            (['loss', 'y_prob', 'y_true'], torch.Size([64, 2, 51]))
        """
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing required batch field: {self.feature_key!r}")
        log_shape, log_rate, gate = self._mixture(
            _value_tensor(kwargs[self.feature_key])
        )
        output = {
            "y_prob": self._event_survival(log_shape, log_rate, gate, self.time_grid)
        }
        has_labels = [k in kwargs for k in (self.times_key, self.indicators_key)]
        if not any(has_labels):
            return output
        if not all(has_labels):
            raise KeyError("Provide both event times and event indicators, or neither")
        times = torch.as_tensor(
            _value_tensor(kwargs[self.times_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, self.n_events)
        indicators = torch.as_tensor(
            _value_tensor(kwargs[self.indicators_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, self.n_events)
        self._check_labels(times, indicators)
        ext_times, ext_indicators = extend_labels(times, indicators)
        log_density = mixture_log_density(
            ext_times, log_shape, log_rate, gate, self.temperature
        )
        log_survival = mixture_log_survival(
            ext_times, log_shape, log_rate, gate, self.temperature
        )
        earlier, later = self.trajectory
        if self.trajectory_variant == "paper":
            log_survival_later = mixture_log_survival(
                ext_times[:, earlier],
                log_shape[:, later],
                log_rate[:, later],
                gate[:, later],
                self.temperature,
            )
        else:
            log_survival_later = log_survival[:, later]
        output["loss"] = mensa_loss(
            log_density,
            log_survival,
            ext_indicators,
            log_survival_later,
            self.state_weights,
            self.trajectory_weight,
            self.trajectory_variant,
            earlier,
            later,
        )
        output["y_true"] = torch.cat([times, indicators], dim=1)
        return output

    @torch.no_grad()
    def predict_survival(self, features: Tensor, times: Tensor) -> Tensor:
        """Predict each event's survival probability at the given times.

        Runs in evaluation mode without gradients, so BatchNorm statistics
        and dropout are not affected, and restores every module's previous
        mode afterwards.

        Args:
            features: ``[B, F]`` raw (unstandardized) features.
            times: ``[T]`` non-negative times, in the label units.

        Returns:
            ``[B, K, T]`` survival probabilities, non-increasing in time,
            equal to 1 at time 0.

        Raises:
            RuntimeError: If :meth:`fit_preprocessing` has not been called.
            ValueError: For malformed features or negative times.

        Examples:
            >>> times = torch.linspace(0, 7000, 50)
            >>> features = batch["features"]  # doctest: +SKIP
            >>> curves = model.predict_survival(features, times)  # doctest: +SKIP
            >>> curves.shape  # doctest: +SKIP
            torch.Size([64, 2, 50])
        """
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        times = times.reshape(-1)
        if not torch.isfinite(times).all() or (times < 0).any():
            raise ValueError("times must be finite and non-negative")
        previous_modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            log_shape, log_rate, gate = self._mixture(features)
            return self._event_survival(log_shape, log_rate, gate, times)
        finally:
            for module, training in previous_modes:
                module.training = training
