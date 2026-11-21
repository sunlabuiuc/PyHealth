# Contributors: Jenny Nguyen (lmn4@illinois.edu),
#   Jordan Alwan (jzalwan2@illinois.edu), Karthik Pullela (kpull2@illinois.edu)
"""Deep Cox Mixtures for right-censored survival data — Decision 2
option (b): the EM loop lives in the example script.


This module ports Deep Cox Mixtures (DCM) from Nagpal, Yadlowsky,
Rostamzadeh and Heller, "Deep Cox Mixtures for Survival Regression"
(MLHC 2021), into PyHealth. The network and the per-group baseline
fitting are ported from the authors' reference implementation in
auton-survival (``auton_survival/models/dcm/``); the EM training driver
itself lives in the example script, not in this file.

Paper: https://proceedings.mlr.press/v149/nagpal21a.html
Reference code: https://github.com/autonlab/auton-survival

Training design (checklist Decision 2, recommended option b — still
open, awaiting team confirmation): the model is a pure network — a
shared MLP encoder plus a gate head and an expert (log hazard ratio)
head — with helper methods for the E-step, the Q-loss, and the
Breslow/spline baseline fit. It is NOT trainable through
``Trainer.train()``: DCM's baselines are dataset-level (risk sets over
all patients) and cannot be computed inside a minibatch ``forward``,
so the example script owns the EM loop, refits the baselines, and
calls :meth:`DeepCoxMixturesScriptEM.fit_baselines`. ``forward`` therefore
requires fitted baselines and raises a clear error until they exist.
The fitted splines are not ``nn.Parameter``s and are not in the
module's ``state_dict()`` — a saved checkpoint alone cannot predict;
refit the baselines on load (or pickle them alongside the checkpoint).

Preprocessing (checklist Decision 3, recommendation a+c — still open,
awaiting team confirmation): this model is dataset-agnostic and takes
``input_dim`` as a constructor argument, so it accepts PyHealth's
existing SUPPORT task (27 features, 8,873 patients) and FLCHAIN's
26-feature task alike. The paper reports 44 features / 9,105
patients on SUPPORT with no preprocessing spec; the existing
``SurvivalTimeToEventSupport2`` task is reused as-is and the
difference is documented, not claimed as an exact match.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from sksurv.linear_model.coxph import BreslowEstimator
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor

__all__ = ["DeepCoxMixturesScriptEM", "DeepCoxMixturesTorch"]


def _value_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Extract the primary value emitted by a PyHealth processor."""
    if isinstance(value, tuple):
        return value[0]
    return value


def _infer_input_dim(dataset: SampleDataset, feature_key: str) -> int:
    """Infer a dense vector width from one processed sample."""
    try:
        sample = dataset[0]
    except Exception as exc:  # pragma: no cover - defensive fallback
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
            "DeepCoxMixturesScriptEM expects each sample's feature field to be a "
            "1D dense numeric vector."
        )
    return int(tensor.shape[0])


def _create_representation(input_dim: int, layers: list[int] | None) -> nn.Module:
    """Build the shared MLP encoder, ported from auton-survival.

    Each hidden layer is ``Linear(bias=False)`` followed by ``ReLU6``.
    ``layers`` empty or ``None`` gives the identity map (a linear model).

    >>> _create_representation(3, [4])
    Sequential(
      (0): Linear(in_features=3, out_features=4, bias=False)
      (1): ReLU6()
    )
    """
    if layers is None:
        layers = []
    modules: list[nn.Module] = []
    prev_dim = input_dim
    for hidden in layers:
        modules.append(nn.Linear(prev_dim, hidden, bias=False))
        modules.append(nn.ReLU6())
        prev_dim = hidden
    return nn.Sequential(*modules)


def _partial_ll_loss(log_hazards: Tensor, durations: Tensor, events: Tensor) -> Tensor:
    """Cox partial log-likelihood loss for one latent group.

    Ported from auton-survival's ``partial_ll_loss``. Times are sorted
    descending; the denominator at each event time is the log-sum-exp
    of log hazard ratios over the risk set. A small random jitter is
    added to tied times, as in the reference code, so the sort order
    among ties is not arbitrary-but-fixed.

    >>> loss = _partial_ll_loss(
    ...     torch.tensor([0.1, -0.2, 0.3]),
    ...     torch.tensor([5.0, 3.0, 4.0]),
    ...     torch.tensor([1.0, 0.0, 1.0]),
    ... )
    >>> isinstance(loss, torch.Tensor)
    True
    """
    durations = durations.reshape(-1)
    events = events.reshape(-1)
    jitter = 1e-2 * torch.rand(
        durations.shape[0],
        device=durations.device,
        dtype=durations.dtype,
    )
    order = torch.argsort(-(durations + jitter))
    sorted_hazards = log_hazards[order]
    sorted_events = events[order]
    log_denominator = torch.logcumsumexp(sorted_hazards, dim=0)
    per_sample = sorted_hazards - log_denominator
    return -per_sample[sorted_events == 1].sum()


def _fit_spline(times: np.ndarray, survival: np.ndarray, s: float):
    """Linear spline through a Breslow baseline survival curve."""
    return UnivariateSpline(times, survival, s=s, ext=3, k=1)


def _repair_log_probs(log_probs: Tensor) -> Tensor:
    """Clamp NaN / very negative log-likelihoods, as in auton-survival."""
    log_probs = log_probs.clone()
    log_probs[torch.isnan(log_probs)] = -10.0
    log_probs[log_probs < -10.0] = -10.0
    return log_probs


class DeepCoxMixturesTorch(nn.Module):
    """PyTorch network for Deep Cox Mixtures (the paper's model).

    A shared MLP encoder ``Phi(x)`` feeds two bias-free linear heads:
    a gate head giving ``log P(Z = k | x)`` (``LogSoftmax``) and an
    expert head giving log hazard ratios, clamped to ``+/- gamma`` (or
    ``gamma * tanh(...)`` when ``use_activation`` is set). Ported
    verbatim from auton-survival's ``DeepCoxMixturesTorch``.

    Args:
        input_dim: Width of the dense input vector.
        k: Number of latent mixture groups (``K`` in the paper).
        gamma: Clamp bound for the log hazard ratios.
        use_activation: If ``True``, use ``gamma * tanh`` instead of
            clamping for the expert head.
        layers: Hidden widths of the shared encoder MLP.

    Examples:
        >>> net = DeepCoxMixturesTorch(input_dim=3, k=2, layers=[4])
        >>> log_gate, log_hazards = net(torch.randn(5, 3))
        >>> log_gate.shape
        torch.Size([5, 2])
        >>> log_hazards.shape
        torch.Size([5, 2])
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 3,
        gamma: float = 10.0,
        use_activation: bool = False,
        layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.k = k
        self.gamma = float(gamma)
        self.use_activation = bool(use_activation)
        if layers is None:
            layers = []
        self.layers = list(layers)
        last_dim = layers[-1] if layers else input_dim
        self.embedding = _create_representation(input_dim, layers)
        self.gate = nn.Linear(last_dim, k, bias=False)
        self.expert = nn.Linear(last_dim, k, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(log_gate_prob, log_hazard_ratios)``, both ``(n, k)``."""
        hidden = self.embedding(x)
        if self.use_activation:
            log_hazards = self.gamma * self.tanh(self.expert(hidden))
        else:
            log_hazards = torch.clamp(
                self.expert(hidden), min=-self.gamma, max=self.gamma
            )
        log_gate_prob = self.log_softmax(self.gate(hidden))
        return log_gate_prob, log_hazards


class DeepCoxMixturesScriptEM(BaseModel):
    """Deep Cox Mixtures model for PyHealth, option (b) (see module
    docstring).

    A mixture of ``k`` Cox models with a shared encoder: conditioned on
    latent group ``Z = k`` the proportional-hazards assumption holds,
    and each group's baseline survival is fitted non-parametrically
    with a spline-interpolated Breslow estimator. The model's
    prediction is ``S(t | x) = sum_k P(Z = k | x) * S_k(t) ** exp(f_k(x))``.

    Args:
        dataset: Fitted PyHealth ``SampleDataset``. The model expects
            one dense numeric feature vector and two tensor outputs:
            duration and event.
        feature_key: Input field containing the dense feature vector.
        duration_key: Output field containing strictly positive
            durations.
        event_key: Output field containing ``1`` for an observed event
            and ``0`` for right censoring.
        input_dim: Width of the dense input vector. If omitted, the
            model infers it from the first processed sample.
        k: Number of latent mixture groups. The paper sweeps {3, 4, 6};
            ``k=1`` is the single-Cox control arm for Ablation 1.
        layers: Hidden widths of the shared encoder MLP. ``[100]`` is
            the paper's grid default; ``[]`` is a linear model.
        gamma: Clamp bound for the log hazard ratios (paper default 10).
        use_activation: If ``True``, use ``gamma * tanh`` instead of
            clamping for the expert head.
        smoothing_factor: Smoothing parameter ``s`` for the baseline
            splines (auton-survival default ``1e-4``).
        time_grid: Times at which ``y_prob`` returns survival
            probabilities. If omitted, a 64-point grid from 1 to 365
            (days) is used — override it for your dataset (SUPPORT
            event-time quantiles are 14 / 58 / 252 days; FLCHAIN's
            are 903 / 2085 / 3246 days).

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
        ...     {
        ...         "patient_id": "p2",
        ...         "features": [0.5, -1.0, 0.3],
        ...         "duration": [3.0],
        ...         "event": [1.0],
        ...     },
        ...     {
        ...         "patient_id": "p3",
        ...         "features": [-0.4, 0.7, 1.1],
        ...         "duration": [4.0],
        ...         "event": [0.0],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
        >>> model.k
        2

    References:
        Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K.
        Deep Cox Mixtures for Survival Regression. Proceedings of
        Machine Learning for Healthcare, 2021.
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """None: the survival outputs are scored by the model, not as a
        label, so the Trainer reports only the loss."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        k: int = 3,
        layers: list[int] | None = None,
        gamma: float = 10.0,
        use_activation: bool = False,
        smoothing_factor: float = 1e-4,
        time_grid: Tensor | None = None,
    ) -> None:
        super().__init__(dataset)
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.k = int(k)
        self.smoothing_factor = float(smoothing_factor)

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
        if layers is None:
            layers = [100]
        self.network = DeepCoxMixturesTorch(
            input_dim=self.input_dim,
            k=self.k,
            gamma=gamma,
            use_activation=use_activation,
            layers=list(layers),
        )
        # Fitted per-group baseline splines, set by fit_baselines().
        # Not nn.Parameters and not in state_dict(): see module docstring.
        self.breslow_splines: dict[int, UnivariateSpline] | None = None
        if time_grid is None:
            time_grid = torch.linspace(1.0, 365.0, 64)
        time_grid = torch.as_tensor(time_grid, dtype=torch.float32)
        if (
            time_grid.numel() == 0
            or not torch.isfinite(time_grid).all()
            or torch.any(time_grid <= 0)
        ):
            raise ValueError("time_grid must contain finite strictly positive values")
        self.register_buffer("time_grid", time_grid.reshape(-1))

    # -- Network outputs -------------------------------------------------
    def encode(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(log_gate_prob, log_hazard_ratios)``, both ``(n, k)``.

        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(
        ...     samples=[{"patient_id": "p0", "features": [0.0, 1.0],
        ...               "duration": [1.0], "event": [1.0]}],
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
        >>> log_gate, log_hazards = model.encode(torch.randn(3, 2))
        >>> log_gate.shape
        torch.Size([3, 2])
        """
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must have shape [batch, input_dim]; "
                f"received {tuple(features.shape)}"
            )
        return self.network(features)

    @torch.no_grad()
    def predict_latent_z(self, features: Tensor) -> Tensor:
        """Return gate probabilities ``P(Z = k | x)`` with shape ``(n, k)``.

        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(
        ...     samples=[{"patient_id": "p0", "features": [0.0, 1.0],
        ...               "duration": [1.0], "event": [1.0]}],
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
        >>> probs = model.predict_latent_z(torch.randn(3, 2))
        >>> probs.shape
        torch.Size([3, 2])
        """
        log_gate, _ = self.encode(features)
        return log_gate.exp()

    # -- Baselines --------------------------------------------------------
    def fit_baselines(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
        posteriors: Tensor | None = None,
        smoothing_factor: float | None = None,
        typ: str = "soft",
    ) -> dict[int, UnivariateSpline]:
        """Fit per-group Breslow baselines and store their splines.

        Ported from auton-survival's ``fit_breslow``. Each patient is
        assigned to one group — sampled from the posteriors when
        ``typ='soft'`` (auton-survival's default, which is a *sampled
        hard* assignment, not a true soft one) or by argmax when
        ``typ='hard'`` — and each group's baseline is a
        ``BreslowEstimator`` fit on that group's patients, smoothed
        with a linear ``UnivariateSpline``. A group with no assigned
        patients gets a degenerate spline that is identically 1.0
        (certain survival), so a later prediction cannot crash on an
        empty group; the reference code would instead raise.

        Args:
            features: Dense input tensor with shape ``(n, input_dim)``.
            durations: Observed times with shape ``(n,)``.
            events: Event indicators (1 = event, 0 = censored).
            posteriors: Log posterior group probabilities with shape
                ``(n, k)``. If ``None``, the gate log-probabilities
                are used.
            smoothing_factor: Spline smoothing ``s``; defaults to the
                model's ``smoothing_factor``.
            typ: ``'soft'`` (sampled assignment) or ``'hard'`` (argmax).

        Returns:
            The fitted ``{group: UnivariateSpline}`` dict, also stored
            on ``self.breslow_splines`` for :meth:`forward` and
            :meth:`predict_survival`.
        """
        if smoothing_factor is None:
            smoothing_factor = self.smoothing_factor
        if typ not in ("soft", "hard"):
            raise ValueError("typ must be 'soft' or 'hard'")
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        durations = torch.as_tensor(
            durations, dtype=torch.float32, device=self.device
        ).reshape(-1)
        events = torch.as_tensor(
            events, dtype=torch.float32, device=self.device
        ).reshape(-1)
        n = features.shape[0]
        if durations.shape[0] != n or events.shape[0] != n:
            raise ValueError("features, durations and events must agree on n")
        with torch.no_grad():
            log_gate, log_hazards = self.network(features)
        if posteriors is None:
            assignment_probs = log_gate
        else:
            assignment_probs = torch.as_tensor(
                posteriors, dtype=torch.float32, device=self.device
            )
            if assignment_probs.shape != (n, self.k):
                raise ValueError(
                    "posteriors must have shape (n, k); "
                    f"received {tuple(assignment_probs.shape)}"
                )
        if typ == "soft":
            group = torch.multinomial(assignment_probs.exp(), num_samples=1)[:, 0]
        else:
            group = torch.argmax(assignment_probs, dim=1)
        # Breslow/spline fitting is numpy work: move to CPU, no grad.
        hazards_np = log_hazards.detach().cpu().numpy()
        durations_np = durations.detach().cpu().numpy()
        events_np = events.detach().cpu().numpy()
        group_np = group.detach().cpu().numpy()
        splines: dict[int, UnivariateSpline] = {}
        max_time = float(durations_np.max()) if n else 1.0
        for i in range(self.k):
            mask = group_np == i
            if not mask.any() or events_np[mask].sum() == 0:
                splines[i] = _fit_spline(
                    np.array([0.0, max_time]),
                    np.array([1.0, 1.0]),
                    s=smoothing_factor,
                )
                continue
            breslow = BreslowEstimator().fit(
                hazards_np[:, i][mask],
                events_np[mask],
                durations_np[mask],
            )
            baseline = breslow.baseline_survival_
            splines[i] = _fit_spline(baseline.x, baseline.y, s=smoothing_factor)
        self.breslow_splines = splines
        return splines

    def set_baselines(self, splines: dict[int, UnivariateSpline]) -> None:
        """Install previously fitted per-group baseline splines.

        Use this to restore baselines after loading a checkpoint,
        since they are not in ``state_dict()``.
        """
        if set(splines) != set(range(self.k)):
            raise ValueError(f"splines must have exactly the keys 0..{self.k - 1}")
        self.breslow_splines = dict(splines)

    # -- E-step / Q-loss (used by the example script's EM loop) -----------
    @torch.no_grad()
    def compute_posteriors(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
    ) -> Tensor:
        """E-step: log posterior group probabilities, shape ``(n, k)``.

        Ported from auton-survival's ``get_likelihood`` /
        ``get_posteriors``. With no fitted baselines (first EM
        iteration) the posteriors are random, exactly as in the
        reference code. Otherwise, for each patient the per-group
        likelihood is the event density if the event was observed and
        the survival probability if censored, times the gate prior.
        This is numpy/spline work and runs on CPU under ``no_grad``.
        """
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        durations = torch.as_tensor(
            durations, dtype=torch.float32, device=self.device
        ).reshape(-1)
        events = torch.as_tensor(
            events, dtype=torch.float32, device=self.device
        ).reshape(-1)
        n = features.shape[0]
        if self.breslow_splines is None:
            return _repair_log_probs(
                torch.log(
                    torch.rand(n, self.k, device=self.device, dtype=torch.float32)
                )
            )
        log_gate, log_hazards = self.network(features)
        hazards_np = log_hazards.detach().cpu().numpy()
        durations_np = durations.detach().cpu().numpy()
        events_np = events.detach().cpu().numpy()
        risks = np.exp(hazards_np)
        survival = np.zeros((n, self.k))
        density = np.zeros((n, self.k))
        for i in range(self.k):
            spline = self.breslow_splines[i]
            base = spline(durations_np)
            survival[:, i] = base ** risks[:, i]
            derivative = spline.derivative()(durations_np)
            density[:, i] = (-risks[:, i]) * (base ** (risks[:, i] - 1)) * derivative
        event_probs = np.where(events_np.astype(bool)[:, None], density, survival)
        log_probs = log_gate.detach().cpu().numpy() + np.log(
            np.clip(event_probs, 1e-10, None)
        )
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        repaired = _repair_log_probs(log_probs)
        return repaired - torch.logsumexp(repaired, dim=1, keepdim=True)

    def q_loss(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
        posteriors: Tensor,
        typ: str = "soft",
    ) -> Tensor:
        """M-step loss: negative Q-function for one batch.

        Ported from auton-survival's ``q_function``. Patients are
        assigned to one group — sampled from the posteriors when
        ``typ='soft'`` (the reference default; sampled-hard, not true
        soft) or by argmax when ``typ='hard'`` — and each group's Cox
        partial log-likelihood is summed, plus a gate term
        ``-sum(posteriors.exp() * log_gate_prob)``.

        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(
        ...     samples=[{"patient_id": "p0", "features": [0.0, 1.0],
        ...               "duration": [1.0], "event": [1.0]}],
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixturesScriptEM(dataset, k=2, layers=[4])
        >>> feats = torch.randn(4, 2)
        >>> post = torch.log(torch.full((4, 2), 0.5))
        >>> loss = model.q_loss(feats, torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ...                     torch.tensor([1.0, 0.0, 1.0, 0.0]), post)
        >>> isinstance(loss, torch.Tensor)
        True
        """
        if typ not in ("soft", "hard"):
            raise ValueError("typ must be 'soft' or 'hard'")
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        durations = torch.as_tensor(
            durations, dtype=torch.float32, device=self.device
        ).reshape(-1)
        events = torch.as_tensor(
            events, dtype=torch.float32, device=self.device
        ).reshape(-1)
        posteriors = torch.as_tensor(
            posteriors, dtype=torch.float32, device=self.device
        )
        log_gate, log_hazards = self.network(features)
        if typ == "hard":
            group = torch.argmax(posteriors, dim=1)
        else:
            group = torch.multinomial(posteriors.exp(), num_samples=1)[:, 0]
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for i in range(self.k):
            mask = group == i
            if not mask.any():
                continue
            loss = loss + _partial_ll_loss(
                log_hazards[mask][:, i],
                durations[mask],
                events[mask],
            )
        gate_loss = -torch.sum(posteriors.exp() * log_gate)
        return loss + gate_loss

    def soft_q_loss(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
        posteriors: Tensor,
    ) -> Tensor:
        """True-soft M-step loss (new code for Ablation 2).

        Unlike :meth:`q_loss`, no patient is assigned to a single
        group: every patient contributes to every group's Cox partial
        log-likelihood, weighted by its posterior probability. For
        group ``k`` the weighted partial log-likelihood is
        ``sum_i w_ik * [f_ik - log sum_{j in R(t_i)} w_jk exp(f_jk)]``
        over the group's events, plus the same gate term as
        :meth:`q_loss`. This is the proposal's "soft assignment";
        auton-survival's ``typ='soft'`` is sampled-hard instead, so
        this loss does not exist in the reference code.
        """
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        durations = torch.as_tensor(
            durations, dtype=torch.float32, device=self.device
        ).reshape(-1)
        events = torch.as_tensor(
            events, dtype=torch.float32, device=self.device
        ).reshape(-1)
        weights = torch.as_tensor(
            posteriors, dtype=torch.float32, device=self.device
        ).exp()
        log_gate, log_hazards = self.network(features)
        jitter = 1e-2 * torch.rand(
            durations.shape[0],
            device=durations.device,
            dtype=durations.dtype,
        )
        order = torch.argsort(-(durations + jitter))
        sorted_events = events[order]
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for i in range(self.k):
            sorted_weights = weights[order][:, i]
            sorted_hazards = log_hazards[order][:, i]
            weighted = sorted_hazards + torch.log(
                torch.clamp(sorted_weights, min=1e-10)
            )
            log_denominator = torch.logcumsumexp(weighted, dim=0)
            per_event = sorted_weights * (sorted_hazards - log_denominator)
            loss = loss - per_event[sorted_events == 1].sum()
        gate_loss = -torch.sum(weights * log_gate)
        return loss + gate_loss

    # -- Prediction -------------------------------------------------------
    @torch.no_grad()
    def predict_survival(self, features: Tensor, times: Tensor) -> Tensor:
        """Predict survival probabilities with shape ``(n, len(times))``.

        ``S(t | x) = sum_k P(Z = k | x) * S_k(t) ** exp(f_k(x))``,
        where ``S_k`` is group ``k``'s fitted baseline spline.
        Requires fitted baselines (see :meth:`fit_baselines`).

        Args:
            features: Dense input tensor with shape ``(n, input_dim)``.
            times: Strictly positive time points.

        Returns:
            Survival probabilities, clamped to ``[0, 1]``.
        """
        if self.breslow_splines is None:
            raise RuntimeError(
                "No baselines fitted: call fit_baselines() (or "
                "set_baselines() with saved splines) before predicting."
            )
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(
            -1
        )
        if (
            times.numel() == 0
            or not torch.isfinite(times).all()
            or torch.any(times <= 0)
        ):
            raise ValueError("times must be finite strictly positive")
        log_gate, log_hazards = self.network(features)
        gate_probs = log_gate.exp().detach().cpu().numpy()
        hazards_np = log_hazards.detach().cpu().numpy()
        times_np = times.detach().cpu().numpy()
        risks = np.exp(hazards_np)
        survival = np.zeros((features.shape[0], times_np.shape[0], self.k))
        for i in range(self.k):
            base = self.breslow_splines[i](times_np)
            survival[:, :, i] = base[None, :] ** risks[:, i][:, None]
        mixture = (gate_probs[:, None, :] * survival).sum(axis=2)
        return torch.as_tensor(
            np.clip(mixture, 0.0, 1.0),
            dtype=torch.float32,
            device=self.device,
        )

    # -- PyHealth forward -------------------------------------------------
    def forward(self, **kwargs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Run one Deep-Cox-Mixtures forward pass.

        Requires fitted baselines: ``loss`` is the Q-loss of the
        current batch under the current posteriors, and ``y_prob`` is
        the survival curve on ``self.time_grid``. This model is NOT
        trainable through ``Trainer.train()`` — train it with the
        example script's EM loop, which calls :meth:`fit_baselines`
        and :meth:`q_loss` directly (checklist Decision 2b).

        Args:
            **kwargs: Batch fields produced by the PyHealth
                dataloader. The batch must include ``feature_key``,
                ``duration_key``, and ``event_key``.

        Returns:
            Dictionary containing ``loss``, ``logit``, ``y_prob``,
            ``y_true``, ``log_gate_prob`` and ``log_hazard_ratios``.
        """
        missing = [
            key
            for key in (self.feature_key, self.duration_key, self.event_key)
            if key not in kwargs
        ]
        if missing:
            raise KeyError(f"Missing required batch fields: {missing}")
        if self.breslow_splines is None:
            raise RuntimeError(
                "No baselines fitted: call fit_baselines() before forward(). "
                "Train this model with the example script's EM loop, not "
                "Trainer.train() (checklist Decision 2b)."
            )
        features = _value_tensor(kwargs[self.feature_key])
        duration = _value_tensor(kwargs[self.duration_key])
        event = _value_tensor(kwargs[self.event_key])
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        duration = torch.as_tensor(
            duration, dtype=torch.float32, device=self.device
        ).reshape(-1)
        event = torch.as_tensor(event, dtype=torch.float32, device=self.device).reshape(
            -1
        )
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must have shape [batch, input_dim]; "
                f"received {tuple(features.shape)}"
            )
        if features.shape[0] == 0 or not torch.isfinite(features).all():
            raise ValueError("features must be nonempty and finite")
        if torch.any((event != 0) & (event != 1)):
            raise ValueError("event must contain only 0/1 indicators")

        posteriors = self.compute_posteriors(features, duration, event)
        loss = self.q_loss(features, duration, event, posteriors)
        y_prob = self.predict_survival(features, self.time_grid)
        y_true = torch.stack([duration, event], dim=-1)
        log_gate, log_hazards = self.network(features)
        return {
            "loss": loss,
            "logit": log_hazards,
            "y_prob": y_prob,
            "y_true": y_true,
            "log_gate_prob": log_gate,
            "log_hazard_ratios": log_hazards,
        }

    def forward_from_embedding(
        self,
        embeddings: Tensor,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run ``forward`` with dense features supplied directly."""
        kwargs[self.feature_key] = embeddings
        return self.forward(**kwargs)
