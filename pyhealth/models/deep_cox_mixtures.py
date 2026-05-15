"""Deep Cox Mixtures for Survival Regression.

Reference:
    Nagpal, C., Yadlowsky, S., Rostamzadeh, N., & Heller, K. (2021).
    Deep Cox Mixtures for Survival Regression.
    Proceedings of Machine Learning for Healthcare (MLHC).
    https://proceedings.mlr.press/v149/nagpal21a/nagpal21a.pdf
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import UnivariateSpline

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class _BreslowSpline:
    """Smoothed non-parametric cumulative hazard H_0^k(t) for one component."""

    def __init__(self, times: np.ndarray, cum_hazard: np.ndarray, smoothing: float):
        self._min_t = float(times[0])
        self._max_t = float(times[-1])
        self._times = times
        self._cum_hazard = cum_hazard
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._spline = UnivariateSpline(
                times, cum_hazard, k=1, s=smoothing, ext="const"
            )

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        out = self._spline(np.clip(t, self._min_t, self._max_t))
        return np.clip(out, 0.0, None)


def _fit_breslow_one_component(
    times: np.ndarray,
    events: np.ndarray,
    log_hr: np.ndarray,
    weights: np.ndarray,
    smoothing: float,
) -> Optional[_BreslowSpline]:
    """Fit a weighted Breslow cumulative hazard for a single mixture component."""
    mask = weights > 1e-8
    if mask.sum() < 2 or (events[mask] > 0.5).sum() < 2:
        return None

    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    w_sorted = weights[order]
    r_sorted = np.exp(log_hr[order]) * w_sorted

    reverse_cum_risk = np.cumsum(r_sorted[::-1])[::-1]
    safe_risk = np.where(reverse_cum_risk > 1e-12, reverse_cum_risk, 1e-12)
    cum_hazard = np.cumsum((e_sorted * w_sorted) / safe_risk)

    # Spline fit requires strictly increasing x; for tied times keep the last
    # cumulative hazard reached.
    unique_t = np.unique(t_sorted)
    if unique_t.size < 4:
        return None
    last_per_unique = np.zeros_like(unique_t, dtype=np.float64)
    for i, t in enumerate(t_sorted):
        idx = int(np.searchsorted(unique_t, t))
        last_per_unique[idx] = max(last_per_unique[idx], cum_hazard[i])
    if not np.any(last_per_unique > 0):
        return None
    return _BreslowSpline(unique_t, last_per_unique, smoothing)


def _cox_partial_log_likelihood(
    log_hr: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted Cox partial log-likelihood with Breslow tie handling."""
    sort_idx = torch.argsort(times, descending=True)
    log_hr_s = log_hr[sort_idx]
    events_s = events[sort_idx]
    weights_s = weights[sort_idx]

    weighted_hr = log_hr_s + torch.log(weights_s.clamp_min(1e-12))
    running_lse = torch.logcumsumexp(weighted_hr, dim=0)

    contrib = weights_s * events_s * (log_hr_s - running_lse)
    denom = (weights_s * events_s).sum().clamp_min(1e-8)
    return contrib.sum() / denom


class DeepCoxMixtures(BaseModel):
    """Deep Cox Mixtures survival model.

    Combines a shared neural embedding with a softmax gate and ``k`` Cox
    experts. Each expert parameterises a proportional-hazards component
    with a non-parametric Breslow baseline cumulative hazard. Mixture
    weights are learned per sample from the input features.

    The model expects a ``SampleDataset`` whose ``output_schema`` declares
    two keys: the survival duration (as ``regression``) and the event
    indicator (as ``binary``). Any number of input features are supported;
    each feature must be encoded by a processor that yields a dense tensor
    (for example the ``tensor`` processor).

    Args:
        dataset: The ``SampleDataset`` to train on. Its ``output_schema``
            must define exactly one regression-mode key (event time) and
            one binary-mode key (event indicator).
        k: Number of mixture components. Default is 3.
        hidden_dims: Width of each hidden layer in the shared embedding
            MLP. Default is ``(32,)``.
        gamma: Hard clamp applied to per-component log-hazard-ratios to
            keep ``exp(log_hr)`` finite. Default is 10.0.
        entropy_weight: Coefficient on the gate entropy regularizer that
            discourages posterior collapse. Default is 0.01.
        spline_smoothing: Smoothing factor passed to
            ``scipy.interpolate.UnivariateSpline`` when fitting Breslow
            cumulative hazards. Default is 1e-4.
        time_key: Override for the regression-mode label key. If ``None``
            the key is inferred from ``dataset.output_schema``.
        event_key: Override for the binary-mode label key. If ``None``
            the key is inferred from ``dataset.output_schema``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import DeepCoxMixtures
        >>> samples = [
        ...     {"patient_id": f"p{i}",
        ...      "features": [float(i), float(i) * 0.5],
        ...      "time": float(i + 1),
        ...      "event": i % 2}
        ...     for i in range(10)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"time": "regression", "event": "binary"},
        ... )
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=4, shuffle=True)
        >>> model = DeepCoxMixtures(dataset=dataset, k=2, hidden_dims=(16,))
        >>> _ = model.fit(loader, epochs=3, learning_rate=1e-2)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        k: int = 3,
        hidden_dims: Sequence[int] = (32,),
        gamma: float = 10.0,
        entropy_weight: float = 0.01,
        spline_smoothing: float = 1e-4,
        time_key: Optional[str] = None,
        event_key: Optional[str] = None,
    ) -> None:
        super().__init__(dataset)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer width")

        self.k = k
        self.gamma = float(gamma)
        self.entropy_weight = float(entropy_weight)
        self.spline_smoothing = float(spline_smoothing)

        self.time_key, self.event_key = self._resolve_label_keys(time_key, event_key)

        input_dim = self._resolve_input_dim()
        self._feature_key = self.feature_keys[0]
        self.input_dim = input_dim

        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU6())
            prev = width
        self.embedding = nn.Sequential(*layers)
        self.gate_head = nn.Linear(prev, k)
        self.expert_head = nn.Linear(prev, k)

        self._breslow_splines: List[Optional[_BreslowSpline]] = [None] * k

    def _resolve_label_keys(
        self, time_key: Optional[str], event_key: Optional[str]
    ) -> Tuple[str, str]:
        if time_key is not None and event_key is not None:
            return time_key, event_key

        regression_keys: List[str] = []
        binary_keys: List[str] = []
        for key in self.label_keys:
            mode = self._resolve_mode(self.dataset.output_schema[key])
            if mode == "regression":
                regression_keys.append(key)
            elif mode == "binary":
                binary_keys.append(key)
        if time_key is None:
            if len(regression_keys) != 1:
                raise ValueError(
                    "DeepCoxMixtures needs exactly one regression-mode label "
                    f"(event time); found {regression_keys}. Pass time_key= explicitly."
                )
            time_key = regression_keys[0]
        if event_key is None:
            if len(binary_keys) != 1:
                raise ValueError(
                    "DeepCoxMixtures needs exactly one binary-mode label "
                    f"(event indicator); found {binary_keys}. Pass event_key= explicitly."
                )
            event_key = binary_keys[0]
        return time_key, event_key

    def _resolve_input_dim(self) -> int:
        if len(self.feature_keys) != 1:
            raise ValueError(
                "DeepCoxMixtures currently supports a single input feature key; "
                f"dataset.input_schema has {self.feature_keys}."
            )
        key = self.feature_keys[0]
        sample_value = self.dataset[0][key]
        if isinstance(sample_value, torch.Tensor):
            return int(sample_value.numel())
        return int(np.asarray(sample_value).size)

    def _encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        features = features.to(self.device).float()
        embedded = self.embedding(features)
        log_gates = F.log_softmax(self.gate_head(embedded), dim=-1)
        log_hr = torch.tanh(self.expert_head(embedded)) * self.gamma
        return log_gates, log_hr

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run a forward pass and, when labels are present, the DCM loss.

        Returns a dict with keys ``logit`` (per-component log-hazard-ratios),
        ``y_prob`` (gate-weighted scalar risk score), ``gate_probs``, and,
        when labels are present, ``loss`` and ``y_true`` (stacked
        ``[time, event]``).
        """
        features = kwargs[self._feature_key]
        log_gates, log_hr = self._encode(features)
        gate_probs = log_gates.exp()
        risk_score = (gate_probs * log_hr).sum(dim=-1, keepdim=True)

        out: Dict[str, torch.Tensor] = {
            "logit": log_hr,
            "y_prob": risk_score,
            "gate_probs": gate_probs,
        }

        time = kwargs.get(self.time_key)
        event = kwargs.get(self.event_key)
        if time is not None and event is not None:
            time = time.to(self.device).float().view(-1)
            event = event.to(self.device).float().view(-1)
            posteriors = self._posteriors(log_gates.detach(), log_hr.detach(), time, event)
            cox_terms: List[torch.Tensor] = []
            for comp in range(self.k):
                weights = posteriors[:, comp]
                if weights.sum() < 1e-6:
                    continue
                pl = _cox_partial_log_likelihood(log_hr[:, comp], time, event, weights)
                cox_terms.append(pl)
            if cox_terms:
                cox_loss = -torch.stack(cox_terms).mean()
            else:
                cox_loss = torch.zeros((), device=self.device)

            nll_gate = -(posteriors * log_gates).sum(dim=-1).mean()
            entropy = -(gate_probs * log_gates).sum(dim=-1).mean()

            loss = cox_loss + nll_gate - self.entropy_weight * entropy
            out["loss"] = loss
            out["y_true"] = torch.stack([time, event], dim=-1)
        return out

    def _posteriors(
        self,
        log_gates: torch.Tensor,
        log_hr: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Compute P(z | x, t, e) for each sample and component."""
        if all(spline is None for spline in self._breslow_splines):
            return log_gates.exp()
        times_np = time.detach().cpu().numpy()
        events_np = event.detach().cpu().numpy()
        hr = log_hr.exp().detach().cpu().numpy()
        log_lik = np.zeros((times_np.shape[0], self.k), dtype=np.float64)
        for comp, spline in enumerate(self._breslow_splines):
            if spline is None:
                log_lik[:, comp] = -1e6
                continue
            h_t = spline(times_np)
            # density f = h * S; approximate h with a central finite difference
            # on the smoothed cumulative hazard.
            eps = max(1e-3, float(times_np.mean()) * 1e-3)
            dH = (spline(times_np + eps) - spline(np.maximum(times_np - eps, 0.0))) / (2 * eps)
            dH = np.clip(dH, 1e-8, None)
            log_lik[:, comp] = np.where(
                events_np > 0.5,
                np.log(dH) + np.log(hr[:, comp] + 1e-12) - h_t * hr[:, comp],
                -h_t * hr[:, comp],
            )
        log_joint = log_lik + log_gates.detach().cpu().numpy()
        log_norm = np.logaddexp.reduce(log_joint, axis=1, keepdims=True)
        posteriors = np.exp(log_joint - log_norm)
        return torch.from_numpy(posteriors).to(self.device).float()

    def fit(
        self,
        loader,
        epochs: int = 20,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Fit the model via alternating E-step / M-step iterations.

        Between epochs the Breslow baseline hazards are re-fit on the full
        training set. Returns a history dict with per-epoch mean M-step loss.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history: Dict[str, List[float]] = {"epoch_loss": []}
        for epoch in range(epochs):
            self.train()
            losses: List[float] = []
            for batch in loader:
                optimizer.zero_grad()
                loss = self.forward(**batch).get("loss")
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if losses:
                history["epoch_loss"].append(sum(losses) / len(losses))
            self.refit_breslow(loader)
            if verbose and history["epoch_loss"]:
                print(
                    f"[DCM] epoch {epoch + 1}/{epochs} "
                    f"loss={history['epoch_loss'][-1]:.4f}"
                )
        return history

    def refit_breslow(self, loader) -> None:
        """Recompute Breslow baseline cumulative hazards for each component."""
        self.eval()
        times_all: List[np.ndarray] = []
        events_all: List[np.ndarray] = []
        hr_all: List[np.ndarray] = []
        gates_all: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                time = batch[self.time_key].to(self.device).float().view(-1)
                event = batch[self.event_key].to(self.device).float().view(-1)
                log_gates, log_hr = self._encode(batch[self._feature_key])
                posteriors = self._posteriors(log_gates, log_hr, time, event)
                times_all.append(time.detach().cpu().numpy())
                events_all.append(event.detach().cpu().numpy())
                hr_all.append(log_hr.detach().cpu().numpy())
                gates_all.append(posteriors.detach().cpu().numpy())
        if not times_all:
            return
        times_np = np.concatenate(times_all)
        events_np = np.concatenate(events_all)
        hr_np = np.concatenate(hr_all, axis=0)
        post_np = np.concatenate(gates_all, axis=0)
        new_splines: List[Optional[_BreslowSpline]] = []
        for comp in range(self.k):
            spline = _fit_breslow_one_component(
                times=times_np,
                events=events_np,
                log_hr=hr_np[:, comp],
                weights=post_np[:, comp],
                smoothing=self.spline_smoothing,
            )
            if spline is None and self._breslow_splines[comp] is not None:
                warnings.warn(
                    f"Breslow refit failed for component {comp}; reusing previous spline.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                new_splines.append(self._breslow_splines[comp])
            else:
                new_splines.append(spline)
        self._breslow_splines = new_splines

    @torch.no_grad()
    def predict_latent_z(self, features: torch.Tensor) -> torch.Tensor:
        """Return the gate probabilities P(z | x) for each component."""
        self.eval()
        log_gates, _ = self._encode(features)
        return log_gates.exp().cpu()

    @torch.no_grad()
    def predict_survival_curve(
        self, features: torch.Tensor, times: Sequence[float]
    ) -> np.ndarray:
        """Return S(t | x) of shape ``(batch, len(times))``.

        If Breslow splines have not been fit yet, returns an all-ones array.
        """
        self.eval()
        log_gates, log_hr = self._encode(features)
        gates = log_gates.exp().cpu().numpy()
        hr = log_hr.exp().cpu().numpy()
        times_np = np.asarray(list(times), dtype=np.float64)
        result = np.zeros((hr.shape[0], times_np.size), dtype=np.float64)
        for comp, spline in enumerate(self._breslow_splines):
            if spline is None:
                result += gates[:, comp : comp + 1]
                continue
            comp_survival = np.exp(-np.outer(hr[:, comp], spline(times_np)))
            result += gates[:, comp : comp + 1] * comp_survival
        return np.clip(result, 0.0, 1.0)

    @torch.no_grad()
    def predict_risk(self, features: torch.Tensor) -> np.ndarray:
        """Return a per-sample scalar risk score (higher == higher hazard)."""
        self.eval()
        log_gates, log_hr = self._encode(features)
        return (log_gates.exp() * log_hr).sum(dim=-1).cpu().numpy()
