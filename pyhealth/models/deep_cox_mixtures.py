"""Deep Cox Mixtures for Survival Regression.

Reference:
    Nagpal et al. (2021).
    Deep Cox Mixtures for Survival Regression.
    Machine Learning for Healthcare.

    https://proceedings.mlr.press/v149/nagpal21a.html
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel


def cox_partial_loss(
    log_risk: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Compute negative Cox partial log-likelihood.

    Uses the Breslow approximation for tied event times.

    Args:
        log_risk: Predicted log hazard ratios of shape (N,).
        duration: Observed follow-up times of shape (N,).
        event: Event indicators of shape (N,). A value of
            1 indicates death and 0 indicates censoring.

    Returns:
        Scalar negative partial log-likelihood.

        Examples:
        >>> import torch
        >>> log_risk = torch.zeros(3)
        >>> duration = torch.tensor([1.0, 2.0, 3.0])
        >>> event = torch.ones(3)
        >>> loss = cox_partial_loss(log_risk, duration, event)
        >>> round(loss.item(), 4)
        0.5973
    """
    log_risk = log_risk.reshape(-1)
    duration = duration.reshape(-1)
    event = event.reshape(-1).bool()

    if not event.any():
        return log_risk.sum() * 0.0

    order = torch.argsort(duration, descending=True)

    log_risk = log_risk[order]
    duration = duration[order]
    event = event[order]

    log_risk_sum = torch.logcumsumexp(log_risk, dim=0)

    _, counts = torch.unique_consecutive(
        duration,
        return_counts=True,
    )

    group_end = torch.cumsum(counts, dim=0) - 1

    log_denominator = torch.repeat_interleave(
        log_risk_sum[group_end],
        counts,
    )

    log_likelihood = (log_risk[event] - log_denominator[event]).sum()

    return -log_likelihood / event.sum()


class DeepCoxMixtures(BaseModel):
    """Deep Cox Mixtures model for time-to-event prediction.

    Models heterogeneous patient survival distributions using
    multiple latent Cox proportional hazards components.

    Args:
        dataset: PyHealth sample dataset.
        input_dim: Number of input features.
        num_components: Number of latent Cox components.
        hidden_dims: Hidden dimensions of the shared encoder.
        dropout: Dropout probability.

        Examples:
        >>> import torch
        >>> model = DeepCoxMixtures(
        ...     dataset=None,
        ...     input_dim=4,
        ...     num_components=3,
        ... )
        >>> features = torch.randn(5, 4)
        >>> output = model(features)
        >>> output["mixture_probs"].shape
        torch.Size([5, 3])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_dim: int = 27,
        num_components: int = 3,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(dataset=dataset)

        if input_dim < 1:
            raise ValueError("input_dim must be positive")

        if num_components < 1:
            raise ValueError("num_components must be positive")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.num_components = num_components
        self.hidden_dims = tuple(hidden_dims)

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim < 1:
                raise ValueError("Hidden dimensions must be positive")

            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        self.gate = nn.Linear(
            previous_dim,
            num_components,
        )

        self.risk = nn.Linear(
            previous_dim,
            num_components,
        )
        self.baselines = [BreslowBaseline() for _ in range(num_components)]

        self.baselines_fitted = False
        self.training_history = []

    @property
    def mode(self) -> None:
        """Return None for custom survival regression."""
        return None

    def encode(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mixture probabilities and log hazard ratios.

        Args:
            features: Patient features of shape (N, input_dim).

        Returns:
            Log mixture probabilities and log hazard ratios,
            both of shape (N, num_components).
        """
        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )

        embedding = self.encoder(features)

        log_pi = F.log_softmax(
            self.gate(embedding),
            dim=-1,
        )

        log_risk = self.risk(embedding)

        return log_pi, log_risk

    def forward(
        self,
        features: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute mixture probabilities and component log risks.

        Args:
            features: Patient feature matrix of shape (N, D).
            **kwargs: Additional dataset fields.

        Returns:
            A dictionary containing component log risks,
            mixture probabilities and log mixture probabilities.

        Note:
            Training requires fit_em() because the model uses
            alternating posterior estimation, Cox partial
            likelihood optimization and baseline hazard updates.
            The standard PyHealth Trainer is not supported.
            Use predict_survival() for survival probabilities.
        """
        log_pi, log_risk = self.encode(features)

        mixture_probs = log_pi.exp()

        return {
            "logit": log_risk,
            "y_prob": mixture_probs,
            "log_pi": log_pi,
            "mixture_probs": mixture_probs,
            "log_risk": log_risk,
        }

    def evaluate_baselines(
        self,
        duration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate component-specific baseline hazards.

        Args:
            duration: Observed patient follow-up times.

        Returns:
            Cumulative and instantaneous baseline hazards,
            both of shape (N, K).
        """
        if not self.baselines_fitted:
            raise RuntimeError("Baseline functions are not fitted")

        times = duration.detach().cpu().numpy().reshape(-1)

        cumulative = []
        hazard = []

        for baseline in self.baselines:
            cumulative.append(baseline.predict_cumulative_hazard(times))

            hazard.append(baseline.predict_hazard(times))

        cumulative = torch.as_tensor(
            np.column_stack(cumulative),
            dtype=torch.float32,
            device=self.device,
        )

        hazard = torch.as_tensor(
            np.column_stack(hazard),
            dtype=torch.float32,
            device=self.device,
        )

        return cumulative, hazard

    @torch.no_grad()
    def e_step(
        self,
        log_pi: torch.Tensor,
        log_risk: torch.Tensor,
        duration: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate posterior subgroup membership probabilities.

        Args:
            log_pi: Log mixture probabilities of shape (N, K).
            log_risk: Component log hazard ratios of shape (N, K).
            duration: Observed follow-up times.
            event: Binary event indicators.

        Returns:
            Posterior probabilities of shape (N, K).
        """
        cumulative, hazard = self.evaluate_baselines(duration)

        log_risk = log_risk.clamp(-50, 50)
        risk = torch.exp(log_risk)

        event = event.reshape(-1, 1).float()

        log_survival = -cumulative * risk

        log_hazard = torch.log(hazard.clamp_min(1e-12)) + log_risk

        log_likelihood = log_survival + event * log_hazard

        posterior = torch.softmax(
            log_pi + log_likelihood,
            dim=1,
        )

        return posterior

    def m_step_loss(
        self,
        log_pi: torch.Tensor,
        log_risk: torch.Tensor,
        posterior: torch.Tensor,
        assignments: torch.Tensor,
        duration: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the approximate EM optimization objective.

        Args:
            log_pi: Log mixture probabilities.
            log_risk: Component log hazard ratios.
            posterior: Posterior membership probabilities.
            assignments: Sampled hard component assignments.
            duration: Observed follow-up times.
            event: Binary event indicators.

        Returns:
            Combined gating and Cox survival loss.
        """
        duration = duration.reshape(-1)
        event = event.reshape(-1).float()
        log_risk = log_risk.clamp(-50, 50)

        # Train the gating network to approximate the posterior.
        gate_loss = -(posterior.detach() * log_pi).sum(dim=1).mean()

        survival_loss = log_risk.sum() * 0.0

        total_events = event.sum().clamp_min(1.0)

        # Optimize the Cox partial likelihood in each subgroup.
        for k in range(self.num_components):
            mask = assignments == k

            if not mask.any():
                continue

            group_events = event[mask].sum()

            if group_events == 0:
                continue

            component_loss = cox_partial_loss(
                log_risk[mask, k],
                duration[mask],
                event[mask],
            )

            survival_loss = survival_loss + (
                component_loss * group_events / total_events
            )

        return gate_loss + survival_loss

    @torch.no_grad()
    def update_baselines(
        self,
        features: torch.Tensor,
        duration: torch.Tensor,
        event: torch.Tensor,
        assignments: torch.Tensor,
    ) -> None:
        """Refit component-specific baseline hazard functions.

        Args:
            features: Patient feature matrix.
            duration: Observed follow-up times.
            event: Binary event indicators.
            assignments: Hard subgroup assignments.
        """
        self.eval()

        _, log_risk = self.encode(features)

        log_risk = log_risk.detach().cpu().numpy()
        duration = duration.detach().cpu().numpy().reshape(-1)
        event = event.detach().cpu().numpy().reshape(-1)
        assignments = assignments.detach().cpu().numpy()

        for k in range(self.num_components):
            mask = assignments == k

            # Retain the previous baseline for an empty component.
            if not mask.any() or event[mask].sum() == 0:
                if self.baselines_fitted:
                    continue

                # Pooled fallback during initialization.
                self.baselines[k].fit(
                    duration,
                    event,
                    log_risk[:, k],
                )
                continue

            self.baselines[k].fit(
                duration[mask],
                event[mask],
                log_risk[mask, k],
            )

        self.baselines_fitted = True

    def fit_em(
        self,
        features: torch.Tensor,
        duration: torch.Tensor,
        event: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[float]:
        """Train Deep Cox Mixtures using approximate EM.

        Uses full-batch training to preserve Cox risk sets.

        Args:
            features: Patient features of shape (N, D).
            duration: Observed follow-up times.
            event: Binary event indicators.
            epochs: Number of training epochs.
            learning_rate: Adam learning rate.
            weight_decay: L2 regularization coefficient.
            seed: Random seed for subgroup sampling.
            verbose: Whether to print training progress.

        Returns:
            Training loss history.
        """
        torch.manual_seed(seed)

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

        if features.ndim != 2:
            raise ValueError("Features must be a 2D tensor")

        if features.shape[1] != self.input_dim:
            raise ValueError("Incorrect number of input features")

        if not len(features) == len(duration) == len(event):
            raise ValueError("Input lengths must match")

        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")

        if not torch.isfinite(duration).all():
            raise ValueError("Durations must be finite")

        if (duration <= 0).any():
            raise ValueError("Durations must be positive")

        if not ((event == 0) | (event == 1)).all():
            raise ValueError("Events must be binary")

        if event.sum() == 0:
            raise ValueError("At least one observed event is required")

        if epochs < 1:
            raise ValueError("Epochs must be positive")

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Initialize latent subgroup membership.
        assignments = torch.randint(
            self.num_components,
            (len(features),),
            device=self.device,
        )

        self.baselines_fitted = False
        self.training_history = []

        self.update_baselines(
            features,
            duration,
            event,
            assignments,
        )

        for epoch in range(epochs):
            # E-step: estimate posterior membership.
            self.eval()

            with torch.no_grad():
                log_pi, log_risk = self.encode(features)

                posterior = self.e_step(
                    log_pi,
                    log_risk,
                    duration,
                    event,
                )

                assignments = torch.multinomial(
                    posterior,
                    num_samples=1,
                ).squeeze(1)

            # M-step: optimize neural network parameters.
            self.train()

            log_pi, log_risk = self.encode(features)

            loss = self.m_step_loss(
                log_pi,
                log_risk,
                posterior,
                assignments,
                duration,
                event,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            # Update component baseline hazards.
            self.update_baselines(
                features,
                duration,
                event,
                assignments,
            )

            loss_value = loss.item()
            self.training_history.append(loss_value)

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs):
                counts = torch.bincount(
                    assignments,
                    minlength=self.num_components,
                ).tolist()

                print(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Loss: {loss_value:.6f} | "
                    f"Components: {counts}"
                )

        self.eval()

        return self.training_history

    @torch.no_grad()
    def predict_survival(
        self,
        features: torch.Tensor,
        horizons: Sequence[float],
    ) -> torch.Tensor:
        """Predict patient-specific survival probabilities.

        Args:
            features: Patient features of shape (N, D).
            horizons: Follow-up times for survival prediction.

        Returns:
            Survival probabilities of shape (N, T), where
            T is the number of evaluation horizons.
        """
        if not self.baselines_fitted:
            raise RuntimeError("Train the model before prediction")

        self.eval()

        horizons = np.asarray(horizons, dtype=float).reshape(-1)

        if not np.isfinite(horizons).all():
            raise ValueError("Horizons must be finite")

        if np.any(horizons < 0):
            raise ValueError("Horizons must be non-negative")

        log_pi, log_risk = self.encode(features)

        pi = log_pi.exp()
        risk = log_risk.clamp(-50, 50).exp()

        cumulative = np.stack(
            [
                baseline.predict_cumulative_hazard(horizons)
                for baseline in self.baselines
            ]
        )

        cumulative = torch.as_tensor(
            cumulative,
            dtype=torch.float32,
            device=self.device,
        )

        component_survival = torch.exp(-risk[:, :, None] * cumulative[None, :, :])

        survival = (pi[:, :, None] * component_survival).sum(dim=1)

        return survival.clamp(0.0, 1.0)

    def get_extra_state(self) -> dict:
        """Return baseline functions for checkpoint persistence.

        Returns:
            Serializable component baseline parameters and
            training history.
        """
        baseline_states = []

        for baseline in self.baselines:
            if baseline.event_times is None:
                baseline_states.append(None)
                continue

            baseline_states.append(
                {
                    "event_times": baseline.event_times.tolist(),
                    "cumulative_hazard": (baseline.cumulative_hazard.tolist()),
                    "epsilon": baseline.epsilon,
                }
            )

        return {
            "baselines": baseline_states,
            "baselines_fitted": self.baselines_fitted,
            "training_history": list(self.training_history),
        }

    def set_extra_state(self, state: dict) -> None:
        """Restore component baseline functions from a checkpoint.

        Args:
            state: Serialized baseline parameters and
                training history.
        """
        baseline_states = state["baselines"]

        if len(baseline_states) != self.num_components:
            raise ValueError("Checkpoint component count does not match model")

        self.baselines = []

        for baseline_state in baseline_states:

            baseline = BreslowBaseline()

            if baseline_state is not None:

                baseline.epsilon = baseline_state["epsilon"]

                baseline.event_times = np.asarray(
                    baseline_state["event_times"],
                    dtype=float,
                )

                baseline.cumulative_hazard = np.asarray(
                    baseline_state["cumulative_hazard"],
                    dtype=float,
                )

                if len(baseline.event_times) >= 2:
                    baseline.spline = PchipInterpolator(
                        baseline.event_times,
                        baseline.cumulative_hazard,
                        extrapolate=False,
                    )

            self.baselines.append(baseline)

        self.baselines_fitted = state["baselines_fitted"]

        self.training_history = list(state["training_history"])


class BreslowBaseline:
    """Estimate a component-specific baseline survival function.

    Uses Breslow cumulative hazard estimation followed by
    shape-preserving cubic interpolation.

    Args:
        epsilon: Small positive constant for numerical stability.
        Examples:
        >>> import numpy as np
        >>> baseline = BreslowBaseline()
        >>> _ = baseline.fit(
        ...     duration=np.array([10, 20, 30]),
        ...     event=np.array([1, 0, 1]),
        ...     log_risk=np.zeros(3),
        ... )
        >>> baseline.predict_survival(np.array([0.0]))
        array([1.])
    """
    

    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon
        self.event_times = None
        self.cumulative_hazard = None
        self.spline = None

    def fit(
        self,
        duration: np.ndarray,
        event: np.ndarray,
        log_risk: np.ndarray,
    ) -> "BreslowBaseline":
        """Fit the baseline cumulative hazard.

        Args:
            duration: Observed follow-up times.
            event: Binary event indicators.
            log_risk: Predicted log hazard ratios.

        Returns:
            Fitted baseline estimator.
        """
        duration = np.asarray(duration, dtype=float).reshape(-1)
        event = np.asarray(event, dtype=int).reshape(-1)
        log_risk = np.asarray(log_risk, dtype=float).reshape(-1)

        if not len(duration) == len(event) == len(log_risk):
            raise ValueError("Input arrays must have equal lengths")

        if not np.isfinite(duration).all() or np.any(duration <= 0):
            raise ValueError("Durations must be finite and positive")

        if not np.isfinite(log_risk).all():
            raise ValueError("Log risks must be finite")

        if not np.isin(event, [0, 1]).all():
            raise ValueError("Event indicators must be binary")

        risk = np.exp(np.clip(log_risk, -50, 50))

        event_times = np.unique(duration[event == 1])

        increments = []

        for t in event_times:
            deaths = np.sum((duration == t) & (event == 1))

            risk_sum = risk[duration >= t].sum()

            increment = deaths / max(risk_sum, self.epsilon)

            increments.append(increment)

        self.event_times = np.concatenate(([0.0], event_times))

        self.cumulative_hazard = np.concatenate(([0.0], np.cumsum(increments)))

        if len(self.event_times) >= 2:
            self.spline = PchipInterpolator(
                self.event_times,
                self.cumulative_hazard,
                extrapolate=False,
            )
        else:
            self.spline = None

        return self

    def predict_cumulative_hazard(
        self,
        times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the baseline cumulative hazard.

        Args:
            times: Evaluation times.

        Returns:
            Non-negative cumulative hazard estimates.
        """
        if self.event_times is None:
            raise RuntimeError("Baseline must be fitted first")

        times = np.asarray(times, dtype=float)

        if self.spline is None:
            return np.zeros_like(times)

        bounded_times = np.clip(times, 0.0, self.event_times[-1])

        cumulative = self.spline(bounded_times)

        return np.maximum(cumulative, 0.0)

    def predict_hazard(
        self,
        times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the instantaneous baseline hazard.

        Args:
            times: Evaluation times.

        Returns:
            Non-negative baseline hazard estimates.
        """
        if self.event_times is None:
            raise RuntimeError("Baseline must be fitted first")

        times = np.asarray(times, dtype=float)

        if self.spline is None:
            return np.zeros_like(times)

        bounded_times = np.clip(times, 0.0, self.event_times[-1])

        hazard = self.spline.derivative()(bounded_times)

        hazard = np.where(
            (times < 0) | (times > self.event_times[-1]),
            0.0,
            hazard,
        )

        return np.maximum(hazard, 0.0)

    def predict_survival(
        self,
        times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the baseline survival probability.

        Args:
            times: Evaluation times.

        Returns:
            Survival probabilities between zero and one.
        """
        cumulative = self.predict_cumulative_hazard(times)

        return np.exp(-cumulative)
