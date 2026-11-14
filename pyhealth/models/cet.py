"""Neural Conditional Event Time (CET) model for right-censored clinical events.

Based on Engelhard et al., *Neural Conditional Event Time Models* (MLHC 2020).
https://proceedings.mlr.press/v126/engelhard20a.html

Contributors:
    Peter Prudhomme (Georgia Tech, @peterjprudhomme)
    James Sun (Georgia Tech, @jsun2192)

The occurrence head models whether each event can occur. The timing head
predicts a log-normal event-time distribution conditioned on a differentiable
sample of the full occurrence vector. The likelihood follows the public
TensorFlow MIMIC implementation (fpr_likelihood=True, prop_fpr=True).

This model takes engineered dense features; it does not select MIMIC ITEMIDs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor

__all__ = ["CET"]


def _value_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Extract the processed tensor from a PyHealth processor output.

    Args:
        value: A tensor or processor output tuple containing a tensor first.

    Returns:
        The input tensor or the first element of its tuple.
    """
    return value[0] if isinstance(value, tuple) else value


def _sample_width(dataset: SampleDataset, key: str) -> int:
    """Infer the width of a processed sample's vector field.

    Args:
        dataset: The processed PyHealth dataset.
        key: Field name to inspect in the first sample.

    Returns:
        Number of vector elements.

    Raises:
        ValueError: If the field is missing or is not a nonempty 1D vector.
    """
    try:
        value = _value_tensor(dataset[0][key])
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot infer {key!r} width from dataset[0]") from exc
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{key!r} must be a nonempty 1D vector per sample")
    return int(tensor.numel())


class CET(BaseModel):
    """Conditional Event Time network with latent event occurrence indicators.

    Inputs are a numeric feature vector and one duration/observed indicator per
    target. Event times should be positive and share a common unit. A zero
    observed indicator represents right censoring; it is **not** a known
    negative eventual-occurrence label.

    Args:
        dataset: Processed PyHealth SampleDataset with ``features`` as a dense
            tensor input and ``duration`` and ``event`` tensor output fields.
        feature_key: Name of the input feature field.
        duration_key: Name of observed event/censoring times.
        event_key: Name of observed (1) or right-censored (0) indicators.
        input_dim: Feature vector width. Inferred from dataset if omitted.
        n_events: Number of modeled events. Inferred from dataset if omitted.
        hidden_dim: Width of the occurrence and timing hidden layers.
        dropout: Dropout probability for each hidden layer.
        n_samples: Number of binary Concrete samples per training forward pass.
        temperature: Positive Gumbel-Sigmoid temperature.
        fpr: Probability factor (0,1] used in the reference likelihood's
            observed-event penalty when occurrence is sampled as zero.
        logvar_bounds: Optional log-variance clamp for stability. ``None``
            disables the safeguard to approach the original code more closely.

    Returns:
        The forward pass returns ``loss``, ``logit``, ``y_prob``, ``y_true`` and
        ``time_pred``. ``y_prob`` is occurrence probability [B,M], ``y_true``
        is the observed/censored indicator [B,M] (not eventual ground truth),
        and ``time_pred`` is the median of sampled conditional log-normal
        medians [B,M]. Because the targets are censored survival labels,
        automatic classification scoring is disabled (``mode=None``).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> ds = create_sample_dataset(
        ...     samples=[
        ...         {"patient_id": "p0", "features": [0., 1.],
        ...          "duration": [3., 1.], "event": [1., 0.]},
        ...         {"patient_id": "p1", "features": [1., 0.],
        ...          "duration": [4., 2.], "event": [0., 1.]},
        ...     ],
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ...     in_memory=True,
        ... )
        >>> model = CET(ds, hidden_dim=8, n_samples=3)
        >>> model.n_events
        2

    References:
        Engelhard M., Berchuck S., D'Arcy J., Henao R. Neural Conditional
        Event Time Models. MLHC 2020. https://proceedings.mlr.press/v126/
        engelhard20a.html
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Provide no classification label processors for censored outcomes.

        Returns:
            Empty processor mapping; observed status is not eventual status.
        """
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        n_events: int | None = None,
        hidden_dim: int = 750,
        dropout: float = 0.5,
        n_samples: int = 100,
        temperature: float = 0.3,
        fpr: float = 0.7,
        logvar_bounds: tuple[float, float] | None = (-12.0, 12.0),
    ) -> None:
        super().__init__(dataset)
        if feature_key not in dataset.input_schema:
            raise ValueError(f"Unknown input feature_key: {feature_key!r}")
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"Unknown output key: {key!r}")
        if input_dim is None:
            input_dim = _sample_width(dataset, feature_key)
        if n_events is None:
            n_events = _sample_width(dataset, event_key)
        for name, value in (
            ("input_dim", input_dim), ("n_events", n_events),
            ("hidden_dim", hidden_dim), ("n_samples", n_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(dropout) or not (0 <= dropout < 1):
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not math.isfinite(fpr) or not (0 < fpr <= 1):
            raise ValueError("fpr must be in (0, 1]")
        if logvar_bounds is not None and (
            len(logvar_bounds) != 2
            or not all(math.isfinite(v) for v in logvar_bounds)
            or logvar_bounds[0] >= logvar_bounds[1]
        ):
            raise ValueError("logvar_bounds must be (finite min, finite max)")
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.input_dim = input_dim
        self.n_events = n_events
        self.n_samples = n_samples
        self.temperature = temperature
        self.nlog_fpr = -math.log(fpr)
        self.logvar_bounds = logvar_bounds
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.occurrence_head = nn.Linear(hidden_dim, n_events)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + n_events, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.time_mean = nn.Linear(hidden_dim, n_events)
        self.time_logvar = nn.Linear(hidden_dim, n_events)

    def _features(self, value: Tensor | tuple[Tensor, ...]) -> Tensor:
        """Convert a batch of dense features to the model device.

        Args:
            value: Feature tensor or processor tuple.

        Returns:
            Finite float32 tensor of shape ``[batch, input_dim]``.

        Raises:
            ValueError: If the feature shape or values are invalid.
        """
        x = torch.as_tensor(
            _value_tensor(value), device=self.device, dtype=torch.float32,
        )
        if x.ndim != 2 or x.shape[1] != self.input_dim or x.shape[0] == 0:
            raise ValueError("features must have shape [batch, input_dim]")
        if not torch.isfinite(x).all():
            raise ValueError("features must be finite")
        return x

    def _outcome(self, value: Tensor | tuple[Tensor, ...],
                 name: str, batch_size: int) -> Tensor:
        """Validate durations or right-censoring indicators.

        Args:
            value: Outcome tensor or processor tuple.
            name: Either ``duration`` or ``event``.
            batch_size: Expected number of samples in this batch.

        Returns:
            Tensor of shape ``[batch, n_events]`` on the model device.

        Raises:
            ValueError: If the outcome shape, range, or values are invalid.
        """
        result = torch.as_tensor(
            _value_tensor(value), device=self.device, dtype=torch.float32,
        )
        if result.ndim == 1 and self.n_events == 1:
            result = result.unsqueeze(1)
        if result.shape != (batch_size, self.n_events):
            raise ValueError(f"{name} must have shape [batch, n_events]")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain finite values")
        if name == "duration" and not torch.all(result > 0):
            raise ValueError("duration must be strictly positive")
        if name == "event" and not torch.all((result == 0) | (result == 1)):
            raise ValueError("event must contain only 0/1 indicators")
        return result

    def occurrence_logits(self, x: Tensor) -> Tensor:
        """Compute event-occurrence logits.

        Args:
            x: Dense input features with shape ``[batch, input_dim]``.

        Returns:
            Eventual-occurrence logits with shape ``[batch, n_events]``.
        """
        return self.occurrence_head(self.encoder(x))

    def sample_occurrence(self, logits: Tensor, n_samples: int) -> Tensor:
        """Sample relaxed occurrence indicators with the binary Concrete law.

        As in the reference TensorFlow code, each Monte Carlo sample uses one
        shared logistic noise draw across patients and event heads.

        Args:
            logits: Occurrence logits with shape ``[batch, n_events]``.
            n_samples: Positive number of Monte Carlo samples.

        Returns:
            Differentiable samples with shape ``[batch, samples, n_events]``.

        Raises:
            ValueError: If ``n_samples`` is not positive.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        u1 = torch.rand(n_samples, device=logits.device, dtype=logits.dtype)
        u2 = torch.rand(n_samples, device=logits.device, dtype=logits.dtype)
        u1 = u1.clamp(1e-6, 1 - 1e-6)
        u2 = u2.clamp(1e-6, 1 - 1e-6)
        logistic = (-torch.log(-torch.log(u1))
                    + torch.log(-torch.log(u2)))
        return torch.sigmoid(
            (logits[:, None, :] + logistic[None, :, None]) / self.temperature
        )

    def conditional_time_params(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Compute conditional log-normal time parameters.

        Args:
            x: Feature matrix with shape ``[batch, input_dim]``.
            c: Relaxed occurrence samples ``[batch, samples, n_events]``.

        Returns:
            Tuple of mean and log-variance tensors, each shaped like ``c``.

        Raises:
            ValueError: If ``c`` has the wrong shape.
        """
        if (c.ndim != 3 or c.shape[0] != x.shape[0] or
                c.shape[2] != self.n_events):
            raise ValueError("c must have shape [batch, samples, n_events]")
        xs = x[:, None, :].expand(-1, c.shape[1], -1)
        hidden = self.decoder(torch.cat((xs, c), dim=-1))
        logvar = self.time_logvar(hidden)
        if self.logvar_bounds is not None:
            logvar = logvar.clamp(*self.logvar_bounds)
        return self.time_mean(hidden), logvar

    @staticmethod
    def lognormal_negative_terms(
        t: Tensor, mu: Tensor, logvar: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute negative log-density and log-survival for event times.

        The tiny offset inside the logarithm follows the reference code.

        Args:
            t: Positive observed or censored durations; broadcastable to mu.
            mu: Conditional means of log event time.
            logvar: Conditional log-variances of log event time.

        Returns:
            Negative log-density and negative log-survival tensors.

        Raises:
            ValueError: If any duration is nonpositive.
        """
        if torch.any(t <= 0):
            raise ValueError("duration must be strictly positive")
        logt = torch.log(t + 1e-4)
        sd = torch.exp(0.5 * logvar)
        z = (logt - mu) / sd
        nlogpdf = logt + torch.log(sd) + 0.5 * z.square()
        nlogpdf = nlogpdf + 0.5 * math.log(2 * math.pi)
        nlogsurv = -torch.special.log_ndtr(-z)
        return nlogpdf, nlogsurv

    @staticmethod
    def _sample_median_time(mu: Tensor) -> Tensor:
        """Median of sampled conditional times along the sample dimension.

        Preserves the linear-interpolated median of torch.quantile(q=0.5)
        for both odd and even sample counts without calling torch.quantile.

        Args:
            mu: Sampled log-time means [batch, samples, events].

        Returns:
            Predicted event times [batch, events].
        """
        times = torch.exp(mu.clamp(max=40))
        sorted_times = torch.sort(times, dim=1).values

        n_samples = sorted_times.shape[1]
        middle = n_samples // 2

        if n_samples % 2:
            return sorted_times[:, middle, :]

        lower = sorted_times[:, middle - 1, :]
        upper = sorted_times[:, middle, :]
        return lower + 0.5 * (upper - lower)

    def censored_nll(
        self, logits: Tensor, c: Tensor, mu: Tensor, logvar: Tensor,
        duration: Tensor, event: Tensor,
    ) -> Tensor:
        """Evaluate the reference Monte Carlo CET training objective.

        The loss averages over patients, samples, and events. It is an MC
        relaxation, not the exact mixture-cure log-likelihood; censored
        observations are not treated as definite negative occurrence labels.

        Args:
            logits: Event-occurrence logits ``[batch, n_events]``.
            c: Relaxed occurrence samples ``[batch, samples, n_events]``.
            mu: Conditional log-time means with the same shape as ``c``.
            logvar: Conditional log-time variances with the same shape as ``c``.
            duration: Observed or censoring times ``[batch, n_events]``.
            event: Observed-event indicators ``[batch, n_events]``.

        Returns:
            Scalar Monte Carlo CET objective to minimize.
        """
        t, s = duration.unsqueeze(1), event.unsqueeze(1)
        nlogpdf, nlogsurv = self.lognormal_negative_terms(t, mu, logvar)
        penalty = (self.nlog_fpr + F.softplus(-logits)).unsqueeze(1)
        values = (s * nlogpdf + (1 - s) * c * nlogsurv +
                  s * (1 - c) * penalty)
        return values.mean()

    def forward(self, **kwargs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Predict occurrence and conditional timing, computing loss if labeled.

        The labeled loss is a Monte Carlo estimate: relaxed occurrence samples
        are drawn in both training and evaluation modes. Repeated evaluation
        calls can therefore produce different ``loss`` and ``time_pred``
        values unless the random seed is controlled. In evaluation mode,
        ``logit`` and ``y_prob`` do not depend on these samples.

        Args:
            **kwargs: PyHealth batch with ``features`` and optionally both
                ``duration`` and ``event``.
        Returns:
            ``logit`` and ``y_prob`` [B,M], ``time_pred`` [B,M], and, when
            labeled, ``loss`` (scalar) and ``y_true`` (observed indicator).
        """
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing required field {self.feature_key!r}")
        if (self.duration_key in kwargs) != (self.event_key in kwargs):
            raise KeyError("duration and event must both be supplied or omitted")
        x = self._features(kwargs[self.feature_key])
        logits = self.occurrence_logits(x)
        c = self.sample_occurrence(logits, self.n_samples)
        mu, logvar = self.conditional_time_params(x, c)
        output = {
            "logit": logits,
            "y_prob": torch.sigmoid(logits),
            "time_pred": self._sample_median_time(mu),
        }
        if self.duration_key in kwargs:
            duration = self._outcome(kwargs[self.duration_key], "duration", len(x))
            event = self._outcome(kwargs[self.event_key], "event", len(x))
            output["loss"] = self.censored_nll(
                logits, c, mu, logvar, duration, event,
            )
            output["y_true"] = event
        return output

    def forward_from_embedding(
        self, embeddings: Tensor, **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run inference using supplied dense feature embeddings.

        Args:
            embeddings: Dense features shaped ``[batch, input_dim]``.
            **kwargs: Other optional survival batch fields.

        Returns:
            The same output dictionary as :meth:`forward`.
        """
        kwargs[self.feature_key] = embeddings
        return self.forward(**kwargs)

    @torch.no_grad()
    def predict_event_time(self, features: Tensor, n_samples: int = 100) -> Tensor:
        """Predict event times from sampled conditional log-normal medians.

        Temporarily switches to evaluation mode and restores the prior state.
        This prediction does not force a target occurrence indicator to one.

        Args:
            features: Dense feature matrix ``[batch, input_dim]``.
            n_samples: Positive number of Monte Carlo samples.

        Returns:
            Time predictions with shape ``[batch, n_events]``.
        """
        was_training = self.training
        try:
            self.eval()
            x = self._features(features)
            logits = self.occurrence_logits(x)
            c = self.sample_occurrence(logits, n_samples)
            mu, _ = self.conditional_time_params(x, c)
            return self._sample_median_time(mu)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def posterior_occurrence(
        self, features: Tensor, duration: Tensor, event: Tensor,
        n_samples: int = 100,
    ) -> Tensor:
        """Approximate eventual-event posteriors for soft-label analysis.

        For censored events, average survival over samples of other occurrence
        variables while forcing this event to one. Observed events have
        posterior probability one. This is not an exact joint posterior.

        Args:
            features: Dense feature matrix ``[batch, input_dim]``.
            duration: Observed or censoring times ``[batch, n_events]``.
            event: Observed-event indicators ``[batch, n_events]``.
            n_samples: Positive number of Monte Carlo samples.

        Returns:
            Posterior occurrence estimates ``[batch, n_events]``.
        """
        was_training = self.training
        try:
            self.eval()
            x = self._features(features)
            t = self._outcome(duration, "duration", len(x))
            s = self._outcome(event, "event", len(x))
            logits = self.occurrence_logits(x)
            p = torch.sigmoid(logits)
            c = self.sample_occurrence(logits, n_samples)
            survival = []
            for j in range(self.n_events):
                forced = c.clone()
                forced[..., j] = 1
                mu, lv = self.conditional_time_params(x, forced)
                _, neg_log_surv = self.lognormal_negative_terms(
                    t[:, None, j], mu[..., j], lv[..., j],
                )
                survival.append(torch.exp(-neg_log_surv).mean(dim=1))
            surv = torch.stack(survival, dim=1)
            numerator = p * surv
            posterior = numerator / (numerator + 1 - p).clamp_min(1e-12)
            return torch.where(s.bool(), torch.ones_like(posterior), posterior)
        finally:
            self.train(was_training)
