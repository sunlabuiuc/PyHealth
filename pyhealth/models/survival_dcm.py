"""Authors: Murphy Crosby

Deep Cox Mixtures survival model.
Paper: Deep Cox Mixtures for Survival Regression, Nagpal et al., MLHC 2021.
https://proceedings.mlr.press/v149/nagpal21a.html
"""

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pyhealth.datasets import SampleDataset
from pyhealth.processors import LabelProcessor

from .base_model import BaseModel


def cox_partial_log_likelihood(
    log_hazard_ratio: Tensor, duration: Tensor, event: Tensor
) -> Tensor:
    """Cox partial log-likelihood of one group of patients.

    For each patient with an observed event, this compares that patient's
    hazard with the total hazard of everyone who was still at risk at that
    time. Censored patients add no term of their own, but they do count as
    at risk until their follow-up ends.

    Args:
        log_hazard_ratio: Log hazard ratios, shape ``[n]``.
        duration: Follow-up times, shape ``[n]``.
        event: 1 for an observed event and 0 for censoring, shape ``[n]``.

    Returns:
        The partial log-likelihood summed over observed events, a scalar. It
        is 0 when the group has no observed events.

    Examples:
        >>> import torch
        >>> log_hazard_ratio = torch.log(torch.tensor([1.0, 2.0, 3.0]))
        >>> duration = torch.tensor([3.0, 2.0, 1.0])
        >>> event = torch.ones(3)
        >>> value = cox_partial_log_likelihood(log_hazard_ratio, duration, event)
        >>> round(value.item(), 4)
        -1.0986
    """
    total = log_hazard_ratio.new_zeros(())
    for i in range(len(duration)):
        if event[i] == 0:
            continue

        # A patient is still at risk at time duration[i] if their own
        # follow-up lasted at least that long. Patients tied at the same
        # time are all included.
        still_at_risk = duration >= duration[i]

        # log of the summed hazard ratios of the at-risk patients. logsumexp
        # does this without leaving log space, which avoids overflow.
        log_total_risk = torch.logsumexp(log_hazard_ratio[still_at_risk], dim=0)

        total = total + log_hazard_ratio[i] - log_total_risk
    return total


@torch.no_grad()
def breslow_baseline_survival(
    log_hazard_ratio: Tensor, duration: Tensor, event: Tensor
) -> tuple[Tensor, Tensor]:
    """Breslow estimate of the baseline survival curve of one group.

    The baseline curve is the survival of a patient whose hazard ratio is 1.
    It is a step function that drops at each time an event was observed. The
    size of each drop is the number of events at that time divided by the
    total hazard of the patients still at risk.

    Args:
        log_hazard_ratio: Log hazard ratios, shape ``[n]``.
        duration: Follow-up times, shape ``[n]``.
        event: 1 for an observed event and 0 for censoring, shape ``[n]``.

    Returns:
        The distinct event times in increasing order, and the baseline
        survival probability just after each of them. Both have shape
        ``[m]``, where ``m`` is the number of distinct event times. Both are
        empty if the group has no observed events.

    Examples:
        >>> import torch
        >>> times, survival = breslow_baseline_survival(
        ...     torch.zeros(4),
        ...     torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ...     torch.tensor([1.0, 1.0, 0.0, 1.0]),
        ... )
        >>> times.tolist()
        [1.0, 2.0, 4.0]
        >>> [round(value, 3) for value in survival.tolist()]
        [0.779, 0.558, 0.205]
    """
    hazard_ratio = torch.exp(log_hazard_ratio)
    event_times = torch.unique(duration[event == 1])  # sorted, no repeats

    cumulative_hazard = 0.0
    survival = []
    for time in event_times:
        deaths_now = ((duration == time) & (event == 1)).sum()

        # Everyone whose follow-up lasted at least this long is at risk.
        total_risk = hazard_ratio[duration >= time].sum()

        cumulative_hazard = cumulative_hazard + deaths_now / total_risk
        survival.append(torch.exp(-cumulative_hazard))

    if not survival:
        return event_times, hazard_ratio.new_zeros(0)
    return event_times, torch.stack(survival)


def smooth_baseline_survival(
    event_times: Tensor, survival: Tensor, grid: Tensor, smoothing: float = 1e-4
) -> tuple[Tensor, Tensor]:
    """Smooth a Breslow step curve and tabulate it on a time grid.

    The Breslow estimate is a staircase, so it has no usable slope. A cubic
    smoothing spline through its points gives a curve whose slope tells how
    fast survival is falling at any time, which the E-step needs.

    Args:
        event_times: Distinct event times in increasing order, shape ``[m]``.
        survival: Baseline survival after each event time, shape ``[m]``.
        grid: Increasing times at which to tabulate the curve, shape ``[g]``.
        smoothing: Smoothing factor of the spline. 0 passes through every
            point; larger values give a smoother curve.

    Returns:
        The smoothed survival and its slope (change in survival per unit
        time, never positive) at each grid time. Both have shape ``[g]``. A
        group with no events gets a flat curve at 1 with slope 0.

    Examples:
        >>> import torch
        >>> event_times = torch.tensor([10.0, 20.0, 30.0, 40.0])
        >>> survival = torch.tensor([0.9, 0.7, 0.5, 0.4])
        >>> grid = torch.linspace(0.0, 40.0, 5)
        >>> smooth, slope = smooth_baseline_survival(event_times, survival, grid)
        >>> smooth.shape, slope.shape
        (torch.Size([5]), torch.Size([5]))
        >>> bool((slope <= 0).all())
        True
    """
    if len(event_times) == 0:
        return torch.ones_like(grid), torch.zeros_like(grid)

    times = event_times.cpu().numpy().astype(float)
    values = survival.cpu().numpy().astype(float)
    grid_times = grid.cpu().numpy().astype(float)

    # Everyone is alive at time 0, so the curve starts at 1.
    if times[0] > 0:
        times = np.concatenate([[0.0], times])
        values = np.concatenate([[1.0], values])

    # A cubic spline needs at least four points; use a lower degree if not.
    degree = min(3, len(times) - 1)
    if degree == 0:
        smooth = np.full_like(grid_times, values[0])
    else:
        spline = UnivariateSpline(times, values, k=degree, s=smoothing)
        # Past the last event the data says nothing more, so hold the curve
        # at its last value instead of letting the spline extrapolate.
        smooth = spline(np.minimum(grid_times, times[-1]))

    # Keep it a valid survival curve: between 0 and 1, and never rising.
    smooth = np.clip(smooth, 1e-6, 1.0)
    smooth = np.minimum.accumulate(smooth)

    slope = np.gradient(smooth, grid_times)
    return (
        torch.as_tensor(smooth, dtype=grid.dtype, device=grid.device),
        torch.as_tensor(slope, dtype=grid.dtype, device=grid.device),
    )


class SurvivalDCM(BaseModel):
    """Deep Cox Mixtures for right-censored survival data.

    A shared multilayer perceptron maps each patient's features to a latent
    representation. Two linear heads read that representation: a gate that
    gives the log-probability of belonging to each of ``num_components``
    latent subgroups, and an expert that gives one log hazard ratio per
    subgroup. Each subgroup is a Cox model with its own baseline survival
    function, so proportional hazards only has to hold within a subgroup.

    Training follows the approximate EM procedure of the paper. Call
    :meth:`fit_baselines` once before training and again after every epoch,
    from your own training loop. PyHealth's ``Trainer.train`` does not refit
    the baseline curves between epochs, so on its own it would train against
    curves that never change. ``Trainer.inference`` and ``Trainer.evaluate``
    work on a fitted model.

    Note:
        This implementation differs from the authors' in a few ways. The
        baseline curves are refitted once per epoch, not every few batches.
        They are smoothed with a cubic spline, as the paper describes, where
        the authors' released PyTorch code uses a linear one. The optional
        variational auto-encoder term of the paper is not included. The log
        hazard ratio is bounded with a scaled tanh, and the default bound of
        1 matches the authors' default.

    Args:
        dataset: Fitted PyHealth ``SampleDataset`` with one dense feature
            vector and two tensor outputs, duration and event.
        feature_key: Input field holding the feature vector.
        duration_key: Output field holding positive follow-up times.
        event_key: Output field holding 1 for an observed event and 0 for
            right censoring.
        hidden_dim: Width of each hidden layer of the encoder.
        num_hidden_layers: Number of hidden layers in the encoder. With 0
            the hazard ratios are a linear function of the features.
        num_components: Number of latent subgroups, K. With 1, the model is
            a single deep Cox model.
        num_grid_points: Number of evenly spaced times at which each
            subgroup's baseline survival curve is stored.
        max_log_hazard_ratio: Largest absolute log hazard ratio the expert
            can output. A patient's hazard within a subgroup is then between
            ``exp(-max)`` and ``exp(max)`` times the baseline.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "features": [0.0, 1.0, -0.5],
        ...         "duration": [12.0],
        ...         "event": [1.0],
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "features": [1.0, 0.2, 0.4],
        ...         "duration": [30.0],
        ...         "event": [0.0],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = SurvivalDCM(dataset, hidden_dim=8, num_components=3)
        >>> model.input_dim, model.num_components
        (3, 3)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        hidden_dim: int = 100,
        num_hidden_layers: int = 2,
        num_components: int = 3,
        num_grid_points: int = 2049,
        max_log_hazard_ratio: float = 1.0,
    ) -> None:
        super().__init__(dataset)
        if feature_key not in dataset.input_schema:
            raise ValueError(f"feature_key {feature_key!r} is not an input field")
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"{key!r} is not an output field")
        if max_log_hazard_ratio <= 0:
            raise ValueError("max_log_hazard_ratio must be positive")
        if num_grid_points < 2:
            raise ValueError("num_grid_points must be at least 2")
        if hidden_dim < 1 or num_hidden_layers < 0 or num_components < 1:
            raise ValueError(
                "hidden_dim and num_components must be positive, and "
                "num_hidden_layers must not be negative"
            )
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.num_components = num_components
        self.max_log_hazard_ratio = max_log_hazard_ratio
        self.input_dim = dataset.input_processors[feature_key].size()

        layers: list[nn.Module] = []
        width = self.input_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.encoder = nn.Sequential(*layers)
        # With no hidden layers the heads read the features directly.
        self.gate = nn.Linear(width, num_components)
        self.expert = nn.Linear(width, num_components, bias=False)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Each subgroup's baseline survival curve is stored as a table: its
        # value and slope at evenly spaced times. Buffers are saved with the
        # model but are not trained by the optimizer. Until the curves are
        # fitted, the grid runs from 0 to 1 and every curve is flat at 1.
        self.register_buffer("baseline_grid", torch.linspace(0.0, 1.0, num_grid_points))
        self.register_buffer(
            "baseline_survival", torch.ones(num_components, num_grid_points)
        )
        self.register_buffer(
            "baseline_slope", torch.zeros(num_components, num_grid_points)
        )
        # Turns True the first time fit_baselines() runs.
        self.register_buffer("baselines_fitted", torch.tensor(False))

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Empty: duration and event are scored by the model's own loss."""
        return {}

    def baseline_at(self, time: Tensor) -> tuple[Tensor, Tensor]:
        """Look up every subgroup's baseline curve at the given times.

        The curves are stored on a grid, so a time between two grid points is
        read off by linear interpolation. Times beyond the grid use its last
        point.

        Args:
            time: Times, shape ``[n]``.

        Returns:
            The baseline survival and its slope for each time and subgroup.
            Both have shape ``[n, num_components]``.
        """
        last = len(self.baseline_grid) - 1
        step = self.baseline_grid[1] - self.baseline_grid[0]

        # Where each time falls on the grid, as a number of steps from 0.
        position = (time / step).clamp(min=0, max=last)
        left = position.floor().long().clamp(max=last - 1)
        fraction = position - left  # 0 at the left grid point, 1 at the right

        survival = self.baseline_survival[:, left] + fraction * (
            self.baseline_survival[:, left + 1] - self.baseline_survival[:, left]
        )
        slope = self.baseline_slope[:, left] + fraction * (
            self.baseline_slope[:, left + 1] - self.baseline_slope[:, left]
        )
        # Stored as [num_components, n]; return as [n, num_components].
        return survival.T, slope.T

    @torch.no_grad()
    def subgroup_posterior(
        self,
        log_gate: Tensor,
        log_hazard_ratio: Tensor,
        duration: Tensor,
        event: Tensor,
    ) -> Tensor:
        """E-step: how likely each subgroup is, given what happened to a patient.

        For every subgroup this asks how well it explains the patient's
        outcome: dying at the observed time if the event was seen, or
        surviving past it if the patient was censored. That is combined with
        the gate's belief from the features alone and normalized so the
        probabilities of the subgroups sum to 1.

        Args:
            log_gate: Log-probability of each subgroup from the gate, shape
                ``[n, num_components]``.
            log_hazard_ratio: Log hazard ratio under each subgroup, same shape.
            duration: Follow-up times, shape ``[n]``.
            event: 1 for an observed event and 0 for censoring, shape ``[n]``.

        Returns:
            Posterior probability of each subgroup, shape
            ``[n, num_components]``. Each row sums to 1.
        """
        baseline_survival, baseline_slope = self.baseline_at(duration)

        # Surviving past the observed time in a subgroup: the baseline
        # survival raised to the power of the patient's hazard ratio.
        log_survival = torch.exp(log_hazard_ratio) * torch.log(baseline_survival)

        # Hazard at the observed time: how fast the baseline curve is falling
        # relative to its height, scaled by the patient's hazard ratio. The
        # small floor keeps the log finite where a curve is flat.
        baseline_hazard = (-baseline_slope / baseline_survival).clamp(min=1e-8)
        log_hazard = torch.log(baseline_hazard) + log_hazard_ratio

        # A censored patient only tells us they survived this long. A patient
        # with an event also tells us the event happened right then.
        log_likelihood = log_survival + event.unsqueeze(1) * log_hazard

        return torch.softmax(log_gate + log_likelihood, dim=1)

    @torch.no_grad()
    def sample_subgroup(self, posterior: Tensor) -> Tensor:
        """Draw one subgroup per patient at random from the posterior.

        Args:
            posterior: Posterior probabilities, shape ``[n, num_components]``.

        Returns:
            The index of the sampled subgroup for each patient, shape ``[n]``.
        """
        return torch.multinomial(posterior, num_samples=1).squeeze(1)

    @torch.no_grad()
    def fit_baselines(self, dataloader: DataLoader, smoothing: float = 1e-4) -> None:
        """Refit every subgroup's baseline survival curve from training data.

        Call this once before training and again after every epoch. Each
        patient is assigned to a subgroup, then each subgroup's curve is
        estimated from its own patients with the Breslow estimator and
        smoothed with a cubic spline. No gradients are involved.

        The first call has no curves to compute posteriors with, so it
        assigns patients to subgroups uniformly at random. Later calls sample
        each patient's subgroup from the posterior of the E-step.

        Args:
            dataloader: Training data. Shuffling does not matter.
            smoothing: Smoothing factor of the splines.
        """
        was_training = self.training
        self.eval()

        # Run the whole training set through the network.
        log_gates, log_hazard_ratios, durations, events = [], [], [], []
        for batch in dataloader:
            output = self(**{self.feature_key: batch[self.feature_key]})
            log_gates.append(output["log_gate"])
            log_hazard_ratios.append(output["log_hazard_ratio"])
            duration = batch[self.duration_key].to(self.device).float()
            event = batch[self.event_key].to(self.device).float()
            durations.append(duration.reshape(-1))
            events.append(event.reshape(-1))
        log_gate = torch.cat(log_gates)
        log_hazard_ratio = torch.cat(log_hazard_ratios)
        duration = torch.cat(durations)
        event = torch.cat(events)

        # Assign every patient to one subgroup.
        if self.baselines_fitted:
            posterior = self.subgroup_posterior(
                log_gate, log_hazard_ratio, duration, event
            )
            assigned = self.sample_subgroup(posterior)
        else:
            assigned = torch.randint(
                self.num_components, duration.shape, device=self.device
            )

        # The curves are stored from time 0 to the longest follow-up seen.
        grid = torch.linspace(
            0.0, float(duration.max()), len(self.baseline_grid), device=self.device
        )

        for k in range(self.num_components):
            members = assigned == k
            event_times, survival = breslow_baseline_survival(
                log_hazard_ratio[members, k], duration[members], event[members]
            )
            smooth, slope = smooth_baseline_survival(
                event_times, survival, grid, smoothing
            )
            self.baseline_survival[k] = smooth
            self.baseline_slope[k] = slope

        self.baseline_grid.copy_(grid)
        self.baselines_fitted.fill_(True)
        self.train(was_training)

    @torch.no_grad()
    def _mixture_survival(
        self, log_gate: Tensor, log_hazard_ratio: Tensor, times: Tensor
    ) -> Tensor:
        """Survival probability of each patient at each time.

        Within a subgroup, a patient's survival is the baseline survival
        raised to the power of their hazard ratio. The subgroup is unknown at
        prediction time, so the subgroups are averaged with the gate's
        probabilities as weights.

        Args:
            log_gate: Log-probability of each subgroup, shape
                ``[n, num_components]``.
            log_hazard_ratio: Log hazard ratio under each subgroup, same shape.
            times: Times at which to predict, shape ``[num_times]``.

        Returns:
            Survival probabilities, shape ``[n, num_times]``.
        """
        baseline, _ = self.baseline_at(times)  # [num_times, num_components]
        hazard_ratio = torch.exp(log_hazard_ratio)  # [n, num_components]

        # Survival within each subgroup: [n, num_times, num_components].
        within_subgroup = baseline.unsqueeze(0) ** hazard_ratio.unsqueeze(1)

        # Weighted average over the subgroups: [n, num_times].
        gate = torch.exp(log_gate).unsqueeze(1)
        return (gate * within_subgroup).sum(dim=2)

    def predict_survival(self, features: Tensor, times: Tensor) -> Tensor:
        """Predict each patient's survival probability at the given times.

        Args:
            features: Feature vectors, shape ``[n, input_dim]``.
            times: Times at which to predict, shape ``[num_times]``.

        Returns:
            Survival probabilities, shape ``[n, num_times]``. Entry ``[i, j]``
            is the probability that patient ``i`` survives past ``times[j]``.

        Raises:
            RuntimeError: If the baseline curves have not been fitted.
        """
        if not self.baselines_fitted:
            raise RuntimeError("Call fit_baselines() before predicting survival")
        output = self(**{self.feature_key: features})
        times = times.to(device=self.device, dtype=torch.float32)
        return self._mixture_survival(
            output["log_gate"], output["log_hazard_ratio"], times
        )

    def forward(self, **kwargs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Score a batch under every subgroup and compute the training loss.

        The loss is one step of the approximate EM procedure of the paper.
        The E-step gives each patient a posterior over subgroups, and one
        subgroup is drawn from it. The loss then has two parts: the gate is
        trained to predict the posterior from the features alone, and each
        subgroup's hazard ratios are trained with the Cox partial likelihood
        of the patients assigned to it.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader. Must include
                ``feature_key``. The loss is computed when ``duration_key``
                and ``event_key`` are also present.

        Returns:
            ``log_gate``: log-probability of each subgroup, shape
            ``[batch, num_components]``. ``log_hazard_ratio``: log hazard
            ratio under each subgroup, same shape. ``loss``: the training
            loss averaged over the batch, when the labels are present.

            In evaluation mode with labels, also ``y_true``: duration and
            event side by side, shape ``[batch, 2]``, and ``y_prob``: the
            predicted survival curve at every time in ``baseline_grid``,
            shape ``[batch, num_grid_points]``. Column ``j`` of ``y_prob``
            is the survival at ``baseline_grid[j]``. ``fit_baselines`` resets
            that grid, so read it from the same fitted model that produced
            ``y_prob``. To predict at chosen times, use
            :meth:`predict_survival`.

        Raises:
            ValueError: If the features have the wrong shape or are not finite.
            RuntimeError: If called in training mode before the baseline
                curves have been fitted.
        """
        features = kwargs[self.feature_key]
        if isinstance(features, tuple):
            features = features[0]
        features = features.to(device=self.device, dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}]; "
                f"got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(
                "features must be finite; standardize them so that missing "
                "values are imputed before training"
            )
        if self.training and not self.baselines_fitted:
            raise RuntimeError("Call fit_baselines() before training")
        hidden = self.encoder(features)
        log_gate = self.log_softmax(self.gate(hidden))
        # Keep every log hazard ratio between -max and +max. Tanh squashes
        # any number into (-1, 1), and the multiplier sets the range.
        log_hazard_ratio = self.max_log_hazard_ratio * self.tanh(self.expert(hidden))
        output = {"log_gate": log_gate, "log_hazard_ratio": log_hazard_ratio}

        if self.duration_key not in kwargs or self.event_key not in kwargs:
            return output

        # The dataloader gives the labels as [batch, 1]; flatten to [batch].
        duration = kwargs[self.duration_key].to(self.device).float().reshape(-1)
        event = kwargs[self.event_key].to(self.device).float().reshape(-1)

        # E-step: posterior over subgroups, then one subgroup per patient.
        # While training the subgroup is sampled, as in the paper. When
        # evaluating, the most likely subgroup is used so the loss is not
        # random.
        posterior = self.subgroup_posterior(log_gate, log_hazard_ratio, duration, event)
        if self.training:
            assigned = self.sample_subgroup(posterior)
        else:
            assigned = posterior.argmax(dim=1)

        # First part: the gate should predict the posterior.
        gate_term = (posterior * log_gate).sum()

        # Second part: one Cox partial likelihood per subgroup, each over the
        # patients assigned to that subgroup.
        cox_term = log_gate.new_zeros(())
        for k in range(self.num_components):
            members = assigned == k
            cox_term = cox_term + cox_partial_log_likelihood(
                log_hazard_ratio[members, k], duration[members], event[members]
            )

        # Both parts are log-likelihoods to maximize, so the loss is their
        # negative, averaged over the batch.
        output["loss"] = -(gate_term + cox_term) / len(duration)

        # When evaluating, also return what PyHealth's Trainer collects: the
        # labels, and each patient's survival curve at the stored grid times.
        if not self.training:
            output["y_true"] = torch.stack([duration, event], dim=1)
            output["y_prob"] = self._mixture_survival(
                log_gate, log_hazard_ratio, self.baseline_grid
            )
        return output
