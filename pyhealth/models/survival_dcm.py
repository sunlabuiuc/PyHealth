"""Authors: Murphy Crosby

Deep Cox Mixtures survival model.
Paper: Deep Cox Mixtures for Survival Regression, Nagpal et al., MLHC 2021.
https://proceedings.mlr.press/v149/nagpal21a.html
"""

import torch
from torch import Tensor, nn

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


class SurvivalDCM(BaseModel):
    """Deep Cox Mixtures for right-censored survival data.

    A shared multilayer perceptron maps each patient's features to a latent
    representation. Two linear heads read that representation: a gate that
    gives the log-probability of belonging to each of ``num_components``
    latent subgroups, and an expert that gives one log hazard ratio per
    subgroup. Each subgroup is a Cox model with its own baseline survival
    function, so proportional hazards only has to hold within a subgroup.

    Args:
        dataset: Fitted PyHealth ``SampleDataset`` with one dense feature
            vector and two tensor outputs, duration and event.
        feature_key: Input field holding the feature vector.
        duration_key: Output field holding positive follow-up times.
        event_key: Output field holding 1 for an observed event and 0 for
            right censoring.
        hidden_dim: Width of each hidden layer of the encoder.
        num_hidden_layers: Number of hidden layers in the encoder.
        num_components: Number of latent subgroups, K. With 1, the model is
            a single deep Cox model.

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
    ) -> None:
        super().__init__(dataset)
        if feature_key not in dataset.input_schema:
            raise ValueError(f"feature_key {feature_key!r} is not an input field")
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"{key!r} is not an output field")
        if hidden_dim < 1 or num_hidden_layers < 1 or num_components < 1:
            raise ValueError(
                "hidden_dim, num_hidden_layers and num_components must be positive"
            )
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.num_components = num_components
        self.input_dim = dataset.input_processors[feature_key].size()

        layers: list[nn.Module] = []
        width = self.input_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.gate = nn.Linear(hidden_dim, num_components)
        self.expert = nn.Linear(hidden_dim, num_components)
        self.log_softmax = nn.LogSoftmax(dim=1)

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Empty: duration and event are scored by the model's own loss."""
        return {}

    def forward(self, **kwargs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Encode a batch and score each patient under every subgroup.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader. Must include
                ``feature_key``.

        Returns:
            ``log_gate``: log-probability of each subgroup, shape
            ``[batch, num_components]``. ``log_hazard_ratio``: log hazard
            ratio under each subgroup, same shape.

        Raises:
            ValueError: If the features have the wrong shape or are not finite.
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
        hidden = self.encoder(features)
        return {
            "log_gate": self.log_softmax(self.gate(hidden)),
            "log_hazard_ratio": self.expert(hidden),
        }
