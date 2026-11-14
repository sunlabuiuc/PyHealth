"""Time-varying Cox proportional hazards model for MAS-style survival tasks."""

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


def cox_time_varying_loss(
    log_risk: torch.Tensor, intervals: torch.Tensor
) -> torch.Tensor:
    """Calculate Breslow Cox negative partial log likelihood.

    Args:
        log_risk: Log-relative-hazard scores, shape [B] or [B, 1].
        intervals: Columns [start, stop, event], shape [B, 3].
            The interval convention is (start, stop]. All relevant patient
            intervals must be present together in the batch. Ties use Breslow.

    Returns:
        Scalar negative partial log likelihood divided by event count.

    Raises:
        ValueError: If interval data is malformed or contains no events.
    """
    scores = log_risk.reshape(-1)
    if intervals.ndim != 2 or intervals.shape[1] != 3:
        raise ValueError("intervals must have shape [B, 3]")
    if scores.numel() != intervals.shape[0]:
        raise ValueError("scores and intervals must have the same B")
    if not torch.isfinite(scores).all() or not torch.isfinite(intervals).all():
        raise ValueError("scores and intervals must be finite")

    start, stop, event = intervals.unbind(dim=1)
    if not torch.all(start < stop):
        raise ValueError("each interval must satisfy start < stop")
    if not torch.all((event == 0) | (event == 1)):
        raise ValueError("event must be 0 or 1")

    event_mask = event == 1
    if not event_mask.any():
        raise ValueError("Cox loss requires at least one observed event")

    # Each event row supplies its own event time. Breslow handles equal times
    # by counting each event with the same full risk-set denominator.
    event_times = stop[event_mask]
    risk_mask = (start.unsqueeze(0) < event_times.unsqueeze(1)) & (
        stop.unsqueeze(0) >= event_times.unsqueeze(1)
    )
    # Stable log sum exp for each event's entire risk set.
    log_denominator = torch.logsumexp(
        scores.unsqueeze(0)
        .expand(event_times.numel(), -1)
        .masked_fill(~risk_mask, float("-inf")),
        dim=1,
    )
    return (log_denominator - scores[event_mask]).mean()


class MAS(BaseModel):
    """Trainable linear time-varying Cox model for PyHealth survival samples.

    Based on the Model for Allograft Survival described by Gao et al. in
    `Predicting Long-Term Allograft Survival in Liver Transplant Recipients
    <https://proceedings.mlr.press/v252/gao24a.html>`_ (MLHC 2024).

    Uses full-risk-set batches and Breslow tie approximation. The model
    predicts relative log hazard, not an absolute event probability.

    Args:
        dataset: PyHealth SampleDataset with ``features`` input and
            ``survival_interval`` output processors.
        input_dim: Optional override for the number of features.

    Examples:
        >>> from pyhealth.datasets import create_synthetic_liver_dataset, get_dataloader
        >>> from pyhealth.models import MAS
        >>> dataset = create_synthetic_liver_dataset(n_patients=100, seed=0)
        >>> loader = get_dataloader(dataset, batch_size=len(dataset), shuffle=False)
        >>> model = MAS(dataset)
        >>> output = model(**next(iter(loader)))
        >>> output["loss"].backward()
        >>> output["y_prob"].shape == (len(dataset), 1)
        True
    """

    def __init__(self, dataset: SampleDataset, input_dim: int | None = None) -> None:
        super().__init__(dataset=dataset)
        if input_dim is None:
            input_dim = dataset.input_processors["features"].size()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        survival_interval: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute log-risk predictions and Cox survival loss.

        Args:
            features: Standardized features of shape [B, F].
            survival_interval: Labels [start, stop, event] of shape [B, 3].
            **kwargs: Extra metadata fields from PyHealth's dataloader.

        Returns:
            Dictionary with ``loss``, ``y_prob``, ``y_true``, and ``logit``.
            ``y_prob`` is the log-relative-risk score, not a probability.
        """
        if features.ndim != 2 or features.shape[1] != self.linear.in_features:
            raise ValueError("features must be [B, input_dim]")
        log_risk = self.linear(features)
        loss = cox_time_varying_loss(log_risk, survival_interval)
        return {
            "loss": loss,
            "y_prob": log_risk,
            "y_true": survival_interval,
            "logit": log_risk,
        }
