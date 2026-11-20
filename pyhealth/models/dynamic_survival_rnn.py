"""Dynamic survival analysis RNN for early event prediction.

Model of Yeche et al., "Dynamic Survival Analysis for Early Event Prediction",
CHIL 2024, https://proceedings.mlr.press/v248/yeche24a.html

Authors: Sarah Husted, Allyson Moran (Georgia Tech CSE/OAN 6250 project).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import LabelProcessor, transform_batch

from .dynamic_survival import hazard_prior as estimate_hazard_prior
from .rnn import RNN_V2, _FieldRNN, _groups


def hazards_to_risk(logits: torch.Tensor) -> torch.Tensor:
    """Cumulative failure ``1 - prod_k (1 - hazard_k)`` from hazard logits.

    Computed in log space, since ``log(1 - sigmoid(z)) = -softplus(z)``.

    Args:
        logits: Hazard logits, ``[N, horizon]``.

    Returns:
        The probability that the event falls within the horizon, ``[N]``.

    Examples:
        >>> import torch
        >>> hazards_to_risk(torch.zeros(1, 2))  # 1 - 0.5 * 0.5
        tensor([0.7500])
    """
    return -torch.expm1(-F.softplus(logits).sum(dim=-1))


def survival_loss(
    logits: torch.Tensor, time_to_event: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    """Truncated survival loss over hazard logits (Eq. 4 of the paper).

    Binary cross-entropy of each hazard, ``y_k = 1[k == time_to_event and
    event]``, over the hours ``k <= time_to_event`` that were observed, and
    only up to the horizon, the width of ``logits``. The mean is over those
    (step, hour) entries.

    Args:
        logits: Hazard logits, ``[N, horizon]``.
        time_to_event: Hour of the event, or observed hours if censored, ``[N]``.
        event: 1 if the event is observed, else 0, ``[N]``.

    Returns:
        A scalar; zero (with a graph) if no hour counts.

    Examples:
        >>> import torch
        >>> loss = survival_loss(torch.zeros(1, 2), torch.tensor([2.0]), torch.ones(1))
        >>> round(loss.item(), 4)  # log 2
        0.6931
    """
    hours = torch.arange(1, logits.shape[-1] + 1, device=logits.device)
    tte = time_to_event.to(logits.dtype).unsqueeze(1)
    hours = hours.to(logits.dtype).unsqueeze(0)
    weight = (hours <= tte).to(logits.dtype)
    target = ((hours == tte) & (event.unsqueeze(1) > 0.5)).to(logits.dtype)
    counted = weight.sum()
    if counted == 0:
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).sum() / counted


class DynamicSurvivalRNN(RNN_V2):
    """RNN that predicts a hazard for each of the next ``horizon`` hours.

    The encoder is :class:`~pyhealth.models.RNN_V2`: the same inputs, causal
    per-step states and arguments. Its single output is replaced by a hazard
    head with one logit per hour ahead, trained with the survival loss
    truncated to the horizon, as in Yeche et al. (2024). The risk of an event
    within the horizon, ``1 - prod_k (1 - hazard_k)``, is what the model
    returns as ``y_prob``, so it is scored like the binary label of an early
    event prediction model.

    The dataset needs three per-step outputs at the same times, such as
    :class:`~pyhealth.tasks.DecompensationMIMIC3` with
    ``label_mode="survival"``: ``decompensation`` (``timeseries_binary``, the
    label the model is scored on), ``time_to_event`` (``timeseries_regression``,
    the hour of the event, or the observed hours if censored) and ``event``
    (``timeseries_binary``, 1 if the event is observed).

    Args:
        dataset: The sample dataset, with the three outputs above.
        horizon: Hours ahead to model; the width of the hazard head. Use the
            task's horizon.
        hazard_prior: The average hazard of each of the ``horizon`` hours
            (see :func:`~pyhealth.models.dynamic_survival.hazard_prior`,
            computed on training steps). The bias of the hazard head starts at
            its log odds, so the model starts at the base rate. None leaves
            the default initialization, which starts every hazard near 0.5.
        **kwargs: Arguments of :class:`~pyhealth.models.RNN_V2`, e.g.
            ``embedding_dim`` and ``hidden_dim``. ``bidirectional`` must stay
            False, since each step uses only the past.

    Examples:
        >>> from pyhealth.models import DynamicSurvivalRNN  # doctest: +SKIP
        >>> model = DynamicSurvivalRNN(  # doctest: +SKIP
        ...     dataset,
        ...     horizon=24,
        ...     hazard_prior=prior,
        ...     embedding_dim={"dynamic": [{"values": 16}]},
        ... )
    """

    # The label that the model is scored on, and the survival targets.
    _SCORED = "decompensation"
    _TIME = "time_to_event"
    _EVENT = "event"

    # RNN_V2 and the Trainer treat the model as having one label: the binary
    # one. The survival targets are read by this model only.
    @property
    def label_keys(self) -> list[str]:
        return [self._SCORED]

    @label_keys.setter
    def label_keys(self, value: list[str]) -> None:
        pass  # set by BaseModel from the three outputs; fixed to one here

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        return {
            key: processor
            for key, processor in super().label_processors.items()
            if key == self._SCORED
        }

    def __init__(
        self,
        dataset: SampleDataset,
        horizon: int = 24,
        hazard_prior: Sequence[float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        missing = {self._SCORED, self._TIME, self._EVENT} - set(dataset.output_schema)
        if missing:
            raise ValueError(
                f"The dataset needs the outputs {sorted(missing)}; use a task "
                "in survival mode, e.g. DecompensationMIMIC3(label_mode='survival')."
            )
        if int(horizon) != horizon or horizon < 1:
            raise ValueError(f"horizon must be a positive whole number, got {horizon!r}.")
        super().__init__(dataset, **kwargs)
        self.horizon = int(horizon)
        self.fc = nn.Linear(self.fc.in_features, self.horizon)
        if hazard_prior is not None:
            prior = np.asarray(hazard_prior, dtype=float)
            if prior.shape != (self.horizon,) or not ((prior > 0) & (prior < 1)).all():
                raise ValueError(
                    f"hazard_prior needs {self.horizon} values in (0, 1), "
                    f"got shape {prior.shape}."
                )
            logit = torch.as_tensor(np.log(prior / (1.0 - prior)), dtype=torch.float32)
            with torch.no_grad():
                self.fc.bias.copy_(logit)

    @staticmethod
    def prior_from_samples(samples: Any, horizon: int) -> np.ndarray:
        """The hazard prior of a processed training split.

        Args:
            samples: A sample dataset (e.g. the training split) whose items have
                ``time_to_event`` and ``event`` as ``(values, time)`` tensors.
            horizon: Hours ahead to model.

        Returns:
            The prior, ``[horizon]``, for ``hazard_prior``.
        """
        times, events = [], []
        for i in range(len(samples)):
            item = samples[i]
            times.append(item["time_to_event"][0][:, 0].numpy())
            events.append(item["event"][0][:, 0].numpy())
        return estimate_hazard_prior(
            np.concatenate(times), np.concatenate(events), horizon
        )

    def _packed(self, batch: dict[str, Any], key: str) -> tuple[torch.Tensor, ...]:
        return tuple(t.to(self.device) for t in batch[key])

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        """Predicts the hazards of a batch of :func:`~pyhealth.datasets.get_dataloader`.

        Returns:
            ``loss`` (the survival loss), ``y_prob`` (the risk within the
            horizon, ``[label records, 1]``), ``y_true`` (the binary label,
            packed), ``logit`` (the hazard logits, ``[label records,
            horizon]``) and, if the batch has ``embed=True``, ``embed``.
        """
        batch = transform_batch(self.dataset.input_processors, batch)
        embedded = self.embedding(
            {field: batch[field] for field in self.sequence_fields}
        )
        label = self._packed(batch, self._SCORED)
        time_to_event = self._packed(batch, self._TIME)
        event = self._packed(batch, self._EVENT)
        for other in (time_to_event, event):
            if not (
                torch.equal(other[1], label[1]) and torch.equal(other[2], label[2])
            ):
                raise ValueError(
                    "The binary label, time_to_event and event must be at the "
                    "same times."
                )
        label_time = label[1]
        label_sample, _ = _groups(label[2])
        parts = []
        for field in self.sequence_fields:
            encoder = cast(_FieldRNN, self.encoders[field])
            _, states, time = encoder(embedded[field])
            assert states is not None and time is not None
            offsets = embedded[field][encoder.depth]["offsets"]
            parts.append(
                self._at_labels(states, time, offsets, label_time, label_sample)
            )
        embed = torch.cat(parts, dim=-1)
        logits = self.fc(embed)
        if len(label_time) == 0:
            loss = logits.sum() * 0.0
        else:
            loss = survival_loss(logits, time_to_event[0][:, 0], event[0][:, 0])
        results = {
            "loss": loss,
            "y_prob": hazards_to_risk(logits).unsqueeze(1),
            "y_true": label,
            "logit": logits,
        }
        if batch.get("embed", False):
            results["embed"] = embed
        return results
