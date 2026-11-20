"""Processors for censored time-to-event (survival) data.

PyHealth's label processors cover binary, multiclass, multilabel, and
regression targets. Survival models need a censoring-aware label instead:
the time at risk plus whether the event was observed. This module adds:

- :class:`SurvivalIntervalProcessor` (``"survival_interval"``): a
  counting-process label ``[start, stop, event]`` for time-varying Cox models.
- :class:`SurvivalLabelProcessor` (``"survival"``): a ``[duration, event]``
  label for landmark samples and models such as DeepSurv.

Numeric lab features use the existing ``"tensor"`` processor with
``normalization="zscore"`` and ``impute_strategy="zero"`` (missing values become
the training mean).

Survival labels are not one of ``BaseModel``'s built-in modes, so a model using
them has ``mode = None``: the ``Trainer`` then reports only the loss, and
survival metrics (for example :func:`pyhealth.metrics.concordance_index`) are
computed from ``Trainer.inference`` outputs.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor


def _as_float_list(value: Any, length: int, name: str) -> list[float]:
    """Converts a label to a list of floats of a fixed length.

    Args:
        value: A sequence or tensor holding the label.
        length: The required number of entries.
        name: The label layout, used in error messages.

    Returns:
        The label entries as Python floats.

    Raises:
        ValueError: If the label has the wrong length or a non-finite entry.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    values = [float(v) for v in value]
    if len(values) != length:
        raise ValueError(f"Expected {name} with {length} values, got {len(values)}.")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be finite, got {values}.")
    return values


def _check_event(event: float, name: str) -> None:
    """Checks that an event indicator is 0 (censored) or 1 (observed).

    Args:
        event: The event indicator.
        name: The label layout, used in error messages.

    Raises:
        ValueError: If the indicator is not 0 or 1.
    """
    if event not in (0.0, 1.0):
        raise ValueError(f"{name}: event must be 0 or 1, got {event}.")


@register_processor("survival_interval")
class SurvivalIntervalProcessor(FeatureProcessor):
    """Counting-process survival label ``[start, stop, event]``.

    Each sample is one at-risk interval ``(start, stop]`` of a patient.
    ``event`` is 1 if the event happened at ``stop`` and 0 otherwise, so only
    a patient's last interval can carry an event. This is the input format of
    time-varying Cox models such as ``lifelines.CoxTimeVaryingFitter``.

    Output:
        A float32 tensor of shape ``(3,)``; batches stack to ``(B, 3)``.

    Examples:
        >>> from pyhealth.processors import SurvivalIntervalProcessor
        >>> SurvivalIntervalProcessor().process([0.0, 0.5, 1])
        tensor([0.0000, 0.5000, 1.0000])
    """

    stores_tensor = True

    def process(self, value: Sequence[float]) -> torch.Tensor:
        """Validates one interval and converts it to a tensor.

        Args:
            value: ``[start, stop, event]`` with ``0 <= start < stop`` and
                ``event`` in ``{0, 1}``.

        Returns:
            A float32 tensor of shape ``(3,)``.

        Raises:
            ValueError: If the interval is malformed.
        """
        start, stop, event = _as_float_list(value, 3, "[start, stop, event]")
        if start < 0 or stop <= start:
            raise ValueError(
                f"Interval needs 0 <= start < stop, got start={start}, stop={stop}."
            )
        _check_event(event, "[start, stop, event]")
        return torch.tensor([start, stop, event], dtype=torch.float32)

    def size(self) -> int:
        """Returns the label width, 3."""
        return 3

    def is_token(self) -> bool:
        """Survival labels are continuous values, not token indices."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 1D tensor of shape ``(3,)``."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return "SurvivalIntervalProcessor()"


@register_processor("survival")
class SurvivalLabelProcessor(FeatureProcessor):
    """Right-censored survival label ``[duration, event]``.

    ``duration`` is the time from the sample's reference point (for example a
    landmark visit) to the event or censoring, and ``event`` is 1 if the event
    was observed.

    Output:
        A float32 tensor of shape ``(2,)``; batches stack to ``(B, 2)``.

    Examples:
        >>> from pyhealth.processors import SurvivalLabelProcessor
        >>> SurvivalLabelProcessor().process([1.5, 0])
        tensor([1.5000, 0.0000])
    """

    stores_tensor = True

    def process(self, value: Sequence[float]) -> torch.Tensor:
        """Validates one label and converts it to a tensor.

        Args:
            value: ``[duration, event]`` with ``duration > 0`` and ``event``
                in ``{0, 1}``.

        Returns:
            A float32 tensor of shape ``(2,)``.

        Raises:
            ValueError: If the label is malformed.
        """
        duration, event = _as_float_list(value, 2, "[duration, event]")
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}.")
        _check_event(event, "[duration, event]")
        return torch.tensor([duration, event], dtype=torch.float32)

    def size(self) -> int:
        """Returns the label width, 2."""
        return 2

    def is_token(self) -> bool:
        """Survival labels are continuous values, not token indices."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 1D tensor of shape ``(2,)``."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return "SurvivalLabelProcessor()"
