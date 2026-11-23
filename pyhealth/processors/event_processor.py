"""Coded event timelines from numeric records, for the ``"events"`` task view.

EHR foundation models read a patient as a timeline of coded events,
``(time, code, value, unit)``, and map each event to a token of their own
vocabulary. The processors here write a task's numeric time series as such
a timeline: one event per measured value, none for a missing one. The
events carry exactly the measurements the other views of the task hold, so
a foundation model is compared on the same features.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np

from . import register_processor
from .base_processor import FeatureProcessor

#: Time 0 of every series. Samples hold times relative to their own start
#: (e.g. hours since ICU admission), and the models read datetimes, so every
#: timeline is placed at this fixed, arbitrary instant.
ORIGIN = datetime(2000, 1, 1)

_UNIT_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}


@register_processor("event_stream")
class EventStreamProcessor(FeatureProcessor):
    """A numeric time series as a timeline of coded events.

    The input is ``{"time": [T], "values": [T, F]}`` with ``NaN`` for a value
    that was not measured. The output is the list of
    ``(datetime, code, value, unit)`` of the measured values, ordered by time
    and then by feature. Nothing is imputed or scaled: the model's tokenizer
    decides how a value becomes a token. Nothing is learned from the data.

    Args:
        names: Name of each feature, in column order.
        codes: Feature name to the code the model's vocabulary uses for it,
            e.g. ``{"crea": "LOINC/2160-0"}``. A feature without an entry
            gets ``"<vocabulary>/<name>"``.
        vocabulary: Prefix of the default codes.
        units: Feature name to unit, written on its events. Defaults to None.
        time_unit: Unit of the input times: ``"seconds"``, ``"minutes"``,
            ``"hours"`` or ``"days"``.

    Examples:
        >>> from pyhealth.processors import EventStreamProcessor
        >>> nan = float("nan")
        >>> processor = EventStreamProcessor(
        ...     names=["hr", "crea"], codes={"crea": "LOINC/2160-0"},
        ...     units={"crea": "mg/dL"})
        >>> events = processor.process(
        ...     {"time": [0, 2], "values": [[80.0, nan], [nan, 1.5]]})
        >>> [(when.hour, code, value, unit) for when, code, value, unit in events]
        [(0, 'YAIB/hr', 80.0, None), (2, 'LOINC/2160-0', 1.5, 'mg/dL')]
    """

    def __init__(
        self,
        names: Sequence[str],
        codes: Mapping[str, str] | None = None,
        vocabulary: str = "YAIB",
        units: Mapping[str, str] | None = None,
        time_unit: str = "hours",
    ) -> None:
        if time_unit not in _UNIT_SECONDS:
            raise ValueError(
                f"time_unit must be one of {sorted(_UNIT_SECONDS)}, not {time_unit!r}."
            )
        self.names = list(names)
        self.vocabulary = vocabulary
        self.time_unit = time_unit
        codes = dict(codes or {})
        units = dict(units or {})
        unknown = (set(codes) | set(units)) - set(self.names)
        if unknown:
            raise ValueError(f"codes/units name unknown features: {sorted(unknown)}.")
        self.codes = [codes.get(name, f"{vocabulary}/{name}") for name in self.names]
        self.units = [units.get(name) for name in self.names]

    def process(self, value: Any) -> list[tuple[datetime, str, float, str | None]]:
        """Returns the events of one series, ordered by time then feature."""
        values = np.asarray(value["values"], dtype=np.float64)
        if values.size == 0:
            return []
        if values.shape[1] != len(self.names):
            raise ValueError(
                f"Expected {len(self.names)} features per record, got {values.shape[1]}."
            )
        seconds = _UNIT_SECONDS[self.time_unit]
        events = []
        for time, row in zip(value["time"], values):
            when = ORIGIN + timedelta(seconds=float(time) * seconds)
            for column in np.flatnonzero(~np.isnan(row)):
                events.append(
                    (when, self.codes[column], float(row[column]), self.units[column])
                )
        return events

    def __repr__(self) -> str:
        return f"EventStreamProcessor(features={len(self.names)}, vocabulary={self.vocabulary!r})"


@register_processor("event_birth")
class EventBirthProcessor(FeatureProcessor):
    """The birth time of a static record that holds an age.

    The input is one record ``{"values": [[...]]}``; the output is the
    datetime ``age`` years before the origin of the sample's timeline (see
    :class:`EventStreamProcessor`), or None when the age is missing. Models
    that tokenize age read it next to the events.

    Args:
        age_index: Column of the age, in years.

    Examples:
        >>> from pyhealth.processors import EventBirthProcessor
        >>> EventBirthProcessor().process({"values": [[40.5, 0.0]]}).year
        1959
        >>> EventBirthProcessor().process({"values": [[float("nan")]]}) is None
        True
    """

    def __init__(self, age_index: int = 0) -> None:
        self.age_index = age_index

    def process(self, value: Any) -> datetime | None:
        """Returns the birth datetime, or None if the age is missing."""
        age = float(np.asarray(value["values"], dtype=np.float64)[0, self.age_index])
        if np.isnan(age):
            return None
        return ORIGIN - timedelta(days=age * 365.25)

    def __repr__(self) -> str:
        return f"EventBirthProcessor(age_index={self.age_index})"
