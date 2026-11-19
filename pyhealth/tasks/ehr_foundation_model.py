"""Turn a task's samples into raw event timelines for an EHR foundation model.

Pretrained EHR foundation models (FMs) such as CLMBR-T (Wornow et al.,
"EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models",
NeurIPS 2023 Datasets and Benchmarks; https://arxiv.org/abs/2307.02028) and
the hf_ehr models of Context Clues (Wornow et al., "Context Clues: Evaluating
Long Context Models for Clinical Prediction Tasks on EHRs", 2024;
https://arxiv.org/abs/2412.16178) read a patient's timeline as their own
tokens: each ``(code, value)`` is mapped to a token of the FM's vocabulary,
with numeric values binned the way its tokenizer was trained.

A ``"sequence"`` input would build a PyHealth vocabulary that the FM cannot
use, so :class:`EHRFoundationModelTask` keeps the timeline raw: one ``"raw"``
``events`` field of time-ordered ``(timestamp, code, value, unit)`` tuples
and the patient's ``birth_datetime``. The model (e.g.
:class:`~pyhealth.models.HFEHRFoundationModel`) does all FM-specific
processing in its ``forward()``, so the task stays model-agnostic.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base_task import BaseTask
from .benchmark_ehrshot import BenchmarkEHRShot
from .ehr_text_serialization import _Keyed, _name

logger = logging.getLogger(__name__)

# Codes that mark a patient's birth: EHRSHOT/OMOP ("Birth"), the MEDS 0.1-0.2
# birth code that femr 0.2 looks for, and the MEDS 0.3 birth code.
BIRTH_CODES: tuple[str, ...] = ("SNOMED/3950001", "SNOMED/184099003", "MEDS_BIRTH")


@dataclass(frozen=True)
class EventSource:
    """Where an event table keeps the parts of an FM event.

    Args:
        code: Attribute holding the code.
        value: Attribute holding the value (numeric or text), if any.
        unit: Attribute holding the unit, if any.
        vocabulary: Prefix written before each code as ``"VOCAB/code"``,
            e.g. ``"ICD10CM"``. ``None`` keeps codes as they are (EHRSHOT's
            codes are already ``"VOCAB/code"``).

    Examples:
        >>> EventSource("code", value="value", unit="unit")
        EventSource(code='code', value='value', unit='unit', vocabulary=None)
    """

    code: str
    value: str | None = None
    unit: str | None = None
    vocabulary: str | None = None


# EHRSHOT's single event table, as configured in ``configs/ehrshot.yaml``.
EHRSHOT_EVENTS: dict[str, EventSource] = {
    "ehrshot": EventSource("code", value="value", unit="unit"),
}


def clean_value(value: Any) -> Any:
    """A value as stored in ``events``: ``None`` for missing, else unchanged.

    Empty strings, ``NaN`` and ``None`` are missing. Numbers and text stay as
    they are; the model decides how to read them (e.g. a numeric string is
    binned as a number by :class:`~pyhealth.models.HFEHRFoundationModel`).

    Examples:
        >>> clean_value(""), clean_value(float("nan")), clean_value("7.2")
        (None, None, '7.2')
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _ehrshot_prediction_times(
    task: BenchmarkEHRShot, patient: Any, samples: Sequence[Mapping[str, Any]]
) -> list[datetime]:
    """The label time of each BenchmarkEHRShot sample, in sample order.

    BenchmarkEHRShot makes one sample per label event, in event order, and
    uses the codes up to and including the label's timestamp.
    """
    labels = patient.get_events(task.task)
    if len(labels) != len(samples):
        raise RuntimeError(
            f"{task.task_name}: {len(samples)} samples but {len(labels)} labels"
        )
    return [label.timestamp for label in labels]


class EHRFoundationModelTask(BaseTask):
    """Wraps a task so each sample carries its raw event timeline.

    Works with the dataset of the wrapped task, e.g.
    :class:`~pyhealth.datasets.EHRShotDataset` with
    :class:`~pyhealth.tasks.BenchmarkEHRShot`.

    Every sample of ``base_task`` keeps its label (and any keys outside its
    input schema, such as EHRSHOT's ``split``); its inputs are replaced by:

    - ``events``: a list of ``(timestamp, code, value, unit)`` tuples of the
      patient's events at or before the sample's prediction time, in time
      order (ties keep the dataset's row order). Codes are ``"VOCAB/code"``
      strings; ``value`` is a number, a string or ``None``
      (:func:`clean_value`); ``unit`` is a string or ``None``.
    - ``birth_datetime``: the time of the patient's first birth code
      (:data:`BIRTH_CODES`), from ``birth_fn``, or else the earliest event.

    Both use the ``"raw"`` processor, so no PyHealth vocabulary is built:
    the FM maps codes to its own vocabulary in ``forward()``.

    The prediction time of each sample comes from ``prediction_time``. For
    :class:`~pyhealth.tasks.BenchmarkEHRShot` it defaults to each label's
    time, the same cutoff the task uses for its codes, so the timeline holds
    exactly the information the wrapped task allows and cannot leak the
    label.

    Args:
        base_task: The task to wrap, e.g. ``BenchmarkEHRShot("guo_icu")``.
        event_sources: Event table to :class:`EventSource`. Defaults to
            :data:`EHRSHOT_EVENTS` for BenchmarkEHRShot; required otherwise.
        prediction_time: The sample key holding its prediction time (a
            sample without it raises ``ValueError``), or
            ``prediction_time(patient, sample, index) -> datetime | None``
            (``None`` keeps all events, on purpose). Defaults to the label
            times for BenchmarkEHRShot; required otherwise.
        birth_fn: ``birth_fn(patient) -> datetime | None``, e.g. from a
            demographics table. Defaults to the first event with a code in
            ``birth_codes``, then the earliest event.
        birth_codes: Codes that mark birth.
        max_events: Keep only each sample's most recent ``max_events``
            events, to bound the cache size. ``None`` keeps all; the model
            truncates to its own context length anyway. Each sample stores
            its own copy of the events up to its prediction time, so for
            tasks with many labels per patient set this to about the
            model's context length.
        keep_inputs: Also keep the wrapped task's input fields, e.g. for a
            count-based baseline on the same samples.

    Examples:
        >>> from pyhealth.tasks import BenchmarkEHRShot
        >>> task = EHRFoundationModelTask(BenchmarkEHRShot("guo_icu"))
        >>> task.input_schema, task.output_schema
        ({'events': 'raw', 'birth_datetime': 'raw'}, {'label': 'binary'})
        >>> task.task_name
        'BenchmarkEHRShot/guo_icu/fm_events'
    """

    def __init__(
        self,
        base_task: BaseTask,
        event_sources: Mapping[str, EventSource] | None = None,
        prediction_time: str | Callable[[Any, Mapping[str, Any], int], Any]
        | None = None,
        birth_fn: Callable[[Any], Any] | None = None,
        birth_codes: Sequence[str] = BIRTH_CODES,
        max_events: int | None = None,
        keep_inputs: bool = False,
    ) -> None:
        params = json.dumps(vars(base_task), sort_keys=True, default=str)
        self._base = _Keyed(base_task, f"{_name(type(base_task))}{params}")
        self.task_name = f"{base_task.task_name}/fm_events"
        is_ehrshot = isinstance(base_task, BenchmarkEHRShot)
        if event_sources is None:
            if not is_ehrshot:
                raise ValueError(
                    "pass event_sources: which event tables hold the codes, "
                    "values and units of the timeline"
                )
            event_sources = EHRSHOT_EVENTS
        if prediction_time is None and not is_ehrshot:
            raise ValueError(
                "pass prediction_time (a sample key or a function): without "
                "it the timeline could include events after the prediction "
                "time"
            )
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be positive")
        self.event_sources = {k: v for k, v in sorted(event_sources.items())}
        self._prediction_time = _Keyed(
            prediction_time,
            prediction_time if isinstance(prediction_time, str)
            else _name(prediction_time),
        )
        self._birth_fn = _Keyed(birth_fn, _name(birth_fn))
        self.birth_codes = sorted(set(birth_codes))
        self.max_events = max_events
        self.keep_inputs = keep_inputs
        inputs = dict(base_task.input_schema)
        self.input_fields = sorted(inputs)
        self.input_schema = {
            **(inputs if keep_inputs else {}),
            "events": "raw",
            "birth_datetime": "raw",
        }
        self.output_schema = dict(base_task.output_schema)

    @property
    def base_task(self) -> BaseTask:
        """The wrapped task."""
        return self._base.obj

    def pre_filter(self, df: Any) -> Any:
        return self.base_task.pre_filter(df)

    # ------------------------------------------------------------ timeline
    def timeline(self, patient: Any) -> list[tuple[Any, str, Any, Any]]:
        """All of a patient's events from ``event_sources``, in time order.

        Events without a timestamp or a code are left out. Events that share
        a timestamp keep the dataset's order (table, then source row).

        Examples:
            >>> from datetime import datetime
            >>> from types import SimpleNamespace
            >>> event = SimpleNamespace(timestamp=datetime(2020, 1, 1),
            ...                         code="LOINC/2236-8", value="7.2",
            ...                         unit="mg/dL")
            >>> patient = SimpleNamespace(
            ...     get_events=lambda event_type=None, **kw: [event])
            >>> task = EHRFoundationModelTask(
            ...     BenchmarkEHRShot("guo_icu"))
            >>> task.timeline(patient)
            [(datetime.datetime(2020, 1, 1, 0, 0), 'LOINC/2236-8', '7.2', 'mg/dL')]
        """
        rows: list[tuple[Any, int, int, tuple[Any, str, Any, Any]]] = []
        for table_index, (table, source) in enumerate(self.event_sources.items()):
            for row, event in enumerate(patient.get_events(event_type=table)):
                timestamp = getattr(event, "timestamp", None)
                code = clean_value(getattr(event, source.code, None))
                if timestamp is None or code is None:
                    continue
                code = str(code).strip()
                if source.vocabulary:
                    code = f"{source.vocabulary}/{code}"
                value = clean_value(getattr(event, source.value, None)
                                    if source.value else None)
                unit = clean_value(getattr(event, source.unit, None)
                                   if source.unit else None)
                rows.append((timestamp, table_index, row,
                             (timestamp, code, value, unit)))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        return [r[3] for r in rows]

    def birth_datetime(
        self, patient: Any, timeline: Sequence[tuple[Any, str, Any, Any]]
    ) -> Any:
        """The patient's birth time (see the class docstring)."""
        if self._birth_fn.obj is not None:
            birth = self._birth_fn.obj(patient)
            if birth is not None:
                return birth
        codes = set(self.birth_codes)
        for timestamp, code, _, _ in timeline:
            if code in codes:
                return timestamp
        return timeline[0][0] if timeline else None

    def _prediction_times(
        self, patient: Any, samples: Sequence[Mapping[str, Any]]
    ) -> list[Any]:
        spec = self._prediction_time.obj
        if spec is None:
            return _ehrshot_prediction_times(self.base_task, patient, samples)
        if isinstance(spec, str):
            times = [sample.get(spec) for sample in samples]
            if any(t is None for t in times):
                # keeping the whole timeline here could leak the label
                raise ValueError(
                    f"{self.task_name}: a sample has no {spec!r} prediction "
                    "time; use a prediction_time function that returns None "
                    "to keep all of its events on purpose"
                )
            return times
        return [spec(patient, sample, i) for i, sample in enumerate(samples)]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = self.base_task(patient)
        if not samples:
            return []
        timeline = self.timeline(patient)
        birth = self.birth_datetime(patient, timeline)
        times = [t for t, _, _, _ in timeline]
        out = []
        for sample, cutoff in zip(samples, self._prediction_times(patient, samples)):
            if cutoff is None:
                events = list(timeline)
            else:
                # events at or before the cutoff (the timeline is sorted)
                end = bisect.bisect_right(times, cutoff)
                events = timeline[:end]
            if self.max_events is not None:
                events = events[-self.max_events:]
            new = {k: v for k, v in sample.items()
                   if k not in self.input_fields or self.keep_inputs}
            new["events"] = events
            new["birth_datetime"] = birth
            out.append(new)
        return out

