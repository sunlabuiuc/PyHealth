"""ETHOS timeline tokenizer for MEDS patient timelines.

Turns a patient's MEDS rows into an ETHOS Patient Health Timeline: a flat
token sequence where each code is followed by a quantile token (``Q1`` ...
``Q10``) when it has a numeric value, and gaps between events are written as
time-interval tokens.

Paper:
    Renc et al. Foundation Model of Electronic Medical Records for Adaptive
    Risk Estimation. https://arxiv.org/abs/2502.06124

Adapted from ethos-ares (commit 2d54383), ``src/ethos/tokenize/common``:
https://github.com/ipolharvard/ethos-ares

MIT License. Copyright (c) 2024 Paweł Renc.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import torch

from . import register_processor
from .base_processor import FeatureProcessor

DEATH_CODE = "MEDS_DEATH"
DISCHARGE_PREFIX = "HOSPITAL_DISCHARGE"
TIMELINE_END_TOKEN = "TIMELINE_END"

#: Codes ETHOS moves into the static patient context, away from the timeline.
STATIC_CODE_PREFIXES: tuple[str, ...] = (
    "MEDS_BIRTH",
    "GENDER",
    "MARITAL",
    "RACE",
)

#: Interval tokens and their inclusive lower bounds. A gap under the first
#: bound gets no token; a gap past the last repeats its token.
TIME_INTERVALS: tuple[tuple[str, timedelta], ...] = (
    ("5m-15m", timedelta(minutes=5)),
    ("15m-45m", timedelta(minutes=15)),
    ("45m-1h15m", timedelta(minutes=45)),
    ("1h15m-2h", timedelta(hours=1, minutes=15)),
    ("2h-3h", timedelta(hours=2)),
    ("3h-5h", timedelta(hours=3)),
    ("5h-8h", timedelta(hours=5)),
    ("8h-12h", timedelta(hours=8)),
    ("12h-18h", timedelta(hours=12)),
    ("18h-1d", timedelta(hours=18)),
    ("1d-2d", timedelta(days=1)),
    ("2d-4d", timedelta(days=2)),
    ("4d-7d", timedelta(days=4)),
    ("7d-12d", timedelta(days=7)),
    ("12d-20d", timedelta(days=12)),
    ("20d-30d", timedelta(days=20)),
    ("30d-2mt", timedelta(days=30)),
    ("2mt-6mt", timedelta(days=60)),
    ("=6mt", timedelta(days=180)),
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class EthosVocab:
    """Maps token strings to dense ids, with ``<pad>`` at 0 and ``<unk>`` at 1.

    Unknown tokens map to ``unk_id``. :class:`EthosProcessor` keeps one as
    ``vocab`` and saves it with :meth:`to_json`.

    Examples:
        >>> from pyhealth.processors.ethos_processor import EthosVocab
        >>> vocab = EthosVocab()
        >>> vocab.add_token("Q1")
        2
        >>> vocab["Q1"], vocab["unseen"], vocab.vocab_size
        (2, 1, 3)
    """

    token_to_id: dict[str, int] = field(default_factory=dict)
    pad_id: int = 0
    unk_id: int = 1
    _next_id: int = 2

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self.token_to_id = {PAD_TOKEN: self.pad_id, UNK_TOKEN: self.unk_id}
            self._next_id = 2

    def add_token(self, token: str) -> int:
        """Returns the token's id, adding it first if it is new."""
        if token not in self.token_to_id:
            self.token_to_id[token] = self._next_id
            self._next_id += 1
        return self.token_to_id[token]

    def __getitem__(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    @property
    def vocab_size(self) -> int:
        """Returns the number of ids, including ``<pad>`` and ``<unk>``."""
        return self._next_id

    def to_json(self) -> dict[str, Any]:
        """Returns the vocabulary as a JSON-ready dict."""
        return {
            "token_to_id": self.token_to_id,
            "next_id": self._next_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EthosVocab:
        """Rebuilds a vocabulary from :meth:`to_json`'s output."""
        vocab = cls(pad_id=int(data["pad_id"]), unk_id=int(data["unk_id"]))
        vocab.token_to_id = dict(data["token_to_id"])
        vocab._next_id = int(data["next_id"])
        return vocab


_MICROSECONDS = timedelta(microseconds=1)
_INTERVAL_LABELS = [label for label, _ in TIME_INTERVALS]
_INTERVAL_BOUNDS = [bound // _MICROSECONDS for _, bound in TIME_INTERVALS]
_MICROSECONDS_PER_HOUR = timedelta(hours=1) // _MICROSECONDS


def _interval_tokens(gap: int) -> list[str]:
    """Returns the interval tokens for a gap given in microseconds."""
    if gap >= _INTERVAL_BOUNDS[-1]:
        # Half away from zero, as polars rounds; Python's round() would not.
        repeats = math.floor(gap / _INTERVAL_BOUNDS[-1] + 0.5)
        return [_INTERVAL_LABELS[-1]] * repeats
    index = bisect.bisect_right(_INTERVAL_BOUNDS, gap) - 1
    return [] if index < 0 else [_INTERVAL_LABELS[index]]


def _quantile_breaks(values: list[float], num_quantiles: int) -> list[float]:
    """Returns one code's sorted, deduplicated inner quantile breaks.

    Quantiles use the nearest observation, with ties in the position rounded
    up, which is how polars' default ``quantile`` picks them.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    levels = np.linspace(0, 1, num_quantiles + 1)[1:-1]
    positions = np.floor((len(ordered) - 1) * levels + 0.5).astype(np.int64)
    return np.unique(ordered[positions]).tolist()


def _timestamps(times: Sequence[Any]) -> np.ndarray:
    """Returns the times as ``datetime64[us]``, with NaT for missing ones.

    MIMIC dates are shifted past 2100 and birth dates can be far earlier, so
    this avoids pandas' nanosecond range, which ends in 2262.
    """
    if pd.api.types.infer_dtype(times, skipna=True) in (
        "integer",
        "floating",
        "mixed-integer-float",
    ):
        raise TypeError(
            "EthosProcessor needs datetimes in 'time', got numbers. MEDS times "
            "are timestamps; a bare number has no unit to bin gaps with."
        )
    return np.array(
        [None if pd.isna(time) else time for time in times], dtype="datetime64[us]"
    )


def _place_death_before_discharge(
    order: list[int], times: np.ndarray, codes: Sequence[str]
) -> list[int]:
    """Puts a death ahead of a discharge recorded at the same time.

    Both move to the slot of the later of the two, as in ETHOS, so the rest of
    the rows sharing that timestamp keep their order.

    Args:
        order: Row indices sorted by time.
        times: Time of each row.
        codes: Code of each row.

    Returns:
        ``order`` with the affected rows moved.
    """
    death_times = {times[i] for i in order if codes[i] == DEATH_CODE}
    for time in death_times:
        slots = [
            slot
            for slot, i in enumerate(order)
            if times[i] == time
            and (codes[i] == DEATH_CODE or codes[i].startswith(DISCHARGE_PREFIX))
        ]
        rows = [order[slot] for slot in slots]
        deaths = [i for i in rows if codes[i] == DEATH_CODE]
        discharges = [i for i in rows if codes[i] != DEATH_CODE]
        first = next(s for s, i in enumerate(order) if times[i] == time)
        others = [i for i in order[first : slots[-1] + 1] if i not in rows]
        order[first : slots[-1] + 1] = others + deaths + discharges
    return order


@register_processor("ethos")
class EthosProcessor(FeatureProcessor):
    """Tokenizes a MEDS patient timeline the way ETHOS does.

    The input is one patient's rows as columns, named as in MEDS:

    * ``time``: a timestamp per row. Rows without one are static and dropped.
    * ``code``: a string per row. Rows without one are dropped.
    * ``numeric_value``: a number or ``None`` per row. May be left out.

    Rows are sorted by time, and rows sharing a timestamp keep their input
    order, except that ``MEDS_DEATH`` goes ahead of a same-time discharge.
    Each row then becomes its code, followed by a quantile token if it has a
    value. ETHOS has no marker for a shared timestamp: such rows are told
    apart only by having no interval token between them.

    A gap between neighboring rows of at least 5 minutes puts one of 19
    interval tokens before the later row. Gaps are measured row to row, so
    rows 4 minutes apart never get a token however long the run is. A gap of
    6 months or more repeats ``=6mt``, rounded to the nearest count. Timelines
    end with ``TIMELINE_END``.

    :meth:`fit` learns the code vocabulary and, per code, the quantile breaks
    of its values. ``Qk`` means ``k - 1`` breaks are strictly below the value,
    so a value equal to a break takes the lower bin. Values outside the range
    seen in training fall in the first or last bin. Breaks are deduplicated,
    so a code with repeated values has fewer bins, and ``Qk`` is then a rank
    among its bins and not a true decile. Quantile tokens are shared across
    codes. Fit on the training split only:
    ``set_task(task, split=PatientSplit(...))``, or reuse this processor for
    test data through ``input_processors``.

    An unseen code becomes ``<unk>`` with no quantile token. All quantile and
    interval tokens are in the vocabulary, whether or not training saw them.

    :meth:`fit` also learns ``interval_estimates``: each interval token's mean
    duration in hours, as ETHOS's ``IntervalEstimator`` does, so that a
    generated timeline can be converted back to elapsed time. Repeats of
    ``=6mt`` split their gap evenly. A token that training never saw falls
    back to its lower bound.

    Args:
        num_quantiles: Number of bins per code. ETHOS uses 10.
        quantize_prefixes: Quantize only codes with one of these prefixes. By
            default every code with a value is quantized.
        exclude_prefixes: Drop codes with one of these prefixes. By default
            the codes that ETHOS keeps as static context, which this processor
            does not emit.
        add_timeline_end: Whether to append ``TIMELINE_END``. Turn this off if
            the timeline is cut at a prediction time and does not end there.

    Examples:
        >>> from datetime import datetime
        >>> from pyhealth.processors import EthosProcessor
        >>> timeline = {
        ...     "time": [datetime(2024, 1, 1, h) for h in (8, 9, 11)],
        ...     "code": ["LAB//HGB"] * 3,
        ...     "numeric_value": [10.0, 12.0, 14.0],
        ... }
        >>> processor = EthosProcessor(num_quantiles=2)
        >>> processor.fit([{"timeline": timeline}], "timeline")
        >>> time, tokens = processor.process(timeline)
        >>> processor.decode(tokens)
        ['LAB//HGB', 'Q1', '45m-1h15m', 'LAB//HGB', 'Q1', '2h-3h', 'LAB//HGB', 'Q2', 'TIMELINE_END']
        >>> time.tolist()
        [0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    """

    learns_statistics = True
    stores_tensor = True

    def __init__(
        self,
        num_quantiles: int = 10,
        quantize_prefixes: Sequence[str] | None = None,
        exclude_prefixes: Sequence[str] = STATIC_CODE_PREFIXES,
        add_timeline_end: bool = True,
    ) -> None:
        if num_quantiles < 2:
            raise ValueError(f"num_quantiles must be at least 2, got {num_quantiles}.")
        self.num_quantiles = num_quantiles
        self.quantize_prefixes = (
            None if quantize_prefixes is None else tuple(quantize_prefixes)
        )
        self.exclude_prefixes = tuple(exclude_prefixes)
        self.add_timeline_end = add_timeline_end
        self.vocab: EthosVocab | None = None
        self.quantiles: dict[str, list[float]] = {}
        self.interval_estimates: dict[str, float] = {}

    def _read(
        self, timeline: dict[str, Any] | None
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Returns the timeline's kept rows in token order.

        Args:
            timeline: Raw timeline, as in the class docstring.

        Returns:
            Times in microseconds, codes, and values with ``nan`` where a row
            has no usable value.

        Raises:
            ValueError: If the columns differ in length.
        """
        empty = (np.empty(0, dtype=np.int64), [], np.empty(0))
        if not timeline or len(timeline["code"]) == 0:
            return empty

        times, codes = timeline["time"], timeline["code"]
        numbers = timeline.get("numeric_value")
        if numbers is None:
            numbers = [None] * len(codes)
        if not len(times) == len(codes) == len(numbers):
            raise ValueError(
                "EthosProcessor needs time, code and numeric_value of one length, "
                f"got {len(times)}, {len(codes)} and {len(numbers)}."
            )

        parsed = _timestamps(times)
        stamps, missing = parsed.astype(np.int64), np.isnat(parsed)
        values = np.array(numbers, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        kept = [
            i
            for i, code in enumerate(codes)
            if not missing[i]
            and code is not None
            and not code.startswith(self.exclude_prefixes)
        ]
        if not kept:
            return empty

        # A stable sort keeps the input order of rows sharing a timestamp.
        order = [kept[i] for i in np.argsort(stamps[kept], kind="stable")]
        order = _place_death_before_discharge(order, stamps, codes)
        return stamps[order], [codes[i] for i in order], values[order]

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Learns the vocabulary and quantile breaks.

        Args:
            samples: Sample dictionaries. Pass the training split only.
            field: Key of the raw timeline in each sample.
        """
        observed: dict[str, list[float]] = defaultdict(list)
        codes: set[str] = set()
        spans: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            stamps, sample_codes, values = self._read(sample.get(field))
            codes.update(sample_codes)
            for gap in np.diff(stamps).tolist():
                labels = _interval_tokens(gap)
                for label in labels:
                    spans[label] += gap / len(labels)
                    counts[label] += 1
            for code, value in zip(sample_codes, values.tolist()):
                if math.isnan(value):
                    continue
                if self.quantize_prefixes is None or code.startswith(
                    self.quantize_prefixes
                ):
                    observed[code].append(value)

        self.quantiles = {
            code: _quantile_breaks(values, self.num_quantiles)
            for code, values in observed.items()
        }
        self.interval_estimates = {
            label: (spans[label] / counts[label] if counts[label] else bound)
            / _MICROSECONDS_PER_HOUR
            for label, bound in zip(_INTERVAL_LABELS, _INTERVAL_BOUNDS)
        }
        self.vocab = EthosVocab()
        for k in range(1, self.num_quantiles + 1):
            self.vocab.add_token(f"Q{k}")
        for label in _INTERVAL_LABELS:
            self.vocab.add_token(label)
        self.vocab.add_token(TIMELINE_END_TOKEN)
        for code in sorted(codes):
            self.vocab.add_token(code)

    def process(self, value: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes one timeline.

        Args:
            value: Raw timeline, as in the class docstring.

        Returns:
            ``(time, tokens)``, both 1-D and of the same length. ``time`` is
            ``float64`` hours since the first kept row, per token; an interval
            token has the time of the row after it. ``tokens`` is ``long``.
            Both are empty for a timeline with no kept rows, which gets no
            ``TIMELINE_END`` either.

        Raises:
            RuntimeError: If called before :meth:`fit` or :meth:`load`.
        """
        if self.vocab is None:
            raise RuntimeError("EthosProcessor must be fitted or loaded first.")

        stamps, codes, values = self._read(value)
        tokens: list[int] = []
        times: list[int] = []
        previous = None
        for time, code, number in zip(stamps.tolist(), codes, values.tolist()):
            if previous is not None:
                for interval in _interval_tokens(time - previous):
                    tokens.append(self.vocab[interval])
                    times.append(time)
            tokens.append(self.vocab[code])
            times.append(time)
            breaks = self.quantiles.get(code)
            if breaks is not None and not math.isnan(number):
                tokens.append(self.vocab[f"Q{bisect.bisect_left(breaks, number) + 1}"])
                times.append(time)
            previous = time

        if tokens and self.add_timeline_end:
            tokens.append(self.vocab[TIMELINE_END_TOKEN])
            times.append(previous)

        hours = np.asarray(times, dtype=np.int64) - (times[0] if times else 0)
        return (
            torch.from_numpy(hours / _MICROSECONDS_PER_HOUR),
            torch.tensor(tokens, dtype=torch.long),
        )

    def decode(self, tokens: torch.Tensor | Iterable[int]) -> list[str]:
        """Maps token ids back to token strings.

        Args:
            tokens: Token ids, such as the output of :meth:`process`.

        Returns:
            The token strings.

        Examples:
            >>> from pyhealth.processors import EthosProcessor
            >>> processor = EthosProcessor()
            >>> processor.fit([], "timeline")
            >>> processor.decode([0, 1, 2])
            ['<pad>', '<unk>', 'Q1']
        """
        by_id = {index: token for token, index in self.vocab.token_to_id.items()}
        return [by_id[int(token)] for token in tokens]

    def save(self, path: str) -> None:
        """Saves the configuration and fitted state as JSON.

        Args:
            path: File to write.
        """
        state = {
            "num_quantiles": self.num_quantiles,
            "quantize_prefixes": self.quantize_prefixes,
            "exclude_prefixes": self.exclude_prefixes,
            "add_timeline_end": self.add_timeline_end,
            "vocab": None if self.vocab is None else self.vocab.to_json(),
            "quantiles": self.quantiles,
            "interval_estimates": self.interval_estimates,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(orjson.dumps(state, option=orjson.OPT_SORT_KEYS))

    def load(self, path: str) -> None:
        """Restores what :meth:`save` wrote, replacing the current state.

        Args:
            path: File to read.
        """
        state = orjson.loads(Path(path).read_bytes())
        self.num_quantiles = state["num_quantiles"]
        prefixes = state["quantize_prefixes"]
        self.quantize_prefixes = None if prefixes is None else tuple(prefixes)
        self.exclude_prefixes = tuple(state["exclude_prefixes"])
        self.add_timeline_end = state["add_timeline_end"]
        vocab = state["vocab"]
        self.vocab = None if vocab is None else EthosVocab.from_json(vocab)
        self.quantiles = state["quantiles"]
        self.interval_estimates = state["interval_estimates"]

    def size(self) -> int:
        """Returns the vocabulary size."""
        return 0 if self.vocab is None else self.vocab.vocab_size

    def is_token(self) -> bool:
        """Output tokens are discrete indices."""
        return True

    def schema(self) -> tuple[str, ...]:
        return ("time", "value")

    def dim(self) -> tuple[int, ...]:
        return (1, 1)

    def spatial(self) -> tuple[bool, ...]:
        return (True,)

    def __repr__(self) -> str:
        return (
            f"EthosProcessor(num_quantiles={self.num_quantiles}, "
            f"vocab_size={self.size()})"
        )
