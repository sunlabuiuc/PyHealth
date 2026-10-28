"""Unified processor for hierarchical packed records (CSR / jagged layout).

See ``notes/Unified_Sequence_Processor.md`` (Section 7) for the design.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate, repeat
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Required,
    TypedDict,
    get_args,
)

import numpy as np
import numpy.typing as npt
import torch

from . import register_processor
from .base_processor import FeatureProcessor, TokenProcessorInterface

if TYPE_CHECKING:
    from pyhealth.medcode import CrossMap

OrderType = Literal["set", "sequence", "series"]
Role = Literal["tokens", "values", "time", "offsets"]
FieldKey = tuple[int, Role]
CodeMapping = tuple[str, str]
ResampleMethod = Literal["first", "last", "min", "max", "avg", "merge"]
ImputeMethod = Literal[
    "zero", "mean", "min", "max", "forward_fill", "backward_fill", "linear"
]

# Imputation methods that use statistics learned in fit().
_STATISTICS = ("mean", "min", "max")
# Imputation methods that work along time.
_ALONG_TIME = ("forward_fill", "backward_fill", "linear")

# Options reserved in the configuration but not implemented in Phase 1.
_RESERVED = {"normalize": None, "missing_indicator": False}


class Records(TypedDict, total=False):
    """Raw input of one level: a collection of ``N`` records, stored as columns.

    A ``Records`` is a plain ``dict``; this class only provides type hints.
    Level 0 is the innermost, and the one a task returns is the outermost
    level ``L``. The level's :class:`LevelConfig` decides which keys are read;
    a key it needs must be present (its column may be empty), and any other
    key is ignored.

    Keys:
        time: ``[N]`` numbers in the task's own unit. Read on ``series``
            levels.
        tokens: ``[N]`` tokens, or ``[N]`` tuples of ``K`` tokens when the
            level has ``K`` token namespaces.
        values: ``[N, *S]`` numbers. Scalars are ``[N, 1]``; a flat ``[N]``
            list is also accepted when ``S`` is ``(1,)``.
        children: ``[N]`` ``Records``; ``children[i]`` holds record ``i``'s
            records one level down. Read on every level but 0.
    """

    time: Sequence[float] | np.ndarray
    tokens: Sequence[str] | Sequence[tuple[str, ...]]
    values: npt.ArrayLike
    children: Sequence["Records"]


class Resample(TypedDict):
    """Resampling of a ``series`` level onto a time grid; a plain ``dict``.

    Each grid step becomes one record, and the level becomes a ``sequence``
    without time: record ``i`` is the step starting at ``i * step``. The grid
    starts at the task's time 0, so negative time raises.

    Keys:
        step: Grid step, in the task's time unit.
        method: How the records in one step become one record.

            - ``"first"`` / ``"last"``: keep the first / last record, with
              its tokens, values and children. Both are over the whole step,
              not per token.
            - ``"min"`` / ``"max"`` / ``"avg"``: reduce the values. Only for
              a level with values and no tokens or children.
            - ``"merge"``: union the records' children into one set. Only for
              a level with children and no tokens or values, whose child
              level is a ``set``.

            A step without records gets ``<pad>`` tokens, NaN values and no
            children; filling it is up to imputation.
    """

    step: float
    method: ResampleMethod


class Impute(TypedDict, total=False):
    """One imputation step for a level's NaN values; a plain ``dict``.

    Keys:
        method: How to fill.

            - ``"zero"``: 0.0 for values and ``<pad>`` for tokens. Ignores
              ``by_tokens`` and ``by_dims``.
            - ``"mean"`` / ``"min"`` / ``"max"``: a statistic of the raw
              training values (before resampling), learned in ``fit()``
              with NaN skipped.
            - ``"forward_fill"`` / ``"backward_fill"``: the previous / next
              value along time.
            - ``"linear"``: linear interpolation along time, using ``time``
              (or the step index on a resampled level).

            The last three need a ``sequence`` or ``series`` level, never
            cross into another group of records, work value position by
            value position (ignoring ``by_dims``), and leave gaps at the
            edges as NaN.
        by_tokens: Token namespaces. Statistics are kept per token in
            these namespaces, and the time-based methods only use records
            with the same token. None (the default) ignores tokens; the time-
            based methods then also fill ``<pad>`` tokens.
        by_dims: For statistics: the axes of the value shape ``S`` to
            reduce over, indexed like ``values`` and ``spatial``. None (the
            default) reduces over all of them, ``()`` over none (one
            statistic per value position); e.g. for ``values: [3, 64, 64]``,
            ``(1, 2)`` gives one statistic per channel. Time is always
            reduced.

    A key without a statistic (a token or value never seen in ``fit()``)
    keeps its NaN, as does any gap a method cannot fill; chain several steps
    to fill those.
    """

    method: Required[ImputeMethod]
    by_tokens: Sequence[int] | None
    by_dims: Sequence[int] | None


class LevelConfig(TypedDict, total=False):
    """Configuration of one level; a plain ``dict`` with type hints.

    Like :class:`Records`, ``tokens`` and ``values`` declare a role by being
    present. They are fixed by the task, so :meth:`SequenceProcessorV2.fit`
    only builds vocabularies (and impute statistics).

    Keys:
        ordering: ``"set"``, ``"sequence"`` (default) or ``"series"``.
            Describes the raw input; ``series`` levels carry ``time``.
        tokens: Number of token namespaces ``K``; the level's records have
            ``tokens``, plain strings when ``K = 1`` and ``K``-tuples otherwise.
        values: Value shape ``S``, e.g. ``[1]`` for scalars; the level's
            records have ``values`` of shape ``[N, *S]``.
        code_mapping: A ``(source, target)`` pair for
            ``pyhealth.medcode.CrossMap``, or a list with one pair (or None)
            per token namespace. A code that maps to several targets becomes
            several records, so a mapping with several targets per code is
            only allowed on a ``set`` level without values.
        resample: A :class:`Resample` for a ``series`` level: puts its
            records on a time grid, turning it into a ``sequence`` level.
        spatial: One bool per value axis; defaults to all False.
        impute: An :class:`Impute` step, or a list of them applied in order,
            run after resampling.
        normalize: Reserved for future work; only None.
        missing_indicator: Reserved for future work; only False.
    """

    ordering: OrderType
    tokens: int
    values: Sequence[int]
    code_mapping: CodeMapping | Sequence[CodeMapping | None] | None
    resample: Resample | None
    spatial: Sequence[bool] | None
    impute: Impute | Sequence[Impute] | None
    normalize: None
    missing_indicator: Literal[False]


class _ImputeStep(NamedTuple):
    """A checked :class:`Impute`."""

    method: ImputeMethod
    by_tokens: tuple[int, ...] | None
    by_dims: tuple[int, ...] | None


@dataclass(frozen=True)
class _InternalLevelConfig:
    """A checked :class:`LevelConfig` with every default filled in.

    A level without tokens has ``tokens == 0`` and one without values has
    ``values == ()``; neither is a valid setting, since ``K >= 1`` and scalars
    are ``(1,)``.
    """

    ordering: OrderType
    tokens: int
    values: tuple[int, ...]
    code_mapping: tuple[CodeMapping | None, ...] | None
    resample: tuple[float, ResampleMethod] | None  # (step, method)
    spatial: tuple[bool, ...] | None
    impute: tuple[_ImputeStep, ...]

    @staticmethod
    def _is_count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _is_index(value: Any, size: int) -> bool:
        return (
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value < size
        )

    @staticmethod
    def _parse_code_mapping(
        level: int, mapping: Any
    ) -> tuple[CodeMapping | None, ...] | None:
        """Normalizes ``code_mapping`` to one entry (or None) per namespace."""
        if mapping is None:
            return None

        def as_pair(entry: Any) -> CodeMapping | None:
            if (
                isinstance(entry, (tuple, list))
                and len(entry) == 2
                and all(isinstance(code, str) for code in entry)
            ):
                return (entry[0], entry[1])
            return None

        pair = as_pair(mapping)
        if pair is not None:
            return (pair,)
        if isinstance(mapping, (tuple, list)) and mapping:
            pairs = [None if entry is None else as_pair(entry) for entry in mapping]
            if all(
                entry is None or pair is not None for entry, pair in zip(mapping, pairs)
            ):
                return tuple(pairs)
        raise ValueError(
            f"Level {level}: code_mapping must be a (source, target) pair or a list "
            f"with one pair (or None) per token namespace, got {mapping!r}."
        )

    @classmethod
    def parse(cls, level: int, config: LevelConfig) -> "_InternalLevelConfig":
        """Checks one level's configuration and fills in the defaults.

        ``config`` still comes from user data (a task schema), so every key and
        value is checked at runtime despite the annotation.
        """
        if not isinstance(config, dict):
            raise TypeError(
                f"Level {level}: configuration must be a dict, got {type(config).__name__}."
            )
        unknown = set(config) - LevelConfig.__annotations__.keys()
        if unknown:
            raise ValueError(
                f"Level {level}: unknown configuration keys {sorted(unknown)}."
            )

        for key, default in _RESERVED.items():
            if config.get(key, default) != default:
                raise NotImplementedError(
                    f"Level {level}: '{key}' is reserved for future work and only "
                    f"accepts {default!r} for now."
                )

        ordering = config.get("ordering", "sequence")
        if ordering not in get_args(OrderType):
            raise ValueError(
                f"Level {level}: ordering must be one of {get_args(OrderType)}, got {ordering!r}."
            )

        tokens = config.get("tokens", 0)
        if "tokens" in config and not cls._is_count(tokens):
            raise ValueError(
                f"Level {level}: tokens must be a positive number of namespaces, "
                f"got {tokens!r}."
            )

        values = config.get("values", ())
        if "values" in config and (
            not isinstance(values, (tuple, list))
            or not values
            or not all(cls._is_count(size) for size in values)
        ):
            raise ValueError(
                f"Level {level}: values must be a non-empty list of positive "
                f"sizes, e.g. [1] for scalars, got {values!r}."
            )
        values = tuple(values)

        if level == 0 and ordering != "series" and not tokens and not values:
            raise ValueError(
                "Level 0: set 'tokens' or 'values', or use ordering 'series'."
            )

        code_mapping = cls._parse_code_mapping(level, config.get("code_mapping"))
        if code_mapping is not None and len(code_mapping) != tokens:
            raise ValueError(
                f"Level {level}: code_mapping has {len(code_mapping)} entries but the "
                f"level has {tokens} token namespace(s)."
            )
        resample = config.get("resample")
        if resample is not None:
            if not isinstance(resample, dict) or resample.keys() != {"step", "method"}:
                raise ValueError(
                    f"Level {level}: resample must be a dict with 'step' and 'method', "
                    f"got {resample!r}."
                )
            step, method = resample["step"], resample["method"]
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not (np.isfinite(step) and step > 0)
            ):
                raise ValueError(
                    f"Level {level}: resample step must be a positive number, got {step!r}."
                )
            if method not in get_args(ResampleMethod):
                raise ValueError(
                    f"Level {level}: resample method must be one of "
                    f"{get_args(ResampleMethod)}, got {method!r}."
                )
            if ordering != "series":
                raise ValueError(
                    f"Level {level}: resample requires ordering 'series', got {ordering!r}."
                )
            if method in ("min", "max", "avg") and (not values or tokens or level > 0):
                raise ValueError(
                    f"Level {level}: resample method {method!r} reduces values, so the "
                    "level needs values and no tokens or children (level 0 only)."
                )
            if method == "merge" and (level == 0 or tokens or values):
                raise ValueError(
                    f"Level {level}: resample method 'merge' unions children, so the "
                    "level needs children and no tokens or values."
                )
            resample = (float(step), method)

        spatial = config.get("spatial")
        if spatial is not None:
            if not isinstance(spatial, (tuple, list)) or not all(
                isinstance(flag, bool) for flag in spatial
            ):
                raise ValueError(
                    f"Level {level}: spatial must be a list of bools, got {spatial!r}."
                )
            if len(spatial) != len(values):
                raise ValueError(
                    f"Level {level}: spatial needs one flag per value axis, got "
                    f"{len(spatial)} flag(s) for values {values}."
                )
            spatial = tuple(spatial)

        impute = config.get("impute")
        steps = (
            [] if impute is None else [impute] if isinstance(impute, dict) else impute
        )
        if not isinstance(steps, (list, tuple)):
            raise TypeError(
                f"Level {level}: impute must be a dict or a list of dicts, got {impute!r}."
            )
        parsed: list[_ImputeStep] = []
        for step in steps:
            if (
                not isinstance(step, dict)
                or "method" not in step
                or not step.keys() <= Impute.__annotations__.keys()
            ):
                raise ValueError(
                    f"Level {level}: each impute step must be a dict with 'method' and "
                    f"optionally 'by_tokens' and 'by_dims', got {step!r}."
                )
            method = step["method"]
            if method not in get_args(ImputeMethod):
                raise ValueError(
                    f"Level {level}: impute method must be one of "
                    f"{get_args(ImputeMethod)}, got {method!r}."
                )
            by_tokens, by_dims = step.get("by_tokens"), step.get("by_dims")
            if by_tokens is not None:
                if (
                    not isinstance(by_tokens, (tuple, list))
                    or not by_tokens
                    or not all(cls._is_index(k, tokens) for k in by_tokens)
                ):
                    raise ValueError(
                        f"Level {level}: by_tokens must be None or a non-empty list of "
                        f"token namespaces below {tokens}, got {by_tokens!r}."
                    )
                by_tokens = tuple(by_tokens)
            if by_dims is not None:
                if (
                    not isinstance(by_dims, (tuple, list))
                    or not all(cls._is_index(d, len(values)) for d in by_dims)
                    or len(set(by_dims)) != len(by_dims)
                ):
                    raise ValueError(
                        f"Level {level}: by_dims must be None or a list of distinct "
                        f"value axes below {len(values)}, got {by_dims!r}."
                    )
                by_dims = tuple(by_dims)
            if method in (*_STATISTICS, "linear") and not values:
                raise ValueError(
                    f"Level {level}: impute method {method!r} needs values."
                )
            if method in _ALONG_TIME and ordering == "set":
                raise ValueError(
                    f"Level {level}: impute method {method!r} works along time, so the "
                    "level must be a 'sequence' or 'series'."
                )
            parsed.append(_ImputeStep(method, by_tokens, by_dims))

        return cls(
            ordering=ordering,
            tokens=tokens,
            values=values,
            code_mapping=code_mapping,
            resample=resample,
            spatial=spatial,
            impute=tuple(parsed),
        )


class _Resampler:
    """Puts the records of one ``series`` level on a time grid; see :class:`Resample`."""

    def __init__(self, level: int, step: float, method: ResampleMethod):
        self.level, self.step, self.method = level, step, method

    def __call__(
        self, columns: dict[Role, np.ndarray], children: list[list[Records]]
    ) -> tuple[dict[Role, np.ndarray], list[list[Records]]]:
        """Puts the level's records on the grid, one record per step.

        Args:
            columns: The level's ``tokens`` (IDs), ``values`` and ``time`` for
                all its records, and ``offsets``: the record count of each
                group of records (one group per record of the level above).
            children: One group of child ``Records`` per record.

        Returns:
            The columns and children of the grid: one record per step, no
            ``time``, and ``offsets`` holding each group's number of steps. A
            step without records gets ``<pad>`` tokens, NaN values and no
            children, for imputation to fill.
        """
        time, counts = columns["time"], columns["offsets"]
        if (time < 0).any():
            raise ValueError(f"Level {self.level}: resample needs non-negative time.")
        steps = np.floor(time.astype(np.float64) / self.step).astype(np.int64)

        # A group's grid runs from step 0 to its last record's step. Each group
        # gets its own block of the level's grid, so `grid` never decreases.
        sizes = np.zeros(len(counts), dtype=np.int64)
        filled_groups = counts > 0
        sizes[filled_groups] = steps[np.cumsum(counts)[filled_groups] - 1] + 1
        grid = steps + np.repeat(np.cumsum(sizes) - sizes, counts)
        n = int(sizes.sum())

        # The first and last record of each step that has records.
        starts = np.flatnonzero(np.diff(grid, prepend=-1))
        ends = np.flatnonzero(np.diff(grid, append=-1))
        filled = grid[starts]

        resampled: dict[Role, np.ndarray] = {"offsets": sizes}
        grid_children: list[list[Records]] = [[] for _ in range(n if self.level else 0)]
        if self.method == "merge":
            for i, kids in zip(grid.tolist(), children):
                grid_children[i].extend(kids)
        elif self.method == "first" or self.method == "last":
            keep = starts if self.method == "first" else ends
            pad = TokenProcessorInterface.PAD
            for role, fill in (("tokens", pad), ("values", np.nan)):
                if role in columns:
                    column = columns[role]
                    on_grid = np.full((n, *column.shape[1:]), fill, dtype=column.dtype)
                    on_grid[filled] = column[keep]
                    resampled[role] = on_grid
            if self.level:
                for i, kept in zip(filled.tolist(), keep.tolist()):
                    grid_children[i] = children[kept]
        else:
            reduce = {"min": np.minimum, "max": np.maximum, "avg": np.add}[self.method]
            values = reduce.reduceat(
                columns["values"].astype(np.float64), starts, axis=0
            )
            if self.method == "avg":
                values /= (ends - starts + 1).reshape(-1, *[1] * (values.ndim - 1))
            on_grid = np.full((n, *values.shape[1:]), np.nan, dtype=np.float32)
            on_grid[filled] = values
            resampled["values"] = on_grid
        return resampled, grid_children


class _Imputer:
    """Runs the impute steps of one level (see :class:`Impute`).

    It also holds the statistics of the ``mean`` / ``min`` / ``max`` steps:
    :meth:`accumulate` adds one ``Records``' raw values during ``fit()``, and
    :meth:`finish` turns the running state into statistics.
    """

    def __init__(self, level: int, steps: tuple[_ImputeStep, ...], tokens: int):
        self.level, self.steps, self.tokens = level, steps, tokens
        self.learns_statistics = any(step.method in _STATISTICS for step in steps)
        # Per statistic step: its running state during fit(), then, per token
        # key, its statistic.
        self._state = self._new_state()
        self._stats: dict[int, dict[tuple[int, ...], np.ndarray]] = {}

    def _new_state(self) -> dict[int, dict[tuple[int, ...], list[np.ndarray]]]:
        return {
            index: {}
            for index, step in enumerate(self.steps)
            if step.method in _STATISTICS
        }

    def accumulate(self, values: np.ndarray, ids: list[list[int]]) -> None:
        """Adds one ``Records``' raw values (and their token IDs) to the statistics."""
        for index, state in self._state.items():
            step = self.steps[index]
            keys = np.zeros((len(values), 0), dtype=np.int64)
            if step.by_tokens is not None:
                if len(ids) != len(values):
                    raise ValueError(
                        f"Level {self.level}: {len(ids)} tokens but {len(values)} values."
                    )
                table = np.array(ids, dtype=np.int64).reshape(len(ids), self.tokens)
                keys = table[:, list(step.by_tokens)]
            self._accumulate(step, values, keys, state)

    def finish(self) -> None:
        """Turns the running state into statistics, ready for a new fit()."""
        for index, state in self._state.items():
            stats: dict[tuple[int, ...], np.ndarray] = {}
            for key, parts in state.items():
                if self.steps[index].method == "mean":
                    with np.errstate(invalid="ignore", divide="ignore"):
                        stat = parts[0] / parts[1]
                else:
                    stat = np.where(np.isinf(parts[0]), np.nan, parts[0])
                stats[key] = stat.astype(np.float32)
            self._stats[index] = stats
        self._state = self._new_state()

    @staticmethod
    def _split_by_key(keys: np.ndarray) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
        """Yields each distinct row of ``keys`` and the mask of rows equal to it."""
        if len(keys) == 0:
            return
        if keys.shape[1] == 0:
            yield (), np.ones(len(keys), dtype=bool)
            return
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for index, key in enumerate(unique.tolist()):
            yield tuple(key), inverse == index

    def _accumulate(
        self,
        step: _ImputeStep,
        values: np.ndarray,
        keys: np.ndarray,
        state: dict[tuple[int, ...], list[np.ndarray]],
    ) -> None:
        """Adds raw values to the running state of one statistic, per token key."""
        reduced = range(values.ndim - 1) if step.by_dims is None else step.by_dims
        axes = (0, *(1 + d for d in reduced))
        values = values.astype(np.float64)  # sums over a dataset need the range
        finite = np.isfinite(values)
        for key, rows in self._split_by_key(keys):
            x, ok = values[rows], finite[rows]
            if step.method == "mean":
                total = np.where(ok, x, 0.0).sum(axis=axes, keepdims=True)[0]
                count = ok.sum(axis=axes, keepdims=True)[0]
                if key in state:
                    state[key][0] += total
                    state[key][1] += count
                else:
                    state[key] = [total, count]
            else:
                low = step.method == "min"
                fill = np.inf if low else -np.inf
                reduce = np.min if low else np.max
                extreme = reduce(np.where(ok, x, fill), axis=axes, keepdims=True)[0]
                if key in state:
                    combine = np.minimum if low else np.maximum
                    state[key][0] = combine(state[key][0], extreme)
                else:
                    state[key] = [extreme]

    @staticmethod
    def _previous_valid(valid: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Index of the latest valid row at or before each row, in its segment.

        ``valid`` is ``[L, M]``; ``start[i]`` is the first row of row ``i``'s
        segment. Returns ``[L, M]``, with -1 where there is none.
        """
        index = np.arange(len(valid))[:, None]
        latest = np.maximum.accumulate(np.where(valid, index, -1), axis=0)
        return np.where(latest >= start[:, None], latest, -1)

    def _next_valid(
        self, valid: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        """Like :meth:`_previous_valid`, for the earliest valid row at or after."""
        last = len(valid) - 1
        found = self._previous_valid(valid[::-1], (last - end)[::-1])[::-1]
        return np.where(found >= 0, last - found, -1)

    @staticmethod
    def _segments(
        group: np.ndarray, keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orders rows by segment: by group, then by token key, keeping time order.

        Returns the order and, for each row in that order, the first and last
        row of its segment.
        """
        n = len(group)
        order = np.lexsort([np.arange(n), *keys.T[::-1], group])
        group, keys = group[order], keys[order]
        new = np.ones(n, dtype=bool)
        new[1:] = (group[1:] != group[:-1]) | (keys[1:] != keys[:-1]).any(axis=1)
        first = np.flatnonzero(new)
        segment = np.cumsum(new) - 1
        return order, first[segment], np.append(first[1:], n)[segment] - 1

    def __call__(self, columns: dict[Role, np.ndarray]) -> None:
        """Runs the impute steps in place on the level's concatenated columns."""
        pad = TokenProcessorInterface.PAD
        counts = columns["offsets"]
        n = int(counts.sum())
        group = np.repeat(np.arange(len(counts)), counts)
        for index, step in enumerate(self.steps):
            values, tokens = columns.get("values"), columns.get("tokens")
            if step.method == "zero":
                if values is not None:
                    columns["values"] = np.where(np.isnan(values), 0.0, values).astype(
                        np.float32
                    )
                continue  # missing tokens are already <pad>

            keys = (
                tokens[:, list(step.by_tokens)]
                if tokens is not None and step.by_tokens is not None
                else np.zeros((n, 0), dtype=np.int64)
            )
            if step.method in _STATISTICS:
                assert values is not None  # the configuration requires values
                stats = self._stats.get(index, {})
                values = values.copy()
                for key, rows in self._split_by_key(keys):
                    stat = stats.get(key)
                    if stat is not None:
                        x = values[rows]
                        values[rows] = np.where(np.isnan(x), stat, x)
                columns["values"] = values
                continue

            # Along time: work in segment order, so each segment is contiguous.
            order, start, end = self._segments(group, keys)
            skip = (keys[order] == pad).any(axis=1)  # rows without a token
            if values is not None:
                x = values.reshape(n, -1)[order]
                valid = ~np.isnan(x)
                before = self._previous_valid(valid, start)
                after = self._next_valid(valid, start, end)
                if step.method == "linear":
                    time = columns.get("time")
                    position = (time if time is not None else np.arange(n))[order]
                    position = position.astype(np.float64)
                    ok = ~valid & (before >= 0) & (after >= 0)
                    x0 = np.take_along_axis(x, np.maximum(before, 0), axis=0)
                    x1 = np.take_along_axis(x, np.maximum(after, 0), axis=0)
                    t0 = position[np.maximum(before, 0)]
                    t1 = position[np.maximum(after, 0)]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        weight = np.where(
                            t1 > t0, (position[:, None] - t0) / (t1 - t0), 0.0
                        )
                    filled = np.where(ok, x0 + (x1 - x0) * weight, x)
                else:
                    source = before if step.method == "forward_fill" else after
                    taken = np.take_along_axis(x, np.maximum(source, 0), axis=0)
                    filled = np.where(~valid & (source >= 0), taken, x)
                filled[skip] = x[skip]
                result = np.empty_like(filled)
                result[order] = filled
                columns["values"] = result.reshape(values.shape).astype(np.float32)
            if (
                tokens is not None
                and step.by_tokens is None
                and step.method != "linear"
            ):
                # Without by_tokens, <pad> tokens (steps without records) are
                # filled from the same neighbour as the values.
                t = tokens[order]
                valid = (t != pad).any(axis=1)[:, None]
                if step.method == "forward_fill":
                    source = self._previous_valid(valid, start)[:, 0]
                else:
                    source = self._next_valid(valid, start, end)[:, 0]
                t = np.where(
                    (~valid[:, 0] & (source >= 0))[:, None], t[np.maximum(source, 0)], t
                )
                result = np.empty_like(t)
                result[order] = t
                columns["tokens"] = result


def _new_vocab() -> dict[str, int]:
    return {"<pad>": TokenProcessorInterface.PAD, "<unk>": TokenProcessorInterface.UNK}


@register_processor("sequence_v2")
class SequenceProcessorV2(FeatureProcessor):
    """Processor for hierarchical packed records, without padding or truncation.

    Reads a :class:`Records` and emits a tuple of tensors in :meth:`schema`
    order: by level, then by role (``tokens``, ``values``, ``time``,
    ``offsets``). ``offsets`` at level ``l`` partitions level ``l`` records
    into level ``l + 1`` records, in CSR layout. Which roles a level has, the
    number of token namespaces ``K`` and the value shape ``S`` come from the
    configuration; :meth:`fit` only builds the vocabularies and the statistics
    of the impute steps.

    Subclasses are configuration only: their ``__init__`` passes fixed
    ``levels`` to ``super().__init__()`` and adds no logic.

    Args:
        levels: One :class:`LevelConfig` per level, innermost (level 0)
            first.

    Examples:
        >>> proc = SequenceProcessorV2(levels=[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}])
        >>> proc.fit([{"visits": {"children": [{"tokens": ["A", "B"]}, {"tokens": ["C"]}]}}], "visits")
        >>> proc.schema()
        ((0, 'tokens'), (0, 'offsets'))
        >>> proc.process({"children": [{"tokens": ["A", "B"]}, {"tokens": []}, {"tokens": ["C"]}]})
        (tensor([2, 3, 4]), tensor([0, 2, 2, 3]))
    """

    def __init__(self, levels: Sequence[LevelConfig]):
        if not isinstance(levels, (list, tuple)) or not levels:
            raise ValueError("levels must be a non-empty list of dicts.")
        self._config = [
            _InternalLevelConfig.parse(level, config)
            for level, config in enumerate(levels)
        ]
        self.levels = [config.copy() for config in levels]
        self._mappers = [
            self._load_mappers(config.code_mapping) for config in self._config
        ]
        for level, (config, mappers) in enumerate(zip(self._config, self._mappers)):
            # A code with several targets becomes several records, which only
            # makes sense in an unordered level with no values to copy.
            many = [m for m in mappers or () if m is not None and m.has_many_targets]
            if many and (config.ordering != "set" or config.values):
                raise ValueError(
                    f"Level {level}: {many[0].s_vocab}->{many[0].t_vocab} maps some "
                    "codes to several targets, which is only allowed on a 'set' "
                    f"level without values (got ordering {config.ordering!r}"
                    f"{' with values' if config.values else ''}). Put the tokens "
                    "to map in their own 'set' level, one level below this one."
                )

        for level, config in enumerate(self._config):
            merge = config.resample is not None and config.resample[1] == "merge"
            if merge and self._config[level - 1].ordering != "set":
                raise ValueError(
                    f"Level {level}: resample method 'merge' unions the children's "
                    f"records, so level {level - 1} must have ordering 'set'."
                )

        self._resamplers = [
            None if config.resample is None else _Resampler(level, *config.resample)
            for level, config in enumerate(self._config)
        ]
        # Fitted state: one vocabulary per level and token namespace, and the
        # statistics held by each level's imputer.
        self._imputers = [
            _Imputer(level, config.impute, config.tokens) if config.impute else None
            for level, config in enumerate(self._config)
        ]
        self.learns_statistics = any(
            imputer is not None and imputer.learns_statistics
            for imputer in self._imputers
        )
        self._vocabs = [
            [_new_vocab() for _ in range(config.tokens)] for config in self._config
        ]
        self._fitted = False

    @staticmethod
    def _load_mappers(
        code_mapping: tuple[CodeMapping | None, ...] | None,
    ) -> "tuple[CrossMap | None, ...] | None":
        if code_mapping is None:
            return None
        from pyhealth.medcode import CrossMap

        return tuple(
            None if pair is None else CrossMap.load(pair[0], pair[1])
            for pair in code_mapping
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """The outermost level ``L``."""
        return len(self._config) - 1

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}: fit() must be called first.")

    def _output_roles(self, level: int) -> tuple[Role, ...]:
        config = self._config[level]
        present = {
            "tokens": config.tokens > 0,
            "values": len(config.values) > 0,
            "time": config.ordering == "series" and config.resample is None,
            "offsets": level < self.depth,
        }
        return tuple(role for role in get_args(Role) if present[role])

    # FeatureProcessor.schema() returns semantic names for legacy models; V2
    # deliberately returns (level, role) keys instead and is not used by them.
    def schema(self) -> tuple[FieldKey, ...]:  # type: ignore[override]
        """Fields of :meth:`process` output: ``(level, role)``, by level then role."""
        return tuple(
            (level, role)
            for level in range(self.depth + 1)
            for role in self._output_roles(level)
        )

    def batch_schema(self) -> tuple[FieldKey, ...]:
        """Fields of :meth:`collate` output: :meth:`schema` plus the batch offsets."""
        batch_offsets: FieldKey = (self.depth, "offsets")
        return (*self.schema(), batch_offsets)

    def ordering(self) -> dict[int, OrderType]:
        """Ordering of each output level; a resampled level becomes ``sequence``."""
        return {
            level: "sequence" if config.resample is not None else config.ordering
            for level, config in enumerate(self._config)
        }

    def vocab_size(self) -> dict[int, tuple[int, ...] | None]:
        """Vocabulary size per token namespace, including ``<pad>`` and ``<unk>``."""
        self._require_fit()
        return {
            level: tuple(len(vocab) for vocab in self._vocabs[level])
            if config.tokens
            else None
            for level, config in enumerate(self._config)
        }

    def value_dim(self) -> dict[int, tuple[int, ...] | None]:
        """Trailing value shape ``S`` per level."""
        return {
            level: config.values or None for level, config in enumerate(self._config)
        }

    # Like schema(): V2 reports one entry per level instead of the legacy tuple.
    def spatial(self) -> dict[int, tuple[bool, ...] | None]:  # type: ignore[override]
        """Whether each value axis is spatial, per level."""
        return {
            level: None
            if not config.values
            else config.spatial or (False,) * len(config.values)
            for level, config in enumerate(self._config)
        }

    def vocab(self, level: int, namespace: int = 0) -> dict[str, int]:
        """Token-to-index mapping of one namespace at one level."""
        self._require_fit()
        if not self._config[level].tokens:
            raise ValueError(f"Level {level} has no tokens.")
        return dict(self._vocabs[level][namespace])

    # ------------------------------------------------------------------
    # Reading raw records
    # ------------------------------------------------------------------

    def _read_tokens(self, level: int, records: Records) -> list[tuple[str, ...]]:
        """Returns each token as a K-tuple, checking it against the configured K."""
        if "tokens" not in records:
            raise ValueError(f"Level {level}: the records have no 'tokens'.")
        k = self._config[level].tokens
        tokens: list[tuple[str, ...]] = []
        for token in records["tokens"]:
            if k == 1 and isinstance(token, str):
                tokens.append((token,))
            elif (
                k > 1
                and isinstance(token, tuple)
                and len(token) == k
                and all(isinstance(code, str) for code in token)
            ):
                tokens.append(token)
            else:
                expected = "a string" if k == 1 else f"a tuple of {k} strings"
                raise ValueError(
                    f"Level {level}: expected each token as {expected}, got {token!r}."
                )
        return tokens

    def _map_tokens(
        self, level: int, tokens: list[tuple[str, ...]]
    ) -> list[tuple[str, ...]]:
        """Applies the code mapping; a code with several targets becomes several records."""
        mappers = self._mappers[level]
        if mappers is None:
            return tokens
        mapped_tokens: list[tuple[str, ...]] = []
        for token in tokens:
            targets: list[list[str]] = []
            for code, mapper in zip(token, mappers):
                mapped = mapper.map(code) if mapper is not None else []
                targets.append(mapped if mapped else [code])
            if all(len(t) == 1 for t in targets):
                mapped_tokens.append(tuple(t[0] for t in targets))
                continue
            # The configuration rules out time and values on a mapped level.
            if level > 0 or len(token) > 1:
                reason = (
                    "its records also have children"
                    if level > 0
                    else "there are several namespaces"
                )
                raise ValueError(
                    f"Level {level}: {token!r} maps to several codes, which would duplicate "
                    f"records, but {reason}."
                )
            mapped_tokens.extend((target,) for target in targets[0])
        return mapped_tokens

    def _read_values(self, level: int, records: Records) -> np.ndarray:
        """Returns ``float32`` values ``[N, *S]``."""
        if "values" not in records:
            raise ValueError(f"Level {level}: the records have no 'values'.")
        shape = self._config[level].values
        array = np.asarray(records["values"], dtype=np.float64)
        if array.ndim == 1 and shape == (1,):
            array = array.reshape(-1, 1)  # scalars given as [v, ...]
        if array.ndim == 1 and array.size == 0:
            return np.zeros((0, *shape), dtype=np.float32)
        if array.ndim < 2 or tuple(array.shape[1:]) != shape:
            raise ValueError(
                f"Level {level}: expected values of shape [N, *{shape}], got "
                f"{tuple(array.shape)}."
            )
        with np.errstate(over="ignore"):  # overflow becomes inf, which process rejects
            return array.astype(np.float32)

    def _read_time(self, level: int, records: Records) -> np.ndarray:
        if "time" not in records:
            raise ValueError(f"Level {level}: the 'series' records have no 'time'.")
        array = np.asarray(records["time"], dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(
                f"Level {level}: time must be 1-D, got shape {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Level {level}: time must be finite.")
        if (np.diff(array) < 0).any():
            raise ValueError(f"Level {level}: time must be non-decreasing.")
        return array

    def _empty(self, level: int, role: Role) -> np.ndarray:
        """The start of each column, so a level without records still works."""
        config = self._config[level]
        if role == "tokens":
            return np.zeros((0, config.tokens), dtype=np.int64)
        if role == "values":
            return np.zeros((0, *config.values), dtype=np.float32)
        return np.zeros(0, dtype=np.float32)  # time

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Builds the vocabularies and the statistics of the impute steps.

        Vocabularies come from the (code-mapped) tokens, statistics from the
        raw values. Records are not checked here; :meth:`process` checks them.

        Args:
            samples: Iterable of sample dicts, read once.
            field: Key whose values are :class:`Records`.
        """
        for sample in samples:
            # Walk one level at a time, from the outermost level L down to 0.
            level_records: list[Records] = [sample[field]]
            for level in range(self.depth, -1, -1):
                vocabs, mappers = self._vocabs[level], self._mappers[level]
                imputer = self._imputers[level]
                children: list[Records] = []
                for records in level_records:
                    ids: list[list[int]] = []
                    for token in records.get("tokens", ()):
                        codes = (token,) if isinstance(token, str) else token
                        row: list[int] = []
                        for vocab, code, mapper in zip(
                            vocabs, codes, mappers or repeat(None)
                        ):
                            targets = mapper.map(code) if mapper is not None else []
                            for target in targets or [code]:
                                if target not in vocab:
                                    vocab[target] = len(vocab)
                            row.append(vocab[(targets or [code])[0]])
                        ids.append(row)
                    if imputer is not None and imputer.learns_statistics:
                        # Statistics are taken from the raw values, before
                        # resampling.
                        imputer.accumulate(self._read_values(level, records), ids)
                    children.extend(records.get("children", ()))
                level_records = children

        for imputer in self._imputers:
            if imputer is not None:
                imputer.finish()
        self._fitted = True

    def process(self, value: Records) -> tuple[torch.Tensor, ...]:
        """Turns one :class:`Records` into packed tensors in :meth:`schema` order.

        Args:
            value: The sample's :class:`Records` for this field.

        Returns:
            Tuple of tensors: ``int64`` tokens (``[N]``, or ``[N, K]``),
            ``float32`` values ``[N, *S]``, ``float32`` time ``[N]`` and
            ``int64`` offsets ``[N_{l+1} + 1]``.
        """
        self._require_fit()
        unk = TokenProcessorInterface.UNK
        fields: dict[FieldKey, torch.Tensor] = {}
        # Walk one level at a time, from the outermost level L down to 0. Each
        # record of a level owns one group of Records one level down; the
        # groups come in their records' order, which is CSR order.
        groups: list[list[Records]] = [[value]]
        for level in range(self.depth, -1, -1):
            config, vocabs = self._config[level], self._vocabs[level]
            declared: tuple[tuple[Role, bool], ...] = (
                ("tokens", config.tokens > 0),
                ("values", len(config.values) > 0),
                ("time", config.ordering == "series"),
            )
            parts: dict[Role, list[np.ndarray]] = {
                role: [self._empty(level, role)] for role, on in declared if on
            }
            counts: list[int] = []
            children: list[list[Records]] = []
            for group in groups:
                count = 0
                for records in group:
                    # Only what the configuration declares is read; other keys
                    # are ignored.
                    lengths: dict[str, int] = {}
                    tokens = None
                    if config.tokens:
                        tokens = self._read_tokens(level, records)
                        lengths["tokens"] = len(tokens)
                    if config.values:
                        values = self._read_values(level, records)
                        parts["values"].append(values)
                        lengths["values"] = len(values)
                    if config.ordering == "series":
                        time = self._read_time(level, records)
                        parts["time"].append(time)
                        lengths["time"] = len(time)
                    if level > 0:
                        if "children" not in records:
                            raise ValueError(
                                f"Level {level}: the records have no 'children' "
                                f"({self.depth + 1} levels are configured)."
                            )
                        children.extend([kid] for kid in records["children"])
                        lengths["children"] = len(records["children"])
                    if len(set(lengths.values())) > 1:
                        raise ValueError(
                            f"Level {level}: columns have different lengths {lengths}."
                        )
                    n = next(iter(lengths.values()))
                    if tokens is not None:
                        tokens = self._map_tokens(level, tokens)
                        ids = [
                            [vocab.get(code, unk) for vocab, code in zip(vocabs, token)]
                            for token in tokens
                        ]
                        parts["tokens"].append(
                            np.array(ids, dtype=np.int64).reshape(
                                len(tokens), config.tokens
                            )
                        )
                        n = len(tokens)
                    count += n
                counts.append(count)

            columns: dict[Role, np.ndarray] = {
                role: np.concatenate(part) for role, part in parts.items()
            }
            columns["offsets"] = np.array(counts, dtype=np.int64)
            resampler, imputer = self._resamplers[level], self._imputers[level]
            if resampler is not None:
                columns, children = resampler(columns, children)
            if imputer is not None:
                imputer(columns)
            for role in self._output_roles(level):
                array = columns[role]
                if role == "offsets":
                    array = np.concatenate([np.zeros(1, np.int64), np.cumsum(array)])
                elif role == "tokens" and config.tokens == 1:
                    array = array.reshape(-1)
                elif role == "values" and not np.isfinite(array).all():
                    raise ValueError(
                        f"Level {level}: values contain NaN or inf; impute them "
                        "(see Impute) or remove them in the task."
                    )
                fields[(level, role)] = torch.from_numpy(array)
            groups = children
        return tuple(fields[key] for key in self.schema())

    def collate(
        self, items: list[tuple[torch.Tensor, ...]]
    ) -> tuple[torch.Tensor, ...]:
        """Packs a batch of :meth:`process` outputs, in :meth:`batch_schema` order.

        Payloads are concatenated, each sample's offsets are shifted by the
        records before it, and the batch offsets ``[B + 1]`` are appended.
        """
        if not items:
            raise ValueError("collate needs at least one sample.")
        schema = self.schema()
        output = []
        for index, (_, role) in enumerate(schema):
            column = [item[index] for item in items]
            if role != "offsets":
                output.append(torch.cat(column))
                continue
            parts, shift = [column[0][:1]], 0
            for offsets in column:
                parts.append(offsets[1:] + shift)
                shift += int(offsets[-1])
            output.append(torch.cat(parts))
        counts = [self._count(schema, item, self.depth) for item in items]
        output.append(torch.tensor([0, *accumulate(counts)], dtype=torch.long))
        return tuple(output)

    @staticmethod
    def _count(
        schema: tuple[FieldKey, ...], output: tuple[torch.Tensor, ...], level: int
    ) -> int:
        """Number of records at ``level``, read from any field that knows it."""
        for (field_level, role), tensor in zip(schema, output):
            if field_level == level and role != "offsets":
                return tensor.shape[0]
            if field_level == level - 1 and role == "offsets":
                return tensor.shape[0] - 1
        raise ValueError(f"No field records the size of level {level}.")

    def validate(self, output: tuple[torch.Tensor, ...], batched: bool = False) -> None:
        """Checks an output of :meth:`process` (or :meth:`collate` if ``batched``).

        Raises:
            TypeError: If a field is not a tensor.
            ValueError: If the output breaks the packed-record invariants.
        """
        self._require_fit()
        schema = self.batch_schema() if batched else self.schema()
        if not isinstance(output, tuple) or len(output) != len(schema):
            raise ValueError(f"Expected a tuple of {len(schema)} tensors for {schema}.")
        fields = dict(zip(schema, output))

        for (level, role), tensor in fields.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{(level, role)}: expected a tensor, got {type(tensor).__name__}."
                )
            dtype = torch.long if role in ("tokens", "offsets") else torch.float32
            if tensor.dtype != dtype:
                raise ValueError(
                    f"{(level, role)}: expected {dtype}, got {tensor.dtype}."
                )
            if role == "tokens":
                sizes = tuple(len(vocab) for vocab in self._vocabs[level])
                expected_ndim = 1 if len(sizes) == 1 else 2
                if tensor.dim() != expected_ndim or (
                    expected_ndim == 2 and tensor.shape[1] != len(sizes)
                ):
                    raise ValueError(
                        f"{(level, role)}: bad shape {tuple(tensor.shape)} for {len(sizes)} namespace(s)."
                    )
                limits = (
                    torch.tensor(sizes)
                    .expand(tensor.shape[0], -1)
                    .reshape(tensor.shape)
                )
                if (tensor < 0).any() or (tensor >= limits).any():
                    raise ValueError(
                        f"{(level, role)}: token index outside the vocabulary."
                    )
            elif role == "values":
                shape = self._config[level].values
                if tuple(tensor.shape[1:]) != shape or tensor.dim() != 1 + len(shape):
                    raise ValueError(
                        f"{(level, role)}: expected [N, *{shape}], got {tuple(tensor.shape)}."
                    )
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{(level, role)}: values contain NaN or inf.")
            elif role == "time":
                if tensor.dim() != 1 or not torch.isfinite(tensor).all():
                    raise ValueError(f"{(level, role)}: time must be 1-D and finite.")
            else:
                if (
                    tensor.dim() != 1
                    or tensor.numel() == 0
                    or tensor[0] != 0
                    or (tensor.diff() < 0).any()
                ):
                    raise ValueError(
                        f"{(level, role)}: offsets must start at 0 and never decrease."
                    )

        for level in range(self.depth + 1):
            sizes = {
                (field_level, role): tensor.shape[0]
                for (field_level, role), tensor in fields.items()
                if field_level == level and role != "offsets"
            }
            if level > 0:
                sizes[(level - 1, "offsets")] = (
                    fields[(level - 1, "offsets")].shape[0] - 1
                )
            if len(set(sizes.values())) > 1:
                raise ValueError(
                    f"Level {level}: fields disagree on the record count {sizes}."
                )
            offsets = fields.get((level, "offsets"))
            if (
                offsets is not None
                and sizes
                and offsets[-1] != next(iter(sizes.values()))
            ):
                raise ValueError(
                    f"{(level, 'offsets')}: last offset must equal the level's record count."
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(levels={self.levels})"
