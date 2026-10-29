import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, final

from ._serialization import _check_keys, _decode, _dumps


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One unit of work a benchmark runs, identified by its params.

    ``params`` may hold JSON types (``None``, ``bool``, ``int``, finite
    ``float``, ``str``, lists/tuples and str-keyed mappings) plus
    module-level classes and functions. Classes and functions are serialized
    by import path as ``{"__type__": "<module>.<qualname>"}``, and that path
    must import back to the same object. Classes defined in ``__main__``
    (a script run directly or a notebook) are rejected: move them into an
    importable module.

    The ``id`` is the hex SHA-256 of the canonical JSON of ``params``. It is
    computed once at construction, so bad params fail immediately.

    Notes:
        - ``params`` must not be mutated after construction, because the
          id is cached.
        - The path uses the *defining* module: ``pyhealth.models.RNN``
          serializes as ``pyhealth.models.rnn.RNN``. Moving a class to
          another file changes the id.
        - ``1`` and ``1.0`` give different ids.
        - The id is unique only within one benchmark. Storage keys on
          ``(benchmark.name, evaluation.id)``.

    Attributes:
        params: The evaluation's parameters.
        id: Hex SHA-256 of the canonical JSON of ``params``.

    Raises:
        TypeError: If ``params`` holds a value that is not allowed, such as
            an arbitrary object, a lambda, a locally defined class, a class
            defined in ``__main__`` or a class whose path doesn't import
            back to it, or a mapping key that is not a string.
        ValueError: If ``params`` holds NaN or infinity, or uses
            ``"__type__"`` as a key anywhere.

    Examples:
        >>> from collections import OrderedDict
        >>> ev = Evaluation({"model": OrderedDict, "lr": 0.001})
        >>> ev.to_dict()
        {'lr': 0.001, 'model': {'__type__': 'collections.OrderedDict'}}
        >>> ev == Evaluation({"lr": 0.001, "model": OrderedDict})
        True
        >>> Evaluation.from_dict(ev.to_dict()) == ev
        True
    """

    params: Mapping[str, Any]
    id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _check_keys(self.params)
        digest = hashlib.sha256(_dumps(self.params).encode()).hexdigest()
        object.__setattr__(self, "id", digest)

    def to_dict(self) -> dict[str, Any]:
        """Returns ``params`` as JSON-compatible data.

        Returns:
            The params, with classes and functions replaced by their
            ``{"__type__": ...}`` form.
        """
        return json.loads(_dumps(self.params))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evaluation":
        """Builds an evaluation from ``to_dict`` output.

        This is the inverse of :meth:`to_dict`:
        ``Evaluation.from_dict(ev.to_dict()) == ev``. Each
        ``{"__type__": "<dotted.path>"}`` mapping is resolved to the class or
        function at that path. Public re-export paths are accepted, but the
        id always uses the defining-module path.

        Args:
            data: JSON-compatible params, as returned by :meth:`to_dict`.

        Returns:
            The rebuilt evaluation.

        Raises:
            ValueError: If a path can't be resolved, names something that is
                not a class or function, or shares its mapping with other
                keys.

        Examples:
            >>> ev = Evaluation.from_dict({"c": {"__type__": "json.JSONDecoder"}})
            >>> ev.to_dict()
            {'c': {'__type__': 'json.decoder.JSONDecoder'}}
            >>> Evaluation.from_dict(ev.to_dict()) == ev
            True
        """
        return cls(_decode(data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Evaluation):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Result:
    """The outcome of one evaluation.

    By convention, a failed evaluation is returned as
    ``Result("failed", {}, {"error": str(exc), "error_type":
    type(exc).__name__})``, so one failure doesn't discard the batch.

    ``metadata["error"]`` may contain patient data and must never be
    published. ``error_type`` is safe to display.

    Attributes:
        status: ``"success"`` or ``"failed"``.
        metrics: Metric names mapped to values.
        metadata: Any other information about the run.

    Examples:
        >>> Result("success", {"accuracy": 0.9})
        Result(status='success', metrics={'accuracy': 0.9}, metadata={})
        >>> exc = ValueError("bad input")
        >>> Result(
        ...     "failed", {}, {"error": str(exc), "error_type": type(exc).__name__}
        ... ).status
        'failed'
    """

    status: Literal["success", "failed"]
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """A benchmark defines and executes evaluations.

    Subclasses implement :attr:`name` and :meth:`_evaluate`. Callers use
    :meth:`evaluate`, which validates the evaluations, runs them and
    validates the results.

    The ``_evaluate`` contract:
        - Any order, grouping, or internal parallelism is allowed.
        - It returns one ``Result`` per evaluation, and catches
          per-evaluation failures as failed Results.
        - ``cache_dir`` may be shared by concurrent jobs, so writes are
          atomic (temporary file, then ``os.replace``).
        - ``output_dir`` holds per-run artifacts. Results are always
          returned, never only written to disk.

    Examples:
        >>> class Squares(Benchmark):
        ...     @property
        ...     def name(self):
        ...         return "squares"
        ...
        ...     def _evaluate(self, evaluations, cache_dir, output_dir):
        ...         return {
        ...             ev: Result("success", {"square": ev.params["x"] ** 2})
        ...             for ev in evaluations
        ...         }
        >>> ev = Evaluation({"x": 3})
        >>> Squares().evaluate([ev])[ev].metrics
        {'square': 9}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Lowercase identifier for the benchmark, e.g. ``"cpbench"``."""

    def validate_evaluations(self, evaluations: Sequence[Evaluation]) -> None:
        """Checks the evaluations before they run.

        Subclasses extend this via ``super()``.

        Args:
            evaluations: The evaluations to run.

        Raises:
            ValueError: If two evaluations share an id. Every duplicate is
                listed, one per line.
        """
        seen: set[str] = set()
        problems = []
        for i, ev in enumerate(evaluations):
            if ev.id in seen:
                problems.append(f"evaluations[{i}]: duplicate id {ev.id}")
            seen.add(ev.id)
        if problems:
            raise ValueError("\n".join(problems))

    def validate_results(
        self,
        evaluations: Sequence[Evaluation],
        results: Mapping[Evaluation, Result],
    ) -> None:
        """Checks that the results match the evaluations.

        Subclasses extend this via ``super()``.

        Args:
            evaluations: The evaluations that were run.
            results: The results returned by :meth:`_evaluate`.

        Raises:
            ValueError: Unless ``results`` has exactly one entry per
                evaluation. Every missing or extra entry is listed, one per
                line.
        """
        expected = set(evaluations)
        problems = [
            f"missing result for evaluation {ev.id}"
            for ev in evaluations
            if ev not in results
        ]
        problems += [
            f"unexpected result for evaluation {ev.id}"
            for ev in results
            if ev not in expected
        ]
        if problems:
            raise ValueError("\n".join(problems))

    @final
    def evaluate(
        self,
        evaluations: Sequence[Evaluation],
        cache_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[Evaluation, Result]:
        """Validates, runs and returns the results of the evaluations.

        Exceptions escaping :meth:`_evaluate` propagate unchanged.

        Args:
            evaluations: The evaluations to run.
            cache_dir: Directory for reusable intermediate data, possibly
                shared by concurrent jobs.
            output_dir: Directory for per-run artifacts.

        Returns:
            One ``Result`` per evaluation.

        Raises:
            ValueError: If the evaluations or the results fail validation.
        """
        self.validate_evaluations(evaluations)
        results = self._evaluate(evaluations, cache_dir, output_dir)
        self.validate_results(evaluations, results)
        return results

    @abstractmethod
    def _evaluate(
        self,
        evaluations: Sequence[Evaluation],
        cache_dir: Path | None,
        output_dir: Path | None,
    ) -> dict[Evaluation, Result]:
        """Runs the evaluations. See the class docstring for the contract."""
