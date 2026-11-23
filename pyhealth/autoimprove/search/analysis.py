"""Analysis turns: a proposer asks a question of the training data.

The proposal is a Python module that defines::

    def analyze(data, out) -> None

``data`` holds the training split as columns (one numpy array per sample
column, see :func:`sample_columns`), with the label, an opaque patient id,
the fold, and the current selection's out-of-fold prediction when there is
one. The code never runs on validation or test data.

Results leave only through ``out`` (:class:`Out`):

- ``out.table(name, by, values, stats)``: the harness groups the samples,
  counts distinct patients per group, and withholds every group with fewer
  than :data:`~pyhealth.metrics.clinical.MIN_CELL` patients. When exactly one
  group would be withheld, the next smallest is withheld too, so its count
  cannot be worked out from the total (the rule
  :func:`~pyhealth.metrics.subgroup_metrics` follows).
- ``out.value(name, x, n_patients)``: one number over at least ``MIN_CELL``
  patients.
- ``out.note(text)``: up to 300 characters.

Anything else the code computes stays in its process, and the whole output
is capped at :data:`MAX_OUTPUT` characters. The proposer turns the results
into short findings, which later prompts of every kind include.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.sandbox import (
    ContractError,
    check_or_raise,
    run_sandboxed,
    sample_patient_ids,
)

ENTRY = "analyze"

MAX_OUTPUT = 4000
"""Most characters of results returned from one analysis."""

MAX_GROUPS = 40
"""Most groups in one table."""

STATS = ("mean", "median", "std", "rate", "sum")
"""Statistics a table may report per group; no minimum or maximum, which
would quote one patient's value."""

ANALYSIS_CONTRACT = """\
Write one Python module that defines

    def analyze(data, out):
        ...

- `data` holds the TRAINING split as columns: `data.<name>` or
  `data["<name>"]` is a numpy array with one value per sample, for every
  name in `data.columns` (listed below). It always has "label", "patient"
  (an opaque id), "fold", and "oof_prob" (the selected model's out-of-fold
  prediction) when a model has been selected.
- Return results ONLY through `out`; nothing else leaves the process:
  - out.table(name, by={"col": array}, values={"col": array},
              stats=("mean", "rate", "median", "std", "sum"))
    The harness groups the samples by `by` (at most 2 columns, at most 40
    groups: bin continuous columns first), counts distinct patients per
    group, and withholds groups under 11 patients.
  - out.value(name, number, n_patients) for one number over >= 11 patients.
  - out.note(text) for up to 300 characters.
- Allowed imports: numpy, scipy, sklearn, pandas, pyhealth, math,
  statistics, collections, itertools, functools. No file, network or shell
  access."""


@dataclass
class AnalysisSpec:
    """A proposed analysis: its question and code.

    Examples:
        >>> AnalysisSpec.from_dict({"name": "age", "question": "Does error rise with age?"},
        ...                        "def analyze(data, out):\\n    pass\\n").name
        'age'
    """

    name: str
    code: str
    question: str = ""
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], code: str) -> AnalysisSpec:
        """Builds and checks a spec from a reply's JSON and code.

        Raises:
            ValueError: If the name is malformed or the code fails the static
                check.
        """
        name = str(data.get("name", "")).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
            raise ValueError('"name" must be a short identifier, e.g. "error_by_age"')
        check_or_raise(code, ENTRY)
        return cls(name, code, str(data.get("question", "")),
                   str(data.get("rationale", "")))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "question": self.question,
                "rationale": self.rationale}


def sample_columns(dataset: Any, samples: Sequence[Mapping[str, Any]] | None = None,
                   max_width: int = 16) -> dict[str, np.ndarray]:
    """The dataset's samples as numeric columns, for evaluations and analyses.

    - A field whose value is one number becomes a column of that name.
    - A fixed-width vector of at most ``max_width`` floating-point numbers
      becomes columns ``<field>_0``, ``<field>_1``, ...
    - A sequence (e.g. codes or visits) becomes ``n_<field>``, its length.
      Integer tensors are token ids, so they count as sequences even when
      every sample has the same number of them.
    - The label becomes ``label`` (one value per sample, or the first of a
      multi-label row), and the patient becomes ``patient``, an opaque hash.

    Examples:
        >>> cols = sample_columns(train_ds)  # doctest: +SKIP
        >>> sorted(cols)  # doctest: +SKIP
        ['label', 'n_conditions', 'patient']
    """
    if samples is None:
        samples = [dataset[i] for i in range(len(dataset))]
    columns: dict[str, np.ndarray] = {}
    for key in getattr(dataset, "input_schema", {}):
        tokens = any(_is_integer_tensor(s.get(key)) for s in samples[:50])
        values = [None if tokens else _numeric(s.get(key)) for s in samples]
        if all(v is not None and v.size == 1 for v in values):
            columns[key] = np.array([float(v.ravel()[0]) for v in values])
        elif all(v is not None and v.ndim == 1 for v in values) and len(
            {v.size for v in values}
        ) == 1 and values and values[0].size <= max_width:
            for j in range(values[0].size):
                columns[f"{key}_{j}"] = np.array([float(v[j]) for v in values])
        else:
            lengths = [_length(s.get(key)) for s in samples]
            if all(n is not None for n in lengths):
                columns[f"n_{key}"] = np.array(lengths, dtype=float)
    label_key = next(iter(getattr(dataset, "output_schema", {}) or {}), None)
    if label_key is not None:
        labels = [_numeric(s.get(label_key)) for s in samples]
        if all(v is not None and v.size >= 1 for v in labels):
            columns["label"] = np.array([float(v.ravel()[0]) for v in labels])
    columns["patient"] = np.array(sample_patient_ids(
        [s.get("patient_id", i) for i, s in enumerate(samples)]))
    return columns


def _is_integer_tensor(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None or not hasattr(value, "shape") or len(value.shape) == 0:
        return False
    return "int" in str(dtype) and "uint8" not in str(dtype)


def _numeric(value: Any) -> np.ndarray | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.ndim <= 1 else None


def _length(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "shape") and len(value.shape) >= 1:
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return None


class Data:
    """Columns of the training split, by attribute or by key.

    Examples:
        >>> data = Data({"age": np.array([63, 71]), "label": np.array([0, 1])})
        >>> data.age.tolist(), data["label"].tolist()
        ([63, 71], [0, 1])
    """

    def __init__(self, columns: Mapping[str, np.ndarray]):
        self._columns = {k: np.asarray(v) for k, v in columns.items()}

    @property
    def columns(self) -> list[str]:
        return sorted(self._columns)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._columns:
            raise ContractError(f"no column {name!r}; columns: {self.columns}")
        return self._columns[name].copy()

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def __len__(self) -> int:
        return len(next(iter(self._columns.values()), []))


class Out:
    """Where an analysis puts its results; small cells are withheld here.

    Args:
        patients: Patient id per sample, for counting distinct patients.
        min_count: Smallest group reported, in patients.

    Examples:
        >>> out = Out(np.array(["a"] * 20 + ["b"] * 3), min_count=11)
        >>> out.value("mean age", 61.5, n_patients=23)
        >>> out.results[0]["value"]
        61.5
    """

    def __init__(self, patients: np.ndarray, min_count: int | None = None):
        from pyhealth.metrics.clinical import MIN_CELL

        self.patients = np.asarray(patients)
        self.min_count = MIN_CELL if min_count is None else min_count
        self.results: list[dict[str, Any]] = []

    def table(self, name: str, by: Mapping[str, Any], values: Mapping[str, Any] = (),
              stats: Sequence[str] = ("mean",)) -> None:
        """Groups samples by ``by`` and reports ``stats`` of ``values`` per group."""
        n = len(self.patients)
        if not isinstance(by, Mapping) or not 1 <= len(by) <= 2:
            raise ContractError("out.table needs by= a dict of 1 or 2 columns")
        values = dict(values or {})
        bad = [s for s in stats if s not in STATS]
        if bad:
            raise ContractError(f"stats {bad} are not allowed; use {list(STATS)}")
        keys = {}
        for col, arr in {**by, **values}.items():
            arr = np.asarray(arr)
            if arr.shape[:1] != (n,):
                raise ContractError(f"column {col!r} has {arr.shape[:1]} values; "
                                    f"expected one per sample ({n})")
            keys[col] = arr
        labels = [tuple(_label(keys[c][i]) for c in by) for i in range(n)]
        groups = sorted(set(labels))
        if len(groups) > MAX_GROUPS:
            raise ContractError(f"{len(groups)} groups; at most {MAX_GROUPS} (bin the "
                                "column first)")
        rows, sizes = [], {}
        for g in groups:
            mask = np.array([lab == g for lab in labels])
            sizes[g] = len(set(self.patients[mask].tolist()))
            row: dict[str, Any] = {c: v for c, v in zip(by, g)}
            row["patients"] = sizes[g]
            for col in values:
                x = keys[col][mask].astype(float)
                x = x[np.isfinite(x)]
                for s in stats:
                    row[f"{col}_{s}"] = _stat(x, s)
            rows.append(row)
        small = [g for g in groups if sizes[g] < self.min_count]
        if len(small) == 1 and len(groups) > 1:
            rest = sorted((g for g in groups if g not in small), key=lambda g: sizes[g])
            small.append(rest[0])
        for row, g in zip(rows, groups):
            if g in small:
                for k in list(row):
                    if k not in by:
                        row[k] = None
                row["withheld"] = True
        self.results.append({"type": "table", "name": str(name)[:80],
                             "by": list(by), "rows": rows,
                             "min_patients": self.min_count})

    def value(self, name: str, x: float, n_patients: int) -> None:
        """One number, reported only when it covers ``min_count`` patients."""
        try:
            value = float(x)
        except (TypeError, ValueError):
            raise ContractError("out.value needs a number") from None
        withheld = int(n_patients) < self.min_count or not math.isfinite(value)
        self.results.append({"type": "value", "name": str(name)[:80],
                             "value": None if withheld else round(value, 6),
                             "patients": None if withheld else int(n_patients),
                             "withheld": withheld})

    def note(self, text: str) -> None:
        """A short note, cut at 300 characters."""
        self.results.append({"type": "note", "text": str(text)[:300]})

    def serialized(self) -> dict[str, Any]:
        """The results, cut to :data:`MAX_OUTPUT` characters of JSON."""
        kept, size = [], 0
        for item in self.results:
            text = json.dumps(item, default=str)
            if size + len(text) > MAX_OUTPUT:
                return {"results": kept, "truncated": True}
            kept.append(json.loads(text))
            size += len(text)
        return {"results": kept, "truncated": False}


def _label(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return int(f) if f.is_integer() else round(f, 4)
    if isinstance(value, (np.integer, int, np.bool_, bool)):
        return int(value)
    return str(value)[:40]


def _stat(x: np.ndarray, s: str) -> float | None:
    if not len(x):
        return None
    value = {"mean": np.mean, "rate": np.mean, "median": np.median,
             "std": np.std, "sum": np.sum}[s](x)
    return round(float(value), 6)


def run_analysis_job(
    spec: AnalysisSpec,
    columns: Mapping[str, np.ndarray],
    timeout: float = 300.0,
    log_path: str | None = None,
    min_count: int | None = None,
) -> dict[str, Any]:
    """Runs an analysis in its own process and returns its suppressed results.

    Raises:
        SandboxError: If the code raised or timed out.

    Examples:
        >>> results = run_analysis_job(spec, columns, timeout=120)  # doctest: +SKIP
        >>> print(format_results(results))  # doctest: +SKIP
    """
    payload = {"columns": dict(columns), "min_count": min_count}
    return run_sandboxed(spec.code, f"{__name__}:run_analysis",
                         payload, timeout=timeout, log_path=log_path,
                         label=f"analysis_{spec.name}")


def run_analysis(namespace: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """The child side of :func:`run_analysis_job`.

    Examples:
        >>> run_sandboxed(spec.code, "pyhealth.autoimprove.search.analysis:run_analysis",
        ... {"columns": columns, "min_count": 11})  # doctest: +SKIP
    """
    analyze = namespace.get(ENTRY)
    if not callable(analyze):
        raise ContractError(f"the module does not define {ENTRY}()")
    columns = payload["columns"]
    out = Out(columns["patient"], payload.get("min_count"))
    analyze(Data(columns), out)
    return out.serialized()


def format_results(results: dict[str, Any]) -> str:
    """Analysis results as compact text for a prompt.

    Examples:
        >>> format_results({"results": [{"type": "note", "text": "hi"}]})
        '- note: hi'
    """
    lines = []
    for item in results.get("results", []):
        if item["type"] == "note":
            lines.append(f"- note: {item['text']}")
        elif item["type"] == "value":
            shown = "withheld (<11 patients)" if item["withheld"] else (
                f"{item['value']} ({item['patients']} patients)")
            lines.append(f"- {item['name']}: {shown}")
        elif item["type"] == "table":
            lines.append(f"- table {item['name']} by {', '.join(item['by'])}:")
            for row in item["rows"]:
                cells = ", ".join(f"{k}={v}" for k, v in row.items()
                                  if k != "withheld")
                lines.append(f"    {cells}" + (" (withheld)" if row.get("withheld")
                                                else ""))
    if results.get("truncated"):
        lines.append("- (output cut at the size limit)")
    return "\n".join(lines)
