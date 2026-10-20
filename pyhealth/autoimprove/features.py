"""Features as data: specs compiled over PyHealth's event log, with plain-language definitions.

A :class:`FeatureSpec` describes one feature as JSON, built from a fixed set of
operators over the events PyHealth already stores for every patient
(``Patient.get_events``). Nothing a proposer writes is executed: a spec is
validated against the :class:`FeatureSpace` and compiled by this module.

Each spec carries its clinical ``rationale`` and expected ``direction``, and
gets a deterministic ``name`` and plain-language ``definition`` generated from
the spec itself, so the sentence a clinician reviews always describes exactly
what was computed.

Operators (``w`` is a window from the space's menu, in days by default):

============================  ===============================================
``count(select, w)``          events matching the selector in [t - w, t)
``exists(select, w)``         1 if any, else 0
``days_since(select)``        days since the last matching event before t
``stat(select, attr, agg, w)``  min / max / mean / last / slope of a value
``n_distinct(select, attr, w)`` distinct values of an attribute
``trend(select, short, long)``  count in the short window minus the count the
                              long window's rate predicts
``blocks_with(select, block, n)`` how many of the last n blocks of ``block``
                              days have an event
``ratio(a, b)``, ``diff(a, b)``  (a + 1) / (b + 1); a - b
``band(of, cuts)``, ``ge(of, x)``  clinical thresholds
``any(a, b)``, ``cross(a, b)``   either is nonzero; a times b
``column(column)``            an existing task feature
============================  ===============================================

A selector names an ``event_type`` (a table), optionally a ``group`` from the
task's vocabulary, and optionally ``where`` filters with allowed values.
Windows always end strictly before the prediction time t.

Every compiled spec must pass the future-blind test (:meth:`FeatureSpace.
future_blind`): each feature is computed twice for a sample of predictions,
once on the patient's full record and once with every row known at or after t
deleted. Any difference rejects the spec.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import SampleDataset
from pyhealth.datasets.sample_dataset import InMemorySampleDataset
from pyhealth.processors import TensorProcessor

WINDOWS = (30, 90, 180, 365, 730)
"""Window lengths a spec may use, in the space's unit (days by default)."""

BLOCKS = (30, 91, 182, 365)
"""Block lengths for ``blocks_with``."""

AGGS = ("min", "max", "mean", "last", "slope")
"""Aggregates for ``stat``; ``slope`` is the least-squares change per unit time."""

EVENT_OPS = ("count", "exists", "days_since", "stat", "n_distinct", "trend",
             "blocks_with")
COMBINE_OPS = ("ratio", "diff", "any", "cross")
DIRECTION_ALIASES = {"+": "+", "-": "-", "−": "-", "u": "u", "": "",
                     "ambiguous": "", "none": ""}
MAX_DEPTH = 3


@dataclass
class FeatureSpec:
    """One validated feature: its operator tree, name, definition and justification.

    Build specs with :meth:`FeatureSpace.validate`, which checks every part
    and fills in the name and definition.

    Args:
        node: The canonical operator tree.
        name: Column name, generated from the tree.
        definition: Plain-language definition, generated from the tree.
        rationale: The clinical reason for the feature.
        direction: Expected effect on the outcome: ``"+"``, ``"-"``,
            ``"u"`` (U-shaped) or ``""`` (none or ambiguous).
        citations: Sources for the rationale, if any.

    Examples:
        >>> spec = FeatureSpec({"op": "column", "column": "age"}, "age", "age",
        ...                    "risk rises with age", "+")
        >>> FeatureSpec.from_dict(spec.to_dict()) == spec
        True
    """

    node: dict[str, Any]
    name: str
    definition: str
    rationale: str
    direction: str = ""
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The spec as JSON-ready types (the operator tree plus annotations)."""
        return {**self.node, "name": self.name, "definition": self.definition,
                "rationale": self.rationale, "direction": self.direction,
                "citations": list(self.citations)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        """Rebuilds a spec saved with :meth:`to_dict`."""
        data = dict(data)
        meta = {k: data.pop(k) for k in ("name", "definition", "rationale",
                                         "direction", "citations") if k in data}
        return cls(data, meta.get("name", ""), meta.get("definition", ""),
                   meta.get("rationale", ""), meta.get("direction", ""),
                   list(meta.get("citations", [])))

    def key(self) -> str:
        """A short hash of the operator tree, for caching."""
        text = json.dumps(self.node, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class FeatureSpace:
    """What features may be built for a task, and how they are compiled.

    Args:
        dataset: The task's :class:`~pyhealth.datasets.BaseDataset` (its
            event log).
        time_key: The sample field holding the prediction time t.
        time_of: Or a function giving t when samples carry none:
            ``time_of(sample)`` or ``time_of(sample, record_id)``, where the
            record id is the sample's ``visit_id`` / ``record_id`` from the
            dataset's ``record_to_index``.
        vocabulary: Named code groups: ``{group: {"event_type", "attribute",
            "values", "noun"}}``, e.g. ``{"SABA": {"event_type": "med_orders",
            "attribute": "drug_class", "values": ["SABA"], "noun": "SABA
            orders"}}``.
        derived_events: ``f(patient, t) -> [row, ...]``: clinical definitions
            computed from the patient's events before t, returned as rows
            ``{"event_type", "timestamp", attribute: value, ...}``.
        derived_types: The event types ``derived_events`` returns, each with
            its attributes, e.g. ``{"episodes": ["severity"]}``.
        known_at: Per event type, the attribute holding the time a row's
            values became known, when that is later than its timestamp (e.g.
            ``{"admissions": "dischtime"}``). The future-blind test keeps such
            a row but blanks its values when they were known at or after t.
        columns: Named columns of the task's tensor fields, for ``column``
            specs: ``{field: [name, ...]}``.
        nouns: Plain words for event types, e.g. ``{"prescriptions":
            "prescriptions"}``.
        windows: Allowed window lengths.
        unit: The length of one window unit.
        time_word: How definitions name t, e.g. ``"the landmark"``.
        max_features: Most specs per proposal.
        min_patients: Smallest number of patients a ``where`` value must have.

    Examples:
        >>> fs = FeatureSpace(base_dataset, time_key="landmark")  # doctest: +SKIP
        >>> spec = fs.validate({"op": "count", "window": 365,
        ...                     "select": {"event_type": "ed_visits"},
        ...                     "rationale": "prior ED use", "direction": "+"})  # doctest: +SKIP
        >>> spec.definition  # doctest: +SKIP
        'the number of ed visits events in the 365 days before the prediction time'
    """

    def __init__(
        self,
        dataset: Any,
        time_key: str | None = None,
        time_of: Callable[[dict], Any] | None = None,
        vocabulary: dict[str, dict[str, Any]] | None = None,
        derived_events: Callable[[Patient, datetime], list[dict]] | None = None,
        derived_types: dict[str, list[str]] | None = None,
        known_at: dict[str, str] | None = None,
        columns: dict[str, list[str]] | None = None,
        nouns: dict[str, str] | None = None,
        windows: Sequence[int] = WINDOWS,
        unit: timedelta = timedelta(days=1),
        time_word: str = "the prediction time",
        max_features: int = 8,
        min_patients: int = 11,
    ):
        if time_key is None and time_of is None:
            raise ValueError("give time_key or time_of: the prediction time per sample")
        self.dataset = dataset
        self.time_key = time_key
        self.time_of = time_of
        self.vocabulary = dict(vocabulary or {})
        self.derived_events = derived_events
        self.derived_types = {k: list(v) for k, v in (derived_types or {}).items()}
        self.known_at = dict(known_at or {})
        self.columns = {k: list(v) for k, v in (columns or {}).items()}
        self.nouns = dict(nouns or {})
        self.windows = tuple(windows)
        self.unit = unit
        self.time_word = time_word
        self.max_features = max_features
        self.min_patients = min_patients
        self._allowed: dict[tuple[str, str], set] = {}

    @classmethod
    def from_task(cls, dataset: Any, task: Any, **kwargs: Any) -> FeatureSpace:
        """A space from a task's optional hooks and its task card.

        Reads ``time_key``, ``feature_vocabulary()``, ``derived_events`` and
        ``derived_event_types`` (see :class:`~pyhealth.tasks.BaseTask`) and
        the card's named columns.

        Examples:
            >>> fs = FeatureSpace.from_task(base_dataset, task)  # doctest: +SKIP
        """
        card = task.task_card() if hasattr(task, "task_card") else None
        kwargs.setdefault("time_key", getattr(task, "time_key", None))
        if hasattr(task, "feature_vocabulary"):
            kwargs.setdefault("vocabulary", task.feature_vocabulary())
        if getattr(task, "derived_event_types", None):
            kwargs.setdefault("derived_events", task.derived_events)
            kwargs.setdefault("derived_types", task.derived_event_types)
        if card is not None:
            kwargs.setdefault("columns", card.columns)
        return cls(dataset, **kwargs)

    # ------------------------------------------------------------ validation
    def validate(self, spec: dict[str, Any] | FeatureSpec) -> FeatureSpec:
        """Checks a spec and returns it with its name and definition.

        Raises:
            ValueError: With the reason, e.g. an unknown operator or group, a
                window outside the menu, or a missing rationale.
        """
        if isinstance(spec, FeatureSpec):
            spec = spec.to_dict()
        spec = dict(spec)
        rationale = str(spec.pop("rationale", "") or "").strip()
        if not rationale:
            raise ValueError("every feature needs a clinical rationale")
        direction = str(spec.pop("direction", "") or "").strip().lower()
        if direction not in DIRECTION_ALIASES:
            raise ValueError(
                f"direction must be one of +, -, u or ambiguous, not {direction!r}"
            )
        citations = [str(c) for c in spec.pop("citations", []) or []]
        for key in ("name", "definition"):
            spec.pop(key, None)
        node = self._node(spec, depth=1)
        return FeatureSpec(node, _name(node), self._define(node), rationale,
                           DIRECTION_ALIASES[direction], citations)

    def validate_many(self, specs: Sequence[dict[str, Any]]) -> list[FeatureSpec]:
        """Validates a proposal's specs: at most ``max_features``, no duplicates.

        Raises:
            ValueError: With the reason for the first invalid spec.
        """
        if len(specs) > self.max_features:
            raise ValueError(f"at most {self.max_features} features per proposal, "
                             f"not {len(specs)}")
        out, names = [], set()
        for i, spec in enumerate(specs):
            try:
                valid = self.validate(spec)
            except ValueError as e:
                raise ValueError(f"feature {i + 1}: {e}") from None
            if valid.name in names:
                raise ValueError(f"feature {i + 1} repeats {valid.name}")
            names.add(valid.name)
            out.append(valid)
        return out

    def _node(self, spec: dict[str, Any], depth: int) -> dict[str, Any]:
        if depth > MAX_DEPTH:
            raise ValueError(f"specs may nest at most {MAX_DEPTH} operators deep")
        op = spec.get("op")
        if op in ("count", "exists", "days_since"):
            node = {"op": op, "select": self._select(spec.get("select"))}
            if op != "days_since":
                node["window"] = self._window(spec.get("window"))
            return node
        if op in ("stat", "n_distinct"):
            select = self._select(spec.get("select"))
            attr = self._attribute(select["event_type"], spec.get("attr"))
            node = {"op": op, "select": select, "attr": attr,
                    "window": self._window(spec.get("window"))}
            if op == "stat":
                if spec.get("agg") not in AGGS:
                    raise ValueError(f"agg must be one of {list(AGGS)}")
                node["agg"] = spec["agg"]
            return node
        if op == "trend":
            short, long = self._window(spec.get("short")), self._window(spec.get("long"))
            if short >= long:
                raise ValueError("trend needs short < long")
            return {"op": op, "select": self._select(spec.get("select")),
                    "short": short, "long": long}
        if op == "blocks_with":
            block, n = spec.get("block"), spec.get("n")
            if block not in BLOCKS:
                raise ValueError(f"block must be one of {list(BLOCKS)}")
            if not isinstance(n, int) or not 1 <= n <= 8:
                raise ValueError("n must be an integer from 1 to 8")
            return {"op": op, "select": self._select(spec.get("select")),
                    "block": block, "n": n}
        if op in COMBINE_OPS:
            return {"op": op, "a": self._child(spec, "a", depth),
                    "b": self._child(spec, "b", depth)}
        if op == "ge":
            x = spec.get("x")
            if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
                raise ValueError("ge needs a finite number x")
            return {"op": op, "of": self._child(spec, "of", depth), "x": x}
        if op == "band":
            cuts = spec.get("cuts")
            if (not isinstance(cuts, list) or not 1 <= len(cuts) <= 5
                    or not all(isinstance(c, (int, float)) and math.isfinite(c)
                               for c in cuts) or cuts != sorted(set(cuts))):
                raise ValueError("cuts must be 1 to 5 increasing finite numbers")
            return {"op": op, "of": self._child(spec, "of", depth), "cuts": cuts}
        if op == "column":
            name = spec.get("column")
            known = {c for cols in self.columns.values() for c in cols}
            if name not in known:
                raise ValueError(f"unknown column {name!r}; known: {sorted(known)[:20]}")
            return {"op": op, "column": name}
        raise ValueError(
            f"unknown op {op!r}; use one of "
            f"{list(EVENT_OPS + COMBINE_OPS + ('ge', 'band', 'column'))}"
        )

    def _child(self, spec, key, depth):
        child = spec.get(key)
        if not isinstance(child, dict):  # a proposer's error: reported, not raised
            raise ValueError(f"{spec.get('op')} needs a spec in {key!r}")  # noqa: TRY004
        return self._node(child, depth + 1)

    def _window(self, value) -> int:
        if value not in self.windows:
            raise ValueError(f"window must be one of {list(self.windows)}, not {value!r}")
        return int(value)

    def _select(self, select) -> dict[str, Any]:
        if not isinstance(select, dict) or not select.get("event_type"):
            raise ValueError("a selector needs an event_type")
        event_type = str(select["event_type"])
        if event_type not in self._event_types():
            raise ValueError(
                f"unknown event_type {event_type!r}; known: {sorted(self._event_types())}"
            )
        out: dict[str, Any] = {"event_type": event_type}
        group = select.get("group")
        if group is not None:
            entry = self.vocabulary.get(group)
            if entry is None:
                raise ValueError(f"unknown group {group!r}; known: {sorted(self.vocabulary)}")
            if entry["event_type"] != event_type:
                raise ValueError(f"group {group!r} belongs to {entry['event_type']!r}")
            out["group"] = group
        where = select.get("where") or {}
        if where:
            clean = {}
            for attr, values in sorted(where.items()):
                attr = self._attribute(event_type, attr)
                values = values if isinstance(values, list) else [values]
                allowed = self._allowed_values(event_type, attr)
                bad = [v for v in values if str(v) not in allowed]
                if bad:
                    raise ValueError(
                        f"{attr} values {bad} are not allowed: a value must occur "
                        f"for at least {self.min_patients} patients"
                    )
                clean[attr] = sorted(str(v) for v in values)
            out["where"] = clean
        return out

    def _attribute(self, event_type: str, attr) -> str:
        if not attr:
            raise ValueError("this operator needs an attr")
        if attr not in self._attributes(event_type):
            raise ValueError(f"{event_type} has no attribute {attr!r}; known: "
                             f"{sorted(self._attributes(event_type))}")
        return str(attr)

    def _event_types(self) -> set[str]:
        types = {str(t).lower() for t in getattr(self.dataset, "tables", [])}
        return types | set(self.derived_types)

    def _attributes(self, event_type: str) -> set[str]:
        if event_type in self.derived_types:
            return set(self.derived_types[event_type])
        prefix = f"{event_type}/"
        names = self.dataset.global_event_df.collect_schema().names()
        return {c[len(prefix):] for c in names if c.startswith(prefix)}

    def _allowed_values(self, event_type: str, attr: str) -> set[str]:
        key = (event_type, attr)
        if event_type in self.derived_types:
            raise ValueError(f"where filters are not available on derived "
                             f"{event_type!r} rows; use a vocabulary group")
        if key not in self._allowed:
            column = f"{event_type}/{attr}"
            counts = (
                self.dataset.global_event_df
                .filter(pl.col("event_type") == event_type)
                .group_by(column)
                .agg(pl.col("patient_id").n_unique().alias("n"))
                .collect()
            )
            self._allowed[key] = {
                str(v) for v, n in counts.iter_rows() if v is not None
                and n >= self.min_patients
            }
        return self._allowed[key]

    # ------------------------------------------------------------ proposing
    def describe(self, max_values: int = 12) -> str:
        """The space in plain text, for an LLM prompt: operators and data.

        Lists event types with their attributes, vocabulary groups and named
        columns. Attribute values are not listed (they are data); a ``where``
        filter is checked when the spec is validated.

        Examples:
            >>> print(fs.describe())  # doctest: +SKIP
        """
        unit = "days" if self.unit == timedelta(days=1) else f"units of {self.unit}"
        lines = _MENU.format(windows=list(self.windows), unit=unit, t=self.time_word,
                             aggs=list(AGGS), blocks=list(BLOCKS)).splitlines()
        for et in sorted(self._event_types()):
            attrs = sorted(self._attributes(et))
            shown = ", ".join(attrs[:max_values]) + (" ..." if len(attrs) > max_values else "")
            lines.append(f"- {et}: {shown or 'no attributes'}")
        if self.vocabulary:
            lines.append("Groups:")
            for name, entry in sorted(self.vocabulary.items()):
                lines.append(f"- {name}: {entry.get('noun') or name} "
                             f"({entry['event_type']})")
        if self.columns:
            cols = [c for v in self.columns.values() for c in v]
            lines.append("Existing task features (for column): "
                         + ", ".join(cols[:40]) + (" ..." if len(cols) > 40 else ""))
        return "\n".join(lines)

    def sample(self, rng: Any) -> dict[str, Any]:
        """A random valid spec, for random search; it has no clinical rationale.

        Examples:
            >>> import random
            >>> fs.sample(random.Random(0))["op"]  # doctest: +SKIP
            'count'
        """
        choices = [("type", et) for et in sorted(self._event_types())]
        choices += [("group", g) for g in sorted(self.vocabulary)]
        kind, value = rng.choice(choices)
        select = ({"event_type": value} if kind == "type" else
                  {"event_type": self.vocabulary[value]["event_type"], "group": value})
        op = rng.choice(["count", "exists", "days_since", "trend", "blocks_with"])
        spec: dict[str, Any] = {"op": op, "select": select}
        if op in ("count", "exists"):
            spec["window"] = rng.choice(self.windows)
        elif op == "trend":
            short, long = sorted(rng.sample(list(self.windows), 2))
            spec.update(short=short, long=long)
        elif op == "blocks_with":
            spec.update(block=rng.choice(BLOCKS), n=rng.randint(2, 4))
        spec.update(rationale="none: proposed by random search", direction="")
        return spec

    # ------------------------------------------------------------ definition
    def _define(self, node: dict[str, Any]) -> str:
        op, t = node["op"], self.time_word
        if op in EVENT_OPS:
            sel = self._noun(node["select"])
            if op == "count":
                return f"the number of {sel} in the {node['window']} days before {t}"
            if op == "exists":
                return (f"1 if there was any {_singular(sel)} in the {node['window']} "
                        f"days before {t}, else 0")
            if op == "days_since":
                return f"days from the last {_singular(sel)} to {t} (missing if none)"
            if op == "stat":
                agg = {"min": "lowest", "max": "highest", "mean": "mean",
                       "last": "most recent", "slope": "change per day in"}[node["agg"]]
                return (f"the {agg} {node['attr']} of {sel} in the {node['window']} "
                        f"days before {t} (missing if none)")
            if op == "n_distinct":
                return (f"the number of distinct {node['attr']} values among {sel} in "
                        f"the {node['window']} days before {t}")
            if op == "trend":
                return (f"the number of {sel} in the {node['short']} days before {t}, "
                        f"minus the number expected at the rate of the "
                        f"{node['long']} days before {t}")
            return (f"how many of the {node['n']} periods of {node['block']} days "
                    f"before {t} had any {_singular(sel)} (0 to {node['n']})")
        if op == "ge":
            of = node["of"]
            if of["op"] == "count":
                return (f"1 if there were {node['x']:g} or more "
                        f"{self._noun(of['select'])} in the {of['window']} days "
                        f"before {t}, else 0")
            return f"1 if {self._define(of)} is {node['x']:g} or more, else 0"
        if op == "band":
            cuts = node["cuts"]
            parts = [f"0 below {cuts[0]:g}"] + [
                f"{i} from {cuts[i - 1]:g} to below {cuts[i]:g}"
                for i in range(1, len(cuts))
            ] + [f"{len(cuts)} from {cuts[-1]:g}"]
            return f"{self._define(node['of'])}, in bands: " + ", ".join(parts)
        if op == "ratio":
            return (f"({self._define(node['a'])} + 1) divided by "
                    f"({self._define(node['b'])} + 1)")
        if op == "diff":
            return f"{self._define(node['a'])} minus {self._define(node['b'])}"
        if op == "any":
            return (f"1 if either {self._define(node['a'])} or "
                    f"{self._define(node['b'])} is above 0, else 0")
        if op == "cross":
            return f"{self._define(node['a'])} times {self._define(node['b'])}"
        return f"the task's feature {node['column']}"

    def _noun(self, select: dict[str, Any]) -> str:
        et = select["event_type"]
        if "group" in select:
            entry = self.vocabulary[select["group"]]
            text = entry.get("noun") or f"{select['group']} {self.nouns.get(et, et)}"
        else:
            text = self.nouns.get(et, f"{et.replace('_', ' ')} events")
        for attr, values in select.get("where", {}).items():
            text += f" with {attr} {' or '.join(values)}"
        return text

    # ------------------------------------------------------------- compiling
    def compute(
        self,
        specs: Sequence[FeatureSpec],
        samples: SampleDataset,
        indices: Sequence[int] | None = None,
        truncate: bool = False,
    ) -> np.ndarray:
        """Raw values of each spec for each sample: ``(n_samples, n_specs)``.

        Missing values (e.g. ``stat`` with no event) are ``NaN``; they are
        imputed later, per fold. With ``truncate``, every patient's record
        is cut to the rows known before each prediction time first (the
        future-blind test's second pass).

        Examples:
            >>> values = fs.compute(specs, train_ds)  # doctest: +SKIP
        """
        indices = list(range(len(samples))) if indices is None else list(indices)
        out = np.full((len(indices), len(specs)), np.nan)
        record_of = {i: rid for rid, idx in getattr(samples, "record_to_index", {}).items()
                     for i in idx}
        by_patient: dict[str, list[int]] = {}
        sample_rows, times = {}, {}
        for row, i in enumerate(indices):
            sample = samples[i]
            sample_rows[row] = sample
            times[row] = self._time(sample, record_of.get(i))
            by_patient.setdefault(str(sample["patient_id"]), []).append(row)
        patients = self._patients(list(by_patient), specs)
        for pid, rows in by_patient.items():
            for row in rows:
                sample = sample_rows[row]
                view = _EventView(patients[pid], times[row], self, truncate)
                for j, spec in enumerate(specs):
                    out[row, j] = self._evaluate(spec.node, view, sample)
        return out

    def _patients(self, pids: list[str], specs) -> dict[str, Patient]:
        """Every patient's needed events, fetched in one query."""
        types = set()
        for spec in specs:
            types |= _event_types_of(spec.node)
        query = self.dataset.global_event_df.filter(pl.col("patient_id").is_in(pids))
        if self.derived_events is None:  # a hook may read any table
            query = query.filter(pl.col("event_type").is_in(sorted(types)))
        df = query.collect()
        parts = df.partition_by("patient_id", as_dict=True) if df.height else {}
        empty = df.clear()
        return {pid: Patient(pid, parts.get((pid,), empty)) for pid in pids}

    def _time(self, sample: dict, record_id: Any = None) -> datetime:
        if self.time_key:
            return _as_datetime(sample[self.time_key])
        if len(inspect.signature(self.time_of).parameters) > 1:
            return _as_datetime(self.time_of(sample, record_id))
        return _as_datetime(self.time_of(sample))

    def _evaluate(self, node, view, sample) -> float:
        op = node["op"]
        if op in EVENT_OPS:
            return view.evaluate(node)
        if op == "ge":
            v = self._evaluate(node["of"], view, sample)
            return v if math.isnan(v) else float(v >= node["x"])
        if op == "band":
            v = self._evaluate(node["of"], view, sample)
            return v if math.isnan(v) else float(sum(v >= c for c in node["cuts"]))
        if op == "column":
            return self._column(sample, node["column"])
        a = self._evaluate(node["a"], view, sample)
        b = self._evaluate(node["b"], view, sample)
        if op == "ratio":
            return (a + 1) / (b + 1) if b != -1 else math.nan
        if op == "diff":
            return a - b
        if op == "any":
            if math.isnan(a) and math.isnan(b):
                return math.nan
            return float((not math.isnan(a) and a > 0) or (not math.isnan(b) and b > 0))
        return a * b

    def _column(self, sample, name) -> float:
        for key, cols in self.columns.items():
            if name in cols and key in sample:
                value = torch.as_tensor(sample[key]).reshape(-1)
                return float(value[cols.index(name)])
        return math.nan

    # ------------------------------------------------------------ the tests
    def future_blind(
        self,
        specs: Sequence[FeatureSpec],
        samples: SampleDataset,
        n_samples: int = 200,
        seed: int = 0,
    ) -> dict[str, str | None]:
        """Checks that no spec uses information known at or after t.

        For up to ``n_samples`` samples, each spec is computed on the full
        record and on the record cut at t: rows dated at or after t are
        removed, and the values of rows known only at or after t (by the
        ``known_at`` attribute, where declared) are blanked. Derived events
        are rebuilt from the cut record. ``column`` specs read the
        task's own feature and are not re-derived.

        Returns:
            Spec name to ``None`` (passed) or the reason it failed.

        Examples:
            >>> fs.future_blind(specs, train_ds)  # doctest: +SKIP
            {'count_365d__ed_visits': None}
        """
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(samples), min(n_samples, len(samples)),
                                replace=False).tolist())
        full = self.compute(specs, samples, idx)
        cut = self.compute(specs, samples, idx, truncate=True)
        out = {}
        for j, spec in enumerate(specs):
            a, b = full[:, j], cut[:, j]
            same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=False)
            bad = int((~same).sum())
            out[spec.name] = None if bad == 0 else (
                f"uses information known at or after {self.time_word}: {bad} of "
                f"{len(idx)} checked samples change when later rows are removed"
            )
        return out

    def check(
        self, specs: Sequence[dict[str, Any] | FeatureSpec], samples: SampleDataset,
        n_samples: int = 200,
    ) -> list[dict[str, Any]]:
        """Validates specs and reports, for each, what a reviewer needs.

        Returns:
            One dict per spec: ``name``, ``definition``, ``rationale``,
            ``direction``, ``coverage`` (share of samples with a nonzero
            value, rounded to a whole percent) and ``future_blind`` (``None``
            or the reason it failed).

        Examples:
            >>> for row in fs.check(specs, train_ds):  # doctest: +SKIP
            ...     print(row["definition"], row["coverage"], row["future_blind"])
        """
        valid = self.validate_many([s.to_dict() if isinstance(s, FeatureSpec) else s
                                    for s in specs])
        blind = self.future_blind(valid, samples, n_samples)
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(len(samples), min(n_samples, len(samples)),
                                replace=False).tolist())
        values = self.compute(valid, samples, idx)
        out = []
        for j, spec in enumerate(valid):
            col = values[:, j]
            nonzero = np.nan_to_num(col) != 0
            out.append({
                "name": spec.name, "definition": spec.definition,
                "rationale": spec.rationale, "direction": spec.direction,
                "coverage": f"{round(100 * nonzero.mean())}%",
                "future_blind": blind[spec.name],
            })
        return out

    def augment(
        self,
        samples: SampleDataset,
        specs: Sequence[FeatureSpec],
        stats: dict[str, list[float]] | None = None,
        values: np.ndarray | None = None,
    ) -> EngineeredDataset:
        """Adds the specs to the samples as one ``engineered`` tensor field.

        Values are imputed (median) and standardised with ``stats``; by
        default the statistics are fitted on these samples, which is right
        for a training split. Pass the training split's ``stats`` when
        augmenting validation and test splits.

        Examples:
            >>> train_fe = fs.augment(train_ds, specs)  # doctest: +SKIP
            >>> test_fe = fs.augment(test_ds, specs, stats=train_fe.stats)  # doctest: +SKIP
        """
        raw = self.compute(specs, samples) if values is None else np.asarray(values)
        return EngineeredDataset(samples, raw, [s.name for s in specs], stats)


class EngineeredDataset(InMemorySampleDataset):
    """A sample dataset with engineered feature columns added as one tensor field.

    The raw values are kept, so imputation and scaling can be refitted on
    any subset of patients (:meth:`fit_stats`, :meth:`with_stats`). Cross-
    validation fits them on each fold's training part, so a held-out fold
    never informs its own medians and scales.

    Args:
        base: The task's samples.
        raw: Raw engineered values, ``(len(base), n_features)``; ``NaN`` is
            missing.
        names: Column names.
        stats: ``{"median", "mean", "sd"}`` lists; fitted on all rows if
            ``None``.
        base_samples: ``[base[i] for i in range(len(base))]``, if already
            loaded, to avoid reading the base dataset again.

    Examples:
        >>> fe = EngineeredDataset(train_ds, raw, ["count_365d__ed_visits"])  # doctest: +SKIP
        >>> fe[0]["engineered"]  # doctest: +SKIP
        tensor([0.42])
    """

    def __init__(self, base, raw, names, stats=None, base_samples=None):
        self.dataset_name = getattr(base, "dataset_name", "")
        self.task_name = getattr(base, "task_name", "")
        self.input_schema = {**base.input_schema, "engineered": "tensor"}
        self.output_schema = dict(base.output_schema)
        self.input_processors = {**base.input_processors,
                                 "engineered": TensorProcessor()}
        self.output_processors = dict(base.output_processors)
        self.patient_to_index = {k: list(v) for k, v in base.patient_to_index.items()}
        self.record_to_index = {k: list(v) for k, v in base.record_to_index.items()}
        self._base = (list(base_samples) if base_samples is not None
                      else [base[i] for i in range(len(base))])
        self.raw = np.asarray(raw, dtype=float).reshape(len(self._base), -1)
        self.names = list(names)
        self.stats = stats if stats is not None else _fit_stats(self.raw)
        self._data = self._materialize()
        self._shuffle = False
        self._shuffle_seed = 0
        self._epoch = 1

    def _materialize(self) -> list[dict[str, Any]]:
        values = _apply_stats(self.raw, self.stats)
        return [{**sample, "engineered": torch.tensor(row, dtype=torch.float32)}
                for sample, row in zip(self._base, values)]

    def fit_stats(self, indices: Sequence[int]) -> dict[str, list[float]]:
        """Medians, means and SDs of the raw values over some rows."""
        return _fit_stats(self.raw[list(indices)])

    def with_stats(self, stats: dict[str, list[float]]) -> EngineeredDataset:
        """The same samples processed with other statistics."""
        out = copy.copy(self)
        out.stats = stats
        out._data = out._materialize()
        return out

    def subset(self, indices):
        """The given rows, keeping their raw values and the statistics."""
        indices = (list(range(*indices.indices(len(self))))
                   if isinstance(indices, slice) else list(indices))
        out = super().subset(indices)
        out._base = [self._base[i] for i in indices]
        out.raw = self.raw[indices]
        return out


_MENU = """\
A feature is a JSON spec. Operators (w is a window in {windows} {unit}, ending \
strictly before {t}):
- count / exists: {{"op": "count", "window": w, "select": SEL}}
- days_since: {{"op": "days_since", "select": SEL}}
- stat: {{"op": "stat", "select": SEL, "attr": a, "agg": one of {aggs}, "window": w}}
- n_distinct: {{"op": "n_distinct", "select": SEL, "attr": a, "window": w}}
- trend: {{"op": "trend", "select": SEL, "short": w1, "long": w2}}
- blocks_with: {{"op": "blocks_with", "select": SEL, "block": one of {blocks}, "n": 1-8}}
- ratio / diff / any / cross: {{"op": ..., "a": SPEC, "b": SPEC}}
- ge: {{"op": "ge", "x": number, "of": SPEC}}; band: {{"op": "band", "cuts": \
[increasing numbers], "of": SPEC}}
- column: {{"op": "column", "column": name}}
SEL is {{"event_type": t, optional "group": g, optional "where": {{attr: [values]}}}}. \
Every spec also needs "rationale" (the clinical reason) and "direction" (+, -, u \
or ambiguous).
Event types and their attributes:"""


# --------------------------------------------------------------- event view
class _EventView:
    """One patient's events as seen at time t, with windows ending before t."""

    def __init__(self, patient: Patient, t: datetime, space: FeatureSpace,
                 truncate: bool):
        self.t = t
        self.space = space
        if truncate:
            patient = _truncated(patient, t, space.known_at)
        self.patient = patient
        self._derived = None
        self._cache: dict[str, pl.DataFrame] = {}

    def rows(self, select: dict[str, Any]) -> pl.DataFrame:
        et = select["event_type"]
        if et not in self._cache:
            df = self._derived_rows(et)
            if df is None:
                df = self.patient.get_events(event_type=et, return_df=True)
            self._cache[et] = df
        df = self._cache[et]
        if df.height == 0:
            return df
        df = df.filter(pl.col("timestamp") < self.t)
        if "group" in select:
            entry = self.space.vocabulary[select["group"]]
            col = _col(df, et, entry["attribute"])
            df = df.filter(pl.col(col).cast(pl.Utf8).is_in([str(v) for v in entry["values"]]))
        for attr, values in select.get("where", {}).items():
            df = df.filter(pl.col(_col(df, et, attr)).cast(pl.Utf8).is_in(values))
        return df

    def _derived_rows(self, et) -> pl.DataFrame | None:
        if self.space.derived_events is None or et not in self.space.derived_types:
            return None
        if self._derived is None:
            rows = self.space.derived_events(self.patient, self.t) or []
            self._derived = {}
            for r in rows:
                self._derived.setdefault(r["event_type"], []).append(r)
        rows = self._derived.get(et)
        if rows is None:
            return None
        data = [{"timestamp": _as_datetime(r["timestamp"]),
                 **{f"{et}/{k}": v for k, v in r.items()
                    if k not in ("event_type", "timestamp")}} for r in rows]
        return pl.DataFrame(data)

    def evaluate(self, node: dict[str, Any]) -> float:
        op, unit = node["op"], self.space.unit
        df = self.rows(node["select"])
        if op == "days_since":
            if df.height == 0:
                return math.nan
            last = df["timestamp"].max()
            return (self.t - last) / unit
        if op == "trend":
            short = _in_window(df, self.t, node["short"] * unit).height
            long = _in_window(df, self.t, node["long"] * unit).height
            return short - long * node["short"] / node["long"]
        if op == "blocks_with":
            block = node["block"] * unit
            return float(sum(
                _in_window(df, self.t - k * block, block).height > 0
                for k in range(node["n"])
            ))
        df = _in_window(df, self.t, node["window"] * unit)
        if op == "count":
            return float(df.height)
        if op == "exists":
            return float(df.height > 0)
        col = _col(df, node["select"]["event_type"], node["attr"]) if df.height else None
        if op == "n_distinct":
            return float(df[col].drop_nulls().n_unique()) if df.height else 0.0
        if df.height == 0:
            return math.nan
        values = df.select(pl.col("timestamp"), pl.col(col).cast(pl.Float64, strict=False)
                           .alias("v")).drop_nulls()
        if values.height == 0:
            return math.nan
        agg = node["agg"]
        if agg == "min":
            return float(values["v"].min())
        if agg == "max":
            return float(values["v"].max())
        if agg == "mean":
            return float(values["v"].mean())
        if agg == "last":
            return float(values.sort("timestamp")["v"][-1])
        if values.height < 2:
            return math.nan
        days = np.array([(ts - self.t) / unit for ts in values["timestamp"]])
        if np.ptp(days) == 0:
            return math.nan
        return float(np.polyfit(days, values["v"].to_numpy(), 1)[0])


# ------------------------------------------------------------------ helpers
def _event_types_of(node: dict[str, Any]) -> set[str]:
    out = set()
    if "select" in node:
        out.add(node["select"]["event_type"])
    for key in ("a", "b", "of"):
        if isinstance(node.get(key), dict):
            out |= _event_types_of(node[key])
    return out


def _truncated(patient: Patient, t: datetime, known_at: dict[str, str]) -> Patient:
    """The record as known at t: later rows removed, later values blanked."""
    df = patient.data_source.filter(pl.col("timestamp") < t)
    for et, attr in known_at.items():
        col = f"{et}/{attr}"
        if col not in df.columns:
            continue
        known = pl.col(col).cast(pl.Utf8).str.to_datetime(strict=False)
        late = (pl.col("event_type") == et) & known.is_not_null() & (known >= t)
        values = [c for c in df.columns if c.startswith(f"{et}/")]
        df = df.with_columns([
            pl.when(late).then(None).otherwise(pl.col(c)).alias(c) for c in values
        ])
    return Patient(patient.patient_id, df)


def _in_window(df: pl.DataFrame, end: datetime, length: timedelta) -> pl.DataFrame:
    if df.height == 0:
        return df
    return df.filter((pl.col("timestamp") >= end - length) & (pl.col("timestamp") < end))


def _col(df: pl.DataFrame, et: str, attr: str) -> str:
    name = f"{et}/{attr}"
    return name if name in df.columns else attr


def _fit_stats(raw: np.ndarray) -> dict[str, list[float]]:
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        width = raw.shape[1] if raw.ndim == 2 else 0
        return {"median": [0.0] * width, "mean": [0.0] * width, "sd": [1.0] * width}
    with np.errstate(all="ignore"):
        median = np.nanmedian(raw, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    filled = np.where(np.isnan(raw), median, raw)
    mean, sd = filled.mean(axis=0), filled.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return {"median": median.tolist(), "mean": mean.tolist(), "sd": sd.tolist()}


def _apply_stats(raw: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    median, mean, sd = (np.asarray(stats[k], dtype=float) for k in ("median", "mean", "sd"))
    filled = np.where(np.isnan(raw), median, raw)
    return (filled - mean) / sd


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").astype(datetime)
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"cannot read a prediction time from {value!r}")


def _singular(noun: str) -> str:
    head, _, tail = noun.partition(" with ")
    if head.endswith("ies"):
        head = head[:-3] + "y"
    elif head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return head + (f" with {tail}" if tail else "")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def _select_name(select: dict[str, Any]) -> str:
    parts = [select["event_type"]]
    if "group" in select:
        parts.append(select["group"])
    for attr, values in select.get("where", {}).items():
        parts.append(f"{attr}-{'-'.join(values)}")
    return _slug("_".join(parts))


def _name(node: dict[str, Any]) -> str:
    op = node["op"]
    if op in ("count", "exists"):
        return f"{op}_{node['window']}d__{_select_name(node['select'])}"
    if op == "days_since":
        return f"days_since__{_select_name(node['select'])}"
    if op == "stat":
        return (f"{node['agg']}_{_slug(node['attr'])}_{node['window']}d__"
                f"{_select_name(node['select'])}")
    if op == "n_distinct":
        return (f"ndistinct_{_slug(node['attr'])}_{node['window']}d__"
                f"{_select_name(node['select'])}")
    if op == "trend":
        return f"trend_{node['short']}v{node['long']}d__{_select_name(node['select'])}"
    if op == "blocks_with":
        return f"blocks{node['n']}x{node['block']}d__{_select_name(node['select'])}"
    if op == "ge":
        x = _slug(format(node["x"], "g"))
        return f"ge{x}__{_name(node['of'])}"
    if op == "band":
        cuts = "_".join(_slug(format(c, "g")) for c in node["cuts"])
        return f"band{cuts}__{_name(node['of'])}"
    if op == "column":
        return _slug(node["column"])
    return f"{op}__{_name(node['a'])}__{_name(node['b'])}"
