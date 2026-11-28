"""Deterministic fingerprinting of tasks for cache keys.

``BaseDataset.set_task`` names each task cache directory after a fingerprint
of the task, and each processed-samples directory below it after a fingerprint
of the processors, storage layout and split. Both are computed here.

Design goals
------------
1. **Deterministic** across processes and machines: never relies on
   ``PYTHONHASHSEED``-dependent iteration order or on memory addresses leaking
   through ``repr()``.
2. **Lossless**: no truncating ``str()`` of large objects, and no type reduced
   to part of its state (see *Known limits*).
3. **Fail loud, not silent**: an argument that cannot be fingerprinted
   deterministically raises with an actionable message instead of producing a
   key that is either unstable (permanent cache misses) or colliding (silently
   stale samples).
4. **Complete**: covers effective instance state (after ``__init__``
   normalisation), class-level configuration, schemas and an explicit task
   ``version``. Raw ``__init__`` arguments are deliberately left out: some
   tasks normalise them so that equivalent spellings share a cache (e.g.
   ``SeizureDetectionCHBMIT`` stores ``subjects`` as a sorted tuple of
   canonical IDs), and keying on the raw values would undo that.
5. **Legible**: emits a human-readable spec that is written next to the cache,
   so users can see *why* a cache directory exists.

Known limits
------------
The fingerprint describes configuration, not code or data on disk. It does not
see the body of ``__call__`` (bump ``version``), module-level globals read by a
function argument, the contents of a file behind a path, or the attributes of a
class passed as an argument (classes are named by module and qualified name).
Dict order is ignored.

``pyhealth.benchmarks`` has its own canonical JSON for benchmark params. Those
params must be decoded back into Python objects, so it accepts only JSON values
and importable classes and functions; task state also holds arrays, tables and
closures, which only need a digest.

Public API
----------
``task_spec(task)``          -> canonical JSON-safe dict describing the task
``task_fingerprint(task)``   -> stable hex digest of that spec
``task_cache_name(task)``    -> path-safe ``{slug}_{digest}`` directory name
``write_task_metadata(...)`` -> writes ``task_meta.json`` sidecar
"""

from __future__ import annotations

import ast
import collections
import copy
import dataclasses
import datetime as _dt
import decimal
import enum
import functools
import hashlib
import inspect
import json
import linecache
import logging
import os
import re
import sys
import textwrap
import types
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

logger = logging.getLogger(__name__)

# Bump when the fingerprint format itself changes. Every cache key changes,
# so existing caches become unreachable (they are not deleted).
FINGERPRINT_VERSION = 5

# Opt-out for environments with exotic task arguments. Strict mode is the
# default because a non-strict fallback reintroduces silent collisions.
_STRICT = os.environ.get("PYHEALTH_FINGERPRINT_STRICT", "1") not in ("0", "false", "False")

# Include a structural hash of ``__call__``/``pre_filter`` source. Off by
# default: editing a comment should not invalidate a 40-minute cache build.
_HASH_SOURCE = os.environ.get("PYHEALTH_FINGERPRINT_SOURCE", "0") in ("1", "true", "True")

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
_MAX_DEPTH = 64

# Types from these packages are fully described by the branch that handles
# their base type. Any other subclass (namedtuple, defaultdict, a user's
# ``class Codes(dict)``) is also tagged with its type and instance state.
_KNOWN_VALUE_MODULES = frozenset(
    {
        "builtins",
        "datetime",
        "decimal",
        "numpy",
        "pandas",
        "pathlib",
        "polars",
        "torch",
        "uuid",
    }
)


class UnfingerprintableError(TypeError):
    """Raised when a task attribute has no deterministic representation.

    Example:
        >>> from pyhealth.tasks.fingerprint import UnfingerprintableError
        >>> isinstance(UnfingerprintableError("opaque"), TypeError)
        True
    """


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hint(obj: Any, name: str) -> str:
    return (
        f"Cannot deterministically fingerprint attribute {name!r} of type "
        f"{type(obj).__module__}.{type(obj).__qualname__}. Its repr() is not "
        f"stable across processes, so it would either break caching or, worse, "
        f"silently reuse another configuration's cache.\n"
        f"Fix it in one of three ways:\n"
        f"  1. give the class a __pyhealth_fingerprint__() method returning a "
        f"JSON-safe description of its configuration;\n"
        f"  2. list {name!r} in the task's `fingerprint_exclude` if it does not "
        f"affect the generated samples;\n"
        f"  3. store the plain configuration (str/int/tuple) on the task instead "
        f"of the constructed object."
    )


def _canon(
    obj: Any, name: str = "<root>", depth: int = 0, seen: dict[int, int] | None = None
) -> Any:
    """Return a JSON-safe, type-tagged, order-stable representation of ``obj``.

    ``seen`` maps the ``id`` of each enclosing object to its depth, which is how
    cycles are recognised.
    """
    if depth > _MAX_DEPTH:
        raise UnfingerprintableError(f"{name}: nesting deeper than {_MAX_DEPTH}")

    if obj is None:
        return None
    seen = seen if seen is not None else {}

    # Before the scalars: IntEnum and str-mixin members are also int / str.
    if isinstance(obj, enum.Enum):
        value = _canon(obj.value, f"{name}.value", depth + 1, seen)
        return ["enum", _qualname(type(obj)), value]

    # Before the base types, so that a container or array subclass can use it.
    hook = getattr(type(obj), "__pyhealth_fingerprint__", None)
    if callable(hook):
        described = _canon(hook(obj), f"{name}.<hook>", depth + 1, seen)
        return ["hook", _qualname(type(obj)), described]

    # Before the scalars: np.float64 and np.complex128 are also float / complex.
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(obj, numpy.generic):
        return _canon_numpy(obj, name, depth, seen)

    scalar = _canon_scalar(obj)
    if scalar is not None:
        return _tag_subclass(obj, scalar, name, depth, seen)

    if id(obj) in seen:
        raise UnfingerprintableError(f"{name}: circular reference")
    seen = seen | {id(obj): depth}

    value = _canon_scientific(obj, name, depth, seen)
    if value is None:
        value = _canon_container(obj, name, depth, seen)
    if value is not None:
        return _tag_subclass(obj, value, name, depth, seen)

    # -- callables, types and modules --------------------------------------
    if isinstance(obj, functools.partial):
        return [
            "partial",
            _canon(obj.func, f"{name}.func", depth + 1, seen),
            _canon(obj.args, f"{name}.args", depth + 1, seen),
            _canon(obj.keywords, f"{name}.keywords", depth + 1, seen),
        ]
    if isinstance(obj, type):
        return ["class", _qualname(obj)]
    if isinstance(obj, types.ModuleType):
        return ["module", obj.__name__]
    if isinstance(
        obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)
    ):
        return _canon_callable(obj, name, depth, seen)

    # -- plain objects: recurse into their state ---------------------------
    state = _object_state(obj)
    if dataclasses.is_dataclass(obj):
        # Fields first, then whatever a subclass adds outside of them.
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        state = {**(state or {}), **fields}
        tag = "dataclass"
    else:
        tag = "object"
    if callable(obj) and isinstance(getattr(obj, "__qualname__", None), str):
        # Compiled functions (Cython) expose an empty ``__dict__``; their name
        # is the only thing that tells them apart.
        compiled = _canon(state, name, depth + 1, seen)
        return [tag, _qualname(type(obj)), _qualname(obj), compiled]
    if state is not None:
        return [tag, _qualname(type(obj)), _canon(state, name, depth + 1, seen)]

    return _fallback_repr(obj, name)


def _fallback_repr(obj: Any, name: str) -> Any:
    """Last resort: ``repr()``, unless it is visibly unstable."""
    text = repr(obj)
    if type(obj).__repr__ is object.__repr__ or _ADDRESS_RE.search(text):
        if _STRICT:
            raise UnfingerprintableError(_hint(obj, name))
        logger.warning("%s -- falling back to an unstable repr()", _hint(obj, name))
    return ["repr", _qualname(type(obj)), text]


def _canon_scalar(obj: Any) -> Any:
    """Handle immutable standard-library values; None if ``obj`` is not one."""
    if isinstance(obj, bool):  # must precede int
        return ["bool", obj]
    if isinstance(obj, int):
        # int() keeps arbitrary precision and ignores a subclass's __str__
        return ["int", str(int(obj))]
    if isinstance(obj, float):
        return ["float", float(obj).hex()]  # exact, incl. -0.0/nan
    if isinstance(obj, str):
        return ["str", str.__str__(obj)]
    if isinstance(obj, (bytes, bytearray)):
        return ["bytes", _digest(bytes(obj))]
    if isinstance(obj, complex):
        return ["complex", obj.real.hex(), obj.imag.hex()]
    if isinstance(obj, decimal.Decimal):
        return ["decimal", str(obj)]
    if isinstance(obj, _dt.datetime):
        return ["datetime", obj.isoformat(), str(obj.tzinfo)]
    if isinstance(obj, _dt.date):
        return ["date", obj.isoformat()]
    if isinstance(obj, _dt.time):
        return ["time", obj.isoformat(), str(obj.tzinfo)]
    if isinstance(obj, _dt.timedelta):
        # pandas.Timedelta adds nanoseconds to the standard fields
        nanoseconds = getattr(obj, "nanoseconds", 0)
        return ["timedelta", obj.days, obj.seconds, obj.microseconds, nanoseconds]
    if isinstance(obj, PurePath):
        return ["path", str(obj)]
    if isinstance(obj, uuid.UUID):
        return ["uuid", str(obj)]
    if isinstance(obj, range):
        return ["range", obj.start, obj.stop, obj.step]
    if isinstance(obj, re.Pattern):  # its repr() truncates the pattern
        return ["pattern", _canon_scalar(obj.pattern), obj.flags]
    return None


def _canon_container(obj: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    """Handle mappings, sets and sequences; None if ``obj`` is not one."""
    if isinstance(obj, Mapping):
        # Fast path for the common all-string-keys case (e.g. a 100k-token
        # vocabulary): sort the raw keys instead of serialising each one.
        if all(type(k) is str for k in obj):
            items = [
                (["str", k], _canon(obj[k], f"{name}[{k!r}]", depth + 1, seen))
                for k in sorted(obj.keys())
            ]
            return ["dict", items]
        items = [
            (
                _canon(k, f"{name}.<key>", depth + 1, seen),
                _canon(v, f"{name}[{k!r}]", depth + 1, seen),
            )
            for k, v in obj.items()
        ]
        # Sort on the serialised key, never on the raw key: sorting raw keys
        # crashes on mixed int/str keys.
        items.sort(key=lambda kv: json.dumps(kv[0], sort_keys=True))
        return ["dict", items]
    if isinstance(obj, (set, frozenset)):
        tag = "frozenset" if isinstance(obj, frozenset) else "set"
        if all(type(v) is str for v in obj):
            return [tag, [["str", v] for v in sorted(obj)]]
        elems = [_canon(v, f"{name}.<elem>", depth + 1, seen) for v in obj]
        # kills PYTHONHASHSEED dependence
        elems.sort(key=lambda e: json.dumps(e, sort_keys=True))
        return [tag, elems]
    if isinstance(obj, (tuple, list)):
        tag = "tuple" if isinstance(obj, tuple) else "list"
        elems = [
            _canon(v, f"{name}[{i}]", depth + 1, seen) for i, v in enumerate(obj)
        ]
        return [tag, elems]
    return None


def _tag_subclass(
    obj: Any, value: Any, name: str, depth: int, seen: dict[int, int]
) -> Any:
    """Add the type and own state of a subclass to its base type's ``value``."""
    cls = type(obj)
    if cls.__module__.split(".")[0] in _KNOWN_VALUE_MODULES or isinstance(
        obj, re.Pattern
    ):
        return value
    state: dict[str, Any] = {}
    # An abstract Mapping keeps its items in its state, which ``value`` already
    # covers; so does a dict that is its own ``__dict__``.
    own_dict = getattr(obj, "__dict__", None)
    if own_dict is not obj and (isinstance(obj, dict) or not isinstance(obj, Mapping)):
        state = _object_state(obj) or {}
    if isinstance(obj, collections.defaultdict):
        state["default_factory"] = obj.default_factory
    return [
        "subclass",
        _qualname(cls),
        value,
        _canon(state, name, depth + 1, seen | {id(obj): depth}),
    ]


def _canon_digest(obj: Any, name: str, depth: int, seen: dict[int, int]) -> str:
    """Digest of ``obj``'s canonical form, to keep tables out of the spec."""
    value = _canon(obj, name, depth + 1, seen)
    return _digest(json.dumps(value, sort_keys=True).encode())


def _canon_scientific(obj: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    """Handle numpy / torch / pandas / polars without importing them."""
    # The whole MRO, so that a user's ndarray or Tensor subclass is recognised.
    mods = {cls.__module__.split(".")[0] for cls in type(obj).__mro__}
    if "numpy" in mods:
        return _canon_numpy(obj, name, depth, seen)
    if "torch" in mods:
        return _canon_torch(obj, name)
    if "pandas" in mods:
        return _canon_pandas(obj, name, depth, seen)
    if "polars" in mods:
        return _canon_polars(obj, name, depth, seen)
    return None


def _canon_numpy(obj: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    import numpy as np

    if isinstance(obj, np.ma.MaskedArray):
        parts = (np.asarray(obj.data), np.ma.getmaskarray(obj))
        return ["ndarray_masked", _canon(parts, name, depth + 1, seen)]
    if isinstance(obj, np.ndarray):
        dtype = obj.dtype
        # tobytes() would hash pointers for objects and variable-width strings,
        # and padding bytes for structured records.
        if dtype.hasobject or dtype.names is not None or dtype.kind == "T":
            values = _canon(obj.tolist(), name, depth + 1, seen)
            return ["ndarray_object", str(dtype), list(obj.shape), values]
        data = np.ascontiguousarray(obj).tobytes()
        return ["ndarray", dtype.str, list(obj.shape), _digest(data)]
    if isinstance(obj, np.dtype):
        return ["dtype", str(obj)]
    if isinstance(obj, np.generic):
        # item() turns a datetime64[ns] into a bare int, drops the field names
        # of a record and returns a longdouble unchanged.
        # The dtype is kept: np.float32(0.5), np.float64(0.5) and 0.5 differ.
        if obj.dtype.kind in "biufc" and not isinstance(obj.item(), np.generic):
            value = _canon(obj.item(), name, depth + 1, seen)
            return ["numpy_scalar", obj.dtype.str, value]
        return _canon(np.asarray(obj), name, depth + 1, seen)
    return None


def _canon_torch(obj: Any, name: str) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        if obj.layout != torch.strided or obj.is_quantized or obj.is_meta:
            return _fallback_tensor(obj, name)
        data = obj.detach().cpu().resolve_conj().resolve_neg().contiguous()
        # As bytes, since numpy has no bfloat16.
        data = data.as_subclass(torch.Tensor).reshape(-1).view(torch.uint8).numpy()
        return ["tensor", str(obj.dtype), list(obj.shape), _digest(data.tobytes())]
    if isinstance(obj, torch.dtype):
        return ["torch.dtype", str(obj)]
    if isinstance(obj, torch.device):
        return ["torch.device", str(obj)]
    return None


def _fallback_tensor(obj: Any, name: str) -> Any:
    """Sparse, quantized and meta tensors have no plain buffer to digest."""
    if _STRICT:
        raise UnfingerprintableError(_hint(obj, name))
    logger.warning("%s -- falling back to repr()", _hint(obj, name))
    return ["repr", _qualname(type(obj)), repr(obj)]


def _canon_pandas(obj: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    import pandas as pd

    def dtype(values: Any) -> Any:
        if isinstance(values.dtype, pd.CategoricalDtype):
            categories = values.dtype.categories.tolist()
            return [
                "category",
                bool(values.dtype.ordered),
                _canon_digest(categories, name, depth, seen),
            ]
        return str(values.dtype)

    def column(values: Any) -> Any:
        # Not hash_pandas_object: it conflates 1 with "1" in an object column,
        # and int32 with int64, and cannot hash list cells.
        native = type(values.dtype).__module__.split(".")[0] == "numpy"
        data = values.to_numpy() if native else values.tolist()
        return [dtype(values), _canon_digest(data, name, depth, seen)]

    def index(values: Any) -> Any:
        names = _canon(list(values.names), f"{name}.names", depth + 1, seen)
        if isinstance(values, pd.MultiIndex):
            levels = range(values.nlevels)
            return [names, [column(values.get_level_values(i)) for i in levels]]
        return [names, column(values)]

    if isinstance(obj, pd.DataFrame):
        columns = [column(obj.iloc[:, i]) for i in range(obj.shape[1])]
        return ["pandas.DataFrame", index(obj.columns), index(obj.index), columns]
    if isinstance(obj, pd.Series):
        label = _canon(obj.name, f"{name}.name", depth + 1, seen)
        return ["pandas.Series", label, index(obj.index), column(obj)]
    if isinstance(obj, pd.Index):
        return ["pandas.Index", index(obj)]
    return None


def _canon_polars(obj: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    import polars as pl

    def column(values: Any) -> Any:
        dtype = values.dtype
        if isinstance(dtype, pl.Enum):  # its repr() may truncate the categories
            categories = dtype.categories.to_list()
            dtype_key = ["enum", _canon_digest(categories, name, depth, seen)]
        else:
            dtype_key = repr(dtype)
        data = _canon_digest(values.to_list(), name, depth, seen)
        return [values.name, dtype_key, data]

    if isinstance(obj, pl.DataFrame):
        return ["polars.DataFrame", list(obj.shape), [column(c) for c in obj]]
    if isinstance(obj, pl.Series):
        return ["polars.Series", column(obj)]
    return None


def _canon_callable(fn: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    """Name, source and captured state of a function or bound method."""
    if isinstance(fn, types.MethodType):
        return [
            "method",
            _canon(fn.__func__, f"{name}.__func__", depth + 1, seen),
            _canon_captured(fn.__self__, f"{name}.__self__", depth, seen),
        ]
    if isinstance(fn, types.BuiltinFunctionType):
        owner = fn.__self__
        # A module-level C function is bound to its module, or to a PyCapsule
        # when pybind11 made it; neither carries configuration.
        if owner is None or isinstance(owner, types.ModuleType) or (
            type(owner).__name__ == "PyCapsule"
        ):
            return ["function", _qualname(fn), None]
        bound = _canon_captured(owner, f"{name}.__self__", depth, seen)
        return ["method", _qualname(fn), bound]

    qn = _qualname(fn)
    src = _structural_source(fn)
    importable = _is_importable(fn)
    # Scripts and sessions all share the module name ``__main__``.
    if src is None and not (importable and fn.__module__ != "__main__"):
        if _STRICT:
            raise UnfingerprintableError(
                f"{name}: the source of callable {qn!r} cannot be read (it was "
                f"defined without a file, for example by exec(), or its file "
                f"changed after import), so two different functions would share "
                f"a cache key. Use a module-level function, or exclude it via "
                f"`fingerprint_exclude`."
            )
        logger.warning("%s: callable with no source; cache key may collide", name)
    if importable:  # its name and code say it all
        return ["function", qn, src]
    return ["function", qn, src, _captured_state(fn, name, depth, seen)]


def _is_importable(fn: Any) -> bool:
    """Whether ``fn`` is what its module and qualified name resolve to."""
    target = sys.modules.get(fn.__module__)
    for part in fn.__qualname__.split("."):
        target = getattr(target, part, None)
    return target is fn


def _captured_state(fn: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    """What ``fn`` closes over: free variables, defaults and own attributes.

    Two closures made by one factory share their name and source, so these
    values are all that tells them apart.
    """
    captured: dict[str, Any] = {}
    for var, cell in zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True):
        try:
            captured[f"closure {var}"] = cell.cell_contents
        except ValueError:  # a variable that is not assigned yet
            continue
    for i, value in enumerate(fn.__defaults__ or ()):
        captured[f"default {i}"] = value
    for key, value in (fn.__kwdefaults__ or {}).items():
        captured[f"kwdefault {key}"] = value
    for key, value in vars(fn).items():
        if key != "__wrapped__":  # set by functools.wraps; also in the closure
            captured[f"attribute {key}"] = value
    return [
        [key, _canon_captured(captured[key], f"{name}.<{key}>", depth, seen)]
        for key in sorted(captured)
    ]


def _canon_captured(value: Any, name: str, depth: int, seen: dict[int, int]) -> Any:
    """Like ``_canon``, but a reference to an enclosing object is not an error.

    A task may store one of its own methods, and a recursive closure captures
    itself. The enclosing object is already in the fingerprint, so its depth is
    enough to identify it.
    """
    if id(value) in seen:
        return ["enclosing", seen[id(value)]]
    if type(value) is object:  # a sentinel default such as ``_MISSING = object()``
        return ["sentinel"]
    return _canon(value, name, depth + 1, seen)


def _structural_source(fn: Any) -> str | None:
    """Hash the code of ``fn``, ignoring comments, whitespace and docstrings."""
    tree: ast.AST | None
    if getattr(fn, "__name__", None) == "<lambda>":
        tree = _lambda_node(fn)
    else:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        except (OSError, TypeError, SyntaxError, IndentationError):
            tree = None
    if tree is None:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                body.pop(0)
    # Not ast.dump(): its output changed between Python 3.12 and 3.13.
    return _digest(ast.unparse(tree).encode())[:16]


def _lambda_node(fn: Any) -> ast.Lambda | None:
    """Find the ``lambda`` expression that defined ``fn`` in its source file.

    ``inspect.getsource`` returns the lines of a lambda, which are not always
    a complete statement (a lambda passed as the last argument of a call
    that spans several lines) and may hold several lambdas. The whole file
    is parsed instead, and the lambdas that start on the line of ``fn`` are
    told apart by the source positions of its code.
    """
    code = fn.__code__
    lines = linecache.getlines(code.co_filename)
    if not lines:
        return None
    try:
        module = _parse_source("".join(lines))
    except (SyntaxError, ValueError):
        return None
    candidates = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno
    ]
    # Positions of the code, in the same (line, byte column) units as the AST.
    # The first entry of a code object spans no columns and is left out.
    spans = [
        (line, col, end_line, end_col)
        for line, end_line, col, end_col in code.co_positions()
        if None not in (line, end_line, col, end_col)
        and (line, col) != (end_line, end_col)
    ]

    def holds_code(node: ast.Lambda) -> bool:
        body = node.body
        return all(
            (body.lineno, body.col_offset) <= (line, col)
            and (end_line, end_col) <= (node.end_lineno, node.end_col_offset)
            for line, col, end_line, end_col in spans
        )

    def size(node: ast.Lambda) -> tuple[int, int]:
        return (node.end_lineno - node.lineno, node.end_col_offset - node.col_offset)

    if not spans:
        # No positions (python -X no_debug_ranges): only lambdas that read the
        # same can share a key.
        if len({ast.unparse(node) for node in candidates}) != 1:
            return None
        return copy.deepcopy(candidates[0])
    # An enclosing lambda also holds the code of the ones it contains.
    matches = sorted((node for node in candidates if holds_code(node)), key=size)
    if not matches:
        return None
    return copy.deepcopy(matches[0])  # the parsed module is cached; keep it intact


@functools.lru_cache(maxsize=16)
def _parse_source(source: str) -> ast.Module:
    return ast.parse(source)


def _object_state(obj: Any) -> dict[str, Any] | None:
    """Instance state of a plain object, covering ``__dict__`` and ``__slots__``."""
    state: dict[str, Any] = {}
    found = False
    if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__", None), dict):
        state.update(obj.__dict__)
        found = True
    for klass in type(obj).__mro__:
        slots = vars(klass).get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot.startswith("__") and not slot.endswith("__"):
                slot = f"_{klass.__name__.lstrip('_')}{slot}"  # name mangling
            if slot not in ("__dict__", "__weakref__") and hasattr(obj, slot):
                state[slot] = getattr(obj, slot)
                found = True
    return state if found else None


def _qualname(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"


# --------------------------------------------------------------------------
# spec / fingerprint
# --------------------------------------------------------------------------

# Never part of the identity of a task's *output*.
_ALWAYS_EXCLUDED = frozenset(
    {
        "num_workers",
        "n_jobs",
        "verbose",
        "cache_dir",
        "refresh_cache",
        "progress_bar",
    }
)


def _class_config(task: Any, excluded: frozenset) -> dict[str, Any]:
    """Class-level configuration, which ``vars(task)`` cannot see."""
    config: dict[str, Any] = {}
    for klass in reversed(type(task).__mro__):
        if klass in (object,):
            continue
        for key, value in vars(klass).items():
            # Underscore-prefixed class attributes are machinery, not
            # configuration (e.g. ABCMeta's ``_abc_impl``, which is a C object
            # with an address-bearing repr and would otherwise raise).
            if key.startswith("_") or key in excluded:
                continue
            if callable(value) or isinstance(
                value, (property, staticmethod, classmethod, types.MemberDescriptorType)
            ):
                continue
            config[key] = value
    return config


def task_spec(task: Any, *, include_source: bool | None = None) -> dict[str, Any]:
    """Build the canonical, JSON-safe description that identifies ``task``.

    Example:
        >>> class T:
        ...     task_name = "toy"
        ...     input_schema = {}
        ...     output_schema = {}
        >>> task_spec(T())["task_name"]
        ['str', 'toy']
    """
    include_source = _HASH_SOURCE if include_source is None else include_source
    excluded = _ALWAYS_EXCLUDED | frozenset(getattr(task, "fingerprint_exclude", ()) or ())

    instance_state = {
        k: v for k, v in (_object_state(task) or {}).items() if k not in excluded
    }
    class_config = {k: v for k, v in _class_config(task, excluded).items()}

    spec: dict[str, Any] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "task_class": _qualname(type(task)),
        # Through _canon like the rest: a numpy integer or a packaging Version
        # would otherwise make json.dumps fail outside UnfingerprintableError.
        "task_name": _canon(getattr(task, "task_name", None), "task_name"),
        "task_version": _canon(getattr(task, "version", "1"), "version"),
        "class_config": _canon(class_config, "class_config"),
        # The task itself encloses its state, so that storing one of its own
        # methods (or a lambda over ``self``) is not a circular reference.
        "instance_state": _canon(
            instance_state, "instance_state", seen={id(task): -1}
        ),
        "input_schema": _canon(getattr(task, "input_schema", None), "input_schema"),
        "output_schema": _canon(getattr(task, "output_schema", None), "output_schema"),
    }
    if include_source:
        spec["source"] = {
            hook: _structural_source(getattr(type(task), hook, None))
            for hook in ("__call__", "pre_filter")
        }
        if spec["source"]["__call__"] is None:
            logger.warning(
                "No source found for %s.__call__; edits to it will not change the "
                "cache key",
                _qualname(type(task)),
            )
    return spec


def task_fingerprint(task: Any, *, include_source: bool | None = None) -> str:
    """Stable hex digest identifying the task configuration.

    Example:
        >>> class T:
        ...     task_name = "toy"
        ...     input_schema = {}
        ...     output_schema = {}
        >>> len(task_fingerprint(T()))
        64
        >>> task_fingerprint(T()) == task_fingerprint(T())
        True
    """
    return _spec_digest(task_spec(task, include_source=include_source))


def _spec_digest(spec: dict[str, Any]) -> str:
    return _digest(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode())


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str, max_len: int = 48) -> str:
    """Make a task name safe for a single path component.

    Example:
        >>> slugify("BenchmarkEHRShot/guo_los")
        'BenchmarkEHRShot-guo_los'
    """
    slug = _SLUG_RE.sub("-", str(name)).strip("-.") or "task"
    return slug[:max_len]


def task_cache_name(task: Any, *, include_source: bool | None = None) -> str:
    """Path-safe cache directory name: ``{slug}_{digest[:16]}``.

    Example:
        >>> class T:
        ...     task_name = "Bench/mark"
        ...     input_schema = {}
        ...     output_schema = {}
        >>> "/" not in task_cache_name(T())
        True
    """
    return f"{slugify(getattr(task, 'task_name', type(task).__name__))}_" \
           f"{task_fingerprint(task, include_source=include_source)[:16]}"


def processors_fingerprint(
    input_processors: Mapping[str, Any] | None,
    output_processors: Mapping[str, Any] | None,
    *,
    storage_format: Any = None,
    split: Any = None,
) -> str:
    """Fingerprint the processed-samples cache key below a task directory.

    Same failure modes as tasks: ``SequenceProcessor`` holds a ``CrossMap``
    (address-bearing repr) and a ``code_vocab`` typed ``Dict[Any, int]``
    (mixed keys crash ``sort_keys=True``).

    ``storage_format`` and ``split`` are the other two inputs that decide the
    content of a ``samples_*`` directory: the on-disk layout makes older
    ``*.bin`` files unreadable when it changes, and a split decides which
    patients fit the processors. Both are omitted from the payload when left at
    ``None``, so a caller that passes neither keeps the two-argument digest.

    Args:
        input_processors: Pre-fitted input processors, or None.
        output_processors: Pre-fitted output processors, or None.
        storage_format: ``SampleDataset.STORAGE_FORMAT`` of the writer.
        split: ``Split.to_dict()`` of the split in use, or None without one.

    Example:
        >>> processors_fingerprint(None, None) == processors_fingerprint({}, {})
        True
        >>> len(processors_fingerprint(None, None))
        64
        >>> a = processors_fingerprint(None, None, split={"kind": "patient", "seed": 1})
        >>> b = processors_fingerprint(None, None, split={"kind": "patient", "seed": 2})
        >>> a == b
        False
        >>> processors_fingerprint(None, None, storage_format=2) == processors_fingerprint(
        ...     None, None, storage_format=3
        ... )
        False
    """
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "input": _canon(
            {f"{k}:{_qualname(type(v))}": v for k, v in (input_processors or {}).items()}, "input"
        ),
        "output": _canon(
            {f"{k}:{_qualname(type(v))}": v for k, v in (output_processors or {}).items()}, "output"
        ),
    }
    # Kept out of the payload when absent so that an unsplit cache written
    # before these two entered the key keeps its digest.
    if storage_format is not None:
        payload["storage_format"] = _canon(storage_format, "storage_format")
    if split is not None:
        payload["split"] = _canon(split, "split")
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def write_task_metadata(
    cache_dir: Path,
    task: Any,
    overwrite: bool = False,
) -> Path:
    """Write a legible sidecar so a cache directory explains itself.

    Answers the second half of issue #916: an opaque digest tells nobody which
    parameter changed. ``task_meta.json`` does.

    The sidecar is purely diagnostic, so this function never raises: failing to
    write a comment must not abort a multi-hour build. It is also skipped when
    the file already exists -- the directory name *is* the fingerprint, so an
    existing sidecar necessarily describes the same configuration, and
    rewriting it on every cache hit would turn ``created_at`` into a
    last-accessed timestamp.

    Example:
        >>> import tempfile
        >>> class T:
        ...     task_name = "toy"
        ...     input_schema = {}
        ...     output_schema = {}
        >>> path = write_task_metadata(Path(tempfile.mkdtemp()), T())
        >>> path.name
        'task_meta.json'
    """
    path = Path(cache_dir) / "task_meta.json"
    if path.exists() and not overwrite:
        return path

    try:
        from pyhealth import __version__ as pyhealth_version
    except ImportError:  # pragma: no cover
        pyhealth_version = "unknown"

    spec = task_spec(task)  # walked once: a task may hold large tables
    payload = {
        "fingerprint": _spec_digest(spec),
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "pyhealth_version": pyhealth_version,
        "spec": spec,
    }

    # Unique temp name per writer. This function runs outside the build lock,
    # so parallel hyper-parameter jobs reach it concurrently; a shared temp
    # name means they clobber each other's partial writes on POSIX and fail
    # outright on Windows, where a second open() of the same path raises
    # ERROR_SHARING_VIOLATION (WinError 32).
    tmp = path.with_name(f"task_meta.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)  # atomic: concurrent readers never see a partial file
    except OSError as exc:
        logger.debug("Could not write task metadata to %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
    return path
