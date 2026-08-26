"""Deterministic fingerprinting of tasks for cache keys.

Proposed replacement for the inline ``json.dumps(vars(task), default=str)``
fingerprint currently used by ``BaseDataset.set_task``.

Design goals
------------
1. **Deterministic** across processes, machines and Python versions: never
   relies on ``PYTHONHASHSEED``-dependent iteration order or on memory
   addresses leaking through ``repr()``.
2. **Lossless**: no truncating ``str()`` of large objects, so two different
   configurations can never collide.
3. **Fail loud, not silent**: an argument that cannot be fingerprinted
   deterministically raises with an actionable message instead of producing a
   key that is either unstable (permanent cache misses) or colliding (silently
   stale samples).
4. **Complete**: covers recorded ``__init__`` arguments, derived instance
   state, class-level configuration, and an explicit task ``version``.
5. **Legible**: emits a human-readable spec that is written next to the cache,
   so users can see *why* a cache directory exists.

Public API
----------
``task_spec(task)``          -> canonical JSON-safe dict describing the task
``task_fingerprint(task)``   -> stable hex digest of that spec
``task_cache_name(task)``    -> path-safe ``{slug}_{digest}`` directory name
``write_task_metadata(...)`` -> writes ``task_meta.json`` sidecar
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as _dt
import decimal
import enum
import functools
import hashlib
import inspect
import json
import logging
import os
import re
import textwrap
import types
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

logger = logging.getLogger(__name__)

# Bump when the fingerprint format itself changes. Every cache key changes,
# so existing caches become unreachable (they are not deleted).
FINGERPRINT_VERSION = 2

# Attribute set on task instances by ``record_init_args``.
_INIT_ARGS_ATTR = "_pyhealth_init_args"

# Opt-out for environments with exotic task arguments. Strict mode is the
# default because a non-strict fallback reintroduces silent collisions.
_STRICT = os.environ.get("PYHEALTH_FINGERPRINT_STRICT", "1") not in ("0", "false", "False")

# Include a structural hash of ``__call__``/``pre_filter`` source. Off by
# default: editing a comment should not invalidate a 40-minute cache build.
_HASH_SOURCE = os.environ.get("PYHEALTH_FINGERPRINT_SOURCE", "0") in ("1", "true", "True")

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
_MAX_DEPTH = 64


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


def _canon(obj: Any, name: str = "<root>", depth: int = 0, seen: set[int] | None = None) -> Any:
    """Return a JSON-safe, type-tagged, order-stable representation of ``obj``."""
    if depth > _MAX_DEPTH:
        raise UnfingerprintableError(f"{name}: nesting deeper than {_MAX_DEPTH}")

    seen = seen if seen is not None else set()

    # -- scalars ----------------------------------------------------------
    if obj is None:
        return None
    if isinstance(obj, bool):  # must precede int
        return ["bool", obj]
    if isinstance(obj, int):
        return ["int", str(obj)]  # str() keeps arbitrary-precision ints exact
    if isinstance(obj, float):
        return ["float", obj.hex()]  # exact and round-trippable, incl. -0.0/nan
    if isinstance(obj, str):
        return ["str", obj]
    if isinstance(obj, (bytes, bytearray)):
        return ["bytes", _digest(bytes(obj))]
    if isinstance(obj, complex):
        return ["complex", obj.real.hex(), obj.imag.hex()]
    if isinstance(obj, decimal.Decimal):
        return ["decimal", str(obj)]

    # -- stdlib value types ------------------------------------------------
    if isinstance(obj, enum.Enum):
        return ["enum", _qualname(type(obj)), _canon(obj.value, f"{name}.value", depth + 1, seen)]
    if isinstance(obj, _dt.datetime):
        return ["datetime", obj.isoformat(), str(obj.tzinfo)]
    if isinstance(obj, _dt.date):
        return ["date", obj.isoformat()]
    if isinstance(obj, _dt.time):
        return ["time", obj.isoformat()]
    if isinstance(obj, _dt.timedelta):
        return ["timedelta", obj.days, obj.seconds, obj.microseconds]
    if isinstance(obj, PurePath):
        return ["path", str(obj)]
    if isinstance(obj, uuid.UUID):
        return ["uuid", str(obj)]
    if isinstance(obj, range):
        return ["range", obj.start, obj.stop, obj.step]

    # -- cycles ------------------------------------------------------------
    if id(obj) in seen:
        raise UnfingerprintableError(f"{name}: circular reference")
    seen = seen | {id(obj)}

    # -- third-party numerics (duck-typed, no hard dependency) -------------
    special = _canon_scientific(obj, name, depth, seen)
    if special is not None:
        return special

    # -- containers --------------------------------------------------------
    if isinstance(obj, Mapping):
        # Fast path for the common all-string-keys case (e.g. a 100k-token
        # vocabulary): sort the raw keys instead of serialising each one.
        if all(isinstance(k, str) for k in obj):
            keys = sorted(obj.keys())
            items = [
                (["str", k], _canon(obj[k], f"{name}[{k!r}]", depth + 1, seen)) for k in keys
            ]
            return ["dict", items]
        items = [
            (_canon(k, f"{name}.<key>", depth + 1, seen), _canon(v, f"{name}[{k!r}]", depth + 1, seen))
            for k, v in obj.items()
        ]
        # Sort on the serialised key, never on the raw key: sorting raw keys
        # crashes on mixed int/str keys (the current json sort_keys=True bug).
        items.sort(key=lambda kv: json.dumps(kv[0], sort_keys=True))
        return ["dict", items]
    if isinstance(obj, (set, frozenset)):
        tag = "frozenset" if isinstance(obj, frozenset) else "set"
        if all(isinstance(v, str) for v in obj):
            return [tag, [["str", v] for v in sorted(obj)]]
        elems = [_canon(v, f"{name}.<elem>", depth + 1, seen) for v in obj]
        elems.sort(key=lambda e: json.dumps(e, sort_keys=True))  # kills PYTHONHASHSEED dependence
        return [tag, elems]
    if isinstance(obj, tuple):
        return ["tuple", [_canon(v, f"{name}[{i}]", depth + 1, seen) for i, v in enumerate(obj)]]
    if isinstance(obj, list):
        return ["list", [_canon(v, f"{name}[{i}]", depth + 1, seen) for i, v in enumerate(obj)]]

    # -- callables and types ----------------------------------------------
    if isinstance(obj, functools.partial):
        return [
            "partial",
            _canon(obj.func, f"{name}.func", depth + 1, seen),
            _canon(obj.args, f"{name}.args", depth + 1, seen),
            _canon(obj.keywords, f"{name}.keywords", depth + 1, seen),
        ]
    if isinstance(obj, type):
        return ["class", _qualname(obj)]
    if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return _canon_callable(obj, name)

    # -- user hook ---------------------------------------------------------
    hook = getattr(type(obj), "__pyhealth_fingerprint__", None)
    if callable(hook):
        return ["hook", _qualname(type(obj)), _canon(hook(obj), f"{name}.<hook>", depth + 1, seen)]

    if dataclasses.is_dataclass(obj):
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        return ["dataclass", _qualname(type(obj)), _canon(fields, name, depth + 1, seen)]

    # -- plain objects: recurse into their state ---------------------------
    state = _object_state(obj)
    if state is not None:
        return ["object", _qualname(type(obj)), _canon(state, name, depth + 1, seen)]

    # -- last resort -------------------------------------------------------
    text = repr(obj)
    if type(obj).__repr__ is object.__repr__ or _ADDRESS_RE.search(text):
        if _STRICT:
            raise UnfingerprintableError(_hint(obj, name))
        logger.warning("%s -- falling back to an unstable repr()", _hint(obj, name))
    return ["repr", _qualname(type(obj)), text]


def _canon_scientific(obj: Any, name: str, depth: int, seen: set[int]) -> Any:
    """Handle numpy / torch / pandas / polars without importing them."""
    mod = type(obj).__module__.split(".")[0]

    if mod == "numpy":
        import numpy as np

        if isinstance(obj, np.ndarray):
            if obj.dtype == object:  # tobytes() would hash pointers
                return ["ndarray_object", list(obj.shape), _canon(obj.tolist(), name, depth + 1, seen)]
            return ["ndarray", obj.dtype.str, list(obj.shape), _digest(np.ascontiguousarray(obj).tobytes())]
        if isinstance(obj, np.dtype):
            return ["dtype", str(obj)]
        if isinstance(obj, np.generic):
            return _canon(obj.item(), name, depth + 1, seen)

    if mod == "torch":
        import torch

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().contiguous().numpy()
            return ["tensor", str(obj.dtype), list(obj.shape), _digest(arr.tobytes())]
        if isinstance(obj, torch.dtype):
            return ["torch.dtype", str(obj)]
        if isinstance(obj, torch.device):
            return ["torch.device", str(obj)]

    if mod == "pandas":
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            values = pd.util.hash_pandas_object(obj, index=True).values
            return ["pandas", list(getattr(obj, "shape", ())), _digest(values.tobytes())]

    if mod == "polars":
        import polars as pl

        if isinstance(obj, pl.DataFrame):
            return ["polars", list(obj.shape), _digest(obj.hash_rows().to_numpy().tobytes())]

    return None


def _canon_callable(fn: Any, name: str) -> Any:
    qn = _qualname(fn)
    anonymous = "<lambda>" in qn or "<locals>" in qn
    src = _structural_source(fn)
    if src is not None:
        return ["function", qn, src]
    if anonymous:
        if _STRICT:
            raise UnfingerprintableError(
                f"{name}: anonymous callable {qn!r} has no retrievable source, so "
                f"two different lambdas would share a cache key. Use a module-level "
                f"function, or exclude it via `fingerprint_exclude`."
            )
        logger.warning("%s: anonymous callable with no source; cache key may collide", name)
    return ["function", qn, None]


def _structural_source(fn: Any) -> str | None:
    """Hash the AST of ``fn``, ignoring comments, whitespace and docstrings."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
                first.value.value, str
            ):
                body.pop(0)
    return _digest(ast.dump(tree).encode())[:16]


def _object_state(obj: Any) -> dict[str, Any] | None:
    """Instance state of a plain object, covering ``__dict__`` and ``__slots__``."""
    state: dict[str, Any] = {}
    found = False
    if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__", None), dict):
        state.update(obj.__dict__)
        found = True
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()) or ():
            if isinstance(slot, str) and hasattr(obj, slot):
                state[slot] = getattr(obj, slot)
                found = True
    return state if found else None


def _qualname(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"


# --------------------------------------------------------------------------
# init-arg recording
# --------------------------------------------------------------------------


def record_init_args(cls: type) -> type:
    """Record the *effective* ``__init__`` arguments of every instance.

    Wraps ``cls.__init__`` so that bound arguments -- with defaults applied --
    are stored on the instance. Applying defaults means ``Task()`` and
    ``Task(window=timedelta(days=15))`` produce the same key when 15 days is
    the default, and adding a new keyword argument changes the key for
    everyone (which is correct: the behaviour changed).

    Intended to be called from ``BaseTask.__init_subclass__``.

    Example:
        >>> from pyhealth.tasks.fingerprint import record_init_args
        >>> class T:
        ...     def __init__(self, n=0):
        ...         self.n = n
        >>> record_init_args(T) is T
        True
        >>> T()._pyhealth_init_args["n"]
        0
    """
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, "_pyhealth_recorded", False):
        return cls
    try:
        sig = inspect.signature(init)
    except (TypeError, ValueError):
        return cls

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        init(self, *args, **kwargs)
        try:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            recorded = dict(bound.arguments)
            recorded.pop(next(iter(sig.parameters)), None)  # drop 'self'
        except TypeError:  # pragma: no cover - init would have raised already
            return
        # Most-derived __init__ returns last, so it wins.
        object.__setattr__(self, _INIT_ARGS_ATTR, recorded)

    wrapper._pyhealth_recorded = True  # type: ignore[attr-defined]
    cls.__init__ = wrapper  # type: ignore[assignment]
    return cls


# --------------------------------------------------------------------------
# spec / fingerprint
# --------------------------------------------------------------------------

# Never part of the identity of a task's *output*.
_ALWAYS_EXCLUDED = frozenset(
    {
        _INIT_ARGS_ATTR,
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
        'toy'
    """
    include_source = _HASH_SOURCE if include_source is None else include_source
    excluded = _ALWAYS_EXCLUDED | frozenset(getattr(task, "fingerprint_exclude", ()) or ())

    init_args = dict(getattr(task, _INIT_ARGS_ATTR, {}) or {})
    instance_state = {
        k: v for k, v in (_object_state(task) or {}).items() if k not in excluded
    }
    class_config = {k: v for k, v in _class_config(task, excluded).items()}

    spec: dict[str, Any] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "task_class": _qualname(type(task)),
        "task_name": getattr(task, "task_name", None),
        "task_version": getattr(task, "version", "1"),
        "init_args": _canon({k: v for k, v in init_args.items() if k not in excluded}, "init_args"),
        "class_config": _canon(class_config, "class_config"),
        "instance_state": _canon(instance_state, "instance_state"),
        "input_schema": _canon(getattr(task, "input_schema", None), "input_schema"),
        "output_schema": _canon(getattr(task, "output_schema", None), "output_schema"),
    }
    if include_source:
        spec["source"] = {
            hook: _structural_source(getattr(type(task), hook, None))
            for hook in ("__call__", "pre_filter")
        }
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
    spec = task_spec(task, include_source=include_source)
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
) -> str:
    """Fingerprint pre-fitted processors for the ``samples_*`` cache key.

    Same failure modes as tasks: ``SequenceProcessor`` holds a ``CrossMap``
    (address-bearing repr) and a ``code_vocab`` typed ``Dict[Any, int]``
    (mixed keys crash ``sort_keys=True``).

    Example:
        >>> processors_fingerprint(None, None) == processors_fingerprint({}, {})
        True
        >>> len(processors_fingerprint(None, None))
        64
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
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def write_task_metadata(
    cache_dir: Path,
    task: Any,
    extra: dict[str, Any] | None = None,
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

    payload = {
        "fingerprint": task_fingerprint(task),
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "pyhealth_version": pyhealth_version,
        "spec": task_spec(task),
        **(extra or {}),
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
