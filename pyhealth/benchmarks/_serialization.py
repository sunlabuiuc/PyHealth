"""Canonical JSON serialization of Evaluation params.

Classes and functions are encoded by import path as
``{"__type__": "<module>.<qualname>"}`` and resolved back on decode.
"""

import importlib
import json
from collections.abc import Mapping
from types import BuiltinFunctionType, FunctionType
from typing import Any


def _encode(obj: Any) -> Any:
    """Encodes a non-JSON value for ``json.dumps``.

    Module-level classes and functions become
    ``{"__type__": "<module>.<qualname>"}``. Mappings that are not dicts
    become dicts.

    The path must resolve back to ``obj`` in a fresh process, so that
    ``from_dict`` can rebuild it. That rules out classes and functions
    defined in ``__main__`` (a script run directly or a notebook), classes
    made with ``type()`` under a name they aren't bound to, objects whose
    name was later rebound to something else, and built-in types whose
    ``__module__`` and ``__qualname__`` don't name them.

    Args:
        obj: The value ``json.dumps`` could not serialize.

    Returns:
        A JSON-compatible stand-in for ``obj``.

    Raises:
        TypeError: If ``obj`` is not a mapping or a module-level class or
            function whose path resolves back to it.
    """
    if _is_type_ref(obj):
        qualname = obj.__qualname__
        module = obj.__module__
        if not module or "<" in qualname:
            raise TypeError(
                f"{obj!r} is not a module-level class or function and cannot "
                "be serialized by import path."
            )
        if module == "__main__":
            raise TypeError(
                f"{qualname} is defined in __main__ (a script run directly or "
                "a notebook), so it can't be imported from another process. "
                "Move it into an importable module and import it from there."
            )
        path = f"{module}.{qualname}"
        try:
            resolved = _resolve(path)
        except ValueError:
            resolved = None
        if resolved is not obj:
            raise TypeError(
                f"{obj!r} is serialized as {path!r}, but that path does not "
                "resolve back to it. Bind it to a module-level name matching "
                "its __qualname__."
            )
        return {"__type__": path}
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not allowed in Evaluation "
        "params. Use JSON types or module-level classes and functions."
    )


def _is_type_ref(obj: Any) -> bool:
    """Returns whether ``obj`` is a class or function."""
    return isinstance(obj, (type, FunctionType, BuiltinFunctionType))


def _check_keys(obj: Any) -> None:
    """Checks every mapping key anywhere in ``obj``.

    Keys must be strings, because ``json.dumps`` silently turns ``1`` into
    ``"1"``, which would give ``{1: x}`` and ``{"1": x}`` the same id. The
    key ``"__type__"`` is reserved for class and function references, so
    allowing it in user data would make ``to_dict`` output ambiguous.

    Raises:
        TypeError: If a key is not a string.
        ValueError: If a key is ``"__type__"``.
    """
    if isinstance(obj, Mapping):
        for key in obj:
            if not isinstance(key, str):
                raise TypeError(
                    f"Evaluation params keys must be strings, got {key!r} of "
                    f"type {type(key).__name__}."
                )
        if "__type__" in obj:
            raise ValueError(
                '"__type__" is reserved for class and function references and '
                "cannot be used as a key in Evaluation params."
            )
        for value in obj.values():
            _check_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _check_keys(value)


def _resolve(path: str) -> Any:
    """Resolves a dotted path to the class or function it names.

    Imports the longest importable module prefix of ``path``, then resolves
    the remaining parts with ``getattr``, so nested classes such as
    ``pkg.mod.Outer.Inner`` work.

    Args:
        path: Dotted import path, e.g. ``"json.decoder.JSONDecoder"``.

    Returns:
        The class or function at ``path``.

    Raises:
        ValueError: If ``path`` can't be resolved, or names something that is
            not a class or function.
    """
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as exc:
            # Only skip prefixes that are themselves missing. A missing
            # dependency inside an existing module is a real error.
            if exc.name is None or not (
                prefix == exc.name or prefix.startswith(exc.name + ".")
            ):
                raise
            continue
        except ValueError:  # Empty or malformed module name.
            break
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            break
        if not _is_type_ref(obj):
            raise ValueError(f"{path!r} does not name a class or function.")
        return obj
    raise ValueError(f"Cannot resolve {path!r} to a class or function.")


def _decode(obj: Any) -> Any:
    """Rebuilds params from ``to_dict`` output, resolving ``__type__`` refs."""
    if isinstance(obj, Mapping):
        if "__type__" in obj:
            path = obj["__type__"]
            if len(obj) != 1:
                raise ValueError(
                    f"Type reference {path!r} must be the only key in its "
                    f"mapping, got keys {sorted(obj)}."
                )
            if not isinstance(path, str):
                raise ValueError(f"Type reference {path!r} must be a string.")
            return _resolve(path)
        return {key: _decode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_decode(value) for value in obj]
    return obj


def _dumps(params: Mapping[str, Any]) -> str:
    """Serializes params canonically: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_encode,
    )
