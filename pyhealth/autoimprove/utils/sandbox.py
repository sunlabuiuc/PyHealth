"""The sandbox: how code a proposer wrote gets run without endangering the search.

Code, evaluation and analysis turns produce Python source written by an LLM.
This module is the only place that runs it. In plain words:

**What it is.** Not a separate environment or a container: the code runs in
the same Python (same conda env, same packages) as the improver, but in a
*child process*, a copy of Python that already has torch and PyHealth loaded.
If the code crashes, hangs or eats memory, only the child dies; the search
carries on and records the failure.

**Before it runs**, :func:`static_check` reads the code without running it and
rejects:

- imports outside :data:`ALLOWED_IMPORTS` (torch, numpy, scipy, sklearn,
  pandas, polars, pyhealth and a few standard-library modules), and the
  submodules in :data:`BLOCKED_MODULES`;
- file, shell and process access (``open``, ``os``, ``subprocess``,
  ``torch.save``/``load``, ``read_csv``...: :data:`BLOCKED_NAMES` and
  :data:`BLOCKED_ATTRIBUTES`), ``exec``/``eval``, and ``__dunder__``
  attributes other than ``__init__``.

**While it runs** (:func:`run_sandboxed`), the child has:

- no network: it moves into its own Linux user and network namespace, which
  holds only a loopback interface (where the kernel allows this without root;
  GPUs stay usable);
- no credentials: environment variables that look like keys or tokens are
  removed;
- a wall-clock limit, after which it is killed, a 64 MB cap on files it
  writes, and optionally a memory cap;
- its printed output sent to a log file, never to the LLM.

**What comes back** is only what the harness function (the *runner*) returns:
predictions, numbers, or tables with small groups withheld. Errors come back
as text, in two forms: in full for the run's log, and without the message
(which can quote a raw value) for the proposer's prompt.

**What it does not do.** The child still runs as your user and can read what
you can, for example through a library function the static check doesn't know
about. It stops mistakes, runaway code and accidental file, network or
credential use, not a deliberate attacker. With an LLM you don't trust, run
the whole improver in a container or a job that can only see the data it
needs.

Jobs start in a fraction of a second after the first, because children are
forked from a server process that imported torch and PyHealth once. As with
any ``multiprocessing`` code, a script using these turns needs an
``if __name__ == "__main__":`` guard.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import linecache
import logging
import multiprocessing
import os
import pickle
import re
import sys
import time
from collections.abc import Sequence
from typing import Any

from .data import describe_error

logger = logging.getLogger(__name__)

ALLOWED_IMPORTS = frozenset({
    "torch", "numpy", "scipy", "sklearn", "pandas", "polars", "pyhealth",
    "xgboost", "math", "statistics", "collections", "itertools", "functools",
    "operator", "typing", "dataclasses", "copy", "random", "enum", "abc",
    "warnings", "heapq", "bisect", "numbers", "string", "time",
})
"""Top-level modules generated code may import."""

BLOCKED_MODULES = (
    "torch.hub", "torch.utils.cpp_extension", "torch.package", "torch.distributed",
    "torch.multiprocessing", "torch.utils.data.datapipes", "numpy.ctypeslib",
    "numpy.f2py", "numpy.distutils", "sklearn.datasets", "pandas.io",
    "scipy.io", "pyhealth.autoimprove",
)
"""Submodules of allowed packages that read, write or fetch data, or start processes."""

BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "memoryview", "help", "exit", "quit",
})
"""Builtins generated code may not call or refer to."""

BLOCKED_ATTRIBUTES = frozenset({
    "load", "save", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "to_csv", "to_pickle", "to_parquet", "to_json", "to_excel",
    "to_sql", "to_hdf", "to_feather", "read_csv", "read_pickle", "read_parquet",
    "read_json", "read_excel", "read_sql", "read_hdf", "read_feather",
    "write_csv", "write_parquet", "scan_csv", "scan_parquet", "system", "popen",
    "spawn", "fork", "load_library", "dlopen", "hub", "cpp_extension",
    "save_model", "load_model", "dump", "dumps_model",
})
"""Attribute names that read or write files, or start processes, on any object."""

ALLOWED_DUNDERS = frozenset({"__init__", "__name__"})

_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", re.IGNORECASE)

_PRELOAD = [
    "numpy", "torch", "pyhealth.datasets", "pyhealth.models", "pyhealth.trainer",
    "pyhealth.metrics", "pyhealth.autoimprove.utils.sandbox",
    "pyhealth.autoimprove.evaluation.training",
    "pyhealth.autoimprove.evaluation.guardrails", "pyhealth.autoimprove.search.analysis",
]

_CONTEXT = None

_NETWORK_ISOLATED = False
"""In a child: whether it was moved to a network namespace with no network."""


class ContractError(ValueError):
    """Generated code broke its contract, found by a harness check.

    Its message is written by the harness and never quotes data, so unlike
    other errors it is shown to the proposer in full.

    Examples:
        >>> str(ContractError("build_and_train must return a BaseModel"))
        'build_and_train must return a BaseModel'
    """


class SandboxError(RuntimeError):
    """Generated code failed in its process.

    Args:
        full: The error with its message, for the run's log.
        brief: The error without its message (which can quote a raw value
            from a sample), for a proposer's prompt.

    Examples:
        >>> e = SandboxError("ValueError: bad value 7 at x.py:3", "ValueError at x.py:3")
        >>> e.brief
        'ValueError at x.py:3'
    """

    def __init__(self, full: str, brief: str):
        super().__init__(full)
        self.full = full
        self.brief = brief


def static_check(code: str, entry: str) -> list[str]:
    """Problems that stop ``code`` from running; empty if it may run.

    The code must parse, define a function ``entry`` at module level, import
    only from :data:`ALLOWED_IMPORTS` (and none of :data:`BLOCKED_MODULES`),
    and use none of :data:`BLOCKED_NAMES`, :data:`BLOCKED_ATTRIBUTES` or
    dunder attributes other than ``__init__``. ``getattr``, ``setattr``,
    ``delattr`` and ``hasattr`` need a literal attribute name that passes the
    same check.

    Examples:
        >>> static_check("import os\\ndef evaluate(y, p, info):\\n    return {}", "evaluate")
        ['line 1: import of os is not allowed']
        >>> static_check("import numpy as np\\ndef evaluate(y, p, info):\\n    return {}", "evaluate")
        []
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"line {e.lineno}: syntax error: {e.msg}"]
    problems: list[str] = []

    def bad_module(name: str, line: int) -> None:
        root = name.split(".")[0]
        blocked = any(name == m or name.startswith(m + ".") for m in BLOCKED_MODULES)
        if root not in ALLOWED_IMPORTS or blocked:
            problems.append(f"line {line}: import of {name} is not allowed")

    def bad_attribute(name: str, line: int) -> None:
        dunder = name.startswith("__") and name.endswith("__")
        if dunder and name not in ALLOWED_DUNDERS:
            problems.append(f"line {line}: attribute {name} is not allowed")
        elif name in BLOCKED_ATTRIBUTES:
            problems.append(
                f"line {line}: .{name} reads or writes files and is not allowed")

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                bad_module(alias.name, line)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"line {line}: relative imports are not allowed")
                continue
            bad_module(node.module or "", line)
            for alias in node.names:
                if alias.name == "*":
                    problems.append(f"line {line}: import * is not allowed")
                else:
                    # e.g. "from torch import hub" or "from numpy import load"
                    bad_module(f"{node.module}.{alias.name}", line)
                    bad_attribute(alias.name, line)
        elif isinstance(node, ast.Name):
            dunder = (node.id.startswith("__") and node.id.endswith("__")
                      and node.id not in ALLOWED_DUNDERS)
            if node.id in BLOCKED_NAMES or dunder:
                problems.append(f"line {line}: {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            bad_attribute(node.attr, line)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (
            node.func.id in ("getattr", "setattr", "delattr", "hasattr")
        ):
            if len(node.args) < 2 or not (
                isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                problems.append(
                    f"line {line}: {node.func.id} needs a literal attribute name")
            else:
                bad_attribute(node.args[1].value, line)
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body):
        problems.append(f"the code must define a function {entry}(...) at module level")
    return sorted(set(problems), key=lambda p: (_line_of(p), p))


def _line_of(problem: str) -> int:
    match = re.match(r"line (\d+):", problem)
    return int(match.group(1)) if match else 10**9


def check_or_raise(code: str, entry: str) -> None:
    """Raises ``ValueError`` listing every problem :func:`static_check` finds.

    Examples:
        >>> check_or_raise("import os", "build_and_train")  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: line 1: import of os is not allowed; ...
    """
    problems = static_check(code, entry)
    if problems:
        raise ValueError("the code was rejected before running: " + "; ".join(problems))


def _context():
    global _CONTEXT
    if _CONTEXT is None:
        methods = multiprocessing.get_all_start_methods()
        method = "forkserver" if "forkserver" in methods else "spawn"
        _CONTEXT = multiprocessing.get_context(method)
        if method == "forkserver":
            _CONTEXT.set_forkserver_preload(_PRELOAD)
    return _CONTEXT


def run_sandboxed(
    code: str,
    runner: str,
    payload: dict[str, Any],
    timeout: float = 600.0,
    log_path: str | None = None,
    memory_gb: float | None = None,
    label: str = "generated",
) -> Any:
    """Runs ``code`` in a child process and returns what ``runner`` returns.

    The child executes ``code`` as a fresh module, then calls the function
    named by ``runner`` (``"package.module:function"``) with the module's
    namespace and ``payload``. Call :func:`static_check` first; this function
    does not.

    Args:
        code: The generated source.
        runner: Import path of the harness function that uses the module,
            e.g. ``"pyhealth.autoimprove.evaluation.training:run_code_trial"``.
        payload: Arguments for the runner; must pickle.
        timeout: Seconds before the child is killed.
        log_path: File for the child's stdout and stderr; discarded if None.
        memory_gb: Address-space limit for the child (CPU runs; CUDA reserves
            far more address space than it uses, so leave this unset on GPU).
        label: Name of the code in tracebacks, e.g. ``"trial_12"``.

    Raises:
        SandboxError: If the code raised, the child died, or it timed out.

    Examples:
        >>> run_sandboxed(code, "pyhealth.autoimprove.evaluation.guardrails:run_evaluation",
        ...               payload, timeout=60)  # doctest: +SKIP
    """
    ctx = _context()
    receive, send = ctx.Pipe(duplex=False)
    # Plain pickle copies tensors' bytes. multiprocessing's own pickler would
    # share each tensor's memory through a file descriptor, and a process can
    # be handed only a few hundred of those.
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    process = ctx.Process(
        target=_child, args=(code, runner, blob, send, log_path, memory_gb, label),
        daemon=True,
    )
    start = time.perf_counter()
    process.start()
    send.close()
    try:
        if not receive.poll(timeout):
            process.kill()
            process.join(5)
            text = (f"TimeoutError: the code ran longer than {timeout:.0f}s "
                    "and was stopped")
            raise SandboxError(text, text)
        try:
            status, *rest = receive.recv()
        except EOFError:
            process.join(5)
            code_ = process.exitcode
            text = (f"ProcessExit: the code's process died (exit code {code_}); "
                    "it may have run out of memory")
            raise SandboxError(text, text) from None
    finally:
        receive.close()
        if process.is_alive():
            process.join(5)
            if process.is_alive():
                process.kill()
    logger.debug("%s ran in %.1fs", label, time.perf_counter() - start)
    if status == "error":
        raise SandboxError(*rest)
    return pickle.loads(rest[0])


def _child(code, runner, blob, send, log_path, memory_gb, label) -> None:
    """The child process: limits, then the module, then the runner."""
    try:
        _limit(log_path, memory_gb)
        payload = pickle.loads(blob)
        filename = f"<{label}>"
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        namespace: dict[str, Any] = {"__name__": label}
        exec(compile(code, filename, "exec"), namespace)  # noqa: S102 - checked code
        module, _, function = runner.partition(":")
        result = getattr(importlib.import_module(module), function)(namespace, payload)
        send.send(("ok", pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)))
    except BaseException as e:  # noqa: BLE001 - every failure is reported
        brief = (f"ContractError: {e}" if isinstance(e, ContractError)
                 else describe_error(e, message=False))
        with contextlib.suppress(Exception):  # e.g. the pipe is gone
            send.send(("error", describe_error(e), brief))
    finally:
        send.close()


def _limit(log_path: str | None, memory_gb: float | None) -> None:
    for name in list(os.environ):
        if _SECRET.search(name):
            del os.environ[name]
    _isolate_network()
    target = log_path or os.devnull
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        import resource

        mb = 1 << 20
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * mb, 64 * mb))
        if memory_gb:
            limit = int(memory_gb * (1 << 30))
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):  # not on this platform
        pass


def _isolate_network() -> None:
    """Moves this process to new user and network namespaces: no network.

    Needs Linux with unprivileged user namespaces; elsewhere the process
    keeps the network. Devices such as GPUs stay usable.
    """
    global _NETWORK_ISOLATED
    flags = getattr(os, "CLONE_NEWUSER", 0) | getattr(os, "CLONE_NEWNET", 0)
    if not flags or not hasattr(os, "unshare"):
        return
    try:
        os.unshare(flags)
        _NETWORK_ISOLATED = True
    except OSError:  # namespaces not allowed here
        pass


def _network_probe(namespace: dict[str, Any], payload: dict[str, Any]) -> dict:
    """Runner reporting the child's isolation, for tests."""
    import socket

    names = sorted(name for _, name in socket.if_nameindex())
    return {"isolated": _NETWORK_ISOLATED, "interfaces": names}


def warm_up() -> None:
    """Starts the process server now, so the first job doesn't pay for it.

    Examples:
        >>> warm_up()  # doctest: +SKIP
    """
    run_sandboxed("def main(ns, payload):\n    return 1\n",
                  "pyhealth.autoimprove.utils.sandbox:_noop", {}, timeout=300)


def _noop(namespace: dict[str, Any], payload: dict[str, Any]) -> int:
    return 1


def sample_patient_ids(ids: Sequence[Any]) -> list[str]:
    """Patient ids as short one-way hashes, for code that should not see them.

    Examples:
        >>> sample_patient_ids(["p1", "p1", "p2"])[0] == sample_patient_ids(["p1"])[0]
        True
    """
    import hashlib

    return [hashlib.sha256(str(i).encode()).hexdigest()[:12] for i in ids]
