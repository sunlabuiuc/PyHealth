#!/usr/bin/env python3
"""Select the tests a change can affect, so PR CI runs those, not everything.

The whole suite grows with every dataset, model and task; a PR touching one
model should not pay for all of them. This script maps changed files to the
test modules that (transitively) import them, using a static import graph
rebuilt from source on every run — nothing is cached or stored, and parsing
the whole package takes well under a second.

Why symbol-level: tests import through re-exporting package ``__init__``s
(``from pyhealth.models import RNN``), and those ``__init__``s eagerly import
every submodule. A module-level graph would say every test depends on every
file. So an imported name is resolved through the ``__init__``'s bindings to
the module that *defines* it, and the ``__init__`` itself becomes a *shallow*
dependency: editing that file selects the test, but its other imports do not
propagate through it.

What that gives up — an import-time error in an unrelated submodule still
breaks ``import pyhealth.models`` for everyone — is covered by the smoke step
``--run`` performs first (compile + import every subpackage).

Fail-safe defaults: build/CI config, ``tests/base.py``, this script, and
anything ``import pyhealth`` itself loads select ALL tests; so does any path
the rules below do not recognise.

Usage:
    python tools/select_tests.py --base origin/master        # list selection
    python tools/select_tests.py --files pyhealth/models/rnn.py
    python tools/select_tests.py --base origin/master --run  # smoke + tests
    python tools/select_tests.py --all --run                 # full suite
    python tools/select_tests.py ... --github-output         # CI outputs
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "pyhealth"
TESTS = "tests"
# Test directories CI runs (under tests/) -> pixi environment that runs them.
SUITES = {"core": "test", "nlp": "nlp"}

ALL = "all"

# Changing any of these can affect every test.
RUN_ALL_FILES = {
    "pyproject.toml",
    "pixi.lock",
    "makefile",
    "hatch_build.py",
    "requirements.txt",
    ".github/workflows/test.yml",
    "tools/select_tests.py",
    "tests/__init__.py",
    "tests/base.py",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
}

# Non-Python paths outside the package that tests never read.
IGNORE_PREFIXES = (
    "docs/", "examples/", "leaderboard/", "figure/", "hackthon/",
    "chat-assistant/", ".claude/", ".github/",
)
IGNORE_FILES = {
    "README.rst", "LICENSE", "LICENSE.md", ".gitignore", "readthedocs.yml",
    "bounty.yaml",  # bounty board data, read only by its own workflow
}
DOC_SUFFIXES = {".md", ".rst", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf"}

# Directories whose files tests never import or read, unless a test names the
# file explicitly (checked first).
LOOSE_IGNORE_PREFIXES = ("tools/",)

# Non-Python trees read at runtime by specific modules: a change counts as a
# change to those modules. Name the readers, not their package, or the change
# leaks through every importer of the package.
OWNED_TREES = {
    # Shipped as pyhealth/skills/_bundle; only the installer reads it.
    "skills/": ["tests.core.test_skills_install"],
}


@dataclass
class Module:
    name: str
    path: str  # repo-relative, posix
    is_pkg: bool
    source: str
    # Top-level name -> (module, attribute or None for a module binding).
    bindings: dict = field(default_factory=dict)
    star_sources: list = field(default_factory=list)
    # (target module, imported name or None, is_star) for every import.
    imports: list = field(default_factory=list)


def _module_name(rel: str) -> str:
    parts = rel[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _absolute(mod: Module, level: int, target: str | None) -> str:
    if not level:
        return target or ""
    base = mod.name.split(".")
    if not mod.is_pkg:
        base = base[:-1]
    if level > 1:
        base = base[: -(level - 1)]
    return ".".join(base + ([target] if target else []))


def _binding_only(node: ast.stmt) -> bool:
    """Import, ``__all__`` assignment, or docstring: changes to these in an
    ``__init__`` only matter to importers of the names involved."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return True
    targets = (node.targets if isinstance(node, ast.Assign)
               else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
               else [])
    return bool(targets) and all(isinstance(t, ast.Name) and t.id == "__all__"
                                 for t in targets)


def _literal_dynamic_import(node: ast.Call) -> str | None:
    """``importlib.import_module("x")`` / ``__import__("x")`` with a literal."""
    func = node.func
    named = (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}) or (
        isinstance(func, ast.Attribute) and func.attr == "import_module"
    )
    if named and node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str) and not value.startswith("."):
            return value
    return None


class Graph:
    def __init__(self, root: Path = REPO_ROOT, package: str = PACKAGE,
                 tests: str = TESTS):
        self.root = root
        self.package = package
        self.tests = tests
        self.modules: dict[str, Module] = {}
        self.by_path: dict[str, Module] = {}
        self.deep_rdeps: dict[str, set] = defaultdict(set)
        self.shallow_rdeps: dict[str, set] = defaultdict(set)
        for top in (package, tests):
            for path in sorted((root / top).rglob("*.py")):
                rel = path.relative_to(root).as_posix()
                if "__pycache__" in rel:
                    continue
                source = path.read_text(encoding="utf-8", errors="replace")
                mod = Module(_module_name(rel), rel, path.name == "__init__.py", source)
                self.modules[mod.name] = mod
                self.by_path[rel] = mod
        for mod in self.modules.values():
            self._parse(mod)
        for mod in self.modules.values():
            self._link(mod)
        # Everything `import pyhealth` executes; changing any of it is ALL.
        self.root_modules = self._closure(package)

    # -- parsing -----------------------------------------------------------

    def internal(self, name: str) -> bool:
        return name.split(".")[0] in (self.package, self.tests)

    def _parse(self, mod: Module) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # e.g. invalid escapes
                tree = ast.parse(mod.source, filename=mod.path)
        except SyntaxError:
            return  # the smoke step reports it
        self._collect_bindings(mod, tree.body)
        plain = []  # (imported module, local name it binds, prefix that name means)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        plain.append((alias.name, alias.asname, alias.name))
                    else:
                        root = alias.name.split(".")[0]
                        plain.append((alias.name, root, root))
            elif isinstance(node, ast.ImportFrom):
                target = _absolute(mod, node.level, node.module)
                for alias in node.names:
                    if alias.name == "*":
                        mod.imports.append((target, None, True))
                    else:
                        mod.imports.append((target, alias.name, False))
            elif isinstance(node, ast.Call):
                target = _literal_dynamic_import(node)
                if target:
                    mod.imports.append((target, None, False))
        if plain:
            self._attribute_imports(mod, tree, plain)

    def _attribute_imports(self, mod: Module, tree: ast.AST, plain: list) -> None:
        """``import pyhealth.medcode as medcode`` then ``medcode.InnerMap``:
        depend on what the attributes resolve to, not the whole package —
        unless the name is also used bare (passed around, ``getattr``...)."""
        inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        chains: dict[str, set] = defaultdict(set)  # local name -> dotted rests
        bare: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and id(node) not in inner:
                parts, cur = [], node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    chains[cur.id].add(".".join(reversed(parts)))
            elif isinstance(node, ast.Name) and id(node) not in inner:
                bare.add(node.id)
        for imported, local, prefix in plain:
            if not self.internal(imported) or local in bare or not chains[local]:
                mod.imports.append((imported, None, False))
                continue
            for rest in chains[local]:
                parts = f"{prefix}.{rest}".split(".")
                # Longest known-module prefix, then the attribute read from it.
                for i in range(len(parts), 0, -1):
                    head = ".".join(parts[:i])
                    if head in self.modules:
                        name = parts[i] if i < len(parts) else None
                        mod.imports.append((head, name, False))
                        break
                else:
                    mod.imports.append((imported, None, False))

    def _collect_bindings(self, mod: Module, body: list) -> None:
        """Names bound by imports at module level (incl. if/try blocks and a
        lazy module ``__getattr__``), for resolving re-exports."""
        for node in body:
            if isinstance(node, ast.ImportFrom):
                target = _absolute(mod, node.level, node.module)
                for alias in node.names:
                    if alias.name == "*":
                        mod.star_sources.append(target)
                    else:
                        mod.bindings.setdefault(alias.asname or alias.name,
                                                (target, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        mod.bindings.setdefault(alias.asname, (alias.name, None))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    self._collect_bindings(mod, getattr(node, attr, []) or [])
            elif isinstance(node, ast.ExceptHandler):
                self._collect_bindings(mod, node.body)
            elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                inner = [n for n in ast.walk(node) if isinstance(n, ast.ImportFrom)]
                self._collect_bindings(mod, inner)

    # -- linking -----------------------------------------------------------

    def _resolve(self, target: str, name: str | None, seen=frozenset()):
        """Return (shallow, deep) module sets that ``from target import name``
        (or ``import target`` when name is None) depends on."""
        if not self.internal(target):
            return set(), set()
        if target not in self.modules or name is None:
            return set(), {target}  # missing => phantom node (deleted module)
        mod = self.modules[target]
        if not mod.is_pkg:
            return set(), {target}
        shallow, deep = {target}, set()
        sub = f"{target}.{name}"
        if sub in self.modules:
            deep.add(sub)
        if name in mod.bindings and (target, name) not in seen:
            t_mod, t_name = mod.bindings[name]
            s, d = self._resolve(t_mod, t_name, seen | {(target, name)})
            shallow |= s
            deep |= d
        if not deep:
            # Defined in the __init__ itself, star-imported, or unknown:
            # depend on the whole package.
            deep.add(target)
        return shallow, deep

    def _link(self, mod: Module) -> None:
        for target, name, star in mod.imports:
            if star:
                shallow, deep = set(), ({target} if self.internal(target) else set())
            else:
                shallow, deep = self._resolve(target, name)
            # Parent packages' __init__s execute on import.
            for dep in list(deep) + list(shallow):
                parts = dep.split(".")
                shallow |= {".".join(parts[:i]) for i in range(1, len(parts))}
            for dep in deep:
                if dep != mod.name:
                    self.deep_rdeps[dep].add(mod.name)
            for dep in shallow - deep:
                if dep != mod.name:
                    self.shallow_rdeps[dep].add(mod.name)

    def _closure(self, start: str) -> set:
        """Every module importing ``start`` executes (deep and shallow)."""
        forward = defaultdict(set)
        for rdeps in (self.deep_rdeps, self.shallow_rdeps):
            for dep, users in rdeps.items():
                for user in users:
                    forward[user].add(dep)
        seen, queue = {start}, deque([start])
        while queue:
            for dep in forward[queue.popleft()]:
                if dep not in seen:
                    seen.add(dep)
                    queue.append(dep)
        return seen

    # -- selection ---------------------------------------------------------

    def affected(self, changed: set, deep_only=frozenset()) -> dict:
        """Module -> the module it was reached from (None for a seed).
        Seeds in ``deep_only`` skip their shallow importers."""
        parent = {c: None for c in changed}
        for c in changed - set(deep_only):
            for user in self.shallow_rdeps.get(c, ()):
                parent.setdefault(user, c)
        queue = deque(parent)
        while queue:
            node = queue.popleft()
            for user in self.deep_rdeps.get(node, ()):
                if user not in parent:
                    parent[user] = node
                    queue.append(user)
        return parent

    def is_suite_test(self, name: str) -> bool:
        parts = name.split(".")
        return (len(parts) == 3 and parts[0] == self.tests
                and parts[1] in SUITES and parts[2].startswith("test_")
                # Names reach a shell in CI; only plain identifiers pass.
                and all(p.isidentifier() for p in parts))

    def changed_bindings(self, pkg: str, old_source: str | None):
        """Names whose binding differs between ``old_source`` and the current
        ``__init__`` of ``pkg``, or None if anything besides import bindings
        (and ``__all__`` / the docstring) changed."""
        if old_source is None or pkg not in self.modules:
            return None
        sides = []
        for source in (old_source, self.modules[pkg].source):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree = ast.parse(source)
            except SyntaxError:
                return None
            probe = Module(pkg, self.modules[pkg].path, True, source)
            self._collect_bindings(probe, tree.body)
            other = [ast.dump(n) for n in tree.body if not _binding_only(n)]
            sides.append((probe.bindings, probe.star_sources, other))
        (old_b, old_star, old_other), (new_b, new_star, new_other) = sides
        if old_other != new_other or old_star != new_star:
            return None
        return {n for n in old_b.keys() | new_b.keys() if old_b.get(n) != new_b.get(n)}

    def users_of(self, pkg: str, names: set) -> set:
        """Modules importing any of ``names`` from ``pkg`` (or ``*``)."""
        return {m.name for m in self.modules.values()
                for target, name, star in m.imports
                if target == pkg and (star or name in names)}

    def referencing(self, token: str) -> set:
        return {m.name for m in self.modules.values() if token in m.source}


@dataclass
class Selection:
    mode: str  # "none" | "some" | "all"
    tests: list = field(default_factory=list)  # dotted module names
    reasons: list = field(default_factory=list)

    def suite(self, suite: str) -> list:
        """Selected test files of one suite, as paths pytest accepts."""
        return [t.replace(".", "/") + ".py" for t in self.tests
                if t.split(".")[1] == suite]


def _classify(path: str, graph: Graph):
    """Return ALL, or a (possibly empty) set of changed module names."""
    if path in RUN_ALL_FILES:
        return ALL
    parts = path.split("/")
    if parts[-1] == "conftest.py":
        return ALL  # pytest fixtures/hooks reach every test below them
    if (parts[0] == graph.tests and parts[-1] == "__init__.py"
            and len(parts) == 3 and parts[1] in SUITES):
        return ALL
    if path.endswith(".py") and parts[0] in (graph.package, graph.tests):
        name = _module_name(path)
        if name in graph.root_modules:
            return ALL  # loaded by `import pyhealth` itself
        return {name}
    if parts[0] == graph.package and Path(path).suffix.lower() in DOC_SUFFIXES:
        return set()  # package READMEs etc.
    for prefix, owner in OWNED_TREES.items():
        if path.startswith(prefix):
            return set(owner)
    if parts[0] not in (graph.package, graph.tests, "test-resources"):
        if (path in IGNORE_FILES or path.startswith(IGNORE_PREFIXES)
                or Path(path).suffix.lower() in DOC_SUFFIXES):
            return set()
    # Data or config file: whichever modules name it.
    if parts[0] == "test-resources":
        rest = parts[1:]
        if rest and rest[0] in SUITES and len(rest) > 1:
            rest = rest[1:]
        tokens = [rest[0]] if rest else []
    else:
        p = Path(path)
        # The bare stem (a .pyx imported as a module) only inside the
        # package; elsewhere it matches unrelated identifiers too often.
        stem = parts[0] == graph.package and len(p.stem) >= 8
        tokens = [p.name] + ([p.stem] if stem else [])
    for token in tokens:
        hits = graph.referencing(token)
        if hits:
            return hits
    if path.startswith(LOOSE_IGNORE_PREFIXES):
        return set()
    return ALL


def select(paths: list, graph: Graph, base_source=None) -> Selection:
    """``base_source(path)`` returns a file's pre-change text (None if new);
    with it, an ``__init__`` edit that only adds/removes re-exports selects
    the importers of those names instead of every importer of the package."""
    changed: dict[str, str] = {}  # module -> path that changed it
    narrowed: set = set()
    for path in paths:
        result = _classify(path, graph)
        if result == ALL:
            return Selection(ALL, reasons=[f"{path} -> runs everything"])
        for name in result:
            changed.setdefault(name, path)
            if base_source and path.endswith("/__init__.py"):
                names = graph.changed_bindings(name, base_source(path))
                if names is not None:
                    narrowed.add(name)
                    for user in graph.users_of(name, names):
                        changed.setdefault(user, f"{path} [re-exports changed]")
    parent = graph.affected(set(changed), deep_only=narrowed)
    tests = sorted(n for n in parent if graph.is_suite_test(n))
    reasons = []
    for test in tests:
        chain, node = [], test
        while node is not None:
            chain.append(node)
            node = parent[node]
        reasons.append(f"{test}  <-  " + " <- ".join(chain[1:] + [changed[chain[-1]]])
                       if len(chain) > 1 else f"{test}  <-  {changed[test]}")
    return Selection("some" if tests else "none", tests, reasons)


# -- inputs / outputs ------------------------------------------------------


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout


def _show(rev: str, path: str) -> str | None:
    result = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def changed_files(base: str) -> list:
    """Files changed since the merge base with ``base``, including
    uncommitted and untracked ones (handy locally; a no-op in CI)."""
    merge_base = _git("merge-base", base, "HEAD").strip()
    out = _git("diff", "--name-only", "--no-renames", merge_base).splitlines()
    # New, not-yet-added sources count; caches and other local clutter don't.
    out += [p for p in _git("ls-files", "--others", "--exclude-standard").splitlines()
            if p.startswith((f"{PACKAGE}/", f"{TESTS}/", "test-resources/"))
            and "__pycache__" not in p and not p.endswith(".pyc")]
    return sorted({line.strip() for line in out if line.strip()})


def suite_total(graph: Graph) -> int:
    return sum(1 for name in graph.modules if graph.is_suite_test(name))


def report(sel: Selection, n_files: int, total: int, stream=sys.stderr) -> None:
    if sel.mode == ALL:
        print(f"select_tests: running ALL tests ({sel.reasons[0]})", file=stream)
    elif sel.mode == "none":
        print(f"select_tests: {n_files} changed file(s) affect no tests", file=stream)
    else:
        print(f"select_tests: {len(sel.tests)}/{total} test modules affected "
              f"by {n_files} changed file(s):", file=stream)
        for line in sel.reasons:
            print(f"  {line}", file=stream)


def github_output(sel: Selection, n_files: int, total: int) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"mode={sel.mode}\n")
            f.write(f"selected={total if sel.mode == ALL else len(sel.tests)}\n")
            f.write(f"suite_total={total}\n")
            for suite in SUITES:
                paths = [f"{TESTS}/{suite}"] if sel.mode == ALL else sel.suite(suite)
                f.write(f"{suite}={' '.join(paths)}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## Test selection\n\n")
            if sel.mode == ALL:
                f.write(f"Running **all** tests: `{sel.reasons[0]}`\n")
            elif sel.mode == "none":
                f.write(f"{n_files} changed file(s) affect no tests.\n")
            else:
                f.write(f"{len(sel.tests)} test module(s) selected:\n\n```\n")
                f.write("\n".join(sel.reasons) + "\n```\n")


def smoke_imports(graph: Graph, suite: str = "core") -> str:
    """Import every top-level subpackage the suite's tests import, i.e. the
    ones its environment is known to be able to import."""
    used = set()
    for name, mod in graph.modules.items():
        if graph.is_suite_test(name) and name.split(".")[1] == suite:
            for target, _, _ in mod.imports:
                parts = target.split(".")
                if parts[0] == graph.package and len(parts) > 1:
                    top = ".".join(parts[:2])
                    if top in graph.modules:
                        used.add(top)
    return "import " + ", ".join([graph.package, *sorted(used)])


def run(sel: Selection, graph: Graph, pytest_args=()) -> int:
    def sh(*cmd) -> int:
        print("+ " + " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode

    if sel.mode == "none":
        return 0
    # Symbol-level selection skips tests whose only link to a change is an
    # eager __init__; catch import-time breakage everywhere instead.
    for cmd in (("python", "-m", "compileall", "-q", PACKAGE),
                ("python", "-c", smoke_imports(graph))):
        if sh("pixi", "run", "-e", "test", *cmd):
            return 1
    status = 0
    for suite, env in SUITES.items():  # same paths --github-output writes
        tests = [f"{TESTS}/{suite}"] if sel.mode == ALL else sel.suite(suite)
        if tests:
            status |= sh("pixi", "run", "-e", env, "pytest", "-v", *pytest_args, *tests)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="git ref to diff against (merge base)")
    src.add_argument("--files", nargs="+", help="explicit changed paths")
    src.add_argument("--all", action="store_true", help="select every test")
    parser.add_argument("--reason", default="--all",
                        help="why --all was chosen, for the log and summary")
    parser.add_argument("--github-output", action="store_true",
                        help="write mode/test lists to $GITHUB_OUTPUT")
    parser.add_argument("--smoke", action="store_true",
                        help="print the smoke-check import statement and exit")
    parser.add_argument("--run", action="store_true",
                        help="run the smoke check and the selected tests")
    args = parser.parse_args(argv)

    graph = Graph()
    if args.all:
        files, sel = [], Selection(ALL, reasons=[args.reason])
    else:
        files = args.files or changed_files(args.base)
        base_source = None
        if args.base:
            merge_base = _git("merge-base", args.base, "HEAD").strip()
            base_source = lambda path: _show(merge_base, path)  # noqa: E731
        sel = select(files, graph, base_source)
    total = suite_total(graph)
    report(sel, len(files), total)
    if args.github_output:
        github_output(sel, len(files), total)
    if args.smoke:
        print(smoke_imports(graph))
        return 0
    if args.run:
        return run(sel, graph)
    if not args.github_output:
        print("\n".join(sel.tests) if sel.mode != ALL else ALL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
