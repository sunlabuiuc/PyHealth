"""pyhealth.autoimprove's layout: search/, evaluation/ and utils/ depend on each
other only in allowed directions, with no import cycles."""

import ast
import importlib
import unittest
from pathlib import Path

import pyhealth.autoimprove as autoimprove

PKG = Path(autoimprove.__file__).parent
PREFIX = "pyhealth.autoimprove"

# evaluation/ may use these search/ modules: the config types it trains and the
# proposal specs and contract it enforces. Nothing that proposes or runs turns.
EVALUATION_MAY_USE = {"search.space", "search.codegen", "search.features",
                      "search.analysis"}
# search/ may use these evaluation/ modules: records, the selection rule it
# shows proposers, the specs proposers write and the guardrails turns adopt,
# and the report's Section type, which the literature adds a section with.
SEARCH_MAY_USE = {"evaluation.records", "evaluation.selection",
                  "evaluation.guardrails", "evaluation.report"}


def module_name(path: Path) -> str:
    rel = path.relative_to(PKG).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def imports(path: Path, top_level_only: bool) -> set[str]:
    """Modules of this package that ``path`` imports, as names relative to it."""
    tree = ast.parse(path.read_text())
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    here = module_name(path).split(".") if module_name(path) else []
    package = here if path.name == "__init__.py" else here[:-1]
    out = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                target = base + (node.module.split(".") if node.module else [])
            elif node.module and node.module.startswith(PREFIX):
                target = node.module[len(PREFIX):].lstrip(".").split(".")
            else:
                continue
            for alias in node.names:
                full = ".".join(target + [alias.name])
                out.add(full if (PKG / (full.replace(".", "/") + ".py")).exists()
                        else ".".join(target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PREFIX + "."):
                    out.add(alias.name[len(PREFIX) + 1:])
    return {m for m in out if m}


def modules():
    return {module_name(p): p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts}


class TestLayout(unittest.TestCase):
    def test_three_compartments_and_the_improver(self):
        top = sorted(p.name for p in PKG.iterdir()
                     if p.suffix == ".py" or (p.is_dir() and p.name != "__pycache__"))
        self.assertEqual(top, ["__init__.py", "evaluation", "improver.py", "search",
                               "utils"])

    def test_dependency_rule(self):
        problems = []
        for name, path in modules().items():
            area = name.split(".")[0] if "." in name else name
            for target in imports(path, top_level_only=False):
                t_area = target.split(".")[0]
                if target == "improver" and name != "":
                    problems.append(f"{name} imports improver (only __init__ may)")
                elif area == "utils" and t_area in ("search", "evaluation", "improver"):
                    problems.append(f"{name} (utils) imports {target}")
                elif area == "evaluation" and t_area == "search" and \
                        ".".join(target.split(".")[:2]) not in EVALUATION_MAY_USE:
                    problems.append(f"{name} (evaluation) imports {target}")
                elif area == "search" and t_area == "evaluation" and \
                        ".".join(target.split(".")[:2]) not in SEARCH_MAY_USE:
                    problems.append(f"{name} (search) imports {target}")
        self.assertEqual(problems, [])

    def test_no_import_cycles_at_module_level(self):
        graph = {name: {t for t in imports(path, top_level_only=True) if t in modules()}
                 for name, path in modules().items()}
        cycles = []

        def visit(node, stack):
            for nxt in graph.get(node, ()):
                if nxt in stack:
                    cycles.append(stack[stack.index(nxt):] + [nxt])
                elif nxt not in done:
                    visit(nxt, stack + [nxt])
            done.add(node)

        done: set[str] = set()
        for name in sorted(graph):
            if name not in done:
                visit(name, [name])
        self.assertEqual(cycles, [])

    def test_every_submodule_and_export_imports(self):
        for name in sorted(modules()):
            importlib.import_module(f"{PREFIX}.{name}" if name else PREFIX)
        self.assertEqual([n for n in autoimprove.__all__ if not hasattr(autoimprove, n)],
                         [])


if __name__ == "__main__":
    unittest.main()
