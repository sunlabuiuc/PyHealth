#!/usr/bin/env python3
"""
CI gate enforcing PyHealth's PR contribution rules.

Rules enforced whenever a PR touches pyhealth/**/*.py:

    1. Docs/examples: the PR must also modify at least one file under
       docs/** and one file under examples/**.
    2. Lint: lines added or modified in touched pyhealth/**/*.py files must
       be free of ruff violations. Pre-existing violations elsewhere in a
       touched file are not flagged.
    3. Docstring examples: new or modified top-level public classes/
       functions in pyhealth/**/*.py must include a '>>>' usage example in
       their docstring.

Usage:
    python tools/check_pr_rules.py --base <base_sha> --head <head_sha>
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sh(*args):
    return subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def changed_files(base, head):
    out = sh("git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}..{head}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def added_lines(base, head, path):
    """Line numbers in `path` at `head` that were added or modified vs `base`."""
    out = sh("git", "diff", "--unified=0", f"{base}..{head}", "--", path)
    lines = set()
    for line in out.splitlines():
        if not line.startswith("@@"):
            continue
        plus = line.split("+")[1].split(" ")[0]
        if "," in plus:
            start, count = (int(x) for x in plus.split(","))
        else:
            start, count = int(plus), 1
        lines |= set(range(start, start + count))
    return lines


def check_docs_examples(files):
    if not any(f.startswith("pyhealth/") and f.endswith(".py") for f in files):
        return []
    problems = []
    if not any(f.startswith("docs/") for f in files):
        problems.append(
            "PR modifies pyhealth/ source files but no file under docs/ "
            "was updated."
        )
    if not any(f.startswith("examples/") for f in files):
        problems.append(
            "PR modifies pyhealth/ source files but no file under "
            "examples/ was updated."
        )
    return problems


def check_lint(files, base, head):
    py_files = [
        f
        for f in files
        if f.startswith("pyhealth/") and f.endswith(".py") and (REPO_ROOT / f).exists()
    ]
    if not py_files:
        return []
    result = subprocess.run(
        ["ruff", "check", "--output-format=json", *py_files],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return []
    problems = []
    for v in json.loads(result.stdout):
        path = Path(v["filename"]).resolve().relative_to(REPO_ROOT).as_posix()
        line = v["location"]["row"]
        if line in added_lines(base, head, path):
            problems.append(f"{path}:{line}: {v['code']} {v['message']}")
    return problems


def check_docstring_examples(files, base, head):
    problems = []
    for path in files:
        if not (path.startswith("pyhealth/") and path.endswith(".py")):
            continue
        full = REPO_ROOT / path
        if not full.exists():
            continue
        added = added_lines(base, head, path)
        if not added:
            continue
        try:
            tree = ast.parse(full.read_text())
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            span = set(range(node.lineno, node.end_lineno + 1))
            if not span & added:
                continue
            doc = ast.get_docstring(node)
            if not doc or ">>>" not in doc:
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                problems.append(
                    f"{path}:{node.lineno}: public {kind} '{node.name}' is "
                    "new/modified but its docstring has no '>>>' usage example."
                )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base commit SHA")
    parser.add_argument("--head", required=True, help="head commit SHA")
    args = parser.parse_args()

    files = changed_files(args.base, args.head)
    problems = (
        check_docs_examples(files)
        + check_lint(files, args.base, args.head)
        + check_docstring_examples(files, args.base, args.head)
    )

    if problems:
        print("PR contribution rules failed:\n")
        for p in problems:
            print(f"  - {p}")
        print(f"\n{len(problems)} issue(s) found. See CONTRIBUTING.md for details.")
        sys.exit(1)

    print("All PR contribution rules passed.")


if __name__ == "__main__":
    main()
