#!/usr/bin/env python3
"""Print the CI coverage summary, merging a PR's partial run with master's.

PR runs only execute the tests tools/select_tests.py picks, so their
``.coverage`` alone undercounts every file that only skipped tests reach.
This merges it with the baseline master's last full run uploaded:

* files changed since the baseline commit: this run's data only (their line
  numbers moved, and every test that imports them was selected);
* every other file: the union of both. A skipped test depends on nothing that
  changed since the baseline, so it covers the same lines it covered there.

So the total is what a full run would report, except where a change makes an
unchanged file lose coverage (e.g. a model stops calling a helper): the union
keeps the baseline's lines for it.

Prints the JSON summary that .github/workflows/coverage.yml and ai-review.yml
parse from this step's log: ``total``, ``modified_files``, and ``tests``.
Keep the format in sync with their parsers; it must stay the last top-level
``{ ... }`` block of the output.

Inputs, from the environment (all optional except MODE):
    MODE             none | some | all, from select_tests.py
    BASELINE         master's coverage data file (required unless MODE=all)
    BASELINE_SHA     the commit BASELINE was measured at
    BASE_SHA         the PR base; files changed since it are "modified_files"
    SELECTED, SUITE_TOTAL   test modules run / in the suites
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from coverage import Coverage, CoverageData

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_changed(rev: str, *extra) -> list:
    out = subprocess.run(
        ["git", "diff", "--name-only", *extra, rev, "HEAD", "--", "pyhealth/*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT)


def merged_data(mode: str, baseline: str, baseline_sha: str) -> str:
    """Write the merged data file and return its path."""
    fresh = REPO_ROOT / ".coverage"
    if mode == "all":
        return str(fresh)
    out = REPO_ROOT / ".coverage.merged"
    out.unlink(missing_ok=True)
    merged = CoverageData(basename=str(out))

    base = CoverageData(basename=baseline)
    base.read()
    # deleted files too: they must not linger in the total
    changed = set(git_changed(baseline_sha))
    stale = [f for f in base.measured_files() if rel(f) in changed]
    base.purge_files(stale)
    merged.update(base)

    if fresh.exists():
        run = CoverageData(basename=str(fresh))
        run.read()
        merged.update(run)
    # A changed file no selected test executed still counts, at 0%, as the
    # full run's --cov=pyhealth source scan would count it.
    have = {rel(f) for f in merged.measured_files()}
    missing = {str(REPO_ROOT / p) for p in changed
               if p not in have and (REPO_ROOT / p).exists()}
    if missing:
        merged.add_lines({f: set() for f in missing})
    merged.write()
    return str(out)


def summarize(s: dict) -> dict:
    return {
        "percent_covered": round(s["percent_covered"], 2),
        "covered_lines": s["covered_lines"],
        "num_statements": s["num_statements"],
    }


def main() -> int:
    env = os.environ
    mode = env["MODE"]
    data_file = merged_data(mode, env.get("BASELINE", ""), env.get("BASELINE_SHA", ""))

    cov = Coverage(data_file=data_file)
    cov.load()
    json_path = REPO_ROOT / "coverage.json"
    cov.json_report(outfile=str(json_path), ignore_errors=True)
    report = json.loads(json_path.read_text())

    files = {rel(k): v["summary"] for k, v in report["files"].items()}
    base_sha = env.get("BASE_SHA", "")
    modified = git_changed(base_sha, "--diff-filter=d") if base_sha else []

    def count(name: str):
        value = env.get(name, "")
        return int(value) if value.isdigit() else None

    print(json.dumps({
        "total": summarize(report["totals"]),
        "modified_files": {
            p: summarize(files[p]) if p in files else None for p in modified
        },
        "tests": {
            "mode": mode,
            "selected": count("SELECTED"),
            "suite_total": count("SUITE_TOTAL"),
            "baseline_sha": env.get("BASELINE_SHA") or None,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
