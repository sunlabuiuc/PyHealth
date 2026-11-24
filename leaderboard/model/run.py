"""Runs the PyHealth model benchmark on the task files in this directory.

Each (task, model) is trained and tested on every split seed, and skipped
while the version in its saved result.json matches the model's ``version``
(commit SHA) in the task file.

Afterwards, writes leaderboard.json next to leaderboard.yaml: one object per
leaderboard, filled from the saved results of every task file here (not only
the tasks just run). Skipped in dev mode, so dev results never reach it.

Usage:
    python leaderboard/model/run.py                        # every task here
    python leaderboard/model/run.py yaib_mimic4_mortality.yaml
    python leaderboard/model/run.py --dev                  # 1000 patients
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
LEADERBOARD_YAML = HERE / "leaderboard.yaml"
LEADERBOARD_JSON = HERE / "leaderboard.json"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_leaderboards(leaderboards: list[dict], tasks: dict[str, dict]) -> None:
    """Fails before any training if a leaderboard names a task with no task file."""
    for board in leaderboards:
        for column in board["columns"]:
            if column["task"] not in tasks:
                raise SystemExit(
                    f"leaderboard.yaml: {board['id']!r} names task {column['task']!r}, "
                    f"which no task file in {HERE} defines"
                )


def build_leaderboards(
    leaderboards: list[dict], tasks: dict[str, dict], benchmark: Any, output_dir: Path | None
) -> list[dict]:
    """One leaderboard object per entry of leaderboard.yaml.

    Each column carries its task's ``training.metrics`` (the second header
    row); each model maps every column's task id to its saved result, or to
    None when it has no task file entry or no result of its current version.
    """
    evaluations = {(ev.params["task_id"], ev.params["model_id"]): ev for ev in benchmark.run_configs}
    built = []
    for board in leaderboards:
        columns, models = [], {}
        for column in board["columns"]:
            task_id = column["task"]
            task = tasks[task_id]
            metrics = list(task["training"]["metrics"])
            columns.append({"task": task_id, "label": column["label"], "metrics": metrics})
            for model in task["models"]:
                entry = models.setdefault(model["id"], {"id": model["id"], "name": model["name"], "results": {}})
                ev = evaluations[(task_id, model["id"])]
                result = benchmark.saved(ev, output_dir)  # None if missing or of an older version
                entry["results"][task_id] = None if result is None else {
                    "version": ev.params["version"],
                    "metrics": {name: result.metrics.get(name) for name in metrics},
                }
        for entry in models.values():  # a model absent from a column's task file
            entry["results"] = {c["task"]: entry["results"].get(c["task"]) for c in columns}
        built.append(
            {
                "id": board["id"],
                "name": board["name"],
                "tags": list(board.get("tags", [])),
                "columns": columns,
                "models": list(models.values()),
            }
        )
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="PyHealth model benchmark")
    parser.add_argument("tasks", nargs="*", type=Path, help="Task YAML files (default: all here).")
    parser.add_argument("--env", type=Path, default=HERE / "env.yaml")
    parser.add_argument("--dev", action="store_true", help="Override env dev: true.")
    args = parser.parse_args()

    env = load_yaml(args.env)
    env["dev"] = env.get("dev", False) or args.dev
    # A task file is any YAML here with a top-level ``task`` (not env/leaderboard).
    here_tasks = [cfg for p in sorted(HERE.glob("*.yaml")) if "task" in (cfg := load_yaml(p) or {})]
    tasks = [load_yaml(p) for p in args.tasks] if args.tasks else here_tasks

    # Every task file here, so a partial run still writes complete leaderboards.
    all_tasks = {t["task"]["id"]: t for t in here_tasks}
    leaderboards = load_yaml(LEADERBOARD_YAML)["leaderboards"]
    check_leaderboards(leaderboards, all_tasks)

    # Before PyHealth: Polars and torch read their thread counts when imported.
    for name, value in (env.get("env") or {}).items():
        os.environ[name] = str(value)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from pyhealth.benchmarks import ModelBenchmark

    results = ModelBenchmark(env, tasks).run()

    failed = 0
    for ev, result in results.items():
        name = f"{ev.params['task_id']} / {ev.params['model_id']}"
        if result.status == "failed":
            failed += 1
            print(f"{name}: FAILED ({result.metadata['error_type']})")
            continue
        metrics = ", ".join(
            f"{k} {v['mean']:.4f}±{v['std']:.4f}" for k, v in result.metrics.items() if k != "loss"
        )
        print(f"{name}: {metrics}")

    if env["dev"]:
        print(f"dev mode: {LEADERBOARD_JSON.name} not written")
    else:
        output_dir = env.get("paths", {}).get("output_dir")
        board_benchmark = ModelBenchmark(env, list(all_tasks.values()))  # only reads saved results
        built = build_leaderboards(leaderboards, all_tasks, board_benchmark, Path(output_dir) if output_dir else None)
        with open(LEADERBOARD_JSON, "w", encoding="utf-8") as f:
            json.dump(built, f, indent=2)
            f.write("\n")
        print(f"wrote {LEADERBOARD_JSON}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
