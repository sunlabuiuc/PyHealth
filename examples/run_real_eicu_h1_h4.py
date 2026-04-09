#!/usr/bin/env python3
"""
Run H1–H4 ablation experiments for the TPC ICU length-of-stay model on the
real eICU dataset.

This script orchestrates the execution of predefined ablation experiments
(H1–H4) for the Temporal Pointwise Convolution (TPC) model using the real
eICU dataset. It automates experiment execution, log capture, metric parsing,
and result aggregation into structured outputs.

Overview:
    The script performs the following steps:

    1. Defines a set of ablation configurations corresponding to H1–H4.
    2. Executes each configuration via the eICU example training script.
    3. Captures stdout logs for each run.
    4. Parses key evaluation metrics (val_loss, MSLE, MAE, RMSE).
    5. Writes results to CSV and JSON for downstream analysis.
    6. Prints a summary table for quick inspection.

Key Features:
    - Fully automated multi-run experiment execution
    - Standardized evaluation across model variants
    - Log capture for reproducibility and debugging
    - Structured output (CSV + JSON) for reporting and visualization
    - External cache control via XDG_CACHE_HOME

Inputs:
    - Real eICU dataset directory
    - PyHealth project structure
    - Predefined ablation configurations (H1–H4)

Outputs:
    - Per-run log files
    - Aggregated results:
        * real_eicu_h1_h4_results.csv
        * real_eicu_h1_h4_results.json
    - Console summary of all runs

Implementation Notes:
    - This script delegates training to:
        examples/eicu_hourly_los_tpc.py
    - Metrics are extracted from ABLATION_SUMMARY log lines when available.
    - Designed for reproducibility of real-data experiments in the TPC
      replication project.
    - Uses subprocess execution to isolate runs and ensure clean environments.

Example:
    >>> export XDG_CACHE_HOME=/path/to/cache
    >>> PYTHONPATH=. python3 examples/run_real_eicu_h1_h4.py

This script is a core component of the experimental pipeline and supports
the evaluation and reporting of TPC model variants on real eICU data.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(
    "/home/medukonis/Documents/Illinois/Spring_2026/"
    "CS598_Deep_Learning_For_Healthcare/Project"
)
PYHEALTH_ROOT = PROJECT_ROOT / "PyHealth"

# Real eICU root. This matches the default pattern you have used previously.
EICU_REAL_ROOT = PROJECT_ROOT / "Datasets/eicu-collaborative-research-database-2.0"

# External cache root you specified.
CACHE_ROOT = Path("/media/medukonis/Elements/pyhealth-cache-eicu")

OUTPUT_DIR = PROJECT_ROOT / "results" / "real_eicu_h1_h4"
LOG_DIR = OUTPUT_DIR / "logs"


# ---------------------------------------------------------------------
# Common run settings
# Keep these aligned with your earlier real-data draft unless you want to
# deliberately change the regime.
# ---------------------------------------------------------------------

COMMON_ARGS = [
    "--epochs", "3",
    "--batch_size", "8",
    "--max_samples", "128",
    "--train_ratio", "0.75",
    "--seed", "42",
]


# ---------------------------------------------------------------------
# Run definitions: exact same ablations as synthetic, now on real eICU
# ---------------------------------------------------------------------

RUNS = [
    {
        "dataset": "eicu",
        "label": "H1_full",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "full",
            "--channel_mode", "full",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H1_temporal_only",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "temporal_only",
            "--channel_mode", "full",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H1_pointwise_only",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "pointwise_only",
            "--channel_mode", "full",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H2_shared",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "temporal_only",
            "--channel_mode", "full",
            "--shared_temporal",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H3_mse",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "full",
            "--channel_mode", "full",
            "--loss", "mse",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H4_no_skip",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "full",
            "--channel_mode", "full",
            "--no_skip_connections",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H4_no_diag",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "full",
            "--channel_mode", "full",
            "--exclude_diagnoses",
        ],
    },
    {
        "dataset": "eicu",
        "label": "H4_no_decay",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_REAL_ROOT,
        "extra_args": [
            "--model_variant", "full",
            "--channel_mode", "no_decay",
        ],
    },
]


@dataclass
class RunResult:
    dataset: str
    label: str
    status: str
    return_code: int
    started_at: str
    finished_at: str
    duration_sec: float
    log_path: str
    command: str
    val_loss: Optional[float] = None
    test_loss: Optional[float] = None
    msle: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    raw_summary_line: Optional[str] = None
    error: Optional[str] = None


def ensure_dirs() -> None:
    """Create output, log, and cache directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def build_command(run: Dict[str, object]) -> List[str]:
    """Construct the subprocess command for a single experiment run."""
    script = str(run["script"])
    root = str(run["root"])
    extra_args = list(run["extra_args"])  # type: ignore[arg-type]

    cmd = [
        sys.executable,
        script,
        "--root", root,
        *COMMON_ARGS,
        *extra_args,
    ]
    return cmd


def parse_metrics(stdout: str) -> Dict[str, Optional[float]]:
    """Extract evaluation metrics from subprocess stdout logs."""
    metrics: Dict[str, Optional[float]] = {
        "val_loss": None,
        "test_loss": None,
        "msle": None,
        "mae": None,
        "rmse": None,
        "raw_summary_line": None,
    }

    summary_match = re.search(r"^ABLATION_SUMMARY.*$", stdout, re.MULTILINE)
    if summary_match:
        line = summary_match.group(0)
        metrics["raw_summary_line"] = line

        for name in ("val_loss", "test_loss", "msle", "mae", "rmse"):
            m = re.search(rf"{name}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if m:
                metrics[name] = float(m.group(1))
        return metrics

    patterns = {
        "val_loss": [
            r"\bval[_ ]loss[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bvalidation[_ ]loss[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ],
        "test_loss": [
            r"\btest[_ ]loss[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ],
        "msle": [
            r"\bMSLE[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bmsle[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ],
        "mae": [
            r"\bMAE[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bmae[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ],
        "rmse": [
            r"\bRMSE[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\brmse[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ],
    }

    for key, regexes in patterns.items():
        for regex in regexes:
            m = re.search(regex, stdout)
            if m:
                metrics[key] = float(m.group(1))
                break

    return metrics


def run_one(run: Dict[str, object]) -> RunResult:
    """Execute a single experiment run and return structured results."""
    dataset = str(run["dataset"])
    label = str(run["label"])
    cmd = build_command(run)

    started = datetime.now()
    safe_name = f"{dataset}__{label}".replace("/", "_")
    log_path = LOG_DIR / f"{safe_name}.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["XDG_CACHE_HOME"] = str(CACHE_ROOT)

    completed = subprocess.run(
        cmd,
        cwd=PYHEALTH_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    finished = datetime.now()
    duration = (finished - started).total_seconds()
    stdout = completed.stdout or ""
    log_path.write_text(stdout, encoding="utf-8")

    parsed = parse_metrics(stdout)

    status = "ok" if completed.returncode == 0 else "failed"
    error = None if completed.returncode == 0 else "See log."

    return RunResult(
        dataset=dataset,
        label=label,
        status=status,
        return_code=completed.returncode,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        duration_sec=duration,
        log_path=str(log_path),
        command=" ".join(shlex.quote(x) for x in cmd),
        val_loss=parsed["val_loss"],
        test_loss=parsed["test_loss"],
        msle=parsed["msle"],
        mae=parsed["mae"],
        rmse=parsed["rmse"],
        raw_summary_line=parsed["raw_summary_line"],
        error=error,
    )


def write_outputs(results: List[RunResult]) -> None:
    """Write aggregated experiment results to CSV and JSON files."""
    json_path = OUTPUT_DIR / "real_eicu_h1_h4_results.json"
    csv_path = OUTPUT_DIR / "real_eicu_h1_h4_results.csv"

    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n")


def print_summary(results: List[RunResult]) -> None:
    """Print a formatted summary of all experiment results."""
    print("\n=== REAL eICU H1-H4 RUN SUMMARY ===")
    for r in results:
        print(
            f"{r.dataset:7s} | {r.label:24s} | {r.status:6s} | "
            f"val_loss={r.val_loss} | msle={r.msle} | mae={r.mae} | rmse={r.rmse}"
        )


def main() -> int:
    """Run all configured experiments and return exit status."""
    ensure_dirs()

    missing = []
    if not PYHEALTH_ROOT.exists():
        missing.append(f"Missing PyHealth root: {PYHEALTH_ROOT}")
    if not EICU_REAL_ROOT.exists():
        missing.append(f"Missing real eICU root: {EICU_REAL_ROOT}")

    if missing:
        for item in missing:
            print(item)
        return 1

    results: List[RunResult] = []
    for i, run in enumerate(RUNS, start=1):
        print(f"\n[{i}/{len(RUNS)}] Running {run['dataset']} :: {run['label']}")
        result = run_one(run)
        results.append(result)

        print(
            f"  status={result.status} return_code={result.return_code} "
            f"val_loss={result.val_loss} msle={result.msle} "
            f"mae={result.mae} rmse={result.rmse}"
        )

    write_outputs(results)
    print_summary(results)

    failed = [r for r in results if r.status != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
