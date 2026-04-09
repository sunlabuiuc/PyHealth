#!/usr/bin/env python3
"""
Run H1–H4 synthetic ablation experiments for the TPC ICU length-of-stay model.

This script orchestrates the execution of predefined H1–H4 ablation
experiments for the Temporal Pointwise Convolution (TPC) model using
synthetic eICU data. It automates experiment execution, log capture, metric
parsing, and result aggregation into structured outputs for reproducibility,
debugging, and demonstration purposes.

Overview:
    The script performs the following steps:

    1. Defines a fixed set of H1–H4 ablation runs.
    2. Launches the existing eICU example training script for each run.
    3. Captures and stores stdout logs.
    4. Parses key metrics from ABLATION_SUMMARY lines or fallback log text.
    5. Writes aggregated results to CSV and JSON files.
    6. Prints a compact summary of all synthetic runs.

    This script is intended to provide a fast, reproducible synthetic-data
    evaluation pipeline for validating that the ablation setup works before
    running on the full real dataset.

Key Features:
    - Fully automated synthetic H1–H4 experiment execution
    - Standardized log capture and parsing
    - CSV/JSON result export for later analysis
    - Compact printed summary for quick inspection
    - Small run settings for fast turnaround during testing and demos

Inputs:
    - Synthetic eICU dataset directory
    - PyHealth project structure
    - Predefined H1–H4 ablation configurations

Outputs:
    - Per-run log files
    - Aggregated results:
        * synthetic_h1_h4_results.csv
        * synthetic_h1_h4_results.json
    - Console summary of all runs

Implementation Notes:
    - This script delegates training to:
        examples/eicu_hourly_los_tpc.py
    - Metrics are extracted from ABLATION_SUMMARY log lines when available.
    - Designed for reproducibility and quick synthetic validation in the TPC
      replication project.
    - Uses subprocess execution so each run is isolated and easy to inspect.

Example:
    >>> PYTHONPATH=. python3 examples/run_synth_h1_h4.py

This script is part of the experimental support pipeline and is used to
validate and demonstrate H1–H4 ablation behavior on synthetic eICU data.
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

# Adjust this if your actual repo root is different.
PYHEALTH_ROOT = PROJECT_ROOT / "PyHealth"

EICU_SYNTH_ROOT = PROJECT_ROOT / "mnt/data/synth/eicu_demo"
MIMIC4_SYNTH_ROOT = PROJECT_ROOT / "mnt/data/synth/mimic4_demo"

OUTPUT_DIR = PROJECT_ROOT / "results" / "synthetic_h1_h4"
LOG_DIR = OUTPUT_DIR / "logs"


# ---------------------------------------------------------------------
# Common run settings
# Keep these small so the synthetic runs finish quickly.
# ---------------------------------------------------------------------

COMMON_ARGS = [
    "--epochs", "3",
    "--batch_size", "2",
    "--max_samples", "16",
    "--train_ratio", "0.75",
    "--seed", "42",
]

# If your scripts support these, they help keep caches separate.
EICU_CACHE_ARGS = []
MIMIC4_CACHE_ARGS = []


# ---------------------------------------------------------------------
# Run definitions
#
# IMPORTANT:
# These are the likely ablation knobs, based on your project and the paper.
# If one of these flags is named differently in your local scripts, just swap
# out that flag/value pair here.
# ---------------------------------------------------------------------

RUNS = [
    {
        "dataset": "eicu",
        "label": "H1_full",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_SYNTH_ROOT,
        "extra_args": ["--model_variant", "full", "--channel_mode", "full"],
    },
    {
        "dataset": "eicu",
        "label": "H1_temporal_only",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_SYNTH_ROOT,
        "extra_args": ["--model_variant", "temporal_only", "--channel_mode", "full"],
    },
    {
        "dataset": "eicu",
        "label": "H1_pointwise_only",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_SYNTH_ROOT,
        "extra_args": ["--model_variant", "pointwise_only", "--channel_mode", "full"],
    },
    {
        "dataset": "eicu",
        "label": "H2_shared",
        "script": "examples/eicu_hourly_los_tpc.py",
        "root": EICU_SYNTH_ROOT,
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
        "root": EICU_SYNTH_ROOT,
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
        "root": EICU_SYNTH_ROOT,
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
        "root": EICU_SYNTH_ROOT,
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
        "root": EICU_SYNTH_ROOT,
        "extra_args": [
            "--model_variant", "full",
            "--channel_mode", "no_decay",
        ],
    },
]


@dataclass
class RunResult:
    """Container for one synthetic ablation run result."""
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
    """Create output and log directories if they do not already exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def boolish_to_flag_value(value: str) -> str:
    """Normalize a boolean-like string to lowercase."""
    return value.lower()


def build_command(run: Dict[str, object]) -> List[str]:
    """Construct the subprocess command for a single synthetic ablation run."""
    dataset = str(run["dataset"])
    script = str(run["script"])
    root = str(run["root"])
    extra_args = list(run["extra_args"])  # type: ignore[arg-type]

    cache_args = EICU_CACHE_ARGS if dataset == "eicu" else MIMIC4_CACHE_ARGS

    cmd = [
        sys.executable,
        script,
        "--root", root,
        *COMMON_ARGS,
        *cache_args,
        *extra_args,
    ]
    return cmd


def parse_metrics(stdout: str) -> Dict[str, Optional[float]]:
    """Extract evaluation metrics from subprocess stdout.

    This parser first looks for an ``ABLATION_SUMMARY`` line and falls back
    to more generic metric patterns if needed.

    Args:
        stdout: Full captured stdout from one subprocess run.

    Returns:
        Dictionary containing parsed metric values and the raw summary line
        when present.
    """
    metrics: Dict[str, Optional[float]] = {
        "val_loss": None,
        "test_loss": None,
        "msle": None,
        "mae": None,
        "rmse": None,
        "raw_summary_line": None,
    }

    # Preferred: custom compact ablation line
    # Example:
    # ABLATION_SUMMARY channel_mode=full include_categorical_statics=False
    # val_loss=0.0168 mae=1.9643 rmse=2.4660
    summary_match = re.search(r"^ABLATION_SUMMARY.*$", stdout, re.MULTILINE)
    if summary_match:
        line = summary_match.group(0)
        metrics["raw_summary_line"] = line

        for name in ("val_loss", "test_loss", "msle", "mae", "rmse"):
            m = re.search(rf"{name}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if m:
                metrics[name] = float(m.group(1))
        return metrics

    # Generic patterns
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
    """Execute a single synthetic ablation run and return structured results."""
    dataset = str(run["dataset"])
    label = str(run["label"])
    cmd = build_command(run)

    started = datetime.now()
    safe_name = f"{dataset}__{label}".replace("/", "_")
    log_path = LOG_DIR / f"{safe_name}.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

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
    """Write aggregated synthetic ablation results to CSV and JSON files."""
    json_path = OUTPUT_DIR / "synthetic_h1_h4_results.json"
    csv_path = OUTPUT_DIR / "synthetic_h1_h4_results.csv"

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
    """Print a formatted summary of all synthetic ablation runs."""
    print("\n=== SYNTHETIC H1-H4 RUN SUMMARY ===")
    for r in results:
        print(
            f"{r.dataset:7s} | {r.label:24s} | {r.status:6s} | "
            f"val_loss={r.val_loss} | msle={r.msle} | mae={r.mae} | rmse={r.rmse}"
        )


def main() -> int:
    """Run all configured synthetic ablations and return exit status."""
    ensure_dirs()

    missing = []
    if not PYHEALTH_ROOT.exists():
        missing.append(f"Missing PyHealth root: {PYHEALTH_ROOT}")
    if not EICU_SYNTH_ROOT.exists():
        missing.append(f"Missing eICU synthetic root: {EICU_SYNTH_ROOT}")
    if not MIMIC4_SYNTH_ROOT.exists():
        missing.append(f"Missing MIMIC-IV synthetic root: {MIMIC4_SYNTH_ROOT}")

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