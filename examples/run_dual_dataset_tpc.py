"""
Run eICU and MIMIC-IV TPC example pipelines sequentially and summarize results.

This utility script orchestrates back-to-back execution of the eICU and
MIMIC-IV hourly length-of-stay (LoS) Temporal Pointwise Convolution (TPC)
example scripts using the current BaseModel-compatible example interfaces.

Overview:
    The script performs the following steps:

    1. Build subprocess commands for the eICU and MIMIC-IV example scripts.
    2. Run each script sequentially from the repository root.
    3. Stream stdout to the console in real time.
    4. Parse the emitted summary lines from each script.
    5. Print a combined summary table for quick comparison.

Inputs:
    Command-line arguments controlling:
        - optional dataset roots
        - dev mode
        - batch size
        - epochs
        - sample counts
        - whether either dataset run should be skipped

Outputs:
    - Real-time console output from each child script
    - Parsed eICU and MIMIC-IV summary metrics
    - Combined summary table printed to stdout
    - Nonzero exit if one or more subprocess runs fail
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


SUMMARY_PATTERNS = {
    "eicu": re.compile(r"^ABLATION_SUMMARY\s+(.*)$"),
    "mimic": re.compile(r"^MIMIC_SUMMARY\s+(.*)$"),
}


def parse_summary_fields(summary_body: str) -> Dict[str, str]:
    """Parse whitespace-delimited key-value tokens from a summary line."""
    fields: Dict[str, str] = {}
    for token in summary_body.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def run_command(
    label: str,
    cmd: List[str],
    cwd: str,
) -> Tuple[int, List[str], Optional[str]]:
    """Run a subprocess command and capture its printed summary line."""
    print("=" * 80)
    print(f"Running {label}")
    print("=" * 80)
    print("Command:")
    print(" ".join(cmd))
    print("-" * 80)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: List[str] = []
    summary_line: Optional[str] = None

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line.rstrip("\n"))

        pattern = SUMMARY_PATTERNS.get(label)
        if pattern is not None:
            match = pattern.match(line.strip())
            if match:
                summary_line = match.group(1)

    return_code = process.wait()

    print("-" * 80)
    print(f"{label} return code: {return_code}")
    print("=" * 80)

    return return_code, output_lines, summary_line


def format_value(fields: Dict[str, str], key: str) -> str:
    """Return a formatted field value or ``NA`` if the key is missing."""
    return fields.get(key, "NA")


def print_combined_summary(
    eicu_fields: Optional[Dict[str, str]],
    mimic_fields: Optional[Dict[str, str]],
) -> None:
    """Print a combined cross-dataset summary report."""
    print("\n" + "=" * 80)
    print("COMBINED SUMMARY")
    print("=" * 80)

    if eicu_fields is None:
        print("eICU summary: NOT FOUND")
    else:
        print("eICU summary:")
        print(
             "  "
            f"model_variant={format_value(eicu_fields, 'model_variant')} "
            f"shared_temporal={format_value(eicu_fields, 'shared_temporal')} "
            f"no_skip_connections={format_value(eicu_fields, 'no_skip_connections')} "
            f"loss={format_value(eicu_fields, 'loss')} "
            f"val_loss={format_value(eicu_fields, 'val_loss')} "
            f"val_mae={format_value(eicu_fields, 'val_mae')} "
            f"val_rmse={format_value(eicu_fields, 'val_rmse')} "
            f"test_loss={format_value(eicu_fields, 'test_loss')} "
            f"test_mae={format_value(eicu_fields, 'test_mae')} "
            f"test_rmse={format_value(eicu_fields, 'test_rmse')}"
        )

    if mimic_fields is None:
        print("MIMIC-IV summary: NOT FOUND")
    else:
        print("MIMIC-IV summary:")
        print(
            "  "
            f"loss={format_value(mimic_fields, 'loss')} "
            f"val_loss={format_value(mimic_fields, 'val_loss')} "
            f"val_mae={format_value(mimic_fields, 'val_mae')} "
            f"val_rmse={format_value(mimic_fields, 'val_rmse')} "
            f"test_loss={format_value(mimic_fields, 'test_loss')} "
            f"test_mae={format_value(mimic_fields, 'test_mae')} "
            f"test_rmse={format_value(mimic_fields, 'test_rmse')}"
        )

    if eicu_fields is not None and mimic_fields is not None:
        print("\nCompact table:")
        print(
            f"{'dataset':<10} {'loss':<8} {'val_loss':<10} {'val_mae':<10} "
            f"{'val_rmse':<10} {'test_loss':<10} {'test_mae':<10} {'test_rmse':<10}"
        )
        print(
            f"{'eICU':<10} "
            f"{format_value(eicu_fields, 'loss'):<8} "
            f"{format_value(eicu_fields, 'val_loss'):<10} "
            f"{format_value(eicu_fields, 'val_mae'):<10} "
            f"{format_value(eicu_fields, 'val_rmse'):<10} "
            f"{format_value(eicu_fields, 'test_loss'):<10} "
            f"{format_value(eicu_fields, 'test_mae'):<10} "
            f"{format_value(eicu_fields, 'test_rmse'):<10}"
        )
        print(
            f"{'MIMIC-IV':<10} "
            f"{format_value(mimic_fields, 'loss'):<8} "
            f"{format_value(mimic_fields, 'val_loss'):<10} "
            f"{format_value(mimic_fields, 'val_mae'):<10} "
            f"{format_value(mimic_fields, 'val_rmse'):<10} "
            f"{format_value(mimic_fields, 'test_loss'):<10} "
            f"{format_value(mimic_fields, 'test_mae'):<10} "
            f"{format_value(mimic_fields, 'test_rmse'):<10}"
        )

    print("=" * 80)


def build_eicu_command(args) -> List[str]:
    """Build the subprocess command for the eICU example script."""
    cmd = [
        sys.executable,
        "examples/eicu_hourly_los_tpc.py",
        "--epochs",
        str(args.eicu_epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_samples",
        str(args.eicu_max_samples),
        "--loss",
        str(args.eicu_loss),
    ]

    if args.eicu_root:
        cmd.extend(["--root", args.eicu_root])

    if args.dev:
        cmd.append("--dev")

    if args.eicu_model_variant:
        cmd.extend(["--model_variant", args.eicu_model_variant])

    if args.eicu_shared_temporal:
        cmd.append("--shared_temporal")

    if args.eicu_no_skip_connections:
        cmd.append("--no_skip_connections")

    return cmd


def build_mimic_command(args) -> List[str]:
    """Build the subprocess command for the MIMIC-IV example script."""
    cmd = [
        sys.executable,
        "examples/mimic4_hourly_los_tpc.py",
        "--epochs",
        str(args.mimic_epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_samples",
        str(args.mimic_max_samples),
        "--loss",
        str(args.mimic_loss),
    ]

    if args.mimic_root:
        cmd.extend(["--root", args.mimic_root])

    if args.dev:
        cmd.append("--dev")

    return cmd


def parse_args():
    """Parse command-line arguments for the dual-dataset run script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run eICU and MIMIC-IV TPC examples sequentially and print a "
            "combined summary."
        )
    )
    parser.add_argument("--dev", action="store_true", help="Run both scripts in dev mode")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eicu_epochs", type=int, default=1)
    parser.add_argument("--mimic_epochs", type=int, default=1)
    parser.add_argument("--eicu_max_samples", type=int, default=24)
    parser.add_argument("--mimic_max_samples", type=int, default=24)
    parser.add_argument("--eicu_root", type=str, default="")
    parser.add_argument("--mimic_root", type=str, default="")
    parser.add_argument(
        "--eicu_loss",
        type=str,
        choices=["msle", "mse"],
        default="msle",
        help="Loss to pass to the eICU script.",
    )
    parser.add_argument(
        "--mimic_loss",
        type=str,
        choices=["msle", "mse"],
        default="msle",
        help="Loss to pass to the MIMIC-IV script.",
    )
    parser.add_argument(
        "--eicu_model_variant",
        type=str,
        choices=["full", "temporal_only", "pointwise_only"],
        default="full",
    )
    parser.add_argument(
        "--eicu_shared_temporal",
        action="store_true",
        help="Pass --shared_temporal to the eICU script.",
    )
    parser.add_argument(
        "--eicu_no_skip_connections",
        action="store_true",
        help="Pass --no_skip_connections to the eICU script.",
    )
    parser.add_argument("--skip_eicu", action="store_true", help="Skip the eICU run")
    parser.add_argument("--skip_mimic", action="store_true", help="Skip the MIMIC-IV run")
    return parser.parse_args()


def main():
    """Run the configured eICU and MIMIC-IV example scripts and summarize results."""
    args = parse_args()

    eicu_fields: Optional[Dict[str, str]] = None
    mimic_fields: Optional[Dict[str, str]] = None

    eicu_rc = 0
    mimic_rc = 0

    if not args.skip_eicu:
        eicu_cmd = build_eicu_command(args)
        eicu_rc, _, eicu_summary = run_command("eicu", eicu_cmd, cwd=REPO_ROOT)
        if eicu_summary is not None:
            eicu_fields = parse_summary_fields(eicu_summary)

    if not args.skip_mimic:
        mimic_cmd = build_mimic_command(args)
        mimic_rc, _, mimic_summary = run_command("mimic", mimic_cmd, cwd=REPO_ROOT)
        if mimic_summary is not None:
            mimic_fields = parse_summary_fields(mimic_summary)

    print_combined_summary(eicu_fields, mimic_fields)

    if eicu_rc != 0 or mimic_rc != 0:
        failed = []
        if eicu_rc != 0:
            failed.append(f"eICU rc={eicu_rc}")
        if mimic_rc != 0:
            failed.append(f"MIMIC-IV rc={mimic_rc}")
        raise SystemExit("One or more runs failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()