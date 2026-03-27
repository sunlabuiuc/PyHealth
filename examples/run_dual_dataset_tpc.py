import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


SUMMARY_PATTERNS = {
    "eicu": re.compile(r"^ABLATION_SUMMARY\s+(.*)$"),
    "mimic": re.compile(r"^MIMIC_SUMMARY\s+(.*)$"),
}


def parse_summary_fields(summary_body: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for token in summary_body.strip().split():
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def run_command(
    label: str,
    cmd: List[str],
    cwd: str,
) -> Tuple[int, List[str], Optional[str]]:
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
    return fields.get(key, "NA")


def print_combined_summary(
    eicu_fields: Optional[Dict[str, str]],
    mimic_fields: Optional[Dict[str, str]],
) -> None:
    print("\n" + "=" * 80)
    print("COMBINED SUMMARY")
    print("=" * 80)

    if eicu_fields is None:
        print("eICU summary: NOT FOUND")
    else:
        print("eICU summary:")
        print(
            "  "
            f"channel_mode={format_value(eicu_fields, 'channel_mode')} "
            f"include_categorical_statics={format_value(eicu_fields, 'include_categorical_statics')} "
            f"val_loss={format_value(eicu_fields, 'val_loss')} "
            f"mae={format_value(eicu_fields, 'mae')} "
            f"rmse={format_value(eicu_fields, 'rmse')}"
        )

    if mimic_fields is None:
        print("MIMIC-IV summary: NOT FOUND")
    else:
        print("MIMIC-IV summary:")
        print(
            "  "
            f"channel_mode={format_value(mimic_fields, 'channel_mode')} "
            f"include_categorical_statics={format_value(mimic_fields, 'include_categorical_statics')} "
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
            f"{'dataset':<10} {'mode':<10} {'cat':<8} {'val_loss':<10} "
            f"{'val_mae':<10} {'val_rmse':<10} {'test_loss':<10} {'test_mae':<10} {'test_rmse':<10}"
        )
        print(
            f"{'eICU':<10} "
            f"{format_value(eicu_fields, 'channel_mode'):<10} "
            f"{format_value(eicu_fields, 'include_categorical_statics'):<8} "
            f"{format_value(eicu_fields, 'val_loss'):<10} "
            f"{format_value(eicu_fields, 'mae'):<10} "
            f"{format_value(eicu_fields, 'rmse'):<10} "
            f"{'-':<10} {'-':<10} {'-':<10}"
        )
        print(
            f"{'MIMIC-IV':<10} "
            f"{format_value(mimic_fields, 'channel_mode'):<10} "
            f"{format_value(mimic_fields, 'include_categorical_statics'):<8} "
            f"{format_value(mimic_fields, 'val_loss'):<10} "
            f"{format_value(mimic_fields, 'val_mae'):<10} "
            f"{format_value(mimic_fields, 'val_rmse'):<10} "
            f"{format_value(mimic_fields, 'test_loss'):<10} "
            f"{format_value(mimic_fields, 'test_mae'):<10} "
            f"{format_value(mimic_fields, 'test_rmse'):<10}"
        )

    print("=" * 80)


def build_eicu_command(args) -> List[str]:
    cmd = [
        sys.executable,
        "examples/eicu_hourly_los_tpc.py",
        "--max_samples",
        str(args.eicu_max_samples),
        "--channel_mode",
        args.channel_mode,
        "--epochs",
        str(args.eicu_epochs),
    ]

    if args.dev:
        cmd.append("--dev")

    if args.include_categorical_statics:
        cmd.append("--include_categorical_statics")

    return cmd


def build_mimic_command(args) -> List[str]:
    cmd = [
        sys.executable,
        "examples/mimic4_hourly_los_tpc.py",
        "--max_samples",
        str(args.mimic_max_samples),
        "--channel_mode",
        args.channel_mode,
        "--epochs",
        str(args.mimic_epochs),
    ]

    if args.dev:
        cmd.append("--dev")

    if args.include_categorical_statics:
        cmd.append("--include_categorical_statics")

    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eICU and MIMIC-IV TPC examples sequentially and print a combined summary."
    )
    parser.add_argument("--dev", action="store_true", help="Run both scripts in dev mode")
    parser.add_argument(
        "--channel_mode",
        type=str,
        choices=["full", "no_decay", "no_mask"],
        default="full",
        help="Shared channel mode for both runs",
    )
    parser.add_argument(
        "--include_categorical_statics",
        action="store_true",
        help="Pass through to both scripts",
    )
    parser.add_argument("--eicu_max_samples", type=int, default=1000)
    parser.add_argument("--mimic_max_samples", type=int, default=1000)
    parser.add_argument("--eicu_epochs", type=int, default=10)
    parser.add_argument("--mimic_epochs", type=int, default=10)
    parser.add_argument(
        "--skip_eicu",
        action="store_true",
        help="Skip the eICU run",
    )
    parser.add_argument(
        "--skip_mimic",
        action="store_true",
        help="Skip the MIMIC-IV run",
    )
    return parser.parse_args()


def main():
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