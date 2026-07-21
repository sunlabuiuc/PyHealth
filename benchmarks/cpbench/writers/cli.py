"""Push CPBench result JSON files to a Google Sheet, or preview them.

Each run file becomes one row per seed, or use --aggregated for one mean/std row per alpha.
- Run with --dry-run to print the rows without touching a sheet or needing credentials.
- To upload, pass --sheet-id and --credentials, where credentials is a Google service account
JSON file that has edit access to the sheet.
- Rows are appended, so re-running a config adds new rows rather than replacing old ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .rows import (
    aggregated_rows,
    column_order,
    per_seed_rows,
    run_context,
    to_values,
)
from .sheets_writer import SheetsWriter

_EXAMPLES = """\
examples:
  preview one file, one row per seed
    python -m benchmarks.cpbench.writers.cli results/run.json --dry-run

  preview the aggregated mean/std rows
    python -m benchmarks.cpbench.writers.cli results/run.json --aggregated --dry-run

  upload every result file to a sheet
    python -m benchmarks.cpbench.writers.cli results/*.json \\
        --sheet-id 1AbC...xyz --credentials service_account.json
"""


def load_rows(paths: list[Path], aggregated: bool) -> list[dict[str, Any]]:
    flatten = aggregated_rows if aggregated else per_seed_rows
    rows: list[dict[str, Any]] = []

    for path in paths:
        with path.open() as handle:
            record = json.load(handle)
        rows.extend(flatten(record))

    return rows


def print_preview(path: Path, record: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    context = run_context(record)
    result_columns = [column for column in column_order(rows) if column not in context]

    print(f"── {path.name} ──")
    _print_block(context)

    print()
    _print_grid(result_columns, rows)


def _print_block(context: dict[str, Any]) -> None:
    label_width = max(len(key) for key in context)
    for key, value in zip(context, to_values(context, list(context))):
        print(f"  {key.ljust(label_width)}  {value}")


def _print_grid(columns: list[str], rows: list[dict[str, Any]]) -> None:
    cells = [[str(value) for value in to_values(row, columns)] for row in rows]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in cells))
        for i in range(len(columns))
    ]

    header = "  ".join(columns[i].ljust(widths[i]) for i in range(len(columns)))
    print(header)
    print("-" * len(header))

    for row in cells:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results", nargs="+", type=Path, help="result JSON files")
    parser.add_argument(
        "--aggregated",
        action="store_true",
        help="Use the aggregated (mean/std) block instead of one row per seed.",
    )
    parser.add_argument("--sheet-id", help="Target spreadsheet id.")
    parser.add_argument("--worksheet", default="results")
    parser.add_argument("--credentials", help="Path to service account JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rows that would be written and exit.",
    )

    return parser.parse_args(argv)


def _dry_run(paths: list[Path], aggregated: bool) -> int:
    flatten = aggregated_rows if aggregated else per_seed_rows
    total = 0

    for path in paths:
        with path.open() as handle:
            record = json.load(handle)
        rows = flatten(record)
        if not rows:
            continue

        if total:
            print()
        print_preview(path, record, rows)
        total += len(rows)

    if not total:
        print("No rows found in the given files.", file=sys.stderr)
        return 1

    print(f"\n{total} rows (dry run, nothing written)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        return _dry_run(args.results, args.aggregated)

    rows = load_rows(args.results, args.aggregated)
    if not rows:
        print("No rows found in the given files.", file=sys.stderr)
        return 1

    if not args.sheet_id or not args.credentials:
        print(
            "--sheet-id and --credentials are required unless --dry-run is set.",
            file=sys.stderr,
        )
        return 1

    writer = SheetsWriter(args.sheet_id, args.credentials, args.worksheet)
    appended = writer.append(rows, column_order(rows))
    print(f"{appended} rows appended")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
