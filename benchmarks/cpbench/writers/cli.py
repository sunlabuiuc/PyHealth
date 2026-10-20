"""Append a CPBench result JSON file to a Google Sheet, or preview what would be written.

The aggregated and per-seed blocks go to separate worksheets. Rows are appended, so
re-running a config adds rows rather than replacing them.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .parse_json import flatten, load_json
from .sheets_writer import SheetsWriter

_AGGREGATED_WORKSHEET = "Aggregated Worksheet"
_PER_SEED_WORKSHEET = "Per-seed Results"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    return str(value)


def _render_table(header: list[Any], rows: list[list[Any]]) -> str:
    columns = [_format_value(column) for column in header]
    cells = [[_format_value(value) for value in row] for row in rows]

    widths = [len(column) for column in columns]
    for row in cells:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(cell))

    def line(values: list[str]) -> str:
        padded = [value.ljust(widths[i]) for i, value in enumerate(values[: len(widths)])]
        return "  ".join(padded).rstrip()

    body = [line(columns)] + [line(row) for row in cells]
    rule = "-" * max(len(text) for text in body)

    return "\n".join([body[0], rule] + body[1:])


def _metadata_rows(
    header: list[Any], values: list[Any], width: int = 70
) -> list[list[str]]:
    import textwrap

    rows = []
    for field, value in zip(header, values):
        chunks = textwrap.wrap(_format_value(value), width=width) or [""]
        rows.append([_format_value(field), chunks[0]])
        rows.extend([["", chunk] for chunk in chunks[1:]])

    return rows


def _render_section(title: str, body: str) -> str:
    rule = "=" * max(len(title), 40)

    return f"\n{rule}\n{title}\n{rule}\n{body}"


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 3, 1)] + "..."


def _target_block(
    worksheet_name: str,
    header: list[Any],
    rows: list[list[Any]],
    sample: int,
    width: int,
) -> str:
    columns = [_format_value(column) for column in header]
    lines = [
        f"worksheet  {worksheet_name}",
        _clip(f"columns    {len(columns)}  ({', '.join(columns)})", width),
        f"rows       {len(rows)}",
    ]

    if not rows:
        lines.append("")
        lines.append("  nothing to append")

        return "\n".join(lines)

    preview = _render_table(header, rows[:sample]).splitlines()
    lines.append("")
    lines.extend("  " + _clip(line, width - 2) for line in preview)

    remaining = len(rows) - sample
    if remaining > 0:
        lines.append(f"  ... {remaining} more row{'' if remaining == 1 else 's'}")

    return "\n".join(lines)


def print_write_preview(
    data: dict[str, tuple[list[Any], list[Any]]],
    worksheets: dict[str, str] | None = None,
    sample: int = 3,
    width: int | None = None,
) -> None:
    if width is None:
        import shutil

        width = shutil.get_terminal_size((100, 24)).columns

    worksheets = worksheets or {}

    metadata_header, metadata_values = data.get("metadata", ([], []))
    metadata_table = (
        _render_table(["Field", "Value"], _metadata_rows(metadata_header, metadata_values))
        if metadata_header
        else "(no data)"
    )
    print(_render_section("METADATA", metadata_table))

    for section in ("aggregated", "per_seed"):
        header, rows = data.get(section, ([], []))
        worksheet_name = worksheets.get(section) or section
        print(
            _render_section(
                f"WRITE PREVIEW: {section}",
                _target_block(worksheet_name, header, rows, sample, width),
            )
        )


def _sheet_value(value: Any) -> Any:
    """Keep scalars as they are; flatten anything a cell cannot hold to text."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return _format_value(value)


def _with_metadata(
    metadata: tuple[list[Any], list[Any]],
    header: list[Any],
    rows: list[list[Any]],
) -> tuple[list[Any], list[list[Any]]]:
    """Prefix every row with the run's metadata so each sheet row stands alone."""
    metadata_header, metadata_values = metadata
    prefix = [_sheet_value(value) for value in metadata_values]

    return list(metadata_header) + list(header), [prefix + row for row in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.cpbench.writers.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True, metavar="PATH", help="Result JSON.")
    parser.add_argument(
        "--sheet-id",
        metavar="ID",
        help="Spreadsheet id. Required unless --preview.",
    )
    parser.add_argument(
        "--per-seed-worksheet",
        default=_PER_SEED_WORKSHEET,
        metavar="NAME",
        help=f'Per-seed worksheet (default: "{_PER_SEED_WORKSHEET}").',
    )
    parser.add_argument(
        "--aggregated-worksheet",
        default=_AGGREGATED_WORKSHEET,
        metavar="NAME",
        help=f'Aggregated worksheet (default: "{_AGGREGATED_WORKSHEET}").',
    )
    parser.add_argument(
        "--credentials",
        metavar="PATH",
        help="Service account JSON. Required unless --preview.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print what would be written; write nothing.",
    )

    args = parser.parse_args(argv)

    if not args.preview:
        missing = [
            flag
            for flag, value in (
                ("--sheet-id", args.sheet_id),
                ("--credentials", args.credentials),
            )
            if not value
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required unless --preview is set.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        data = flatten(load_json(args.results))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    worksheets = {
        "aggregated": args.aggregated_worksheet,
        "per_seed": args.per_seed_worksheet,
    }

    if args.preview:
        print_write_preview(data, worksheets)
        print("\n(preview only, nothing written)", file=sys.stderr)
        return 0

    writer = SheetsWriter(args.sheet_id, args.credentials)
    metadata = data.get("metadata", ([], []))
    total = 0
    for section, worksheet_name in worksheets.items():
        header, rows = data.get(section, ([], []))
        columns, rows = _with_metadata(metadata, header, rows)
        appended = writer.append(worksheet_name, rows, columns)
        print(f"{appended} rows appended to {worksheet_name}")
        total += appended

    if not total:
        print("No rows found in the given file.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
