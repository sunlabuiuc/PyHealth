from __future__ import annotations

import json
from typing import Any

# Everything at the top level except these two blocks is run-level context that gets copied onto every row.
_RESULT_BLOCKS = ("per_seed", "aggregated")


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}

    flat: dict[str, Any] = {}
    for key, inner in value.items():
        flat.update(_flatten(f"{prefix}.{key}", inner))

    return flat


def run_context(record: dict[str, Any]) -> dict[str, Any]:
    """The run-level fields shared by every row, flattened to dotted keys."""
    context: dict[str, Any] = {}
    for key, value in record.items():
        if key not in _RESULT_BLOCKS:
            context.update(_flatten(key, value))

    return context


def per_seed_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    context = run_context(record)
    rows: list[dict[str, Any]] = []

    for seed, results in sorted(record.get("per_seed", {}).items()):
        for result in results:
            rows.append({**context, "seed": int(seed), **result})

    return rows


def aggregated_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    context = run_context(record)

    return [{**context, **entry} for entry in record.get("aggregated", [])]


def column_order(rows: list[dict[str, Any]]) -> list[str]:
    order: dict[str, None] = {}
    for row in rows:
        for key in row:
            order.setdefault(key, None)

    return list(order)


def to_values(row: dict[str, Any], columns: list[str]) -> list[Any]:
    return [_cell(row.get(column)) for column in columns]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value)
