import json
from typing import Any

def load_json(path: str) -> list[Any] | dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    except OSError as e:
        raise ValueError(f"Could not read {path}: {e}") from e


def flatten(data: dict[Any, Any]) -> dict[str, tuple[list[Any], list[list[Any]]]]:
    aggregated = data.pop("aggregated")
    per_seed = data.pop("per_seed")

    aggregated_header, aggregated_rows = _flatten_rows(aggregated)
    per_seed_header, per_seed_rows = _flatten_per_seed(per_seed)

    metadata_header, metadata_values = _flatten_metadata(data)

    return {
        "aggregated" : (aggregated_header, aggregated_rows),
        "per_seed" : (per_seed_header, per_seed_rows),
        "metadata" : (metadata_header, metadata_values)
            }


def _flatten_metadata(metadata: dict):
    header = list(metadata.keys())
    values = list(metadata.values())

    return header, values


def _flatten_rows(rows: list[dict]):
    if not rows:
        return [], []

    header = list(rows[0].keys())
    flattened_rows = []

    for row in rows:
        flattened_rows.append(list(row.values()))

    return header, flattened_rows


def _flatten_per_seed(seed_to_rows: dict[Any, list[dict]]):
    header, aggregated_flattened_rows = [], []

    for seed, rows in seed_to_rows.items():
        seed_header, flattened_rows = _flatten_rows(rows)

        if not header:
            header = seed_header

        aggregated_flattened_rows.extend(flattened_rows)

    return header, aggregated_flattened_rows


if __name__ == "__main__":
    from pprint import pprint

    data = load_json("benchmarks/cpbench/writers/los_dev_4alpha.json")
    flattened_data = flatten(data)
    pprint(flattened_data)