# Contributor: Arun Vishnu (anv5@illinois.edu)
# Paper: Evaluating Model Performance in Medical Datasets Over Time
# https://proceedings.mlr.press/v209/zhou23a.html
# Utility: Reproducible bounded sampling for the temporal example.

"""Download a reproducible, bounded CDC case-surveillance sample.

The full CDC file has about 100 million rows. This utility requests a fixed
number of Yes and No ``death_yn`` records per month from Socrata dataset
``n8mc-b4w4``. It divides each month/label stratum into deterministic segments
in Socrata's opaque row-ID order and samples a small block from every segment.
This systematic coverage avoids relying on one contiguous storage block, but
it is still a compute-bounded convenience sample rather than a statistically
uniform sample. Equal class counts deliberately change prevalence, so do not
use it for calibration or mortality-rate claims.

Example (4,800 rows):
    python examples/cdc_covid_download_bounded_sample.py \
        /data/cdc/n8mc-b4w4.csv --start 2020-07 --months 24 --per-class 100
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://data.cdc.gov/resource/n8mc-b4w4.json"
FIELDS = (
    "case_month",
    "res_state",
    "state_fips_code",
    "res_county",
    "county_fips_code",
    "age_group",
    "sex",
    "race",
    "ethnicity",
    "case_positive_specimen",
    "case_onset_interval",
    "process",
    "exposure_yn",
    "current_status",
    "symptom_status",
    "hosp_yn",
    "icu_yn",
    "death_yn",
    "underlying_conditions_yn",
)


def query(**parameters: str | int) -> list[dict[str, Any]]:
    """Run one bounded Socrata JSON query."""
    url = f"{ENDPOINT}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": "PyHealth-example"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def month_range(start: str, count: int) -> list[str]:
    """Build consecutive YYYY-MM values from a validated starting month."""
    year, month = (int(part) for part in start.split("-"))
    if not 1 <= month <= 12 or count < 1:
        raise ValueError("start must be YYYY-MM and months must be positive")
    return [
        f"{year + (month - 1 + offset) // 12:04d}-{(month - 1 + offset) % 12 + 1:02d}"
        for offset in range(count)
    ]


def stratum_counts() -> dict[tuple[str, str], int]:
    """Return all public month/outcome counts in one aggregate query."""
    rows = query(
        **{
            "$select": "case_month,death_yn,count(*) as n",
            "$where": 'death_yn in ("Yes","No")',
            "$group": "case_month,death_yn",
            "$limit": 5000,
        }
    )
    return {(row["case_month"], row["death_yn"]): int(row["n"]) for row in rows}


def deterministic_offsets(
    month: str, label: str, count: int, size: int, chunks: int
) -> list[tuple[int, int]]:
    """Choose one stable block from every segment of a row-ID ordering."""
    if count < size or size < chunks or size % chunks:
        raise ValueError(
            f"{month} {label}: count must cover per-class, which must be a "
            "positive multiple of chunks"
        )
    block = size // chunks
    offsets = []
    for index in range(chunks):
        start = index * count // chunks
        stop = (index + 1) * count // chunks
        choices = stop - start - block + 1
        digest = hashlib.sha256(
            f"pyhealth-cdc-v2:{month}:{label}:{index}".encode()
        ).digest()
        offsets.append((start + int.from_bytes(digest[:8], "big") % choices, block))
    return offsets


def download(
    output: Path, start: str, months: int, per_class: int, chunks: int
) -> None:
    """Write every requested stratum to one official-schema CSV."""
    counts = stratum_counts()
    requests = []
    for month in month_range(start, months):
        for label in ("No", "Yes"):
            count = counts.get((month, label), 0)
            for offset, limit in deterministic_offsets(
                month, label, count, per_class, chunks
            ):
                requests.append((month, label, count, offset, limit))

    def fetch(spec: tuple[str, str, int, int, int]) -> tuple[tuple, list[dict]]:
        month, label, _, offset, limit = spec
        rows = query(
            **{
                "$where": f'case_month="{month}" AND death_yn="{label}"',
                "$order": ":id",
                "$offset": offset,
                "$limit": limit,
            }
        )
        if len(rows) != limit:
            raise RuntimeError(
                f"Expected {limit} rows for {month} {label}, got {len(rows)}"
            )
        return spec, rows

    fetched: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for spec, rows in executor.map(fetch, requests):
            month, label, count, offset, _ = spec
            fetched.setdefault((month, label, count), []).extend(rows)
            print(f"{month} {label}: segment at offset {offset} complete", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for month in month_range(start, months):
            for label in ("No", "Yes"):
                count = counts[(month, label)]
                rows = fetched[(month, label, count)]
                if len(rows) != per_class:
                    raise RuntimeError(
                        f"Expected {per_class} rows for {month} {label}, got {len(rows)}"
                    )
                writer.writerows(rows)
                print(f"{month} {label}: wrote {len(rows)} of {count} rows")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", default="2020-07")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--chunks", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download(
        arguments.output,
        arguments.start,
        arguments.months,
        arguments.per_class,
        arguments.chunks,
    )
