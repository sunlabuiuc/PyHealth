"""The dataset cache is rebuilt when a source file or the YAML config changes.

PyHealth caches each dataset's event table. The cache key includes a hash of
the YAML config and the size and modification time of every source file, so
regenerating a source file at the same path (a common step in an analysis
pipeline) gives a fresh cache instead of silently reusing stale data.

Runs in a few seconds on CPU with a tiny synthetic table; no download.

Usage:
    python examples/dataset_cache_refresh.py
"""

import tempfile
from pathlib import Path

from pyhealth.datasets import BaseDataset

CONFIG = """version: "1.0"
tables:
  visits:
    file_path: "visits.csv"
    patient_id: "patient_id"
    timestamp: "visit_time"
    attributes:
      - "diagnosis"
"""


def write_visits(path: Path, patients: list[str]) -> None:
    rows = ["patient_id,visit_time,diagnosis"]
    rows += [f"{p},2024-01-15 09:00:00,J45" for p in patients]
    path.write_text("\n".join(rows) + "\n")


def load(root: Path, cache: Path) -> BaseDataset:
    return BaseDataset(
        root=str(root),
        tables=["visits"],
        dataset_name="cache_refresh_demo",
        config_path=str(root / "config.yaml"),
        cache_dir=str(cache),
    )


def n_patients(dataset: BaseDataset) -> int:
    return dataset.global_event_df.select("patient_id").unique().collect().height


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root, cache = Path(tmp) / "data", Path(tmp) / "cache"
        root.mkdir()
        (root / "config.yaml").write_text(CONFIG)

        write_visits(root / "visits.csv", ["p1", "p2"])
        first = load(root, cache)
        print("v1:", n_patients(first), "patients, cache", first.cache_dir.name)

        # Regenerate the source file at the same path, e.g. after a cohort update.
        write_visits(root / "visits.csv", ["p1", "p2", "p3", "p4"])
        second = load(root, cache)
        print("v2:", n_patients(second), "patients, cache", second.cache_dir.name)

        again = load(root, cache)
        print("unchanged rerun reuses cache:", again.cache_dir == second.cache_dir)


if __name__ == "__main__":
    main()
