#!/usr/bin/env python3
"""Benchmark streaming vs in-memory sample loaders.

This script benchmarks PyHealth's:
1) New streaming data loader (`SampleDataset`)
2) Legacy in-memory loader (`InMemorySampleDataset`)

Metrics collected:
- Wall-clock time (seconds)
- Peak RAM usage tracked by `tracemalloc` (MB)
- Throughput (patients/second)

By default, it runs on dataset sizes of 1k, 10k, and 100k patients.
"""

from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyhealth.datasets import create_sample_dataset, get_dataloader


DEFAULT_SIZES = [1_000, 10_000, 100_000]
INPUT_SCHEMA = {"feature": "raw"}
OUTPUT_SCHEMA = {"label": "raw"}


def parse_sizes(raw: str) -> List[int]:
    """Parse comma-separated patient counts."""
    values = []
    for token in raw.split(","):
        stripped = token.strip().replace("_", "")
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("No valid sizes were provided.")
    return values


def generate_samples(num_patients: int) -> List[Dict[str, Any]]:
    """Generate synthetic samples with one record per patient."""
    samples = []
    for i in range(num_patients):
        samples.append(
            {
                "patient_id": f"p{i}",
                "record_id": f"r{i}",
                "feature": [i % 17, (i + 1) % 17, (i + 2) % 17],
                "label": i % 2,
            }
        )
    return samples


def count_batch_patients(batch: Dict[str, Any]) -> int:
    """Count patients in a collated batch."""
    if "patient_id" in batch:
        return len(batch["patient_id"])
    first_value = next(iter(batch.values()))
    return len(first_value)


def benchmark_loader(
    samples: List[Dict[str, Any]],
    loader_name: str,
    in_memory: bool,
    batch_size: int,
) -> Dict[str, Any]:
    """Benchmark one loader mode on a fixed sample list."""
    dataset = None
    dataloader = None
    tracemalloc.start()
    start_time = time.perf_counter()
    processed_patients = 0

    try:
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            dataset_name="loader_benchmark",
            task_name="loader_benchmark_task",
            in_memory=in_memory,
        )

        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        for batch in dataloader:
            processed_patients += count_batch_patients(batch)
    finally:
        elapsed = time.perf_counter() - start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if dataset is not None:
            dataset.close()
        del dataloader
        del dataset
        gc.collect()

    peak_mb = peak_bytes / (1024**2)
    throughput = processed_patients / elapsed if elapsed > 0 else float("inf")

    return {
        "loader": loader_name,
        "num_patients": len(samples),
        "wall_time_sec": elapsed,
        "peak_ram_mb": peak_mb,
        "throughput_patients_per_sec": throughput,
        "processed_patients": processed_patients,
        "batch_size": batch_size,
    }


def run_benchmark(sizes: Iterable[int], batch_size: int) -> pd.DataFrame:
    """Run all benchmark combinations and return a DataFrame."""
    records: List[Dict[str, Any]] = []
    loader_configs = [
        ("streaming", False),
        ("in_memory", True),
    ]

    for size in sizes:
        samples = generate_samples(size)
        for loader_name, in_memory in loader_configs:
            record = benchmark_loader(
                samples=samples,
                loader_name=loader_name,
                in_memory=in_memory,
                batch_size=batch_size,
            )
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(["num_patients", "loader"]).reset_index(drop=True)
    return df


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    """Create a comparison chart for runtime, memory, and throughput."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted(df["num_patients"].unique())

    metrics = [
        ("wall_time_sec", "Wall-Clock Time (s)"),
        ("peak_ram_mb", "Peak RAM (MB)"),
        ("throughput_patients_per_sec", "Throughput (patients/s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, (metric, title) in zip(axes, metrics):
        for loader_name, group in df.groupby("loader"):
            group = group.sort_values("num_patients")
            axis.plot(
                group["num_patients"],
                group[metric],
                marker="o",
                linewidth=2,
                label=loader_name,
            )
        axis.set_title(title)
        axis.set_xlabel("Patients")
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{value:,}" for value in sizes], rotation=30)
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark streaming vs in-memory loaders in PyHealth."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,10000,100000",
        help="Comma-separated patient counts (default: 1000,10000,100000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="DataLoader batch size (default: 256).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("benchmarks/loader_benchmark_results.csv"),
        help="CSV output path (default: benchmarks/loader_benchmark_results.csv).",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("benchmarks/loader_benchmark_comparison.png"),
        help="Plot output path (default: benchmarks/loader_benchmark_comparison.png).",
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    df = run_benchmark(sizes=sizes, batch_size=args.batch_size)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    plot_results(df, args.plot_out)

    print(df.to_string(index=False))
    print(f"\nSaved CSV: {args.csv_out}")
    print(f"Saved chart: {args.plot_out}")


if __name__ == "__main__":
    main()
