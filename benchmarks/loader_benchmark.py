#!/usr/bin/env python3
"""Benchmark PyHealth streaming and legacy sample loaders.

This script compares:
1) Streaming loader: ``SampleDataset`` (disk-backed)
2) Legacy loader: ``InMemorySampleDataset`` (in-memory)

It benchmarks synthetic patient samples at:
- small: 100 patients
- medium: 1,000 patients
- large: 5,000 patients

Metrics:
- Peak RAM via ``tracemalloc`` (MB)
- Wall-clock time via ``time.perf_counter`` (seconds)
- Throughput (patients/second)
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import os
import sys
import time
import tracemalloc
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# Keep runtime cache local/writable for sandboxed environments.
RUNTIME_CACHE_DIR = Path("benchmarks/.runtime_cache")
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_MPL_DIR = RUNTIME_CACHE_DIR / "matplotlib"
RUNTIME_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_MPL_DIR.resolve()))
# PyHealth uses Path.home()/.cache directly; set HOME to local runtime cache.
RUNTIME_HOME = (RUNTIME_CACHE_DIR / "home").resolve()
RUNTIME_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(RUNTIME_HOME)

# Some environments may not ship Python's optional _lzma module.
# TorchVision imports lzma at import-time; provide a minimal fallback so this
# benchmark can still run when lzma compression is not used.
try:
    import lzma  # noqa: F401
except ModuleNotFoundError:
    lzma_stub = types.ModuleType("lzma")

    def _missing_lzma(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "lzma support is unavailable in this Python build."
        )

    class _MissingLZMAFile:
        def __init__(self, *_args, **_kwargs):
            _missing_lzma()

    class _MissingLZMAError(Exception):
        pass

    lzma_stub.open = _missing_lzma  # type: ignore[attr-defined]
    lzma_stub.LZMAFile = _MissingLZMAFile  # type: ignore[attr-defined]
    lzma_stub.LZMAError = _MissingLZMAError  # type: ignore[attr-defined]
    sys.modules["lzma"] = lzma_stub

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyhealth.datasets as datasets_module
from pyhealth.datasets import (
    BaseDataset,
    InMemorySampleDataset,
    SampleDataset,
    create_sample_dataset,
    get_dataloader,
)


SCALES: Sequence[tuple[str, int]] = (
    ("small", 100),
    ("medium", 1_000),
    ("large", 5_000),
)
INPUT_SCHEMA = {"feature": "raw"}
OUTPUT_SCHEMA = {"label": "raw"}
DEFAULT_BATCH_SIZE = 256
DEFAULT_RESULTS_CSV = Path("benchmarks/results.csv")
DEFAULT_CHART_PATH = Path("benchmarks/benchmark_chart.png")

LEGACY_LOADER = "legacy_in_memory"
STREAMING_LOADER = "streaming"


@dataclass
class BenchmarkRow:
    scale: str
    num_patients: int
    loader: str
    dataset_class: str
    status: str
    wall_time_sec: float | None
    peak_ram_mb: float | None
    throughput_patients_per_sec: float | None
    note: str


def parse_sizes(raw: str) -> List[int]:
    """Parse comma-separated patient counts."""
    parsed: List[int] = []
    for token in raw.split(","):
        token = token.strip().replace("_", "")
        if not token:
            continue
        parsed.append(int(token))
    if not parsed:
        raise ValueError("No valid sizes provided.")
    return parsed


def generate_samples(num_patients: int) -> List[Dict[str, Any]]:
    """Generate synthetic sample data with one sample per patient."""
    return [
        {
            "patient_id": f"p{i}",
            "record_id": f"r{i}",
            "feature": [i % 17, (i * 3) % 23, (i * 7) % 31],
            "label": i % 2,
        }
        for i in range(num_patients)
    ]


def count_batch_patients(batch: Dict[str, Any]) -> int:
    """Count patients in a collated batch."""
    if "patient_id" in batch:
        return len(batch["patient_id"])
    first_value = next(iter(batch.values()))
    return len(first_value)


def discover_streaming_supported_datasets() -> List[str]:
    """Discover dataset classes that inherit BaseDataset."""
    supported: List[str] = []
    for name, obj in inspect.getmembers(datasets_module, inspect.isclass):
        if obj in (BaseDataset, SampleDataset, InMemorySampleDataset):
            continue
        if issubclass(obj, BaseDataset):
            supported.append(name)
    return sorted(set(supported))


def summarize_loader_apis() -> Dict[str, str]:
    """Summarize loader signatures to document API differences."""
    create_sig = inspect.signature(create_sample_dataset)
    streaming_sig = inspect.signature(SampleDataset.__init__)
    legacy_sig = inspect.signature(InMemorySampleDataset.__init__)
    return {
        "create_sample_dataset": str(create_sig),
        "streaming_loader": f"SampleDataset{streaming_sig}",
        "legacy_loader": f"InMemorySampleDataset{legacy_sig}",
        "supports_in_memory_flag": str("in_memory" in create_sig.parameters),
    }


def print_codebase_exploration() -> None:
    """Print codebase-derived info before running benchmarks."""
    supported = discover_streaming_supported_datasets()
    api = summarize_loader_apis()

    print("\nCodebase Exploration")
    print("====================")
    print(
        f"Streaming-capable dataset classes (BaseDataset subclasses): {len(supported)}"
    )
    print(", ".join(supported))
    print("\nLoader API differences:")
    print(f"- Factory helper: create_sample_dataset{api['create_sample_dataset']}")
    print(f"- Streaming loader: {api['streaming_loader']}")
    print(f"- Legacy loader: {api['legacy_loader']}")
    print(
        "- Mode parameter: "
        f"create_sample_dataset(..., in_memory=<bool>) -> "
        "False: streaming / True: legacy in-memory"
    )


def _benchmark_one(
    scale: str,
    num_patients: int,
    samples: List[Dict[str, Any]],
    loader_name: str,
    in_memory: bool,
    batch_size: int,
) -> BenchmarkRow:
    dataset = None
    dataloader = None
    processed = 0
    status = "ok"
    note = ""
    dataset_class = ""

    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                dataset = create_sample_dataset(
                    samples=samples,
                    input_schema=INPUT_SCHEMA,
                    output_schema=OUTPUT_SCHEMA,
                    dataset_name="loader_benchmark",
                    task_name=f"{scale}_{loader_name}",
                    in_memory=in_memory,
                )
                dataset_class = dataset.__class__.__name__

                dataloader = get_dataloader(
                    dataset=dataset,
                    batch_size=min(batch_size, max(1, num_patients)),
                    shuffle=False,
                )
                for batch in dataloader:
                    processed += count_batch_patients(batch)
    except Exception as exc:
        status = "skipped" if loader_name == STREAMING_LOADER else "error"
        note = f"{type(exc).__name__}: {exc}"
    finally:
        wall_time = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if dataset is not None and hasattr(dataset, "close"):
            dataset.close()
        del dataloader
        del dataset
        gc.collect()

    if status != "ok":
        return BenchmarkRow(
            scale=scale,
            num_patients=num_patients,
            loader=loader_name,
            dataset_class=dataset_class or "n/a",
            status=status,
            wall_time_sec=None,
            peak_ram_mb=None,
            throughput_patients_per_sec=None,
            note=note,
        )

    wall_time = float(wall_time)
    peak_ram_mb = peak_bytes / (1024**2)
    throughput = processed / wall_time if wall_time > 0 else None
    return BenchmarkRow(
        scale=scale,
        num_patients=num_patients,
        loader=loader_name,
        dataset_class=dataset_class,
        status=status,
        wall_time_sec=wall_time,
        peak_ram_mb=peak_ram_mb,
        throughput_patients_per_sec=throughput,
        note=note,
    )


def run_benchmark(sizes: Iterable[int], batch_size: int) -> pd.DataFrame:
    """Run benchmark for both loaders on each scale."""
    label_for_size = {size: label for label, size in SCALES}
    records: List[BenchmarkRow] = []

    streaming_available = True
    streaming_skip_note = ""

    for size in sizes:
        scale_label = label_for_size.get(size, f"custom_{size}")
        samples = generate_samples(size)

        records.append(
            _benchmark_one(
                scale=scale_label,
                num_patients=size,
                samples=samples,
                loader_name=LEGACY_LOADER,
                in_memory=True,
                batch_size=batch_size,
            )
        )

        if streaming_available:
            streaming_row = _benchmark_one(
                scale=scale_label,
                num_patients=size,
                samples=samples,
                loader_name=STREAMING_LOADER,
                in_memory=False,
                batch_size=batch_size,
            )
            records.append(streaming_row)
            if streaming_row.status != "ok":
                streaming_available = False
                streaming_skip_note = (
                    streaming_row.note
                    or "Streaming mode unavailable in current environment."
                )
        else:
            records.append(
                BenchmarkRow(
                    scale=scale_label,
                    num_patients=size,
                    loader=STREAMING_LOADER,
                    dataset_class="n/a",
                    status="skipped",
                    wall_time_sec=None,
                    peak_ram_mb=None,
                    throughput_patients_per_sec=None,
                    note=streaming_skip_note,
                )
            )

    df = pd.DataFrame(asdict(row) for row in records)
    df = df.sort_values(["num_patients", "loader"]).reset_index(drop=True)
    return df


def format_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format values for terminal display."""
    display_df = df.copy()
    for col in ["wall_time_sec", "peak_ram_mb", "throughput_patients_per_sec"]:
        display_df[col] = display_df[col].map(
            lambda x: "-" if pd.isna(x) else f"{x:,.4f}"
        )
    return display_df


def _metric_values(
    df: pd.DataFrame, scales: List[str], loader: str, metric: str
) -> List[float]:
    values: List[float] = []
    for scale in scales:
        row = df[
            (df["scale"] == scale) & (df["loader"] == loader) & (df["status"] == "ok")
        ]
        if row.empty:
            values.append(np.nan)
        else:
            values.append(float(row.iloc[0][metric]))
    return values


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart comparing RAM and wall time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scales = [label for label, _ in SCALES if label in set(df["scale"].tolist())]
    if not scales:
        scales = sorted(df["scale"].unique().tolist())

    metrics = [
        ("peak_ram_mb", "Peak RAM (MB)"),
        ("wall_time_sec", "Wall Time (s)"),
    ]
    loaders = [
        (LEGACY_LOADER, "Legacy in-memory"),
        (STREAMING_LOADER, "Streaming"),
    ]
    x = np.arange(len(scales))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, title) in zip(axes, metrics):
        for idx, (loader_key, loader_label) in enumerate(loaders):
            offset = (idx - 0.5) * width
            values = _metric_values(df, scales, loader_key, metric)
            valid_points = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
            if not valid_points:
                continue
            x_positions = [x[i] + offset for i, _ in valid_points]
            y_values = [v for _, v in valid_points]
            ax.bar(x_positions, y_values, width=width, label=loader_label)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scales)
        ax.set_xlabel("Scale")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    streaming_skipped = df[
        (df["loader"] == STREAMING_LOADER) & (df["status"] != "ok")
    ]
    if not streaming_skipped.empty:
        note = streaming_skipped.iloc[0]["note"]
        fig.text(
            0.01,
            0.01,
            f"Note: streaming benchmark skipped in this environment. {note}",
            fontsize=9,
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PyHealth streaming vs legacy in-memory loaders."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,1000,5000",
        help="Comma-separated patient counts (default: 100,1000,5000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"DataLoader batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"CSV output path (default: {DEFAULT_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--chart-out",
        type=Path,
        default=DEFAULT_CHART_PATH,
        help=f"Chart output path (default: {DEFAULT_CHART_PATH}).",
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    print_codebase_exploration()

    results_df = run_benchmark(sizes=sizes, batch_size=args.batch_size)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.csv_out, index=False)

    plot_results(results_df, args.chart_out)

    print("\nBenchmark Results")
    print("=================")
    print(format_results_table(results_df).to_string(index=False))
    print(f"\nSaved CSV: {args.csv_out}")
    print(f"Saved chart: {args.chart_out}")


if __name__ == "__main__":
    main()
