"""Benchmark script for MIMIC-IV length of stay prediction across multiple num_workers.

This benchmark measures:
1. Time to load the base dataset (once)
2. Time to process the task for each num_workers value (optionally repeated)
3. Cache sizes for base dataset and each task run
4. Peak memory usage (RSS, includes child processes)

Typical usage:
  python benchmark_workers_n_length_of_stay.py
  python benchmark_workers_n_length_of_stay.py --workers 1,4,8,12,16 --repeats 3
  python benchmark_workers_n_length_of_stay.py --dev --workers 1,2,4

Notes:
- The task cache directory is recreated for each run and deleted after measuring size.
- The base dataset cache is also deleted before and after each run, so every run
    measures full dataset + task processing time from scratch.
- Peak memory is sampled in a background thread; it reports total RSS of the current
  process plus all child processes.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import psutil

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import LengthOfStayPredictionMIMIC4

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


@dataclass
class RunResult:
    num_workers: int
    repeat_index: int
    dataset_load_s: float
    task_process_s: float
    total_s: float
    base_cache_bytes: int
    task_cache_bytes: int
    peak_rss_bytes: int


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_directory_size(path: str | Path) -> int:
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    try:
        for entry in p.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except FileNotFoundError:
                    # File might disappear if something is concurrently modifying cache.
                    pass
    except Exception as e:
        print(f"Error calculating size for {p}: {e}")
    return total


def set_memory_limit(max_memory_gb: float) -> None:
    """Set a hard virtual memory limit for the process."""
    if not HAS_RESOURCE:
        print(
            "Warning: resource module not available (Windows?). "
            "Memory limit not enforced."
        )
        return

    max_memory_bytes = int(max_memory_gb * 1024**3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        print(f"✓ Memory limit set to {max_memory_gb} GB")
    except Exception as e:
        print(f"Warning: Failed to set memory limit: {e}")


class PeakMemoryTracker:
    """Tracks peak RSS for current process + children."""

    def __init__(self, poll_interval_s: float = 0.1) -> None:
        self._proc = psutil.Process(os.getpid())
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self._peak = 0

    def stop(self) -> None:
        self._stop.set()

    def peak_bytes(self) -> int:
        with self._lock:
            return self._peak

    def _total_rss_bytes(self) -> int:
        total = 0
        try:
            total += self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._total_rss_bytes()
            with self._lock:
                if rss > self._peak:
                    self._peak = rss
            time.sleep(self._poll_interval_s)


def parse_workers(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    workers: list[int] = []
    for p in parts:
        w = int(p)
        if w <= 0:
            raise argparse.ArgumentTypeError("All worker counts must be > 0")
        workers.append(w)
    if not workers:
        raise argparse.ArgumentTypeError("No workers provided")
    return workers


def remove_dir(path: str | Path, retries: int = 3, delay: float = 1.0) -> None:
    """Remove a directory with retry logic for busy file handles."""
    p = Path(path)
    if not p.exists():
        return
    for attempt in range(retries):
        try:
            shutil.rmtree(p)
            return
        except OSError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"Warning: Failed to delete {p} after {retries} attempts: {e}")


def median(values: Iterable[float]) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MIMIC-IV length of stay prediction over multiple num_workers"
    )
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=[1, 4, 8, 12, 16],
        help="Comma-separated list of num_workers values (default: 1,4,8,12,16)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per worker setting (default: 1)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev mode dataset loading (smaller subset)",
    )
    parser.add_argument(
        "--ehr-root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        help="Path to MIMIC-IV root directory",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="/shared/eng/pyhealth/",
        help="Root directory for benchmark caches (default: /shared/eng/pyhealth/)",
    )
    parser.add_argument(
        "--enable-memory-limit",
        action="store_true",
        help="Enforce a hard memory limit via resource.setrlimit (Unix only)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help=(
            "Hard memory limit in GB (only used if --enable-memory-limit is set). "
            "If omitted, no memory limit is applied by default."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="../benchmark_results_los_workers_sweep.csv",
        help="Where to write per-run results as CSV",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")

    if args.enable_memory_limit:
        if args.max_memory_gb is None:
            raise SystemExit(
                "When using --enable-memory-limit, you must also pass "
                "--max-memory-gb (e.g., --max-memory-gb 32)."
            )
        set_memory_limit(args.max_memory_gb)

    tracker = PeakMemoryTracker(poll_interval_s=0.1)
    tracker.start()

    print("=" * 80)
    print("BENCHMARK: Length of Stay Prediction num_workers sweep")
    print(f"workers={args.workers} repeats={args.repeats} dev={args.dev}")
    if args.enable_memory_limit:
        print(f"Memory Limit: {args.max_memory_gb} GB (ENFORCED)")
    else:
        print("Memory Limit: None (unrestricted)")
    print(f"ehr_root: {args.ehr_root}")
    print(f"cache_root: {args.cache_root}")
    print("=" * 80)

    cache_root = Path(args.cache_root)
    base_cache_dir = cache_root / (
        "base_dataset_los_dev" if args.dev else "base_dataset_los"
    )

    total_start = time.time()

    results: list[RunResult] = []

    print("\n[1/1] Sweeping num_workers (each run reloads dataset + task)...")
    for w in args.workers:
        for r in range(args.repeats):
            # Ensure no cache artifacts before this run.
            remove_dir(base_cache_dir)

            tracker.reset()
            run_start = time.time()

            dataset_start = time.time()
            base_dataset = MIMIC4Dataset(
                ehr_root=args.ehr_root,
                ehr_tables=[
                    "patients",
                    "admissions",
                    "diagnoses_icd",
                    "procedures_icd",
                    "prescriptions",
                ],
                dev=args.dev,
                cache_dir=str(base_cache_dir),
            )
            dataset_load_s = time.time() - dataset_start
            base_cache_bytes = get_directory_size(base_cache_dir)

            task_start = time.time()
            sample_dataset = base_dataset.set_task(
                LengthOfStayPredictionMIMIC4(),
                num_workers=w,
            )

            task_process_s = time.time() - task_start
            total_s = time.time() - run_start
            peak_rss_bytes = tracker.peak_bytes()
            tasks_dir = base_cache_dir / "tasks"
            task_cache_bytes = get_directory_size(tasks_dir)

            # Capture sample count BEFORE cleaning up the cache (litdata needs it).
            num_samples = len(sample_dataset)

            # Release the dataset reference to free file handles before cleanup.
            del sample_dataset
            del base_dataset

            # Clean up to avoid disk growth across an overnight sweep.
            remove_dir(base_cache_dir)

            results.append(
                RunResult(
                    num_workers=w,
                    repeat_index=r,
                    dataset_load_s=dataset_load_s,
                    task_process_s=task_process_s,
                    total_s=total_s,
                    base_cache_bytes=base_cache_bytes,
                    task_cache_bytes=task_cache_bytes,
                    peak_rss_bytes=peak_rss_bytes,
                )
            )

            print(
                "✓ "
                f"workers={w:>2} repeat={r+1:>2}/{args.repeats} "
                f"samples={num_samples} "
                f"dataset={dataset_load_s:.2f}s "
                f"task={task_process_s:.2f}s "
                f"total={total_s:.2f}s "
                f"peak_rss={format_size(peak_rss_bytes)} "
                f"base_cache={format_size(base_cache_bytes)} "
                f"task_cache={format_size(task_cache_bytes)}"
            )

    total_sweep_s = time.time() - total_start

    # Write CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for rr in results:
            writer.writerow(asdict(rr))

    # Print a compact summary per worker (median across repeats)
    print("\n" + "=" * 80)
    print("SUMMARY (median across repeats)")
    print("=" * 80)

    for w in args.workers:
        wrs = [rr for rr in results if rr.num_workers == w]
        med_task = median([rr.task_process_s for rr in wrs])
        med_total = median([rr.total_s for rr in wrs])
        med_peak = median([float(rr.peak_rss_bytes) for rr in wrs])
        med_cache = median([float(rr.task_cache_bytes) for rr in wrs])
        print(
            f"workers={w:>2}  "
            f"task_med={med_task:>8.2f}s  "
            f"total_med={med_total:>8.2f}s  "
            f"peak_rss_med={format_size(int(med_peak)):>10}  "
            f"task_cache_med={format_size(int(med_cache)):>10}"
        )

    print("\nArtifacts:")
    print(f"  - CSV: {out_csv}")
    print(f"  - Cache root: {cache_root}")
    print("\nTotals:")
    print(f"  - Sweep wall time: {total_sweep_s:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

