"""Legacy PyHealth 1.1.6 Benchmark script for MIMIC-IV drug recommendation.

This benchmark measures performance across multiple worker counts (via pandarallel):
1. Time to load the base dataset
2. Time to process the task for each num_workers value
3. Cache sizes for base dataset
4. Peak memory usage (RSS, includes child processes)

Typical usage:
  # First, install the legacy version:
  pip install pyhealth==1.1.6

  # Then run the benchmark:
  python benchmark_legacy_drug_rec.py
  python benchmark_legacy_drug_rec.py --workers 1,4,8,12,16 --repeats 3
  python benchmark_legacy_drug_rec.py --dev --workers 1,2,4

API differences from PyHealth 2.0:
- Uses `root` instead of `ehr_root`
- Uses `tables` instead of `ehr_tables`
- Uses `refresh_cache` instead of `cache_dir`
- set_task() takes a `task_fn` function instead of a task class
- Parallelization via pandarallel (not num_workers in set_task)

Notes:
- This uses the PyHealth 1.1.6 legacy API
- Uses the built-in drug_recommendation_mimic4_fn task function
- Pandarallel is re-initialized for each worker count
- Cache is cleared before each run to ensure fresh timing data
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

# Import pandarallel - will be initialized before each run
from pandarallel import pandarallel

# Global variable to store desired worker count for monkey-patching
_DESIRED_NB_WORKERS: int = 16

# Store the original pandarallel.initialize function
_original_pandarallel_initialize = pandarallel.initialize


def _patched_pandarallel_initialize(*args, **kwargs):
    """Patched pandarallel.initialize that enforces our worker count.
    
    The legacy PyHealth code calls pandarallel.initialize() without nb_workers,
    which defaults to all CPUs. This patch ensures our desired worker count is used.
    """
    # Override nb_workers with our desired value
    kwargs['nb_workers'] = _DESIRED_NB_WORKERS
    return _original_pandarallel_initialize(*args, **kwargs)


# Apply the monkey-patch
pandarallel.initialize = _patched_pandarallel_initialize

# Legacy PyHealth 1.1.6 imports (AFTER monkey-patching)
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH
from pyhealth.tasks import drug_recommendation_mimic4_fn

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


# =============================================================================
# Benchmark Infrastructure
# =============================================================================


@dataclass
class RunResult:
    num_workers: int
    repeat_index: int
    dataset_load_s: float
    task_process_s: float
    total_s: float
    base_cache_bytes: int
    peak_rss_bytes: int
    num_samples: int


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
                    pass
    except Exception as e:
        print(f"Error calculating size for {p}: {e}")
    return total


def clear_pyhealth_cache(verbose: bool = True) -> int:
    """Clear all PyHealth cache files.
    
    PyHealth 1.1.6 stores cache as .pkl files in MODULE_CACHE_PATH.
    This function deletes all .pkl files in that directory.
    
    Args:
        verbose: Whether to print information about deleted files.
        
    Returns:
        Number of cache files deleted.
    """
    cache_path = Path(MODULE_CACHE_PATH)
    if not cache_path.exists():
        return 0
    
    deleted_count = 0
    total_size = 0
    
    # Find all .pkl cache files
    for cache_file in cache_path.glob("*.pkl"):
        try:
            file_size = cache_file.stat().st_size
            cache_file.unlink()
            deleted_count += 1
            total_size += file_size
        except OSError as e:
            if verbose:
                print(f"    Warning: Could not delete {cache_file}: {e}")
    
    if verbose and deleted_count > 0:
        print(f"    Cleared {deleted_count} cache files ({format_size(total_size)})")
    
    return deleted_count


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


def parse_workers(value: str) -> list[int]:
    """Parse comma-separated list of worker counts."""
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
        description="Legacy PyHealth 1.1.6 Benchmark for MIMIC-IV drug recommendation"
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
        "--root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        help="Path to MIMIC-IV hosp directory (legacy uses single root)",
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
        default="benchmark_legacy_drug_rec_workers_sweep.csv",
        help="Where to write per-run results as CSV",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Do not clear PyHealth cache before each run (default: clear cache)",
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
    print("LEGACY BENCHMARK: PyHealth 1.1.6 API - Drug Recommendation (Worker Sweep)")
    print(f"workers={args.workers} repeats={args.repeats} dev={args.dev}")
    print(f"clear_cache={not args.no_clear_cache}")
    if args.enable_memory_limit:
        print(f"Memory Limit: {args.max_memory_gb} GB (ENFORCED)")
    else:
        print("Memory Limit: None (unrestricted)")
    print(f"root: {args.root}")
    print(f"cache_path: {MODULE_CACHE_PATH}")
    print("=" * 80)

    # Determine cache directory based on PyHealth's default location
    cache_dir = Path(MODULE_CACHE_PATH)

    total_start = time.time()
    results: list[RunResult] = []

    print("\n[1/1] Sweeping num_workers (pandarallel)...")

    for w in args.workers:
        for r in range(args.repeats):
            # Clear cache before each run to ensure fresh timing data
            if not args.no_clear_cache:
                print(f"\n  Clearing PyHealth cache...")
                clear_pyhealth_cache(verbose=True)

            # Set the desired worker count for pandarallel
            # The monkey-patched initialize() will enforce this when PyHealth calls it
            global _DESIRED_NB_WORKERS
            _DESIRED_NB_WORKERS = w
            print(f"  Set pandarallel worker count to {w} (will be enforced via monkey-patch)")

            tracker.reset()
            run_start = time.time()

            # Step 1: Load base dataset using legacy API
            print(f"  workers={w} repeat={r + 1}/{args.repeats}: Loading dataset...")
            dataset_start = time.time()

            # Legacy PyHealth 1.1.6 API:
            # - Uses `root` instead of `ehr_root`
            # - Uses `tables` instead of `ehr_tables`
            # - Always use refresh_cache=True to force reprocessing (for accurate timing)
            # Drug recommendation uses: diagnoses_icd, procedures_icd, prescriptions
            base_dataset = MIMIC4Dataset(
                root=args.root,
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                dev=args.dev,
                code_mapping={"NDC": "ATC"},
                refresh_cache=True,  # Always refresh to measure processing time
            )
            dataset_load_s = time.time() - dataset_start
            base_cache_bytes = get_directory_size(cache_dir)

            # Step 2: Set task using legacy API (built-in drug_recommendation_mimic4_fn)
            print("    Processing task...")
            task_start = time.time()

            sample_dataset = base_dataset.set_task(
                task_fn=drug_recommendation_mimic4_fn
            )

            task_process_s = time.time() - task_start
            total_s = time.time() - run_start
            peak_rss_bytes = tracker.peak_bytes()

            # Get sample count
            num_samples = len(sample_dataset.samples)

            results.append(
                RunResult(
                    num_workers=w,
                    repeat_index=r,
                    dataset_load_s=dataset_load_s,
                    task_process_s=task_process_s,
                    total_s=total_s,
                    base_cache_bytes=base_cache_bytes,
                    peak_rss_bytes=peak_rss_bytes,
                    num_samples=num_samples,
                )
            )

            print(
                f"  ✓ workers={w:>2} repeat={r + 1:>2}/{args.repeats} "
                f"samples={num_samples} "
                f"dataset={dataset_load_s:.2f}s "
                f"task={task_process_s:.2f}s "
                f"total={total_s:.2f}s "
                f"peak_rss={format_size(peak_rss_bytes)} "
                f"cache={format_size(base_cache_bytes)}"
            )

            # Clean up references
            del sample_dataset
            del base_dataset

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
        med_dataset = median([rr.dataset_load_s for rr in wrs])
        med_task = median([rr.task_process_s for rr in wrs])
        med_total = median([rr.total_s for rr in wrs])
        med_peak = median([float(rr.peak_rss_bytes) for rr in wrs])
        print(
            f"workers={w:>2}  "
            f"dataset_med={med_dataset:>8.2f}s  "
            f"task_med={med_task:>8.2f}s  "
            f"total_med={med_total:>8.2f}s  "
            f"peak_rss_med={format_size(int(med_peak)):>10}"
        )

    print("\nArtifacts:")
    print(f"  - CSV: {out_csv}")
    print(f"  - Cache dir: {cache_dir}")
    print("\nTotals:")
    print(f"  - Sweep wall time: {total_sweep_s:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
