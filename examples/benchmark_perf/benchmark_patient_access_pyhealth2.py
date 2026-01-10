"""Benchmark: PyHealth 2.0 - Data Loading & Single Patient Access

Measures:
1. Time to load/initialize the dataset from raw MIMIC-IV data
2. Time to access a single patient after loading
3. Total time (load + access)

Usage:
  # Activate PyHealth 2.0 environment first
  source activate pyhealth312
  
  # Run benchmark
  python benchmark_patient_access_pyhealth2.py
  python benchmark_patient_access_pyhealth2.py --patient-id 10014729 --workers 8
  python benchmark_patient_access_pyhealth2.py --dev
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

import psutil

from pyhealth.datasets import MIMIC4Dataset


# =============================================================================
# Benchmark Result
# =============================================================================

@dataclass
class BenchmarkResult:
    approach: str
    data_load_s: float
    patient_access_1st_s: float  # First access (cold cache)
    patient_access_2nd_s: float  # Second access (warm cache)
    total_s: float
    peak_rss_bytes: int
    patient_found: bool
    num_events: int


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


class PeakMemoryTracker:
    """Tracks peak RSS for current process + children."""

    def __init__(self, poll_interval_s: float = 0.05) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PyHealth 2.0 data loading and single patient access"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="10014729",
        help="Patient ID to access (default: 10014729)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers (default: 8)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev mode (smaller subset)",
    )
    parser.add_argument(
        "--ehr-root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        help="Path to MIMIC-IV root directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/shared/eng/pyhealth/benchmark_patient_access_2.0",
        help="Cache directory",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Do not clear cache before benchmark",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_patient_access_pyhealth2.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BENCHMARK: PyHealth 2.0 - Data Loading & Patient Access")
    print("=" * 80)
    print(f"Patient ID: {args.patient_id}")
    print(f"Workers: {args.workers}")
    print(f"Dev mode: {args.dev}")
    print(f"EHR Root: {args.ehr_root}")
    print(f"Cache dir: {args.cache_dir}")
    print("=" * 80)

    # Clear cache for accurate timing
    cache_path = Path(args.cache_dir)
    if not args.no_clear_cache:
        if cache_path.exists():
            print("\nClearing cache...")
            shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    tracker = PeakMemoryTracker(poll_interval_s=0.05)
    tracker.start()
    tracker.reset()

    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    load_start = time.time()
    
    dataset = MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=["patients", "admissions", "diagnoses_icd", "procedures_icd", "labevents"],
        dev=args.dev,
        cache_dir=str(cache_path),
        num_workers=args.workers,
    )
    
    data_load_s = time.time() - load_start
    print(f"  Dataset loaded in {data_load_s:.2f}s")

    # Step 2: First patient access (cold cache)
    print(f"\n[Step 2] First access to patient {args.patient_id} (cold cache)...")
    access_1_start = time.time()
    
    patient_ids = dataset.unique_patient_ids
    print(f"  Number of patients: {len(patient_ids)}")
    patient_found = args.patient_id in patient_ids
    
    if patient_found:
        patient = dataset.get_patient(args.patient_id)
        num_events = len(patient.get_events())
        print(f"  Patient found!")
        print(f"  Number of events: {num_events}")
    else:
        num_events = 0
        print(f"  Patient NOT found!")
        # List available patient IDs (first 10)
        print(f"  Available patient IDs (first 10): {patient_ids[:10]}")
    
    patient_access_1st_s = time.time() - access_1_start
    
    # Step 3: Second patient access (warm cache)
    print(f"\n[Step 3] Second access to patient {args.patient_id} (warm cache)...")
    access_2_start = time.time()
    
    if patient_found:
        patient = dataset.get_patient(args.patient_id)
        # Re-count events to ensure we're actually accessing the data
        count = len(patient.get_events())
    
    patient_access_2nd_s = time.time() - access_2_start
    
    total_s = data_load_s + patient_access_1st_s + patient_access_2nd_s
    peak_rss = tracker.peak_bytes()
    
    tracker.stop()

    # Cleanup
    del dataset
    if not args.no_clear_cache and cache_path.exists():
        shutil.rmtree(cache_path)

    result = BenchmarkResult(
        approach="pyhealth_2.0",
        data_load_s=data_load_s,
        patient_access_1st_s=patient_access_1st_s,
        patient_access_2nd_s=patient_access_2nd_s,
        total_s=total_s,
        peak_rss_bytes=peak_rss,
        patient_found=patient_found,
        num_events=num_events,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: PyHealth 2.0")
    print("=" * 80)
    access_1_str = f"{patient_access_1st_s*1000:.2f}ms" if patient_access_1st_s < 1 else f"{patient_access_1st_s:.2f}s"
    access_2_str = f"{patient_access_2nd_s*1000:.2f}ms" if patient_access_2nd_s < 1 else f"{patient_access_2nd_s:.2f}s"
    print(f"  Data load time:            {data_load_s:.2f}s")
    print(f"  Patient access (1st/cold): {access_1_str}")
    print(f"  Patient access (2nd/warm): {access_2_str}")
    print(f"  Total time:                {total_s:.2f}s")
    print(f"  Peak RSS:                  {format_size(peak_rss)}")
    print(f"  Patient found:             {patient_found}")
    print(f"  Events:                    {num_events}")

    # Write CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))
    print(f"\nResults saved to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

