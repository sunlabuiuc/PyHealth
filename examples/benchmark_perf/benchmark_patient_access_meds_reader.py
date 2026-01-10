"""Benchmark: meds_reader - Data Loading & Single Patient Access

Measures:
1. Time to convert MIMIC-IV to MEDS format using meds_etl
2. Time to convert MEDS to meds_reader database
3. Time to access a single patient after loading
4. Total time (load + access)

For meds_reader, "data loading" includes:
- meds_etl_mimic: Convert MIMIC-IV directly to MEDS format
- meds_reader_convert: Convert MEDS to meds_reader database

Usage:
  # Activate meds_reader environment (with meds_etl installed)
  pip install meds_etl meds_reader
  
  # Run benchmark (uses existing DB if available)
  python benchmark_patient_access_meds_reader.py
  
  # Force reconversion of database
  python benchmark_patient_access_meds_reader.py --force-reconvert
  
  # Custom settings
  python benchmark_patient_access_meds_reader.py --patient-id 10014729 --threads 8
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

try:
    import meds_reader
except ImportError:
    raise ImportError(
        "meds_reader not found. Install with: pip install meds_reader\n"
        "Or from source: pip install -e /path/to/meds_reader"
    )


# =============================================================================
# Benchmark Result
# =============================================================================

@dataclass
class BenchmarkResult:
    approach: str
    data_load_s: float  # Full conversion time (or 0 if using cached DB)
    meds_etl_s: float  # meds_etl_mimic conversion time
    meds_reader_convert_s: float  # meds_reader_convert time
    patient_access_1st_s: float  # First access (cold cache)
    patient_access_2nd_s: float  # Second access (warm cache)
    total_s: float
    peak_rss_bytes: int
    patient_found: bool
    num_events: int
    used_cached_db: bool


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


# =============================================================================
# Data Conversion Functions
# =============================================================================

def run_meds_etl_mimic(
    src_mimic: str,
    output_dir: str,
    num_shards: int = 100,
    num_proc: int = 1,
    backend: str = "polars",
) -> float:
    """Run meds_etl_mimic to convert MIMIC-IV to MEDS format.
    
    Args:
        src_mimic: Path to MIMIC-IV root (containing 2.2/ subdirectory)
        output_dir: Path to output MEDS dataset
        num_shards: Number of shards for processing
        num_proc: Number of processes to use
        backend: Backend to use (polars or cpp)
    
    Returns:
        Time taken in seconds
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    print(f"    Running meds_etl_mimic (shards={num_shards}, proc={num_proc}, backend={backend})...")
    print(f"    Source: {src_mimic}")
    print(f"    Destination: {output_dir}")
    
    start = time.time()
    result = subprocess.run(
        [
            "meds_etl_mimic",
            src_mimic,
            output_dir,
            "--num_shards", str(num_shards),
            "--num_proc", str(num_proc),
            "--backend", backend,
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"    STDOUT: {result.stdout}")
        print(f"    STDERR: {result.stderr}")
        raise RuntimeError(f"meds_etl_mimic failed with code {result.returncode}")
    
    print(f"    meds_etl_mimic completed in {elapsed:.2f}s")
    return elapsed


def run_meds_reader_convert(input_dir: str, output_dir: str, num_threads: int = 10) -> float:
    """Run meds_reader_convert. Returns time taken."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    print(f"    Running meds_reader_convert (threads={num_threads})...")
    start = time.time()
    result = subprocess.run(
        ["meds_reader_convert", input_dir, output_dir, "--num_threads", str(num_threads)],
        capture_output=True, text=True,
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        raise RuntimeError(f"meds_reader_convert failed: {result.stderr}")
    
    print(f"    meds_reader_convert completed in {elapsed:.2f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark meds_reader data loading and single patient access"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="10014729",
        help="Patient ID to access (default: 10014729)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for meds_reader (default: 8)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for meds_etl_mimic (default: 8)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=100,
        help="Number of shards for meds_etl_mimic (default: 100)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="polars",
        choices=["polars", "cpp"],
        help="Backend for meds_etl_mimic (default: polars)",
    )
    parser.add_argument(
        "--mimic-root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv",
        help="Path to MIMIC-IV root directory (containing 2.2/ subdirectory)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/shared/eng/pyhealth",
        help="Cache directory for MEDS databases",
    )
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="Force reconversion even if database exists",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_patient_access_meds_reader.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    meds_dir = f"{args.cache_dir}/mimic4_meds"
    meds_reader_dir = f"{args.cache_dir}/mimic4_meds_reader"

    print("=" * 80)
    print("BENCHMARK: meds_reader - Data Loading & Patient Access")
    print("=" * 80)
    print(f"Patient ID: {args.patient_id}")
    print(f"Threads: {args.threads}")
    print(f"Num proc: {args.num_proc}")
    print(f"Num shards: {args.num_shards}")
    print(f"Backend: {args.backend}")
    print(f"MIMIC root: {args.mimic_root}")
    print(f"MEDS dir: {meds_dir}")
    print(f"meds_reader dir: {meds_reader_dir}")
    print("=" * 80)

    # Verify MIMIC-IV structure
    mimic_version_path = os.path.join(args.mimic_root, "2.2")
    if not os.path.exists(mimic_version_path):
        print(f"\nWARNING: Expected MIMIC-IV version directory not found: {mimic_version_path}")
        print("meds_etl_mimic expects the MIMIC-IV data to be in {mimic_root}/2.2/")
        print("Please ensure the directory structure is correct.")

    tracker = PeakMemoryTracker(poll_interval_s=0.05)
    tracker.start()
    tracker.reset()

    # Step 1: Data loading (conversion if needed)
    need_convert = args.force_reconvert or not Path(meds_reader_dir).exists()
    used_cached_db = not need_convert
    
    meds_etl_s = 0.0
    meds_reader_convert_s = 0.0
    
    if need_convert:
        print("\n[Step 1] Converting MIMIC-IV -> MEDS -> meds_reader database...")
        load_start = time.time()
        
        # Step 1a: meds_etl_mimic
        meds_etl_s = run_meds_etl_mimic(
            src_mimic=args.mimic_root,
            output_dir=meds_dir,
            num_shards=args.num_shards,
            num_proc=args.num_proc,
            backend=args.backend,
        )
        
        # Step 1b: meds_reader_convert
        meds_reader_convert_s = run_meds_reader_convert(
            meds_dir, meds_reader_dir, num_threads=args.threads
        )
        
        data_load_s = time.time() - load_start
    else:
        print("\n[Step 1] Using existing meds_reader database")
        print(f"  (use --force-reconvert to rebuild)")
        data_load_s = 0.0

    # Convert patient_id to integer for meds_reader
    subject_id = int(args.patient_id) if args.patient_id.isdigit() else hash(args.patient_id) % (10**9)
    
    with meds_reader.SubjectDatabase(meds_reader_dir, num_threads=args.threads) as database:
        print(f"  Database opened with {len(database)} subjects")
        
        # Step 2: First patient access (cold cache)
        print(f"\n[Step 2] First access to patient {args.patient_id} (cold cache)...")
        access_1_start = time.time()
        
        try:
            subject = database[subject_id]
            patient_found = True
            num_events = len(subject.events)
            print(f"  Patient found!")
            print(f"  Subject ID: {subject.subject_id}")
            print(f"  Number of events: {num_events}")
        except KeyError:
            patient_found = False
            num_events = 0
            print(f"  Patient NOT found with subject_id={subject_id}")
            # List some available subject IDs
            available_ids = list(database)[:10]
            print(f"  Available subject IDs (first 10): {available_ids}")
        
        patient_access_1st_s = time.time() - access_1_start
        
        # Step 3: Second patient access (warm cache)
        print(f"\n[Step 3] Second access to patient {args.patient_id} (warm cache)...")
        access_2_start = time.time()
        
        if patient_found:
            subject = database[subject_id]
            # Re-count events to ensure we're actually accessing the data
            count = len(subject.events)
        
        patient_access_2nd_s = time.time() - access_2_start
    
    total_s = data_load_s + patient_access_1st_s + patient_access_2nd_s
    peak_rss = tracker.peak_bytes()
    
    tracker.stop()

    result = BenchmarkResult(
        approach="meds_reader",
        data_load_s=data_load_s,
        meds_etl_s=meds_etl_s,
        meds_reader_convert_s=meds_reader_convert_s,
        patient_access_1st_s=patient_access_1st_s,
        patient_access_2nd_s=patient_access_2nd_s,
        total_s=total_s,
        peak_rss_bytes=peak_rss,
        patient_found=patient_found,
        num_events=num_events,
        used_cached_db=used_cached_db,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: meds_reader")
    print("=" * 80)
    access_1_str = f"{patient_access_1st_s*1000:.2f}ms" if patient_access_1st_s < 1 else f"{patient_access_1st_s:.2f}s"
    access_2_str = f"{patient_access_2nd_s*1000:.2f}ms" if patient_access_2nd_s < 1 else f"{patient_access_2nd_s:.2f}s"
    print(f"  Used cached DB:            {used_cached_db}")
    if not used_cached_db:
        print(f"  meds_etl_mimic:            {meds_etl_s:.2f}s")
        print(f"  meds_reader_convert:       {meds_reader_convert_s:.2f}s")
    print(f"  Total data load:           {data_load_s:.2f}s")
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
