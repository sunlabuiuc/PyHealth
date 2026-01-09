"""Benchmark script for MIMIC-IV length of stay prediction using meds_reader.

This benchmark measures performance across multiple thread counts:
1. Time to process the task
2. Peak memory usage (RSS, includes child processes)
3. Number of samples generated

This script uses meds_etl for data conversion:
- Converts MIMIC-IV directly to MEDS format via meds_etl_mimic
- Runs meds_reader_convert to prepare the database
- Then runs the benchmark

Typical usage:
  # First install dependencies:
  pip install meds_etl meds_reader
  
  # Run benchmark:
  python benchmark_meds_reader_los.py
  python benchmark_meds_reader_los.py --threads 1,4,8,12,16 --repeats 3
  python benchmark_meds_reader_los.py --force-reconvert  # Force fresh conversion
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
from typing import Iterable, Iterator

import psutil
import torch

try:
    import meds_reader
except ImportError:
    raise ImportError(
        "meds_reader not found. Install with: pip install meds_reader\n"
        "Or from source: pip install -e /path/to/meds_reader"
    )


# =============================================================================
# Processor Classes (matching PyHealth's SequenceProcessor for fair comparison)
# =============================================================================

class SequenceProcessor:
    """Matches PyHealth's SequenceProcessor for vocabulary building and tokenization."""
    
    def __init__(self):
        self.code_vocab = {"<pad>": 0}
        self._next_index = 1
    
    def fit(self, samples, field):
        """Build vocabulary from all samples (first pass through data)."""
        for sample in samples:
            if field not in sample:
                continue
            for token in sample[field]:
                if token is None:
                    continue
                if token not in self.code_vocab:
                    self.code_vocab[token] = self._next_index
                    self._next_index += 1
        self.code_vocab["<unk>"] = len(self.code_vocab)
    
    def process(self, value):
        """Convert code strings to tensor of indices."""
        indices = []
        for token in value:
            if token in self.code_vocab:
                indices.append(self.code_vocab[token])
            else:
                indices.append(self.code_vocab["<unk>"])
        return torch.tensor(indices, dtype=torch.long)
    
    def size(self):
        return len(self.code_vocab)


try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


# =============================================================================
# Data Conversion (MIMIC-IV -> MEDS -> meds_reader via meds_etl)
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
    
    print(f"  Running meds_etl_mimic (shards={num_shards}, proc={num_proc}, backend={backend})...")
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
    """Run meds_reader_convert CLI tool. Returns time taken."""
    print(f"  Running meds_reader_convert (threads={num_threads})...")
    print(f"    {input_dir} -> {output_dir}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    start = time.time()
    try:
        result = subprocess.run(
            ["meds_reader_convert", input_dir, output_dir, "--num_threads", str(num_threads)],
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start
        print(f"    meds_reader_convert completed in {elapsed:.2f}s")
        return elapsed
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: meds_reader_convert failed:")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"    ERROR: meds_reader_convert not found in PATH")
        raise


def ensure_meds_database(
    mimic_root: str,
    meds_dir: str,
    meds_reader_dir: str,
    num_shards: int,
    num_proc: int,
    backend: str,
    force_reconvert: bool,
) -> bool:
    """Ensure MEDS database exists, converting if necessary."""
    
    # Check if final meds_reader database exists
    if Path(meds_reader_dir).exists() and not force_reconvert:
        print(f"✓ MEDS database exists: {meds_reader_dir}")
        return True
    
    print(f"\n{'='*60}")
    print(f"MEDS database not found or reconvert requested")
    print(f"{'='*60}")
    
    # Verify MIMIC-IV structure
    mimic_version_path = os.path.join(mimic_root, "2.2")
    if not os.path.exists(mimic_version_path):
        print(f"\nERROR: Expected MIMIC-IV version directory not found: {mimic_version_path}")
        print("meds_etl_mimic expects the MIMIC-IV data to be in {mimic_root}/2.2/")
        return False
    
    # Step 1: Convert MIMIC-IV -> MEDS using meds_etl
    print(f"\n[Step 1/2] Converting MIMIC-IV to MEDS format using meds_etl...")
    run_meds_etl_mimic(
        src_mimic=mimic_root,
        output_dir=meds_dir,
        num_shards=num_shards,
        num_proc=num_proc,
        backend=backend,
    )
    
    # Step 2: Run meds_reader_convert
    print(f"\n[Step 2/2] Running meds_reader_convert...")
    run_meds_reader_convert(meds_dir, meds_reader_dir, num_threads=num_proc)
    
    print(f"\n✓ MEDS database ready: {meds_reader_dir}")
    return True


# =============================================================================
# Task Function - Length of Stay Prediction
# =============================================================================

def get_los_samples(subjects: Iterator[meds_reader.Subject]):
    """Process subjects for length of stay prediction task.
    
    Uses MEDS-ETL code conventions:
    - Admission codes are like "MIMIC_IV_Admission/..."
    - Diagnosis codes are like "ICD10CM/..." or "ICD9CM/..."
    - Procedure codes are like "ICD10PCS/..." or "ICD9Proc/..."
    - Prescriptions are like "NDC/..." or "MIMIC_IV_Drug/..."
    """
    samples = []
    
    for subject in subjects:
        admission_data = {}
        
        # First pass: identify admissions and their discharge times
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                # Get admission metadata
                visit_id = getattr(event, 'visit_id', None)
                end_time = getattr(event, 'end', None)
                
                if visit_id is not None and event.time is not None and end_time is not None:
                    los_days = (end_time - event.time).days
                    
                    # Categorize LOS (matching PyHealth's categorization)
                    if los_days < 1: los_label = 0
                    elif los_days <= 7: los_label = los_days
                    elif los_days <= 14: los_label = 8
                    else: los_label = 9
                    
                    admission_data[visit_id] = {
                        'start': event.time,
                        'los_days': los_days,
                        'label': los_label,
                        'conditions': set(),
                        'procedures': set(),
                        'drugs': set(),
                    }
        
        # Second pass: collect features per admission
        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id is None or visit_id not in admission_data:
                continue
            
            code = event.code
            if code.startswith("ICD"):  # ICD9CM, ICD10CM, ICD9Proc, ICD10PCS
                if "CM" in code:
                    admission_data[visit_id]['conditions'].add(code)
                else:
                    admission_data[visit_id]['procedures'].add(code)
            elif code.startswith("NDC/") or code.startswith("MIMIC_IV_Drug/"):
                admission_data[visit_id]['drugs'].add(code)
        
        # Create samples for admissions with sufficient data
        for visit_id, data in admission_data.items():
            conditions = list(data['conditions'])
            procedures = list(data['procedures'])
            drugs = list(data['drugs'])
            
            # Match PyHealth's filtering: require conditions, procedures, and drugs
            if len(conditions) == 0 or len(procedures) == 0 or len(drugs) == 0:
                continue
            
            samples.append({
                "visit_id": visit_id,
                "patient_id": subject.subject_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": data['label'],
                "los_days": data['los_days'],
            })
    
    return samples


# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class RunResult:
    num_threads: int
    repeat_index: int
    task_process_s: float
    peak_rss_bytes: int
    num_samples: int


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def set_memory_limit(max_memory_gb: float) -> None:
    if not HAS_RESOURCE:
        print("Warning: resource module not available. Memory limit not enforced.")
        return
    max_memory_bytes = int(max_memory_gb * 1024**3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        print(f"✓ Memory limit set to {max_memory_gb} GB")
    except Exception as e:
        print(f"Warning: Failed to set memory limit: {e}")


class PeakMemoryTracker:
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


def parse_threads(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    threads = []
    for p in parts:
        t = int(p)
        if t <= 0:
            raise argparse.ArgumentTypeError("All thread counts must be > 0")
        threads.append(t)
    if not threads:
        raise argparse.ArgumentTypeError("No threads provided")
    return threads


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
        description="Benchmark meds_reader for MIMIC-IV length of stay prediction"
    )
    parser.add_argument(
        "--threads", type=parse_threads, default=[1, 4, 8, 12, 16],
        help="Comma-separated list of num_threads values (default: 1,4,8,12,16)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of repeats per thread setting (default: 1)",
    )
    parser.add_argument(
        "--mimic-root", type=str,
        default="/srv/local/data/physionet.org/files/mimiciv",
        help="Path to MIMIC-IV root directory (containing 2.2/ subdirectory)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="datasets",
        help="Directory for MEDS cache (default: datasets)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=100,
        help="Number of shards for meds_etl_mimic (default: 100)",
    )
    parser.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for meds_etl_mimic (default: 8)",
    )
    parser.add_argument(
        "--backend", type=str, default="polars", choices=["polars", "cpp"],
        help="Backend for meds_etl_mimic (default: polars)",
    )
    parser.add_argument(
        "--force-reconvert", action="store_true",
        help="Force reconversion even if MEDS database exists",
    )
    parser.add_argument(
        "--enable-memory-limit", action="store_true",
        help="Enforce a hard memory limit via resource.setrlimit (Unix only)",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=None,
        help="Hard memory limit in GB (only used if --enable-memory-limit is set)",
    )
    parser.add_argument(
        "--output-csv", type=str,
        default="benchmark_meds_reader_los_threads_sweep.csv",
        help="Where to write per-run results as CSV",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")

    if args.enable_memory_limit:
        if args.max_memory_gb is None:
            raise SystemExit(
                "When using --enable-memory-limit, you must also pass --max-memory-gb"
            )
        set_memory_limit(args.max_memory_gb)

    # MEDS paths
    meds_dir = f"{args.cache_dir}/mimic4_meds"
    meds_reader_dir = f"{args.cache_dir}/mimic4_meds_reader"

    print("=" * 80)
    print("BENCHMARK: meds_reader - Length of Stay Prediction (Thread Sweep)")
    print(f"threads={args.threads} repeats={args.repeats}")
    print(f"mimic_root: {args.mimic_root}")
    print(f"backend: {args.backend}, num_proc: {args.num_proc}, num_shards: {args.num_shards}")
    print("=" * 80)

    # Ensure MEDS database exists
    if not ensure_meds_database(
        mimic_root=args.mimic_root,
        meds_dir=meds_dir,
        meds_reader_dir=meds_reader_dir,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
        backend=args.backend,
        force_reconvert=args.force_reconvert,
    ):
        raise SystemExit("Failed to prepare MEDS database")

    tracker = PeakMemoryTracker(poll_interval_s=0.1)
    tracker.start()

    total_start = time.time()
    results: list[RunResult] = []

    print(f"\n{'='*60}")
    print("Running benchmark...")
    print(f"{'='*60}")

    for t in args.threads:
        for r in range(args.repeats):
            tracker.reset()
            
            print(f"\n  threads={t} repeat={r + 1}/{args.repeats}: Processing...")
            task_start = time.time()

            # Step 1: Extract samples using meds_reader (parallel)
            samples = []
            with meds_reader.SubjectDatabase(meds_reader_dir, num_threads=t) as database:
                for s in database.map(get_los_samples):
                    samples.extend(s)
            
            # Step 2: Build vocabularies (matching PyHealth's processor.fit())
            conditions_processor = SequenceProcessor()
            procedures_processor = SequenceProcessor()
            drugs_processor = SequenceProcessor()
            
            conditions_processor.fit(samples, "conditions")
            procedures_processor.fit(samples, "procedures")
            drugs_processor.fit(samples, "drugs")
            
            # Step 3: Tokenize samples (matching PyHealth's processor.process())
            processed_samples = []
            for sample in samples:
                processed_sample = {
                    "visit_id": sample["visit_id"],
                    "patient_id": sample["patient_id"],
                    "conditions": conditions_processor.process(sample["conditions"]),
                    "procedures": procedures_processor.process(sample["procedures"]),
                    "drugs": drugs_processor.process(sample["drugs"]),
                    "label": torch.tensor(sample["label"], dtype=torch.long),
                    "los_days": sample["los_days"],
                }
                processed_samples.append(processed_sample)

            task_process_s = time.time() - task_start
            peak_rss_bytes = tracker.peak_bytes()
            num_samples = len(processed_samples)

            results.append(
                RunResult(
                    num_threads=t,
                    repeat_index=r,
                    task_process_s=task_process_s,
                    peak_rss_bytes=peak_rss_bytes,
                    num_samples=num_samples,
                )
            )

            print(
                f"  ✓ threads={t:>2} repeat={r + 1:>2}/{args.repeats} "
                f"samples={num_samples} "
                f"task={task_process_s:.2f}s "
                f"peak_rss={format_size(peak_rss_bytes)} "
                f"vocab_sizes=({conditions_processor.size()},{procedures_processor.size()},{drugs_processor.size()})"
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

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (median across repeats)")
    print("=" * 80)

    for t in args.threads:
        trs = [rr for rr in results if rr.num_threads == t]
        med_task = median([rr.task_process_s for rr in trs])
        med_peak = median([float(rr.peak_rss_bytes) for rr in trs])
        print(
            f"threads={t:>2}  "
            f"task_med={med_task:>8.2f}s  "
            f"peak_rss_med={format_size(int(med_peak)):>10}"
        )

    print("\nArtifacts:")
    print(f"  - CSV: {out_csv}")
    print(f"  - MEDS database: {meds_reader_dir}")
    print("\nTotals:")
    print(f"  - Sweep wall time: {total_sweep_s:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
