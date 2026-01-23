"""Benchmark script for MIMIC-IV drug recommendation using meds_reader.

This benchmark measures performance across multiple thread counts:
1. Time for MEDS ETL conversion (MIMIC-IV -> MEDS format)
2. Time for meds_reader database conversion (MEDS -> meds_reader format)
3. Time to process the task
4. Peak memory usage (RSS, includes child processes)
5. Number of samples generated

IMPORTANT: For fair comparison with PyHealth, conversion time MUST be included.
PyHealth's dataset loading includes parsing raw MIMIC-IV CSVs, so we must
account for the equivalent preprocessing time in meds_reader.

This script uses meds_etl for data conversion:
- Converts MIMIC-IV directly to MEDS format via meds_etl_mimic
- Runs meds_reader_convert to prepare the database
- Then runs the benchmark

Typical usage:
  # First install dependencies:
  pip install meds_etl meds_reader
  
  # Run benchmark (includes conversion time by default):
  python benchmark_meds_reader_drug_rec.py
  python benchmark_meds_reader_drug_rec.py --threads 1,4,8,12,16 --repeats 3
  
  # Skip conversion (only for debugging, not fair benchmarking):
  python benchmark_meds_reader_drug_rec.py --skip-conversion
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import psutil
import torch
from torch.utils.data import Dataset

try:
    import meds_reader
except ImportError:
    raise ImportError(
        "meds_reader not found. Install with: pip install meds_reader\n"
        "Or from source: pip install -e /path/to/meds_reader"
    )


# =============================================================================
# PyTorch Dataset Wrapper
# =============================================================================

class MedsReaderSampleDataset(Dataset):
    """PyTorch Dataset wrapper for meds_reader samples.
    
    Provides a standard PyTorch Dataset interface for model training.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        input_processors: Dict[str, Any],
        output_processors: Dict[str, Any],
        dataset_name: str = "",
        task_name: str = "",
    ):
        self.samples = samples
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_processors = input_processors
        self.output_processors = output_processors
        self.dataset_name = dataset_name
        self.task_name = task_name
        
        self.patient_to_index: Dict[Any, List[int]] = collections.defaultdict(list)
        self.record_to_index: Dict[Any, List[int]] = collections.defaultdict(list)
        
        for idx, sample in enumerate(samples):
            if "patient_id" in sample:
                self.patient_to_index[sample["patient_id"]].append(idx)
            if "visit_id" in sample:
                self.record_to_index[sample["visit_id"]].append(idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]
    
    def __repr__(self) -> str:
        return f"MedsReaderSampleDataset({self.dataset_name}, {self.task_name}, n={len(self)})"


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
            # Handle nested sequences (list of lists)
            values = sample[field]
            if values and isinstance(values[0], list):
                for visit_values in values:
                    for token in visit_values:
                        if token is None:
                            continue
                        if token not in self.code_vocab:
                            self.code_vocab[token] = self._next_index
                            self._next_index += 1
            else:
                for token in values:
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
    
    def process_nested(self, values):
        """Convert nested sequences to list of tensors."""
        return [self.process(v) for v in values]
    
    def size(self):
        return len(self.code_vocab)


class MultilabelProcessor:
    """Processor for multilabel outputs (matching PyHealth's MultiLabelProcessor)."""
    
    def __init__(self):
        self.label_vocab = {}
    
    def fit(self, samples, field):
        """Build vocabulary from all label values."""
        for sample in samples:
            if field in sample:
                for val in sample[field]:
                    if val not in self.label_vocab:
                        self.label_vocab[val] = len(self.label_vocab)
    
    def process(self, value):
        """Convert label list to multi-hot tensor."""
        multi_hot = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for val in value:
            if val in self.label_vocab:
                multi_hot[self.label_vocab[val]] = 1.0
        return multi_hot
    
    def size(self):
        return len(self.label_vocab)


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


@dataclass
class ConversionResult:
    """Holds timing information for the MEDS conversion process."""
    meds_etl_s: float
    meds_reader_convert_s: float
    total_conversion_s: float
    was_cached: bool  # True if conversion was skipped due to existing cache


def run_meds_conversion(
    mimic_root: str,
    meds_dir: str,
    meds_reader_dir: str,
    num_shards: int,
    num_proc: int,
    backend: str,
    force_reconvert: bool,
    skip_conversion: bool,
) -> ConversionResult:
    """Run MEDS conversion and return timing information.
    
    Args:
        mimic_root: Path to MIMIC-IV root directory
        meds_dir: Path for intermediate MEDS output
        meds_reader_dir: Path for final meds_reader database
        num_shards: Number of shards for meds_etl
        num_proc: Number of processes for meds_etl
        backend: Backend for meds_etl (polars or cpp)
        force_reconvert: If True, always reconvert even if cache exists
        skip_conversion: If True, skip conversion (for debugging only)
    
    Returns:
        ConversionResult with timing information
    """
    # Check if we should skip conversion
    if skip_conversion:
        if not Path(meds_reader_dir).exists():
            raise SystemExit(
                f"Cannot skip conversion: MEDS database does not exist at {meds_reader_dir}\n"
                "Run without --skip-conversion first."
            )
        print(f"✓ Skipping conversion (using cached MEDS database: {meds_reader_dir})")
        print("  WARNING: For fair benchmarking, conversion time should be included!")
        return ConversionResult(
            meds_etl_s=0.0,
            meds_reader_convert_s=0.0,
            total_conversion_s=0.0,
            was_cached=True,
        )
    
    # Check if we can reuse existing cache
    if Path(meds_reader_dir).exists() and not force_reconvert:
        print(f"✓ MEDS database exists: {meds_reader_dir}")
        print("  NOTE: Using cached data (subsequent repeat or --skip-conversion).")
        return ConversionResult(
            meds_etl_s=0.0,
            meds_reader_convert_s=0.0,
            total_conversion_s=0.0,
            was_cached=True,
        )
    
    print(f"\n{'='*60}")
    print(f"Converting MIMIC-IV to MEDS format")
    print(f"{'='*60}")
    
    # Clear existing cache directories to avoid interference
    if Path(meds_dir).exists():
        print(f"  Clearing existing MEDS cache: {meds_dir}")
        shutil.rmtree(meds_dir)
    if Path(meds_reader_dir).exists():
        print(f"  Clearing existing meds_reader cache: {meds_reader_dir}")
        shutil.rmtree(meds_reader_dir)
    
    # Verify MIMIC-IV structure
    mimic_version_path = os.path.join(mimic_root, "2.2")
    if not os.path.exists(mimic_version_path):
        raise SystemExit(
            f"ERROR: Expected MIMIC-IV version directory not found: {mimic_version_path}\n"
            f"meds_etl_mimic expects the MIMIC-IV data to be in {{mimic_root}}/2.2/"
        )
    
    # Step 1: Convert MIMIC-IV -> MEDS using meds_etl
    print(f"\n[Step 1/2] Converting MIMIC-IV to MEDS format using meds_etl...")
    meds_etl_s = run_meds_etl_mimic(
        src_mimic=mimic_root,
        output_dir=meds_dir,
        num_shards=num_shards,
        num_proc=num_proc,
        backend=backend,
    )
    
    # Step 2: Run meds_reader_convert
    print(f"\n[Step 2/2] Running meds_reader_convert...")
    meds_reader_convert_s = run_meds_reader_convert(
        meds_dir, meds_reader_dir, num_threads=num_proc
    )
    
    total_conversion_s = meds_etl_s + meds_reader_convert_s
    print(f"\n✓ MEDS database ready: {meds_reader_dir}")
    print(f"  Total conversion time: {total_conversion_s:.2f}s")
    
    return ConversionResult(
        meds_etl_s=meds_etl_s,
        meds_reader_convert_s=meds_reader_convert_s,
        total_conversion_s=total_conversion_s,
        was_cached=False,
    )


# =============================================================================
# Task Function - Drug Recommendation
# =============================================================================

def get_drug_rec_samples(subjects: Iterator[meds_reader.Subject]):
    """Process subjects for drug recommendation task.
    
    Uses MEDS-ETL code conventions:
    - Admission codes are like "MIMIC_IV_Admission/..."
    - Diagnosis codes are like "ICD10CM/..." or "ICD9CM/..."
    - Procedure codes are like "ICD10PCS/..." or "ICD9Proc/..."
    - Prescriptions are like "NDC/..." or "MIMIC_IV_Drug/..."
    
    Drug recommendation predicts drugs for current visit based on
    cumulative history of conditions, procedures, and past drugs.
    """
    samples = []
    
    for subject in subjects:
        # Collect all admissions with their data
        admissions = {}  # visit_id -> {time, conditions, procedures, drugs}
        
        # First pass: identify admissions
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                visit_id = getattr(event, 'visit_id', None)
                if visit_id is not None and event.time is not None:
                    admissions[visit_id] = {
                        'time': event.time,
                        'conditions': set(),
                        'procedures': set(),
                        'drugs': set(),
                    }
        
        # Second pass: collect features per admission
        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id is None or visit_id not in admissions:
                continue
            
            code = event.code
            if code.startswith("ICD"):  # ICD9CM, ICD10CM, ICD9Proc, ICD10PCS
                if "CM" in code:
                    admissions[visit_id]['conditions'].add(code)
                else:
                    admissions[visit_id]['procedures'].add(code)
            elif code.startswith("NDC/") or code.startswith("MIMIC_IV_Drug/"):
                admissions[visit_id]['drugs'].add(code)
        
        # Sort admissions by time
        sorted_visits = sorted(
            [(vid, data) for vid, data in admissions.items()],
            key=lambda x: x[1]['time']
        )
        
        # Filter to visits with complete data
        valid_visits = [
            (vid, data) for vid, data in sorted_visits
            if len(data['conditions']) > 0 and len(data['procedures']) > 0 and len(data['drugs']) > 0
        ]
        
        # Need at least 2 visits for drug recommendation
        if len(valid_visits) < 2:
            continue
        
        # Create samples with cumulative history
        for i, (visit_id, data) in enumerate(valid_visits):
            # Build cumulative history
            conditions_hist = []
            procedures_hist = []
            drugs_hist = []
            
            for j in range(i + 1):
                _, hist_data = valid_visits[j]
                conditions_hist.append(list(hist_data['conditions']))
                procedures_hist.append(list(hist_data['procedures']))
                # For drugs_hist, current visit's drugs are emptied (target)
                if j < i:
                    drugs_hist.append(list(hist_data['drugs']))
                else:
                    drugs_hist.append([])  # Empty for current visit
            
            samples.append({
                "visit_id": visit_id,
                "patient_id": subject.subject_id,
                "conditions": conditions_hist,  # Nested: [[codes_v1], [codes_v2], ...]
                "procedures": procedures_hist,  # Nested
                "drugs_hist": drugs_hist,  # Nested (current visit empty)
                "drugs": list(data['drugs']),  # Target drugs (flat list)
            })
    
    return samples


# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class RunResult:
    num_threads: int
    repeat_index: int
    meds_etl_s: float  # Time for MIMIC-IV -> MEDS conversion
    meds_reader_convert_s: float  # Time for MEDS -> meds_reader conversion
    task_process_s: float  # Time to run the ML task
    total_s: float  # Total time (conversion + task)
    peak_rss_bytes: int
    num_samples: int
    conversion_cached: bool  # True if conversion was skipped


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
        description="Benchmark meds_reader for MIMIC-IV drug recommendation"
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
        "--cache-dir", type=str, default="/srv/local/data/REDACTED_USER/meds_reader",
        help="Directory for MEDS cache",
    )
    parser.add_argument(
        "--num-shards", type=int, default=100,
        help="Number of shards for meds_etl_mimic (default: 100)",
    )
    # Note: --num-proc is deprecated; conversion now uses the current thread count
    # being benchmarked (from --threads) to ensure fair comparison
    parser.add_argument(
        "--backend", type=str, default="polars", choices=["polars", "cpp"],
        help="Backend for meds_etl_mimic (default: polars)",
    )
    # Note: Cache is always cleared before first repeat of each thread count
    # for accurate benchmarking. Use --skip-conversion to skip conversion entirely.
    parser.add_argument(
        "--skip-conversion", action="store_true",
        help="Skip conversion entirely (for debugging only - NOT fair benchmarking)",
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
        default="benchmark_meds_reader_drug_rec_threads_sweep.csv",
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
    # Use task-specific cache directories to avoid interference between tasks
    meds_dir = f"{args.cache_dir}/mimic4_meds_drug_rec"
    meds_reader_dir = f"{args.cache_dir}/mimic4_meds_reader_drug_rec"

    print("=" * 80)
    print("BENCHMARK: meds_reader - Drug Recommendation (Thread Sweep)")
    print(f"threads={args.threads} repeats={args.repeats}")
    print(f"mimic_root: {args.mimic_root}")
    print(f"backend: {args.backend}, num_shards: {args.num_shards}")
    print("NOTE: meds_etl uses the same thread count as task processing for fair comparison")
    if args.skip_conversion:
        print("WARNING: --skip-conversion is set. Conversion time will NOT be included.")
        print("         This is NOT a fair comparison with PyHealth!")
    print("=" * 80)

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
            run_start = time.time()
            
            # Step 0: Convert MIMIC-IV to MEDS format (part of total time)
            # For fair comparison with PyHealth, we must include this conversion time
            # since PyHealth's dataset loading includes parsing raw MIMIC-IV CSVs.
            # Use current thread count (t) for meds_etl conversion to match task parallelism
            # This ensures fair comparison: ETL + task both use the same thread count
            #
            # IMPORTANT: Always clear cache and run fresh conversion for EVERY run
            # to ensure fair benchmarking with no cached data influence.
            # Each run (thread count x repeat) gets a completely fresh conversion.
            if not args.skip_conversion:
                if os.path.exists(meds_dir):
                    print(f"  Clearing MEDS cache: {meds_dir}")
                    shutil.rmtree(meds_dir)
                if os.path.exists(meds_reader_dir):
                    print(f"  Clearing meds_reader cache: {meds_reader_dir}")
                    shutil.rmtree(meds_reader_dir)
            
            conversion = run_meds_conversion(
                mimic_root=args.mimic_root,
                meds_dir=meds_dir,
                meds_reader_dir=meds_reader_dir,
                num_shards=args.num_shards,
                num_proc=t,  # Use current thread count for fair benchmarking
                backend=args.backend,
                force_reconvert=not args.skip_conversion,  # Always reconvert unless skip mode
                skip_conversion=args.skip_conversion,
            )
            
            print(f"\n  threads={t} repeat={r + 1}/{args.repeats}: Processing task...")
            task_start = time.time()

            # Step 1: Extract samples using meds_reader (parallel)
            samples = []
            with meds_reader.SubjectDatabase(meds_reader_dir, num_threads=t) as database:
                for s in database.map(get_drug_rec_samples):
                    samples.extend(s)
            
            # Step 2: Build vocabularies (matching PyHealth's processor.fit())
            conditions_processor = SequenceProcessor()
            procedures_processor = SequenceProcessor()
            drugs_processor = SequenceProcessor()
            drugs_label_processor = MultilabelProcessor()
            
            conditions_processor.fit(samples, "conditions")  # Nested
            procedures_processor.fit(samples, "procedures")  # Nested
            drugs_processor.fit(samples, "drugs_hist")  # Nested
            drugs_processor.fit(samples, "drugs")  # Flat (adds more vocab)
            drugs_label_processor.fit(samples, "drugs")  # For multilabel output
            
            # Step 3: Tokenize samples (matching PyHealth's processor.process())
            processed_samples = []
            for sample in samples:
                processed_sample = {
                    "visit_id": sample["visit_id"],
                    "patient_id": sample["patient_id"],
                    "conditions": conditions_processor.process_nested(sample["conditions"]),
                    "procedures": procedures_processor.process_nested(sample["procedures"]),
                    "drugs_hist": drugs_processor.process_nested(sample["drugs_hist"]),
                    "drugs": drugs_label_processor.process(sample["drugs"]),  # Multilabel
                }
                processed_samples.append(processed_sample)
            
            # Step 4: Wrap in PyTorch Dataset for model training compatibility
            dataset = MedsReaderSampleDataset(
                samples=processed_samples,
                input_schema={
                    "conditions": "sequence",
                    "procedures": "sequence",
                    "drugs_hist": "sequence",
                },
                output_schema={"drugs": "multilabel"},
                input_processors={
                    "conditions": conditions_processor,
                    "procedures": procedures_processor,
                    "drugs_hist": drugs_processor,
                },
                output_processors={"drugs": drugs_label_processor},
                dataset_name="MIMIC-IV",
                task_name="DrugRecommendation",
            )

            task_process_s = time.time() - task_start
            total_s = time.time() - run_start
            peak_rss_bytes = tracker.peak_bytes()
            num_samples = len(dataset)

            results.append(
                RunResult(
                    num_threads=t,
                    repeat_index=r,
                    meds_etl_s=conversion.meds_etl_s,
                    meds_reader_convert_s=conversion.meds_reader_convert_s,
                    task_process_s=task_process_s,
                    total_s=total_s,
                    peak_rss_bytes=peak_rss_bytes,
                    num_samples=num_samples,
                    conversion_cached=conversion.was_cached,
                )
            )

            # Build output message
            timing_str = f"task={task_process_s:.2f}s"
            if not conversion.was_cached:
                timing_str = (
                    f"meds_etl={conversion.meds_etl_s:.2f}s "
                    f"convert={conversion.meds_reader_convert_s:.2f}s "
                    + timing_str + f" total={total_s:.2f}s"
                )
            
            print(
                f"  ✓ threads={t:>2} repeat={r + 1:>2}/{args.repeats} "
                f"samples={num_samples} "
                f"{timing_str} "
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
    
    # Check if any results have conversion times
    has_conversion = any(not rr.conversion_cached for rr in results)
    
    if has_conversion:
        print("\n  NOTE: Conversion time included for fair comparison with PyHealth.")
        print("        PyHealth's dataset_load_s ≈ meds_etl_s + meds_reader_convert_s")
    else:
        print("\n  NOTE: Conversion was skipped (--skip-conversion mode).")

    print()
    for t in args.threads:
        trs = [rr for rr in results if rr.num_threads == t]
        med_task = median([rr.task_process_s for rr in trs])
        med_total = median([rr.total_s for rr in trs])
        med_peak = median([float(rr.peak_rss_bytes) for rr in trs])
        
        # Get conversion times (from first repeat which has them)
        first_run = [rr for rr in trs if rr.repeat_index == 0][0]
        
        if not first_run.conversion_cached:
            print(
                f"threads={t:>2}  "
                f"meds_etl={first_run.meds_etl_s:>7.2f}s  "
                f"convert={first_run.meds_reader_convert_s:>7.2f}s  "
                f"task_med={med_task:>7.2f}s  "
                f"total={med_total:>7.2f}s  "
                f"peak_rss={format_size(int(med_peak)):>10}"
            )
        else:
            print(
                f"threads={t:>2}  "
                f"task_med={med_task:>8.2f}s  "
                f"(conversion cached)  "
                f"peak_rss_med={format_size(int(med_peak)):>10}"
            )

    print("\nArtifacts:")
    print(f"  - CSV: {out_csv}")
    print(f"  - MEDS database: {meds_reader_dir}")
    print("\nTotals:")
    print(f"  - Sweep wall time: {total_sweep_s:.2f}s")
    
    # Print comparison note
    print("\nFor comparison with PyHealth:")
    print("  PyHealth total_s = dataset_load_s + task_process_s")
    print("  meds_reader total_s = meds_etl_s + meds_reader_convert_s + task_process_s")
    print("=" * 80)


if __name__ == "__main__":
    main()
