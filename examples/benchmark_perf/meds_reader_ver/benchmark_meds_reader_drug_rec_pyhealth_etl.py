"""Benchmark script for MIMIC-IV drug recommendation using meds_reader.

This is a FALLBACK version that uses PyHealth 1.1.6 for ETL instead of meds_etl_mimic.
Use this if meds_etl_mimic fails to run properly.

Pipeline:
1. Load MIMIC-IV data using PyHealth 1.1.6 (MIMIC4Dataset)
2. Convert PyHealth data structures to MEDS format (parquet files)
3. Run meds_reader_convert to create meds_reader database
4. Process the task using meds_reader
5. Return samples in a PyTorch-compatible Dataset

IMPORTANT: For fair comparison with PyHealth, conversion time MUST be included.

Typical usage:
  # First install dependencies:
  pip install pyhealth==1.1.6 meds_reader pyarrow

  # Run benchmark:
  python benchmark_meds_reader_drug_rec_pyhealth_etl.py
  python benchmark_meds_reader_drug_rec_pyhealth_etl.py --threads 1,4,8,12,16
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

try:
    import meds_reader
except ImportError:
    raise ImportError(
        "meds_reader not found. Install with: pip install meds_reader\n"
        "Or from source: pip install -e /path/to/meds_reader"
    )

# Import PyHealth 1.1.6
try:
    from pyhealth.datasets import MIMIC4Dataset
except ImportError:
    raise ImportError(
        "PyHealth not found. Install with: pip install pyhealth==1.1.6"
    )


# =============================================================================
# PyTorch Dataset Wrapper
# =============================================================================

class MedsReaderSampleDataset(Dataset):
    """PyTorch Dataset wrapper for meds_reader samples."""

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
# Processor Classes
# =============================================================================

class SequenceProcessor:
    """Matches PyHealth's SequenceProcessor for vocabulary building."""

    def __init__(self):
        self.code_vocab = {"<pad>": 0}
        self._next_index = 1

    def fit(self, samples, field):
        for sample in samples:
            if field not in sample:
                continue
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
        indices = []
        for token in value:
            if token in self.code_vocab:
                indices.append(self.code_vocab[token])
            else:
                indices.append(self.code_vocab["<unk>"])
        return torch.tensor(indices, dtype=torch.long)

    def process_nested(self, values):
        return [self.process(v) for v in values]

    def size(self):
        return len(self.code_vocab)


class MultilabelProcessor:
    """Processor for multilabel outputs."""

    def __init__(self):
        self.label_vocab = {}

    def fit(self, samples, field):
        for sample in samples:
            if field in sample:
                for val in sample[field]:
                    if val not in self.label_vocab:
                        self.label_vocab[val] = len(self.label_vocab)

    def process(self, value):
        multi_hot = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for val in value:
            if val in self.label_vocab:
                multi_hot[self.label_vocab[val]] = 1.0
        return multi_hot

    def size(self):
        return len(self.label_vocab)


# =============================================================================
# Data Conversion (PyHealth 1.1.6 -> MEDS -> meds_reader)
# =============================================================================

def pyhealth_to_meds(
    pyhealth_root: str,
    output_dir: str,
    tables: List[str],
    dev: bool = False,
    num_shards: int = 100,
) -> float:
    """Convert MIMIC-IV data via PyHealth 1.1.6 to MEDS format."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("  Loading MIMIC-IV via PyHealth 1.1.6...")
    print(f"    Root: {pyhealth_root}")
    print(f"    Tables: {tables}")
    print(f"    Dev mode: {dev}")

    start = time.time()

    dataset = MIMIC4Dataset(
        root=pyhealth_root,
        tables=tables,
        dev=dev,
        refresh_cache=True,
    )

    pyhealth_load_time = time.time() - start
    print(f"    PyHealth load completed in {pyhealth_load_time:.2f}s")

    print("  Converting to MEDS format...")
    convert_start = time.time()

    results = collections.defaultdict(list)

    for patient_id, patient in dataset.patients.items():
        subject_id = int(patient_id)

        # Birth event
        if patient.birth_datetime is not None:
            birth_obj = {
                'subject_id': subject_id,
                'code': 'meds/birth',
                'time': patient.birth_datetime,
            }
            if hasattr(patient, 'gender') and patient.gender:
                birth_obj['gender'] = patient.gender
            if hasattr(patient, 'ethnicity') and patient.ethnicity:
                birth_obj['ethnicity'] = patient.ethnicity
            results[subject_id].append(birth_obj)

        # Death event
        if patient.death_datetime is not None:
            results[subject_id].append({
                'subject_id': subject_id,
                'code': 'meds/death',
                'time': patient.death_datetime,
            })

        # Process visits
        for visit_id, visit in patient.visits.items():
            visit_id_int = int(visit_id)

            visit_event = {
                'subject_id': subject_id,
                'code': 'MIMIC_IV_Admission/unknown',
                'time': visit.encounter_time,
                'visit_id': visit_id_int,
            }
            if visit.discharge_time:
                visit_event['end'] = visit.discharge_time
            if hasattr(visit, 'discharge_status'):
                visit_event['discharge_status'] = visit.discharge_status

            results[subject_id].append(visit_event)

            for table in visit.available_tables:
                for event in visit.get_event_list(table):
                    event_obj = {
                        'subject_id': subject_id,
                        'visit_id': visit_id_int,
                        'code': f'{event.vocabulary}/{event.code}',
                        'time': event.timestamp or visit.discharge_time,
                    }

                    if hasattr(event, 'attr_dict') and event.attr_dict:
                        for k, v in event.attr_dict.items():
                            if v == v:  # Skip NaN
                                event_obj[k] = v

                    results[subject_id].append(event_obj)

        results[subject_id].sort(
            key=lambda a: a['time'] if a['time'] else datetime.datetime.min
        )

    # Write to parquet shards
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/metadata", exist_ok=True)
    os.makedirs(f"{output_dir}/data", exist_ok=True)

    all_subjects = list(results.keys())
    subject_ids_per_shard = np.array_split(all_subjects, num_shards)

    attr_map = {
        str: pa.string(),
        int: pa.int64(),
        np.int64: pa.int64(),
        float: pa.float64(),
        datetime.datetime: pa.timestamp('us'),
    }

    attr_schema = {}
    for subject_values in results.values():
        for row in subject_values:
            for k, v in row.items():
                if k not in {'subject_id', 'time'} and v is not None:
                    pa_type = attr_map.get(type(v), pa.string())
                    if k not in attr_schema:
                        attr_schema[k] = pa_type

    schema = pa.schema([
        ('subject_id', pa.int64()),
        ('time', pa.timestamp('us')),
    ] + [(k, v) for k, v in sorted(attr_schema.items())])

    for i, subject_ids in enumerate(subject_ids_per_shard):
        if len(subject_ids) == 0:
            continue
        rows = [v for subject_id in subject_ids for v in results[subject_id]]
        if rows:
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, f"{output_dir}/data/{i}.parquet")

    convert_time = time.time() - convert_start
    total_time = time.time() - start

    print(f"    MEDS conversion completed in {convert_time:.2f}s")
    print(f"    Total PyHealth ETL time: {total_time:.2f}s")

    return total_time


def run_meds_reader_convert(
    input_dir: str, output_dir: str, num_threads: int = 10
) -> float:
    """Run meds_reader_convert CLI tool."""
    print(f"  Running meds_reader_convert (threads={num_threads})...")
    print(f"    {input_dir} -> {output_dir}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    start = time.time()
    try:
        subprocess.run(
            ["meds_reader_convert", input_dir, output_dir,
             "--num_threads", str(num_threads)],
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start
        print(f"    meds_reader_convert completed in {elapsed:.2f}s")
        return elapsed
    except subprocess.CalledProcessError as e:
        print("    ERROR: meds_reader_convert failed:")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("    ERROR: meds_reader_convert not found in PATH")
        raise


@dataclass
class ConversionResult:
    """Holds timing information for the conversion process."""
    pyhealth_etl_s: float
    meds_reader_convert_s: float
    total_conversion_s: float
    was_cached: bool


def run_pyhealth_meds_conversion(
    pyhealth_root: str,
    meds_dir: str,
    meds_reader_dir: str,
    tables: List[str],
    dev: bool,
    num_shards: int,
    num_threads: int,
    force_reconvert: bool,
    skip_conversion: bool,
) -> ConversionResult:
    """Run PyHealth-based MEDS conversion."""

    if skip_conversion:
        if not Path(meds_reader_dir).exists():
            raise SystemExit(
                f"Cannot skip conversion: MEDS database does not exist at "
                f"{meds_reader_dir}\nRun without --skip-conversion first."
            )
        print("✓ Skipping conversion (using cached MEDS database)")
        return ConversionResult(0.0, 0.0, 0.0, True)

    if Path(meds_reader_dir).exists() and not force_reconvert:
        print(f"✓ MEDS database exists: {meds_reader_dir}")
        return ConversionResult(0.0, 0.0, 0.0, True)

    print("\n" + "=" * 60)
    print("Converting MIMIC-IV to MEDS format via PyHealth 1.1.6")
    print("=" * 60)

    if Path(meds_dir).exists():
        print(f"  Clearing existing MEDS cache: {meds_dir}")
        shutil.rmtree(meds_dir)
    if Path(meds_reader_dir).exists():
        print(f"  Clearing existing meds_reader cache: {meds_reader_dir}")
        shutil.rmtree(meds_reader_dir)

    print("\n[Step 1/2] Loading via PyHealth and converting to MEDS...")
    pyhealth_etl_s = pyhealth_to_meds(
        pyhealth_root=pyhealth_root,
        output_dir=meds_dir,
        tables=tables,
        dev=dev,
        num_shards=num_shards,
    )

    print("\n[Step 2/2] Running meds_reader_convert...")
    meds_reader_convert_s = run_meds_reader_convert(
        meds_dir, meds_reader_dir, num_threads=num_threads
    )

    total = pyhealth_etl_s + meds_reader_convert_s
    print(f"\n✓ MEDS database ready. Total conversion: {total:.2f}s")

    return ConversionResult(pyhealth_etl_s, meds_reader_convert_s, total, False)


# =============================================================================
# Task Function - Drug Recommendation
# =============================================================================

def get_drug_rec_samples(subjects: Iterator[meds_reader.Subject]):
    """Process subjects for drug recommendation task."""
    samples = []

    for subject in subjects:
        admissions = {}

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

        for event in subject.events:
            visit_id = getattr(event, 'visit_id', None)
            if visit_id is None or visit_id not in admissions:
                continue

            code = event.code
            if code.startswith("ICD"):
                if "CM" in code:
                    admissions[visit_id]['conditions'].add(code)
                else:
                    admissions[visit_id]['procedures'].add(code)
            elif code.startswith("NDC/") or code.startswith("MIMIC_IV_Drug/"):
                admissions[visit_id]['drugs'].add(code)

        sorted_visits = sorted(
            [(vid, data) for vid, data in admissions.items()],
            key=lambda x: x[1]['time']
        )

        if len(sorted_visits) < 2:
            continue

        for i, (visit_id, current) in enumerate(sorted_visits):
            conditions = list(current['conditions'])
            procedures = list(current['procedures'])
            drugs = list(current['drugs'])

            if len(conditions) == 0 or len(procedures) == 0 or len(drugs) == 0:
                continue

            conditions_hist = []
            procedures_hist = []
            drugs_hist = []

            for j in range(i + 1):
                conditions_hist.append(list(sorted_visits[j][1]['conditions']))
                procedures_hist.append(list(sorted_visits[j][1]['procedures']))
                if j < i:
                    drugs_hist.append(list(sorted_visits[j][1]['drugs']))
                else:
                    drugs_hist.append([])

            samples.append({
                "visit_id": visit_id,
                "patient_id": subject.subject_id,
                "conditions": conditions_hist,
                "procedures": procedures_hist,
                "drugs_hist": drugs_hist,
                "drugs": drugs,
            })

    return samples


# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class RunResult:
    num_threads: int
    repeat_index: int
    pyhealth_etl_s: float
    meds_reader_convert_s: float
    task_process_s: float
    total_s: float
    peak_rss_bytes: int
    num_samples: int
    conversion_cached: bool


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


class PeakMemoryTracker:
    def __init__(self, poll_interval_s: float = 0.1):
        self._proc = psutil.Process(os.getpid())
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def reset(self):
        with self._lock:
            self._peak = 0

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

    def _run(self):
        while not self._stop.is_set():
            rss = self._total_rss_bytes()
            with self._lock:
                if rss > self._peak:
                    self._peak = rss
            time.sleep(self._poll_interval_s)


def parse_threads(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts if int(p) > 0]


def median(values: Iterable[float]) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 == 1 else (xs[mid - 1] + xs[mid]) / 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark meds_reader Drug Rec using PyHealth 1.1.6 ETL"
    )
    parser.add_argument(
        "--threads", type=parse_threads, default=[1, 4, 8, 12, 16],
        help="Comma-separated list of thread counts",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--pyhealth-root", type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        help="Path to MIMIC-IV hosp directory (for PyHealth 1.1.6)",
    )
    parser.add_argument("--cache-dir", type=str, default="/srv/local/data/johnwu3/meds_reader")
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--force-reconvert", action="store_true")
    parser.add_argument("--skip-conversion", action="store_true")
    parser.add_argument(
        "--output-csv", type=str,
        default="benchmark_meds_reader_drug_rec_pyhealth_etl.csv",
    )
    args = parser.parse_args()

    meds_dir = f"{args.cache_dir}/mimic4_meds_drug_rec_pyhealth"
    meds_reader_dir = f"{args.cache_dir}/mimic4_meds_reader_drug_rec_pyhealth"

    print("=" * 80)
    print("BENCHMARK: meds_reader Drug Rec (PyHealth 1.1.6 ETL - Fallback)")
    print(f"threads={args.threads} repeats={args.repeats} dev={args.dev}")
    print(f"pyhealth_root: {args.pyhealth_root}")
    print("=" * 80)

    tracker = PeakMemoryTracker()
    tracker.start()

    total_start = time.time()
    results: List[RunResult] = []

    # Tables needed for drug recommendation task
    tables = ["diagnoses_icd", "procedures_icd", "prescriptions"]

    for t in args.threads:
        for r in range(args.repeats):
            tracker.reset()
            run_start = time.time()

            conversion = run_pyhealth_meds_conversion(
                pyhealth_root=args.pyhealth_root,
                meds_dir=meds_dir,
                meds_reader_dir=meds_reader_dir,
                tables=tables,
                dev=args.dev,
                num_shards=args.num_shards,
                num_threads=args.num_threads,
                force_reconvert=args.force_reconvert and r == 0,
                skip_conversion=args.skip_conversion or r > 0,
            )

            print(f"\n  threads={t} repeat={r + 1}/{args.repeats}: Processing...")
            task_start = time.time()

            samples = []
            with meds_reader.SubjectDatabase(meds_reader_dir, num_threads=t) as db:
                for s in db.map(get_drug_rec_samples):
                    samples.extend(s)

            conditions_proc = SequenceProcessor()
            procedures_proc = SequenceProcessor()
            drugs_proc = SequenceProcessor()
            drugs_label_proc = MultilabelProcessor()

            conditions_proc.fit(samples, "conditions")
            procedures_proc.fit(samples, "procedures")
            drugs_proc.fit(samples, "drugs_hist")
            drugs_proc.fit(samples, "drugs")
            drugs_label_proc.fit(samples, "drugs")

            processed = []
            for sample in samples:
                processed.append({
                    "visit_id": sample["visit_id"],
                    "patient_id": sample["patient_id"],
                    "conditions": conditions_proc.process_nested(sample["conditions"]),
                    "procedures": procedures_proc.process_nested(sample["procedures"]),
                    "drugs_hist": drugs_proc.process_nested(sample["drugs_hist"]),
                    "drugs": drugs_label_proc.process(sample["drugs"]),
                })

            dataset = MedsReaderSampleDataset(
                samples=processed,
                input_schema={
                    "conditions": "sequence",
                    "procedures": "sequence",
                    "drugs_hist": "sequence",
                },
                output_schema={"drugs": "multilabel"},
                input_processors={
                    "conditions": conditions_proc,
                    "procedures": procedures_proc,
                    "drugs_hist": drugs_proc,
                },
                output_processors={"drugs": drugs_label_proc},
                dataset_name="MIMIC-IV",
                task_name="DrugRecommendation",
            )

            task_process_s = time.time() - task_start
            total_s = time.time() - run_start
            peak_rss = tracker.peak_bytes()

            results.append(RunResult(
                num_threads=t,
                repeat_index=r,
                pyhealth_etl_s=conversion.pyhealth_etl_s,
                meds_reader_convert_s=conversion.meds_reader_convert_s,
                task_process_s=task_process_s,
                total_s=total_s,
                peak_rss_bytes=peak_rss,
                num_samples=len(dataset),
                conversion_cached=conversion.was_cached,
            ))

            timing = f"task={task_process_s:.2f}s"
            if not conversion.was_cached:
                timing = (f"pyhealth_etl={conversion.pyhealth_etl_s:.2f}s "
                         f"convert={conversion.meds_reader_convert_s:.2f}s "
                         f"{timing} total={total_s:.2f}s")

            print(f"  ✓ threads={t:>2} samples={len(dataset)} {timing} "
                  f"peak_rss={format_size(peak_rss)}")

    total_sweep_s = time.time() - total_start

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for rr in results:
            writer.writerow(asdict(rr))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for t in args.threads:
        trs = [rr for rr in results if rr.num_threads == t]
        med_task = median([rr.task_process_s for rr in trs])
        first = [rr for rr in trs if rr.repeat_index == 0][0]
        if not first.conversion_cached:
            print(f"threads={t:>2}  pyhealth_etl={first.pyhealth_etl_s:.2f}s  "
                  f"convert={first.meds_reader_convert_s:.2f}s  "
                  f"task_med={med_task:.2f}s")
        else:
            print(f"threads={t:>2}  task_med={med_task:.2f}s  (cached)")

    print(f"\nSweep time: {total_sweep_s:.2f}s")
    print(f"CSV: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

