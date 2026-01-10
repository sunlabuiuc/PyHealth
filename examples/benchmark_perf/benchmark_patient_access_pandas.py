"""Benchmark: Pure Pandas - Data Loading & Single Patient Access (with Parquet Caching)

Measures:
1. Time to load raw MIMIC-IV CSV tables with pandas
2. Time to cache tables as parquet files
3. Time to reload from parquet cache
4. Time to join tables and access a single patient's events
5. Total time

This benchmark mimics a realistic workflow where:
- Raw CSV data is loaded once
- Data is cached as parquet for faster subsequent access
- Patient queries are performed on the cached data

Usage:
  python benchmark_patient_access_pandas.py
  python benchmark_patient_access_pandas.py --patient-id 10014729
  python benchmark_patient_access_pandas.py --data-root /path/to/mimiciv/hosp
  python benchmark_patient_access_pandas.py --skip-parquet  # Skip parquet caching step
  python benchmark_patient_access_pandas.py --use-temp-dir  # Use temp dir (auto-cleaned)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import psutil


# =============================================================================
# Benchmark Result
# =============================================================================

@dataclass
class BenchmarkResult:
    approach: str
    csv_load_s: float           # Time to load raw CSVs
    parquet_write_s: float      # Time to write parquet cache
    parquet_read_s: float       # Time to reload from parquet
    patient_access_1st_s: float # First access (includes joins)
    patient_access_2nd_s: float # Second access (warm cache)
    total_s: float
    peak_rss_bytes: int
    patient_found: bool
    num_events: int
    num_visits: int
    num_tables_loaded: int
    parquet_cache_bytes: int    # Size of parquet cache


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
# Data Loading and Parquet Caching
# =============================================================================

def get_directory_size(path: str | Path) -> int:
    """Calculate total size of a directory."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for entry in p.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except FileNotFoundError:
                pass
    return total


def write_tables_to_parquet(
    tables: Dict[str, pd.DataFrame],
    cache_dir: str,
) -> float:
    """Write DataFrames to parquet files for caching.
    
    Args:
        tables: Dictionary mapping table name to DataFrame
        cache_dir: Directory to write parquet files
    
    Returns:
        Time taken in seconds
    """
    start = time.time()
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in tables.items():
        parquet_path = cache_path / f"{table_name}.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    
    return time.time() - start


def load_tables_from_parquet(
    cache_dir: str,
    tables: List[str],
) -> Dict[str, pd.DataFrame]:
    """Load tables from parquet cache.
    
    Args:
        cache_dir: Directory containing parquet files
        tables: List of table names to load
    
    Returns:
        Dictionary mapping table name to DataFrame
    """
    loaded = {}
    cache_path = Path(cache_dir)
    
    for table in tables:
        parquet_path = cache_path / f"{table}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            loaded[table] = df
    
    return loaded


def load_mimic_tables(
    data_root: str,
    tables: List[str],
) -> Dict[str, pd.DataFrame]:
    """Load MIMIC-IV tables from CSV files.
    
    Args:
        data_root: Path to MIMIC-IV hosp directory
        tables: List of table names to load
    
    Returns:
        Dictionary mapping table name to DataFrame
    """
    loaded = {}
    
    for table in tables:
        # Try both .csv and .csv.gz extensions
        csv_path = os.path.join(data_root, f"{table}.csv")
        csv_gz_path = os.path.join(data_root, f"{table}.csv.gz")
        
        if os.path.exists(csv_gz_path):
            path = csv_gz_path
        elif os.path.exists(csv_path):
            path = csv_path
        else:
            print(f"    WARNING: Table {table} not found at {csv_path} or {csv_gz_path}")
            continue
        
        print(f"    Loading {table}...")
        start = time.time()
        
        # Use low_memory=False for tables that might have mixed types
        df = pd.read_csv(path, low_memory=False)
        
        elapsed = time.time() - start
        print(f"      -> {len(df):,} rows in {elapsed:.2f}s")
        
        loaded[table] = df
    
    return loaded


def get_patient_events_with_joins(
    patient_id: str,
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Get all events for a patient, joining tables to build visit hierarchy.
    
    This mimics what PyHealth does: building a patient -> visits -> events structure
    by joining clinical tables with admissions to get visit context.
    
    Args:
        patient_id: Patient ID (subject_id in MIMIC-IV)
        tables: Dictionary of loaded DataFrames
    
    Returns:
        Dictionary with patient data organized by visits
    """
    subject_id = int(patient_id)
    
    patient_data = {
        "subject_id": subject_id,
        "visits": {},  # hadm_id -> visit data with events
        "demographics": None,
        "total_events": 0,
    }
    
    # Get patient demographics
    if "patients" in tables:
        patients_df = tables["patients"]
        patient_demo = patients_df[patients_df["subject_id"] == subject_id]
        if len(patient_demo) > 0:
            patient_data["demographics"] = patient_demo.iloc[0].to_dict()
    
    # Get admissions (visits) for this patient
    if "admissions" not in tables:
        return patient_data
    
    admissions_df = tables["admissions"]
    patient_admissions = admissions_df[admissions_df["subject_id"] == subject_id].copy()
    
    if len(patient_admissions) == 0:
        return patient_data
    
    # Parse datetime columns for admissions
    patient_admissions["admittime"] = pd.to_datetime(patient_admissions["admittime"])
    patient_admissions["dischtime"] = pd.to_datetime(patient_admissions["dischtime"])
    
    # Initialize visit structure
    for _, admission in patient_admissions.iterrows():
        hadm_id = admission["hadm_id"]
        patient_data["visits"][hadm_id] = {
            "hadm_id": hadm_id,
            "admittime": admission["admittime"],
            "dischtime": admission["dischtime"],
            "events": {},
        }
    
    hadm_ids = set(patient_admissions["hadm_id"].tolist())
    
    # Join diagnoses_icd with admissions context
    if "diagnoses_icd" in tables:
        diagnoses_df = tables["diagnoses_icd"]
        
        # Filter to patient first, then join with admissions
        patient_diagnoses = diagnoses_df[diagnoses_df["subject_id"] == subject_id].copy()
        
        # Join to get admission context (admittime, dischtime)
        patient_diagnoses = patient_diagnoses.merge(
            patient_admissions[["hadm_id", "admittime", "dischtime"]],
            on="hadm_id",
            how="inner"
        )
        
        # Organize by visit
        for hadm_id in hadm_ids:
            visit_diagnoses = patient_diagnoses[patient_diagnoses["hadm_id"] == hadm_id]
            patient_data["visits"][hadm_id]["events"]["diagnoses_icd"] = visit_diagnoses
            patient_data["total_events"] += len(visit_diagnoses)
    
    # Join procedures_icd with admissions context
    if "procedures_icd" in tables:
        procedures_df = tables["procedures_icd"]
        
        patient_procedures = procedures_df[procedures_df["subject_id"] == subject_id].copy()
        
        patient_procedures = patient_procedures.merge(
            patient_admissions[["hadm_id", "admittime", "dischtime"]],
            on="hadm_id",
            how="inner"
        )
        
        for hadm_id in hadm_ids:
            visit_procedures = patient_procedures[patient_procedures["hadm_id"] == hadm_id]
            patient_data["visits"][hadm_id]["events"]["procedures_icd"] = visit_procedures
            patient_data["total_events"] += len(visit_procedures)
    
    # Join labevents with admissions context
    if "labevents" in tables:
        labevents_df = tables["labevents"]
        
        # Filter to patient (labevents is large, so filter first)
        patient_labs = labevents_df[labevents_df["subject_id"] == subject_id].copy()
        
        # Join with admissions to get visit context
        # Note: Some lab events may not have hadm_id (outpatient)
        patient_labs = patient_labs.merge(
            patient_admissions[["hadm_id", "admittime", "dischtime"]],
            on="hadm_id",
            how="inner"  # Only keep labs with valid admission
        )
        
        for hadm_id in hadm_ids:
            visit_labs = patient_labs[patient_labs["hadm_id"] == hadm_id]
            patient_data["visits"][hadm_id]["events"]["labevents"] = visit_labs
            patient_data["total_events"] += len(visit_labs)
    
    # Join prescriptions with admissions context
    if "prescriptions" in tables:
        prescriptions_df = tables["prescriptions"]
        
        patient_prescriptions = prescriptions_df[prescriptions_df["subject_id"] == subject_id].copy()
        
        patient_prescriptions = patient_prescriptions.merge(
            patient_admissions[["hadm_id", "admittime", "dischtime"]],
            on="hadm_id",
            how="inner"
        )
        
        for hadm_id in hadm_ids:
            visit_prescriptions = patient_prescriptions[patient_prescriptions["hadm_id"] == hadm_id]
            patient_data["visits"][hadm_id]["events"]["prescriptions"] = visit_prescriptions
            patient_data["total_events"] += len(visit_prescriptions)
    
    return patient_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pure pandas data loading and single patient access (with parquet caching)"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="10014729",
        help="Patient ID to access (default: 10014729)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.2/hosp",
        help="Path to MIMIC-IV hosp directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_patient_access_pandas.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip the parquet caching step",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/shared/eng/pyhealth/pandas_parquet_cache",
        help="Directory for parquet cache (default: /shared/eng/pyhealth/pandas_parquet_cache)",
    )
    parser.add_argument(
        "--use-temp-dir",
        action="store_true",
        help="Use a temporary directory for parquet cache (auto-cleaned after benchmark)",
    )
    args = parser.parse_args()

    # Tables to load (matching what PyHealth loads for mortality/LOS tasks)
    tables_to_load = [
        "patients",
        "admissions", 
        "diagnoses_icd",
        "procedures_icd",
        "labevents",
    ]

    # Set up parquet cache directory
    use_temp_dir = args.use_temp_dir
    if use_temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="mimic_parquet_cache_")
        cache_dir = temp_dir
    else:
        cache_dir = args.cache_dir
        temp_dir = None

    print("=" * 80)
    print("BENCHMARK: Pure Pandas - Data Loading & Patient Access (with Parquet Caching)")
    print("=" * 80)
    print(f"Patient ID: {args.patient_id}")
    print(f"Data root: {args.data_root}")
    print(f"Tables: {tables_to_load}")
    print(f"Skip parquet: {args.skip_parquet}")
    print(f"Parquet cache: {cache_dir} {'(temp, will be deleted)' if use_temp_dir else ''}")
    print("=" * 80)

    tracker = PeakMemoryTracker(poll_interval_s=0.05)
    tracker.start()
    tracker.reset()

    try:
        # Step 1: Load raw CSV tables
        print("\n[Step 1] Loading MIMIC-IV tables from CSV...")
        csv_load_start = time.time()
        
        tables = load_mimic_tables(args.data_root, tables_to_load)
        
        csv_load_s = time.time() - csv_load_start
        num_tables_loaded = len(tables)
        
        total_rows = sum(len(df) for df in tables.values())
        print(f"\n  Loaded {num_tables_loaded} tables ({total_rows:,} total rows) in {csv_load_s:.2f}s")

        # Step 2: Write to parquet cache (if not skipping)
        parquet_write_s = 0.0
        parquet_read_s = 0.0
        parquet_cache_bytes = 0
        
        if not args.skip_parquet:
            print("\n[Step 2] Writing tables to parquet cache...")
            parquet_write_start = time.time()
            
            write_tables_to_parquet(tables, cache_dir)
            
            parquet_write_s = time.time() - parquet_write_start
            parquet_cache_bytes = get_directory_size(cache_dir)
            
            print(f"  Wrote {num_tables_loaded} parquet files in {parquet_write_s:.2f}s")
            print(f"  Cache size: {format_size(parquet_cache_bytes)}")
            
            # Step 3: Reload from parquet cache (simulating future access)
            print("\n[Step 3] Reloading tables from parquet cache...")
            
            # Clear the in-memory tables to simulate a fresh load
            del tables
            
            parquet_read_start = time.time()
            
            tables = load_tables_from_parquet(cache_dir, tables_to_load)
            
            parquet_read_s = time.time() - parquet_read_start
            print(f"  Reloaded {len(tables)} tables from parquet in {parquet_read_s:.2f}s")
            
            next_step = 4
        else:
            print("\n[Step 2] Skipping parquet caching (--skip-parquet)")
            next_step = 3

        # Step N: First patient access (cold - includes joining tables)
        print(f"\n[Step {next_step}] First access to patient {args.patient_id} (with table joins)...")
        access_1_start = time.time()
        
        patient_data = get_patient_events_with_joins(args.patient_id, tables)
        
        patient_found = patient_data["total_events"] > 0 or len(patient_data["visits"]) > 0
        num_events = patient_data["total_events"]
        num_visits = len(patient_data["visits"])
        
        if patient_found:
            print(f"  Patient found!")
            print(f"  Number of visits: {num_visits}")
            print(f"  Total events: {num_events}")
            
            # Show events per visit
            for hadm_id, visit in patient_data["visits"].items():
                visit_events = sum(len(df) for df in visit["events"].values())
                print(f"    Visit {hadm_id}: {visit_events} events")
                for table_name, events_df in visit["events"].items():
                    if len(events_df) > 0:
                        print(f"      - {table_name}: {len(events_df)} rows")
        else:
            print(f"  Patient NOT found!")
            if "patients" in tables:
                available_ids = tables["patients"]["subject_id"].head(10).tolist()
                print(f"  Available patient IDs (first 10): {available_ids}")
        
        patient_access_1st_s = time.time() - access_1_start
        
        # Step N+1: Second patient access (warm)
        print(f"\n[Step {next_step + 1}] Second access to patient {args.patient_id} (repeat with joins)...")
        access_2_start = time.time()
        
        if patient_found:
            patient_data_2 = get_patient_events_with_joins(args.patient_id, tables)
            count = patient_data_2["total_events"]
            print(f"  Verified {count} events")
        
        patient_access_2nd_s = time.time() - access_2_start
        
        # Calculate total time
        total_s = csv_load_s + parquet_write_s + parquet_read_s + patient_access_1st_s + patient_access_2nd_s
        peak_rss = tracker.peak_bytes()
        
    finally:
        tracker.stop()
        
        # Clean up temporary parquet cache
        if use_temp_dir and temp_dir and os.path.exists(temp_dir):
            print(f"\n[Cleanup] Removing temporary parquet cache: {temp_dir}")
            shutil.rmtree(temp_dir)

    result = BenchmarkResult(
        approach="pandas_with_parquet_cache",
        csv_load_s=csv_load_s,
        parquet_write_s=parquet_write_s,
        parquet_read_s=parquet_read_s,
        patient_access_1st_s=patient_access_1st_s,
        patient_access_2nd_s=patient_access_2nd_s,
        total_s=total_s,
        peak_rss_bytes=peak_rss,
        patient_found=patient_found,
        num_events=num_events,
        num_visits=num_visits,
        num_tables_loaded=num_tables_loaded,
        parquet_cache_bytes=parquet_cache_bytes,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Pure Pandas (with Parquet Caching)")
    print("=" * 80)
    access_1_str = f"{patient_access_1st_s*1000:.2f}ms" if patient_access_1st_s < 1 else f"{patient_access_1st_s:.2f}s"
    access_2_str = f"{patient_access_2nd_s*1000:.2f}ms" if patient_access_2nd_s < 1 else f"{patient_access_2nd_s:.2f}s"
    print(f"  CSV load time:             {csv_load_s:.2f}s")
    if not args.skip_parquet:
        print(f"  Parquet write time:        {parquet_write_s:.2f}s")
        print(f"  Parquet read time:         {parquet_read_s:.2f}s")
        print(f"  Parquet cache size:        {format_size(parquet_cache_bytes)}")
    print(f"  Patient access (1st/cold): {access_1_str}")
    print(f"  Patient access (2nd/warm): {access_2_str}")
    print(f"  Total time:                {total_s:.2f}s")
    print(f"  Peak RSS:                  {format_size(peak_rss)}")
    print(f"  Patient found:             {patient_found}")
    print(f"  Visits:                    {num_visits}")
    print(f"  Events:                    {num_events}")
    print(f"  Tables loaded:             {num_tables_loaded}")

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
