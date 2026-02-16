"""Benchmark script for MIMIC-IV mortality prediction with num_workers=4.

This benchmark measures:
1. Time to load base dataset
2. Time to process task with num_workers=4
3. Total processing time
4. Cache sizes
5. Peak memory usage (with optional memory limit)
"""

import time
import os
import threading
from pathlib import Path
import psutil
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


PEAK_MEM_USAGE = 0
SELF_PROC = psutil.Process(os.getpid())


def track_mem():
    """Background thread to track peak memory usage."""
    global PEAK_MEM_USAGE
    while True:
        m = SELF_PROC.memory_info().rss
        if m > PEAK_MEM_USAGE:
            PEAK_MEM_USAGE = m
        time.sleep(0.1)


def set_memory_limit(max_memory_gb):
    """Set hard memory limit for the process.

    Args:
        max_memory_gb: Maximum memory in GB (e.g., 8 for 8GB)

    Note:
        If limit is exceeded, the process will raise MemoryError.
        Only works on Unix-like systems (Linux, macOS).
    """
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


def get_directory_size(path):
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in Path(path).rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Error calculating size for {path}: {e}")
    return total


def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    """Main benchmark function."""
    # Configuration
    dev = False  # Set to True for development/testing
    enable_memory_limit = False  # Set to True to enforce memory limit
    max_memory_gb = 32  # Memory limit in GB (if enable_memory_limit=True)

    # Apply memory limit if enabled
    if enable_memory_limit:
        set_memory_limit(max_memory_gb)

    # Start memory tracking thread
    mem_thread = threading.Thread(target=track_mem, daemon=True)
    mem_thread.start()

    print("=" * 80)
    print(f"BENCHMARK: num_workers=4, dev={dev}")
    if enable_memory_limit:
        print(f"Memory Limit: {max_memory_gb} GB (ENFORCED)")
    else:
        print("Memory Limit: None (unrestricted)")
    print("=" * 80)

    # Define cache directories based on dev mode
    cache_root = "./benchmark_cache/workers_4"
    if dev:
        cache_root += "_dev"

    # Track total time
    total_start = time.time()  # STEP 1: Load MIMIC-IV base dataset
    print("\n[1/2] Loading MIMIC-IV base dataset...")
    dataset_start = time.time()

    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        dev=dev,
        # cache_dir=f"{cache_root}/base_dataset",
    )

    dataset_time = time.time() - dataset_start
    print(f"✓ Dataset loaded in {dataset_time:.2f} seconds")

    # STEP 2: Apply StageNet mortality prediction task with num_workers=4
    print("\n[2/2] Applying mortality prediction task (num_workers=4)...")
    task_start = time.time()

    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=4,
    )

    task_time = time.time() - task_start
    print(f"✓ Task processing completed in {task_time:.2f} seconds")

    # Measure cache sizes
    print("\n[3/3] Measuring cache sizes...")
    base_cache_dir = f"{cache_root}/base_dataset"
    task_cache_dir = f"{base_cache_dir}/tasks"

    base_cache_size = get_directory_size(base_cache_dir)
    task_cache_size = get_directory_size(task_cache_dir)
    total_cache_size = base_cache_size + task_cache_size

    print(f"✓ Base dataset cache: {format_size(base_cache_size)}")
    print(f"✓ Task samples cache: {format_size(task_cache_size)}")
    print(f"✓ Total cache size: {format_size(total_cache_size)}")

    # Total time and peak memory
    total_time = time.time() - total_start
    peak_mem = PEAK_MEM_USAGE

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print("Configuration:")
    print("  - num_workers: 4")
    print(f"  - dev mode: {dev}")
    print(f"  - Total samples: {len(sample_dataset)}")
    print("\nTiming:")
    print(f"  - Dataset loading: {dataset_time:.2f}s")
    print(f"  - Task processing: {task_time:.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    print("\nCache Sizes:")
    print(f"  - Base dataset cache: {format_size(base_cache_size)}")
    print(f"  - Task samples cache: {format_size(task_cache_size)}")
    print(f"  - Total cache: {format_size(total_cache_size)}")
    print("\nMemory:")
    print(f"  - Peak memory usage: {format_size(peak_mem)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
