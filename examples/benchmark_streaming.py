"""Benchmark Streaming Mode vs Normal Mode Performance

This script compares PyHealth's streaming mode vs normal mode using the
StageNet mortality prediction task on MIMIC-IV (dev mode: 1000 patients).

Measures:
- Processing time
- Peak memory usage
- Sample throughput

Usage:
    python benchmark_streaming.py
"""

import argparse
import time
from typing import Dict, Any

import psutil
from torch.utils.data import DataLoader

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def benchmark_mode(
    mode_name: str,
    ehr_root: str,
    cache_dir: str,
    stream: bool,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Benchmark a single mode (streaming or normal).

    Args:
        mode_name: Name for display ("Streaming" or "Normal")
        ehr_root: Root directory of MIMIC-IV dataset
        cache_dir: Directory for cache
        stream: Whether to use streaming mode
        num_workers: Number of worker processes for parallel processing

    Returns:
        Dictionary containing benchmark metrics
    """
    print(f"\n{'='*70}")
    print(f"{mode_name} Mode (Dev: 1000 patients, {num_workers} workers)")
    print(f"{'='*70}\n")

    ehr_tables = [
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "labevents",
    ]

    task = MortalityPredictionStageNetMIMIC4()

    # Get process for memory tracking
    process = psutil.Process()

    # Start tracking
    baseline_memory = process.memory_info().rss
    total_start = time.time()

    # Phase 1: Initialize dataset
    print(f"[1/2] Initializing MIMIC4Dataset (stream={stream})...")
    init_start = time.time()
    init_memory_before = process.memory_info().rss

    dataset = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=ehr_tables,
        stream=stream,
        cache_dir=cache_dir,
        dev=False,  # Change to True for dev mode (1000 patients)
    )

    init_time = time.time() - init_start
    init_memory_after = process.memory_info().rss
    init_memory_delta = init_memory_after - init_memory_before
    print(f"      Time: {init_time:.2f}s")
    print(f"      Memory Delta: {format_bytes(init_memory_delta)}")

    # Phase 2: Apply task
    print(f"[2/2] Generating samples with {task.task_name}...")
    task_start = time.time()
    task_memory_before = process.memory_info().rss

    sample_dataset = dataset.set_task(
        task, cache_dir=cache_dir, num_workers=num_workers
    )

    task_time = time.time() - task_start
    task_memory_after = process.memory_info().rss
    task_memory_delta = task_memory_after - task_memory_before
    print(f"      Time: {task_time:.2f}s")
    print(f"      Memory Delta: {format_bytes(task_memory_delta)}")

    # Get final stats
    total_time = time.time() - total_start
    final_memory = process.memory_info().rss
    total_memory_delta = final_memory - baseline_memory

    num_samples = len(sample_dataset)

    # Print summary
    print(f"\n{'='*70}")
    print(f"{mode_name} Mode Results")
    print(f"{'='*70}")
    print(f"Total Time:        {total_time:.2f}s")
    print(f"  - Init:          {init_time:.2f}s")
    print(f"  - Task/Samples:  {task_time:.2f}s")
    print(f"Total Memory:      {format_bytes(final_memory)}")
    print(f"Memory Delta:      {format_bytes(total_memory_delta)}")
    print(f"  - Init Delta:    {format_bytes(init_memory_delta)}")
    print(f"  - Task Delta:    {format_bytes(task_memory_delta)}")
    print(f"Samples:           {num_samples}")
    if total_time > 0:
        throughput = num_samples / total_time
        print(f"Throughput:        {throughput:.2f} samples/sec")
    print(f"{'='*70}\n")

    return {
        "mode": mode_name,
        "total_time": total_time,
        "init_time": init_time,
        "task_time": task_time,
        "total_memory": final_memory,
        "memory_delta": total_memory_delta,
        "init_memory_delta": init_memory_delta,
        "task_memory_delta": task_memory_delta,
        "num_samples": num_samples,
        "throughput": num_samples / total_time if total_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming vs normal mode")
    parser.add_argument(
        "--ehr_root",
        type=str,
        default="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        help="Root directory of MIMIC-IV dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../benchmark_cache",
        help="Directory for cache files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PyHealth Streaming vs Normal Mode Benchmark")
    print("=" * 70)
    print(f"Dataset: MIMIC-IV (Dev Mode: 1000 patients)")
    print(f"Task: StageNet Mortality Prediction")
    print(f"EHR Root: {args.ehr_root}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"Workers: {args.num_workers}")
    print("=" * 70)

    # Benchmark streaming mode
    streaming_results = benchmark_mode(
        mode_name="STREAMING",
        ehr_root=args.ehr_root,
        cache_dir=args.cache_dir,
        stream=True,
        num_workers=args.num_workers,
    )

    # Benchmark normal mode
    normal_results = benchmark_mode(
        mode_name="NORMAL",
        ehr_root=args.ehr_root,
        cache_dir=args.cache_dir,
        stream=False,
        num_workers=args.num_workers,
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("Mode          Time      Memory Delta  Samples  Throughput")
    print("-" * 70)
    s_time = streaming_results["total_time"]
    s_mem = streaming_results["memory_delta"]
    s_samp = streaming_results["num_samples"]
    s_through = streaming_results["throughput"]

    print(
        f"Streaming     {s_time:6.2f}s  "
        f"{format_bytes(s_mem):>12}  "
        f"{s_samp:>7}  "
        f"{s_through:>6.2f} samp/s"
    )

    n_time = normal_results["total_time"]
    n_mem = normal_results["memory_delta"]
    n_samp = normal_results["num_samples"]
    n_through = normal_results["throughput"]

    print(
        f"Normal        {n_time:6.2f}s  "
        f"{format_bytes(n_mem):>12}  "
        f"{n_samp:>7}  "
        f"{n_through:>6.2f} samp/s"
    )
    print("-" * 70)

    # Calculate improvements
    time_diff = s_time - n_time
    mem_reduction = (1 - s_mem / n_mem) * 100 if n_mem > 0 else 0

    print("\nStreaming vs Normal:")
    print(f"  Time difference: {time_diff:+.2f}s")
    print(f"  Memory reduction: {mem_reduction:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
