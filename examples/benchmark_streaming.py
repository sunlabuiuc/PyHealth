"""Benchmark PyHealth Dataset Processing Performance

This script benchmarks PyHealth dataset processing using the StageNet mortality
prediction task on MIMIC-IV.

You can benchmark either streaming mode or normal mode:
- Streaming mode: Memory-efficient disk-backed processing (use --stream flag)
- Normal mode: Traditional in-memory processing (default)

Measures:
- Processing time
- Peak memory usage
- Sample throughput

Usage:
    # Benchmark normal mode
    python benchmark_streaming.py

    # Benchmark streaming mode
    python benchmark_streaming.py --stream

    # Benchmark streaming mode with batch_size=1000
    python benchmark_streaming.py --stream --batch_size 1000

    # Dev mode with 5000 patients
    python benchmark_streaming.py --stream --dev --dev_max_patients 5000

    # Use cached processors to skip fitting
    python benchmark_streaming.py --stream --use_cached_processors
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import psutil
from torch.utils.data import DataLoader

from pyhealth.datasets import (
    MIMIC4Dataset,
    load_processors,
    save_processors,
)
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
    batch_size: int = 100,
    use_cached_processors: bool = False,
    processor_dir: str = None,
    dev: bool = False,
    dev_max_patients: int = 1000,
) -> Dict[str, Any]:
    """Benchmark a single mode (streaming or normal).

    Args:
        mode_name: Name for display ("Streaming" or "Normal")
        ehr_root: Root directory of MIMIC-IV dataset
        cache_dir: Directory for cache
        stream: Whether to use streaming mode
        num_workers: Number of worker processes for parallel processing
        batch_size: Number of patients to process per batch (streaming only)
        use_cached_processors: Whether to load pre-fitted processors
        processor_dir: Directory containing cached processors
        dev: Whether to enable dev mode (limit patients)
        dev_max_patients: Maximum number of patients in dev mode

    Returns:
        Dictionary containing benchmark metrics
    """
    print(f"\n{'='*70}")
    print(
        f"{mode_name} Mode "
        f"({num_workers} workers, "
        f"batch_size={batch_size if stream else 'N/A'}, "
        f"cached_procs={use_cached_processors})"
    )
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
        dev=dev,
        dev_max_patients=dev_max_patients,
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

    # Load processors if requested
    input_processors = None
    output_processors = None

    if use_cached_processors and processor_dir:
        processor_dir_path = Path(processor_dir)
        input_procs_file = processor_dir_path / "input_processors.pkl"
        output_procs_file = processor_dir_path / "output_processors.pkl"

        if input_procs_file.exists() and output_procs_file.exists():
            print(f"      Loading processors from {processor_dir}...")
            load_start = time.time()
            input_processors, output_processors = load_processors(processor_dir)
            load_time = time.time() - load_start
            print(f"      Processors loaded in {load_time:.2f}s")
            print(f"      ✓ Skipping processor fitting!")
        else:
            print(f"      WARNING: Processor files not found in {processor_dir}")
            print(f"      Will create new processors")

    sample_dataset = dataset.set_task(
        task,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=batch_size,
        input_processors=input_processors,
        output_processors=output_processors,
    )

    # Save processors if they were newly created
    if use_cached_processors and processor_dir:
        if input_processors is None and output_processors is None:
            print(f"      Saving processors to {processor_dir}...")
            save_start = time.time()
            save_processors(sample_dataset, processor_dir)
            save_time = time.time() - save_start
            print(f"      Processors saved in {save_time:.2f}s")

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
    parser = argparse.ArgumentParser(
        description="Benchmark PyHealth dataset processing"
    )
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
        "--stream",
        action="store_true",
        help="Use streaming mode (default: normal mode)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of patients to process per batch in streaming mode",
    )
    parser.add_argument(
        "--use_cached_processors",
        action="store_true",
        help="Load pre-fitted processors instead of fitting from scratch",
    )
    parser.add_argument(
        "--processor_dir",
        type=str,
        default="../output/processors/stagenet_mortality_mimic4_benchmark_10k",
        help="Directory to load/save cached processors",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Enable dev mode to limit number of patients (default: False)",
    )
    parser.add_argument(
        "--dev_max_patients",
        type=int,
        default=1000,
        help="Maximum number of patients in dev mode (default: 1000)",
    )

    args = parser.parse_args()

    mode_name = "STREAMING" if args.stream else "NORMAL"

    print("\n" + "=" * 70)
    print(f"PyHealth {mode_name} Mode Benchmark")
    print("=" * 70)
    print(
        f"Dataset: MIMIC-IV (Dev Mode: {args.dev}, Max Patients: {args.dev_max_patients})"
    )
    print("Task: StageNet Mortality Prediction")
    print(f"Mode: {mode_name}")
    print(f"EHR Root: {args.ehr_root}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"Workers: {args.num_workers}")
    if args.stream:
        print(f"Batch Size: {args.batch_size}")
    print(f"Use Cached Processors: {args.use_cached_processors}")
    if args.use_cached_processors:
        print(f"Processor Dir: {args.processor_dir}")
    print("=" * 70)

    # Benchmark the selected mode
    results = benchmark_mode(
        mode_name=mode_name,
        ehr_root=args.ehr_root,
        cache_dir=args.cache_dir,
        stream=args.stream,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        use_cached_processors=args.use_cached_processors,
        processor_dir=args.processor_dir,
        dev=args.dev,
        dev_max_patients=args.dev_max_patients,
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Mode:              {mode_name}")
    print(f"Total Time:        {results['total_time']:.2f}s")
    print(f"Peak Memory:       {format_bytes(results['total_memory'])}")
    print(f"Memory Delta:      {format_bytes(results['memory_delta'])}")
    print(f"Samples Generated: {results['num_samples']}")
    print(f"Throughput:        {results['throughput']:.2f} samples/sec")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
