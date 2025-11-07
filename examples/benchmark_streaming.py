"""Benchmark Streaming Mode Performance with StageNet Task

This script benchmarks PyHealth's streaming mode implementation using the
StageNet mortality prediction task on MIMIC-IV, measuring:
- Peak memory usage (should be < 2GB regardless of dataset size)
- Processing time (first run vs cached run)
- Cache size on disk
- Memory reduction compared to normal mode

Based on the mortality_mimic4_stagenet_v2.py example.

Usage:
    python benchmark_streaming.py --ehr_root /path/to/mimic4 --cache_dir ./cache
"""

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Optional

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from tqdm import tqdm


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_cache_size(cache_dir: Path) -> int:
    """Get total size of cache directory."""
    total_size = 0
    for item in cache_dir.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


def benchmark_streaming_mode(
    ehr_root: str,
    cache_dir: str,
    compare_normal: bool = True,
) -> Dict[str, Any]:
    """Benchmark streaming mode performance with StageNet task.

    Args:
        ehr_root: Root directory of MIMIC-IV dataset
        cache_dir: Directory for streaming cache
        compare_normal: Whether to compare with normal mode (may fail with large datasets)

    Returns:
        Dictionary containing benchmark metrics
    """
    print(f"\n{'='*70}")
    print("Benchmarking Streaming Mode: MIMIC-IV StageNet Mortality Prediction")
    print(f"{'='*70}\n")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Use same tables as StageNet v2 example
    ehr_tables = [
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "labevents",
    ]

    # Use StageNet mortality prediction task
    task = MortalityPredictionStageNetMIMIC4()

    results = {}

    # === Phase 1: First Run (Cache Building) ===
    print("Phase 1: First run with streaming mode (building cache)...")
    print(f"Tables: {', '.join(ehr_tables)}")
    tracemalloc.start()
    start_time = time.time()

    # Initialize dataset in streaming mode
    print("  Initializing MIMIC4Dataset with stream=True...")
    dataset = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=ehr_tables,
        stream=True,
        cache_dir=cache_dir,
    )

    # Apply StageNet mortality prediction task
    print("  Applying MortalityPredictionStageNetMIMIC4 task...")
    sample_dataset = dataset.set_task(task, cache_dir=cache_dir)

    # Iterate through all samples
    sample_count = 0
    for sample in tqdm(sample_dataset, desc="Processing samples"):
        sample_count += 1

    first_run_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["first_run_time"] = first_run_time
    results["first_run_peak_memory"] = peak
    results["sample_count"] = sample_count

    print(f"✓ First run complete:")
    print(f"  - Time: {first_run_time:.2f}s")
    print(f"  - Peak memory: {format_bytes(peak)}")
    print(f"  - Samples processed: {sample_count:,}")

    # === Phase 2: Cache Size ===
    cache_size = get_cache_size(cache_path)
    results["cache_size"] = cache_size
    print(f"  - Cache size: {format_bytes(cache_size)}")

    # === Phase 3: Second Run (Cache Reuse) ===
    print("\nPhase 2: Second run with streaming mode (reusing cache)...")
    tracemalloc.start()
    start_time = time.time()

    # Initialize dataset in streaming mode (cache already exists)
    dataset2 = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=ehr_tables,
        stream=True,
        cache_dir=cache_dir,
    )

    # Apply task (will reuse cached samples)
    sample_dataset2 = dataset2.set_task(task, cache_dir=cache_dir)

    # Iterate through samples again
    sample_count2 = 0
    for sample in tqdm(sample_dataset2, desc="Processing samples"):
        sample_count2 += 1

    second_run_time = time.time() - start_time
    current, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["second_run_time"] = second_run_time
    results["second_run_peak_memory"] = peak2
    results["speedup"] = first_run_time / second_run_time if second_run_time > 0 else 0

    print(f"✓ Second run complete:")
    print(f"  - Time: {second_run_time:.2f}s")
    print(f"  - Peak memory: {format_bytes(peak2)}")
    print(f"  - Speedup: {results['speedup']:.2f}x")

    # === Phase 4: Comparison with Normal Mode (Optional) ===
    if compare_normal:
        print("\nPhase 3: Comparing with normal mode (stream=False)...")
        print("  WARNING: This may fail with large datasets due to memory constraints!")
        try:
            tracemalloc.start()
            start_time = time.time()

            # Initialize dataset in normal mode (loads all data to memory)
            print("  Initializing MIMIC4Dataset with stream=False...")
            dataset_normal = MIMIC4Dataset(
                ehr_root=ehr_root,
                ehr_tables=ehr_tables,
                stream=False,  # Normal mode - loads everything to memory
            )

            print("  Applying task in normal mode...")
            sample_dataset_normal = dataset_normal.set_task(task)

            normal_time = time.time() - start_time
            current_normal, peak_normal = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results["normal_mode_time"] = normal_time
            results["normal_mode_peak_memory"] = peak_normal
            results["memory_reduction"] = (peak_normal - peak) / peak_normal * 100

            print(f"✓ Normal mode completed:")
            print(f"  - Time: {normal_time:.2f}s")
            print(f"  - Peak memory: {format_bytes(peak_normal)}")
            print(f"  - Memory reduction with streaming: {results['memory_reduction']:.1f}%")

        except MemoryError as e:
            print("✗ Normal mode failed due to insufficient memory")
            print(f"    Error: {e}")
            results["normal_mode_failed"] = True
        except Exception as e:
            print(f"✗ Normal mode failed: {e}")
            results["normal_mode_failed"] = True
    else:
        print("\nPhase 3: Skipping normal mode comparison (--no-compare-normal flag set)")

    # === Summary ===
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset: MIMIC-IV")
    print(f"Task: StageNet Mortality Prediction")
    print(f"Tables: {', '.join(ehr_tables)}")
    print(f"Samples: {sample_count:,}")
    print(f"\nStreaming Mode Performance:")
    print(f"  First run:  {first_run_time:.2f}s @ {format_bytes(peak)}")
    print(f"  Cached run: {second_run_time:.2f}s @ {format_bytes(peak2)} ({results['speedup']:.2f}x faster)")
    print(f"  Cache size: {format_bytes(cache_size)}")

    if "memory_reduction" in results:
        print(f"\nMemory Efficiency Comparison:")
        print(f"  Normal mode:    {format_bytes(results['normal_mode_peak_memory'])}")
        print(f"  Streaming mode: {format_bytes(peak)}")
        print(f"  Reduction:      {results['memory_reduction']:.1f}%")
        print(f"  Memory saved:   {format_bytes(results['normal_mode_peak_memory'] - peak)}")

    print(f"\n✓ Streaming peak memory < 2GB: {peak < 2e9}")
    if peak < 2e9:
        print(f"  SUCCESS: Peak memory is {format_bytes(peak)}, well within the 2GB target!")
    else:
        print(f"  WARNING: Peak memory {format_bytes(peak)} exceeds 2GB target")
    
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyHealth streaming mode with MIMIC-IV StageNet task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python benchmark_streaming.py --ehr_root /srv/local/data/physionet.org/files/mimiciv/2.2/
  python benchmark_streaming.py --ehr_root /path/to/mimic4 --cache_dir ./my_cache --no-compare-normal
        """
    )
    parser.add_argument(
        "--ehr_root",
        type=str,
        required=True,
        help="Root directory of MIMIC-IV dataset (e.g., /srv/local/data/physionet.org/files/mimiciv/2.2/)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./benchmark_cache",
        help="Directory for streaming cache (default: ./benchmark_cache)",
    )
    parser.add_argument(
        "--no-compare-normal",
        action="store_true",
        help="Skip comparison with normal mode (useful for very large datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to save benchmark results",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("PyHealth Streaming Mode Benchmark")
    print("Based on mortality_mimic4_stagenet_v2.py example")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  EHR Root:       {args.ehr_root}")
    print(f"  Cache Dir:      {args.cache_dir}")
    print(f"  Compare Normal: {not args.no_compare_normal}")
    print()

    # Run benchmark
    results = benchmark_streaming_mode(
        ehr_root=args.ehr_root,
        cache_dir=args.cache_dir,
        compare_normal=not args.no_compare_normal,
    )

    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()

