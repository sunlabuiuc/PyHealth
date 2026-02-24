# Loader Benchmark

This benchmark compares PyHealth's streaming sample loader against the legacy in-memory loader using synthetic patient data. It focuses on:

- Peak RAM usage (MB) using `tracemalloc`
- Wall-clock time (seconds) using `time.perf_counter`
- Throughput (patients/second)

Why this matters:
- The streaming loader is designed to reduce memory pressure on larger datasets.
- The legacy in-memory loader can be faster on small datasets but may use more RAM.
- This benchmark provides a reproducible baseline for tradeoff decisions by dataset size.

## How to run

```bash
python benchmarks/loader_benchmark.py
```

The script benchmarks three scales by default:
- `small`: 100 patients
- `medium`: 1,000 patients
- `large`: 5,000 patients

Outputs:
- `benchmarks/results.csv`
- `benchmarks/benchmark_chart.png`

## Sample terminal output

```text
 scale  num_patients           loader         dataset_class status wall_time_sec peak_ram_mb throughput_patients_per_sec note
 small           100 legacy_in_memory InMemorySampleDataset     ok        0.0022      0.1253                 45,313.8046
 small           100        streaming         SampleDataset     ok        7.7841      1.3903                     12.8466
medium          1000 legacy_in_memory InMemorySampleDataset     ok        0.0155      0.8346                 64,692.1166
medium          1000        streaming         SampleDataset     ok        8.2625      1.3849                    121.0291
 large          5000 legacy_in_memory InMemorySampleDataset     ok        0.0774      4.0510                 64,578.5552
 large          5000        streaming         SampleDataset     ok       11.8215      5.5277                    422.9585
```

If streaming mode is unavailable in the environment, streaming rows are marked as `skipped` with a note, and the script still completes successfully.

## Key findings

- Placeholder: summarize RAM and time differences after running in your environment.
- Placeholder: note when streaming becomes beneficial by scale.
