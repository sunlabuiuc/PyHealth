# Performance benchmarks

These scripts measure how fast PyHealth turns raw MIMIC-IV tables into task
samples, and how much memory that takes. They compare PyHealth 2.x against
PyHealth 1.1.6, plain pandas and [meds_reader](https://github.com/som-shahlab/meds_reader).
They're kept for reproducing those comparisons. They are not tutorials. To learn
the library, start with the [tutorial series](../tutorials/series/README.md).

All three tasks are covered: in-hospital mortality (StageNet-style, with labs),
drug recommendation and length of stay.

## What's here

| Path | What it runs | Needs |
|---|---|---|
| `benchmark_workers_n*.py` | PyHealth 2.x, sweeping `num_workers` in `set_task`; reports time, cache size, peak RSS, open file descriptors | current PyHealth |
| `benchmark_workers_{1,4,12}.py` | the same as above, fixed worker count | current PyHealth |
| `benchmark_split_set_task.py` | `set_task` with and without `split=PatientSplit(...)` on synthetic MIMIC-III tables (no real data) | current PyHealth |
| `benchmark_pandas*.py` | the same tasks written in plain pandas | pandas |
| `memtest.py` | StageNet mortality on MIMIC-IV, for memory profiling | current PyHealth |
| `legacy_ver/` | the same tasks on PyHealth 1.1.6 (`task_fn=` API) | `pip install pyhealth==1.1.6` in a **separate** environment |
| `meds_reader_ver/` | the same tasks through MEDS ETL + meds_reader | `pip install meds_etl meds_reader` |
| `patient_exploration/` | time to load a dataset and fetch one patient, for each library | as per file name |
| `loc/` | minimal versions of each task, used to compare lines of code; `minimal_legacy_*` need PyHealth 1.1.6 | as per file name |

## Running

The defaults point to the authors' cluster paths
(`/srv/local/data/physionet.org/files/mimiciv/2.2/`). Pass your own paths:

```bash
python examples/benchmark_perf/benchmark_workers_n.py \
    --ehr-root /path/to/mimiciv/2.2/ --cache-root /tmp/pyhealth_bench \
    --workers 1,4,8 --repeats 3
```

To check that a script runs without credentialed data, point it at the
MIMIC-IV demo tables in the repo (the numbers are meaningless at that size):

```bash
python examples/benchmark_perf/benchmark_workers_n.py --workers 1 \
    --ehr-root test-resources/core/mimic4demo --cache-root /tmp/pyhealth_bench
```

`benchmark_split_set_task.py` generates its own synthetic data:

```bash
python examples/benchmark_perf/benchmark_split_set_task.py --patients 50000
```

Scripts without command-line options (`benchmark_workers_{1,4,12}.py`,
`memtest.py`, `loc/*`, `patient_exploration/clean_*`) have the data path
as a constant near the top. Edit it before running.

MIMIC-IV is credentialed: request access at
<https://physionet.org/content/mimiciv/>.
