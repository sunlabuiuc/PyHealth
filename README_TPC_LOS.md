# TPC Hourly Length-of-Stay (LoS) Replication

This contribution implements the **Temporal Pointwise Convolution (TPC)** model for hourly ICU remaining length-of-stay prediction using the PyHealth framework.

## Quick Start (Synthetic Data)

The example scripts support fast execution using synthetic datasets.

### eICU Example

```bash
EICU_ROOT=/path/to/synthetic/eicu \
PYTHONPATH=. python3 examples/eicu_hourly_los_tpc.py \
  --epochs 1 \
  --batch_size 2 \
  --max_samples 8 \
  --channel_mode full
```

### MIMIC-IV Example

```bash
MIMIC4_ROOT=/path/to/synthetic/mimic4 \
PYTHONPATH=. python3 examples/mimic4_hourly_los_tpc.py \
  --epochs 1 \
  --batch_size 2 \
  --max_samples 8 \
  --channel_mode full
```

### Dual Dataset Run

```bash
EICU_ROOT=/path/to/synthetic/eicu \
MIMIC4_ROOT=/path/to/synthetic/mimic4 \
PYTHONPATH=. python3 examples/run_dual_dataset_tpc.py \
  --eicu_max_samples 8 \
  --mimic_max_samples 8 \
  --eicu_epochs 1 \
  --mimic_epochs 1 \
  --channel_mode full
```

## Key Files

* `pyhealth/models/tpc.py` – TPC model implementation
* `pyhealth/tasks/hourly_los.py` – hourly LoS prediction task
* `examples/eicu_hourly_los_tpc.py` – eICU pipeline
* `examples/mimic4_hourly_los_tpc.py` – MIMIC-IV pipeline
* `examples/run_dual_dataset_tpc.py` – combined run

## Notes

* Uses environment variables (`EICU_ROOT`, `MIMIC4_ROOT`) for dataset paths
* Designed for fast synthetic-data testing
* Increase `--epochs` and `--max_samples` for full runs
