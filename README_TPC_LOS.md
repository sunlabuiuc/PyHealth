# TPC Hourly Length-of-Stay Replication

CS598 Deep Learning for Healthcare project contribution for reproducing:

**Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit**

## Contributors

* Michael Edukonis (`meduk2`)
* Keon Young Lee
* Tanmay Thareja

## Project Overview

This contribution adds a PyHealth implementation of the **Temporal Pointwise Convolution (TPC)** model and an **hourly ICU remaining length-of-stay task** for eICU and MIMIC-IV style data.

The repository additions include:

* a new TPC model implementation,
* a custom hourly LoS task,
* dataset configuration files for eICU and MIMIC-IV,
* synthetic-data tests,
* runnable example scripts for eICU, MIMIC-IV, and a combined dual-dataset run.

## File Structure

```text
.
├── pyhealth/models/tpc.py
├── pyhealth/tasks/hourly_los.py
├── pyhealth/datasets/configs/eicu_tpc.yaml
├── pyhealth/datasets/configs/mimic4_ehr_tpc.yaml
├── examples/eicu_hourly_los_tpc.py
├── examples/mimic4_hourly_los_tpc.py
├── examples/run_dual_dataset_tpc.py
├── tests/models/test_tpc.py
├── tests/tasks/test_hourly_los.py
├── docs/api/models/pyhealth.models.tpc.rst
├── docs/api/tasks/pyhealth.tasks.hourly_los.rst
└── README_TPC_LOS.md
```

## Setup

1. Clone the repository.
2. Create and activate a python virtual environment.
3. Install project dependencies.
4. Set dataset paths with environment variables for the example scripts.

## Quick Start (Synthetic Data)

### eICU example

```bash
EICU_ROOT=/path/to/synthetic/eicu \
PYTHONPATH=. python3 examples/eicu_hourly_los_tpc.py \
  --epochs 1 \
  --batch_size 2 \
  --max_samples 8 \
  --channel_mode full
```

### MIMIC-IV example

```bash
MIMIC4_ROOT=/path/to/synthetic/mimic4 \
PYTHONPATH=. python3 examples/mimic4_hourly_los_tpc.py \
  --epochs 1 \
  --batch_size 2 \
  --max_samples 8 \
  --channel_mode full
```

### Combined dual-dataset run

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

## Notes on Real Data

For full eICU or MIMIC-IV experiments, point `EICU_ROOT` or `MIMIC4_ROOT` to the real dataset locations and increase settings such as `--epochs`, `--batch_size`, and `--max_samples` as needed.

## Tests

Run the project-specific tests with:

```bash
python3 -m pytest tests/models/test_tpc.py tests/tasks/test_hourly_los.py -q
```

## Documentation

API documentation entries were added for:

* `pyhealth.models.tpc`
* `pyhealth.tasks.hourly_los`

## Output

The example scripts print compact summary lines for quick validation:

* `ABLATION_SUMMARY` for eICU
* `MIMIC_SUMMARY` for MIMIC-IV

The dual-dataset runner parses both and prints a combined summary table.

## Environment

This project is designed to run within the PyHealth environment.

Recommended setup:

- Python 3.12
- pyhealth (>= 2.0.0)
- torch
- standard scientific Python stack (numpy, pandas)

Install PyHealth and dependencies following the main repository instructions.
