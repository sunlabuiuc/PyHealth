# CS598 Deep Learning for Healthcare projet

This PR reproduces and contributes an implementation of:

**Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit
Rocheteau et al., ACM CHIL 2021**
Paper: https://arxiv.org/abs/2007.09483


## Contributors

* Michael Edukonis (`meduk2@illinois.edu`)
* Keon Young Lee (`kylee7@illinois.edu`)
* Tanmay Thareja (`tanmayt3@illinois.edu`)

# PR Overview

## Contribution Types

This PR includes:

- Model contribution
- Standalone Task contribution
- Synthetic data tests
- End to End Pipeline (Model + Task + Dataset configs) example scripts + combined dual dataset run

## Problem Overview


Efficient ICU bed management depends critically on estimating how long a patient will remain in the ICU. 

This is formulated as:

	•	Input: Patient data up to hour _t_
	•	Output: Remaining ICU length of stay _(LoS)_ at time _t_

We follow the formulation in the original paper, predicting remaining LoS at each hour of the ICU stay.

## Implementation Details

**1) Model Contribution**

_pyhealth.models.tpc_

We implement the Temporal Pointwise Convolution (TPC) model as a PyHealth-compatible model by extending BaseModel and follows the original paper’s architecture with adaptations for PyHealth’s input/output interfaces. Index files were updated to include the new modules accordingly.

**2) Task Contribution**

We implement a custom PyHealth task: Hourly Remaining Length of Stay. Index files were updated to include the new modules accordingly.

_pyhealth.tasks.hourly_los_

Task Definition

	•	Predict remaining LoS at every hour
	•	Predictions start after first 5 hours
	•	Continuous regression task

Motivation

	•	Mimics real-world ICU monitoring
	•	Enables dynamic prediction updates

**3) Ablation Study/ Example Usage**

We implemented scripts for runnning the pipeline end to end with support for different experiemental setups or ablations.

_examples/eicu_hourly_los_tpc.py_ : This provides an end-to-end script for reproducing and evaluating pipeline on EICU dataset.
_examples/mimic4_hourly_los_tpc.py_ : This provides an end-to-end script for reproducing and evaluating pipeline on MIMIC-IV dataset.
_examples/run_dual_dataset_tpc.py_ : This utility scripts runs the pipeline on both datasets and produces a combined report.

**4) Test Cases**

We implemented fast performing test cases for our Model and Task contribution using Sythentic Data.

_tests/models/test_tpc.py_
_tests/tasks/test_hourly_los.py_


## Experimental Setup and Findings

1) Ablations to include/exclude domain specific engineering (skip connections, decay indicators, etc)
<br>
<img width="306" height="140" alt="image" src="https://github.com/user-attachments/assets/e607fc67-7d56-4ce7-ae19-eaa29dc02a1f" />
<br>
2) Comparison across using combined temporal and pointwise convolutions vs using either architecture alone.
<br>
<img width="293" height="140" alt="image" src="https://github.com/user-attachments/assets/67d2c90b-d2d9-449f-add5-7413ec0db0db" />
<br>
3) Feature independant (no weight-sharing) vs weight-shared temporal convolutions.
<br>
<img width="306" height="140" alt="image" src="https://github.com/user-attachments/assets/5c6ca146-56b6-4257-94a7-09c20c7c7533" />
<br>
4) Evaluating MSLE loss  vs MSE loss for skewed LoS target regression.
<br>
<img width="306" height="108" alt="image" src="https://github.com/user-attachments/assets/454f834a-b496-4a84-843e-84f4fe767031" />
<br>
## Testing model with varying hyperparameters.

We varied key optimization and architectural hyperparameters (e.g. learning rate, dropout rate, etc) while keeping preprocessing and data splits fixed.

<img width="431" height="278" alt="image" src="https://github.com/user-attachments/assets/17dcb797-b762-47dc-b43d-475dbfd89ce9" />


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
├── docs/api/models.rst
├── docs/api/models/pyhealth.models.tpc.rst
├── docs/api/tasks.rst
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
