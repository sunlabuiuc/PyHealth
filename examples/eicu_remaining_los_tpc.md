# TPC for remaining ICU length of stay

Contributor: Audrey Labrie.

Paper: Rocheteau, Lio, and Hyland (2021), *Temporal Pointwise Convolutional
Networks for Length of Stay Prediction in the Intensive Care Unit*.
[Paper](https://arxiv.org/abs/2007.09483) ·
[Authors' code](https://github.com/EmmaRocheteau/TPC-LoS-prediction).

## About

This example demonstrates TPC in PyHealth and compares five kernel-size and
dilation settings using prepared eICU training and validation files. It computes
MSLE and MAD. An LSTM baseline and held-out test evaluation are outside this
example's scope.

TPC extends `BaseModel`. The example uses `create_sample_dataset`, raw tensor
processors, and a custom batch function that pads shorter stays. Raw eICU
extraction, hourly resampling, scaling, and encoding must be done beforehand.
The example does not load raw tables through `EICUDataset`.

### Relation to the existing YAIB task

PyHealth's `YAIBLengthOfStayEICU` in `pyhealth/tasks/yaib_eicu.py` uses a
seven-day-capped, min-max-scaled target. This example instead uses remaining
days without that cap or target scaling and trains with squared ordinary-log
error. It provides a different target setup and does not replace the YAIB task.

## Run

Run these commands from the PyHealth repository with its dependencies installed.

**Synthetic check:** with no arguments, run one forward/backward pass on two
made-up stays on CPU. This checks shapes and gradients; it does not run training
epochs or save a model, and it needs no eICU files.

```bash
python examples/eicu_remaining_los_tpc.py
```

**Short training run:** train the reference TPC for two epochs with seed 9
using prepared training and validation data.

```bash
python examples/eicu_remaining_los_tpc.py \
  --data-dir /path/to/eicu_demo_trainfit \
  --output-dir /path/to/new_tpc_example \
  --device cuda
```

**Ablations:** train five TPC settings with three seeds, for 15 runs. Each run
uses at most 30 epochs and stops after seven epochs without improved validation
MSLE.

```bash
python examples/eicu_remaining_los_tpc.py \
  --data-dir /path/to/eicu_demo_trainfit \
  --output-dir /path/to/new_tpc_ablation \
  --device cuda --ablation --seeds 9 19 29 --epochs 30 --patience 7
```

Use `--device cpu` when CUDA is unavailable. Each `--output-dir` must be a new
directory. Data runs read only `train/` and `val/`; no test split is read.

| Setting | Kernel size | Dilation rates |
| --- | ---: | --- |
| `tpc_reference` | 4 | 1, 3, 6, 9, 12, 15, 18, 21, 24 |
| `tpc_kernel2` | 2 | Same as reference |
| `tpc_kernel3` | 3 | Same as reference |
| `tpc_no_dilation` | 4 | Nine 1s |
| `tpc_gradual_dilation` | 4 | 1, 2, 3, 4, 5, 6, 7, 8, 9 |

## Prepared file format

The data directory must contain these three JSON files. Each must have a
`features` array of ordered column names; other JSON fields are allowed.

| JSON file | Ordered feature names |
| --- | --- |
| `timeseries_scaler.json` | Measurement names, called `M` below |
| `flat_encoder.json` | Static feature names, called `S` below |
| `diagnosis_encoder.json` | Diagnosis feature names, called `D` below |

The loader reads these name lists only. The CSV inputs must already be scaled
and encoded using transformations fitted on the training data.

Both `train/` and `val/` must contain the following CSV files:

| File | Required columns |
| --- | --- |
| `timeseries.csv` | Exact order: `patient`, `time`, all `M` columns, all corresponding `<measurement>_mask` columns in the same order, `hour` |
| `flat.csv` | Exact order: `patient`, then all `S` columns; one row per ICU stay |
| `diagnoses.csv` | Exact order: `patient`, then all `D` columns; one row per ICU stay, including stays with no diagnosis records |
| `labels.csv` | `patient`, `uniquepid`, `actualiculos`; column order and extra columns are allowed |

- `patient` is an integer ICU-stay ID. `uniquepid` is a nonempty person ID.
  The training and validation sets must contain different people.
- All four CSVs must cover the same stays within each split. Stay IDs must be
  unique in `labels.csv`, `flat.csv`, and `diagnoses.csv`.
- Numeric inputs must be finite. `actualiculos` is the total ICU stay in days
  and must be finite and positive.
- Time-series rows must be grouped by stay. Within each stay, `time` contains
  consecutive integer hours between 1 and 335, including an hour after hour 5.
- The `_mask` columns contain observation-recency indicators in `[0, 1]`, not
  the padding or loss mask. `hour` is the prepared, scaled hour-of-day feature.

The loader converts `time` to days for model input and creates a target at each
hour as `max(actualiculos - time / 24, 1 / 48)`. Only hours with `time > 5` are
scored; earlier hours remain as input, and padding is excluded from the loss.
Targets have no seven-day upper cap. Model predictions are separately bounded
to about 1/48 through 100 days.

## Training and output

Data runs use Adam with learning rate 0.001, batch size 8, gradient clipping at
5, and float32. TPC normalization is off. The checkpoint is selected by the
lowest validation MSLE. MSLE uses `log(y)`, not `log1p(y)`; MAD is mean absolute
error in days. Epoch metrics average over all scored hours.

Each setting and seed saves `config.json`, `history.csv`, and `best.pt` in its
own subdirectory. The output directory also contains `validation.csv` with the
selected results and `validation_summary.csv` with the mean and sample standard
deviation across seeds. With one seed, the sample standard deviation is undefined.
