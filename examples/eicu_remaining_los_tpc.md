# TPC for remaining ICU length of stay

Contributor: Audrey Labrie.

Paper: Rocheteau, Lio, and Hyland (2021), *Temporal Pointwise Convolutional
Networks for Length of Stay Prediction in the Intensive Care Unit*.
[Paper](https://arxiv.org/abs/2007.09483) ·
[Authors' code](https://github.com/EmmaRocheteau/TPC-LoS-prediction).

## About

This project tests TPC kernel sizes and dilation rates on eICU demo v2.0.1.
It also includes an LSTM baseline. It does not test mortality or MIMIC-IV.
TPC extends PyHealth's `BaseModel`. Prepared stays are loaded with
`create_sample_dataset` and raw tensor processors. A custom batch function
pads shorter stays. This example needs prepared files; it does not load raw
eICU tables through `EICUDataset` or prepare the data from start to finish.

## Run

From the PyHealth repository, these commands run a small CPU example with
made-up data, a two-epoch demo, and the five TPC settings:

```bash
python examples/eicu_remaining_los_tpc.py

python examples/eicu_remaining_los_tpc.py \
  --data-dir /path/to/eicu_demo_trainfit \
  --output-dir /path/to/new_tpc_example \
  --device cuda

python examples/eicu_remaining_los_tpc.py \
  --data-dir /path/to/eicu_demo_trainfit \
  --output-dir /path/to/new_tpc_ablation \
  --device cuda --ablation --seeds 9 19 29 --epochs 30 --patience 7
```

Data runs use only the existing training and validation sets, and need a new
output folder. They never open test files. The completed study is already saved.

The data folder needs `timeseries_scaler.json`, `flat_encoder.json`,
`diagnosis_encoder.json`, and `train`/`val` folders. Each split folder needs
`timeseries.csv`, `flat.csv`, `diagnoses.csv`, and `labels.csv`. Use the saved
patient splits and encoders fitted on training data.

## Data

The demo had 2,520 ICU stays. SQL kept 1,820, and hourly preprocessing kept
1,780. This includes 82 stays with no diagnosis records.

| Partition | ICU stays | People | Hourly input rows |
| --- | ---: | ---: | ---: |
| Training | 1,234 | 1,032 | 75,593 |
| Validation | 267 | 222 | 15,777 |
| Test | 279 | 222 | 17,454 |

The split used `uniquepid` and seed 9. No person was in more than one set.
Python scaling and category/diagnosis lists were fitted on training data.
SQL feature selection happened before the split, so that step was not based
on training data alone.

Inputs include 83 measurements, 64 static features, and 277 diagnosis indicators.
The 168 time-series channels are time since admission in days, 83 measurements,
83 indicators for how recent each measurement is, and scaled hour of day.
Those indicators are 1 when observed, 0 if never observed, and
`1 / (gap_hours * 4/3)` otherwise. They differ from the mask used to score hours.

Hours run from 1 to at most 335, before discharge. The target is
`max(actualiculos - elapsed_hours/24, 1/48)` days. Only hours where
`elapsed_hours > 5` are scored; padding is excluded. Earlier hours stay in the
input. Each of the 18 runs was tested on 16,059 valid hours.

## Training

TPC uses 9 layers, 12 temporal channels per feature group, 13 pointwise channels
per layer, 64 diagnosis units, and 17 hidden units before the final prediction.
Main dropout is 0.45 and temporal dropout is 0.05. Normalization is off.
Outputs are limited in log space to about 1/48 through 100 days.

All runs used Adam, learning rate 0.001, batch size 8, gradient clipping at 5,
float32, at most 30 epochs, and patience 7. A saved model had to improve
validation MSLE. Loss averages squared log error over valid hours in each batch;
epoch scores use all valid hours. Validation selected LSTM overall and kernel 3
among TPC settings. These choices were fixed before testing.

LSTM uses 2 layers, 128 hidden units, dropout 0.2, 64 diagnosis units, 17 final
hidden units, and main dropout 0.45. It used the same data splits, targets,
training budget, and rule for choosing the saved model.

| Configuration | Kernel | Dilations | Receptive field (hours) | Parameters |
| --- | ---: | --- | ---: | ---: |
| TPC reference | 4 | 1,3,6,9,12,15,18,21,24 | 328 | 982,100 |
| TPC kernel 2 | 2 | 1,3,6,9,12,15,18,21,24 | 110 | 624,932 |
| TPC kernel 3 | 3 | 1,3,6,9,12,15,18,21,24 | 219 | 803,516 |
| TPC no dilation | 4 | 1,1,1,1,1,1,1,1,1 | 28 | 982,100 |
| TPC gradual dilation | 4 | 1,2,3,4,5,6,7,8,9 | 136 | 982,100 |
| LSTM | — | — | — | 306,851 |

Each setting used seeds 9, 19, and 29 on the same patient split. Kernel size
changes the parameter count and how far back the model can look. Changing
dilation keeps the parameter count fixed. The paper also tested kernel sizes.

## Results

Values are mean ± sample SD over three training seeds. All 18 runs are included.
The SD shows variation between seeds; it is not a patient confidence interval.

| Configuration | MAD (days) | MSLE | MAPE (%) |
| --- | --- | --- | --- |
| LSTM | 2.2298 ± 0.0487 | 1.5911 ± 0.0415 | 120.0222 ± 12.7224 |
| TPC kernel 3 | 2.3319 ± 0.0832 | 1.6725 ± 0.1488 | 120.3331 ± 8.3155 |
| TPC reference | 2.3295 ± 0.0874 | 1.6934 ± 0.1324 | 119.3847 ± 8.9843 |
| TPC kernel 2 | 2.3693 ± 0.1015 | 1.7555 ± 0.1529 | 130.0173 ± 1.1646 |
| TPC gradual dilation | 2.3840 ± 0.0762 | 1.7715 ± 0.1255 | 124.4424 ± 8.4217 |
| TPC no dilation | 2.4282 ± 0.0003 | 1.8442 ± 0.0005 | 129.2414 ± 0.1277 |

| Configuration | MSE (days²) | R² | Linear-weighted kappa |
| --- | --- | --- | --- |
| LSTM | 15.8761 ± 1.0981 | -0.0572 ± 0.0731 | 0.1576 ± 0.0351 |
| TPC kernel 3 | 17.3892 ± 1.0568 | -0.1579 ± 0.0704 | 0.0793 ± 0.0733 |
| TPC reference | 17.4192 ± 1.0527 | -0.1599 ± 0.0701 | 0.0716 ± 0.0665 |
| TPC kernel 2 | 17.8029 ± 1.3662 | -0.1855 ± 0.0910 | 0.0310 ± 0.0537 |
| TPC gradual dilation | 18.0991 ± 0.8564 | -0.2052 ± 0.0570 | 0.0292 ± 0.0506 |
| TPC no dilation | 18.5964 ± 0.0057 | -0.2383 ± 0.0004 | 0.0000 ± 0.0000 |

MAD is mean absolute error. MSLE uses `log`, not `log1p`. MAPE divides by at
least 4/24 days. Kappa uses linear weights and bin edges
`[1,2,3,4,5,6,7,8,14]` days; a value on an edge goes in the higher bin.

## Notes

LSTM had the lowest mean test MSLE and MAD. Kernel 3 had the lowest TPC MSLE;
reference TPC had slightly lower MAD. This demo study does not show that TPC
is better than LSTM. The data and training choices differ from the paper.

Nine of 15 TPC runs gave constant test predictions: reference seed 19;
kernel 2 seeds 9 and 19; kernel 3 seed 9; all no-dilation seeds; and gradual
dilation seeds 9 and 19. No LSTM run was constant. These outputs were away from
the prediction limits. The cause was not proved. The no-dilation results mix
the effect of dilation with this training problem. Their very small SD comes
from similar constant predictions, so it does not show a reliable model.

Every setting had negative mean R², though some individual runs did not.
Hours from the same stay are related, and longer stays contribute more hours.
Three seeds on one small split are not enough to prove a significant difference
or usefulness in clinical care. All runs were kept, and settings were not
changed after looking at the test results.

## Environment

Training used Python 3.13.16, PyTorch 2.14.1+cu130, PyHealth 2.1.0, NumPy 2.4.6,
pandas 2.2.3, and scikit-learn 1.6.1 on a Tesla T4. Using other package versions
may change results even with the same seeds. The later model checks have
separate environment logs. The small synthetic example does not need this
exact training setup.
