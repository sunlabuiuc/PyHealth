# CDC COVID-19 mortality temporal ablation

This example is a bounded partial reproduction inspired by Zhou, Chen, and
Lipton, [*Evaluating Model Performance in Medical Datasets Over
Time*](https://proceedings.mlr.press/v209/zhou23a.html) (CHIL 2023). It tests
strictly future-month performance with PyHealth's existing logistic-regression
model. It is not an exact reproduction of the paper's CDC experiment.

## Reproduce the bounded sample

The public CDC source contains about 100 million deidentified case rows. On a
16 GiB Intel Mac without a supported current PyTorch GPU stack, downloading and
processing the full file was not practical. The course instructor explicitly
permits a compute-bounded subset when the limitation is disclosed.

The following command downloads 4,800 labeled rows: 24 months from 2020-07
through 2022-06, with 100 `death_yn=Yes` and 100 `death_yn=No` cases per month.
Each month/label stratum is divided into five segments in Socrata's opaque row
ID order, and a deterministic 20-row block is taken from each segment.

```bash
python examples/cdc_covid_download_bounded_sample.py \
  /data/cdc/n8mc-b4w4.csv \
  --start 2020-07 --months 24 --per-class 100 --chunks 5
```

The analyzed file had SHA-256
`074f36c065cbef60a988436d59527157f7d9ffe214136b18223fcc349f79c5ac`.
Every month had exactly 200 cases and both outcome classes. Of the 4,800 rows,
4,252 had a blank `underlying_conditions_yn` value.

This is a deterministic, systematically dispersed convenience sample, not a
uniform probability sample. Outcome balancing deliberately changes prevalence;
the results must not be used for mortality-rate or calibration claims. Socrata
row order can also change, so the hash records the exact analyzed artifact.

## Run the ablation

```bash
python examples/cdc_covid_mortality_logistic_regression.py \
  --data-root /data/cdc \
  --cache-dir /data/cache/cdc-covid \
  --epochs 3 --seed 7 --results /data/cdc-results.csv
```

The example compares two feature sets and two temporal training regimes. The
test cohort is identical across regimes for each month, and every training
month strictly precedes its test month.

- `demographics`: age group, sex, race, and ethnicity.
- `preoutcome`: demographics plus exposure, case status, symptom status,
  underlying-condition status, and two case-timing intervals.
- `expanding`: all earlier months.
- `sliding_6m`: only the six most recent earlier months.

Hospitalization, ICU admission, and the mortality label are never predictors.

## Measured results

The command above ran on 2026-11-22 in 121.22 seconds on CPU (Intel Core
i7-9750H, 16 GiB RAM, macOS 26.6.2, Python 3.12.15, PyTorch 2.2.2). The local
PyTorch version is an Intel-macOS compatibility constraint; upstream PyHealth
CI should use its declared supported dependency versions.

| Feature set | Training regime | 2022-03 | 2022-04 | 2022-05 | 2022-06 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| demographics | expanding | 0.9979 | 0.9010 | 0.9601 | 0.8888 | 0.9370 |
| demographics | sliding 6m | 0.9979 | 0.9010 | 0.9472 | 0.9380 | 0.9460 |
| preoutcome | expanding | 0.9910 | 0.9080 | 0.9679 | 0.9000 | 0.9417 |
| preoutcome | sliding 6m | 0.9952 | 0.9092 | 0.9455 | 0.8716 | 0.9304 |

Each test month contains 200 cases and 100 recorded deaths. The results show
month-to-month variation and no regime that wins for every feature set and
month. With only four small, balanced test cohorts, differences are descriptive
and have substantial sampling uncertainty; no confidence interval or
significance claim is made.

## Reproduction limits

The public geography file has 19 columns and no person identifier, whereas the
paper used a different 33-column restricted CDC extract. This implementation
therefore models independent deidentified cases, not longitudinal people. The
public fields do not establish a uniform prospective prediction time, death
reporting may be delayed, missingness and surveillance practice change over
time, and the bounded sampling design limits generalizability. These results
demonstrate a runnable PyHealth temporal evaluation and an additional feature
ablation; they do not reproduce the paper's numeric results.
