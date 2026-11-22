# SUPPORT2 five-fold study for Deep Cox Mixtures

Partial reproduction of Nagpal et al., *Deep Cox Mixtures for Survival
Regression* ([MLHC 2021](https://proceedings.mlr.press/v149/nagpal21a.html)) on
SUPPORT2, plus a missing-laboratory extension that is not in the paper.
Results: [RESULTS.md](RESULTS.md).

For the PyHealth pipeline (`Support2Dataset` -> `Support2SurvivalTask` ->
`DeepCoxMixtures`) and a small ablation that runs on generated data, use
[`examples/support2_survival_dcm.py`](../support2_survival_dcm.py) instead.

## Setup

Python 3.12 or 3.13, from the repository root:

```sh
pip install -e '.[dcm]' pytest
# The authors' package, at the tested commit. --no-deps because its metadata
# pins torch<2 and numpy<2, which this model does not need.
pip install --no-deps \
  'auton-survival @ git+https://github.com/autonlab/auton-survival@5dde465f7223601717abddc1d075e837707c403b'
```

Download the public [SUPPORT2 archive](https://hbiostat.org/data/repo/support2csv.zip)
(SHA-256 `8ed43980742a18e1847a8dfc5530bc4b30564ad9e4ad1b1b50bbc5d29d8c86fe`)
and extract `support2.csv`.

## What the study driver does

The driver in `experiment/` reads `support2.csv` with pandas and works on
NumPy arrays. **It does not build a PyHealth dataset and does not call
`Support2SurvivalTask`.** It imports the covariate list from that task, so
both use the same 24 covariates, and it trains DCM through
`pyhealth.models._dcm_reference.DCMReference`, the EM engine that
`DeepCoxMixtures.fit_survival` also calls, with the same fixed settings.

- All 9,105 rows; five event-stratified outer folds (seed 20261102).
- Imputation, scaling and one-hot encoding are fit on training rows only.
- Inside each outer fold, a 20% validation split selects one configuration
  per family: DCM K in {3, 4, 6} x width in {50, 100}; Cox PH with six
  penalties; random survival forest with six settings. The selected DCM is
  refit for its early-stopped number of epochs.
- Metrics: mean IPCW Brier score and IPCW concordance at the three quartiles
  of training event times, with censoring weights from training data only.
- Extension (not in the paper): the selected DCM is also refit with eight
  laboratory missing-value indicators, and every model is scored after masking
  10% or 30% of observed test-fold lab values (seeds 71/172/273), without
  retraining.

```sh
cd examples/support2_dcm
python -m experiment.run --data /path/to/support2.csv --output /path/to/new-dir
python -m experiment.summarize --results /path/to/new-dir
python -m pytest test_protocol.py   # preprocessing, masking and metric checks
```

The full run is 110 fits, roughly 30-60 CPU-minutes; folds can run in
parallel with `--folds 0`, `--folds 1`, ... into the same output directory.
`--smoke` runs one configuration per family on 600 rows. Each finished fold is
checkpointed; use a new output directory after changing any setting.

## Limits

Only SUPPORT2 is reproduced, with different preprocessing, splits and software
versions from the paper; FLCHAIN, SEER and the fairness analyses are not.
Fold SDs are descriptive, not confidence intervals. Most SUPPORT2 measurements
are from study day 3, so this is not an admission-time model. The authors'
posterior-floor and spline numerics are kept as they are; the model rejects
nonfinite or materially invalid curves and projects tiny overshoots (none were
needed on the held-out folds).

The network and curve code in `pyhealth/models/_dcm_reference.py` follow the
authors' MIT-licensed auton-survival implementation (Copyright (c) 2020
Carnegie Mellon University Auton Lab); training calls their `train_dcm`
unchanged.
