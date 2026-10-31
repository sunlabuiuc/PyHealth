# Survival MDN, Cox and DeepSurv benchmarks

Authors: Aritra Saha and Ugochukwu Njoku.
Paper: [Survival Mixture Density Networks](https://proceedings.mlr.press/v182/han22a.html),
Han, Goldstein and Ranganath, MLHC 2022.

These examples use native PyHealth survival datasets/tasks and a shared evaluator.
Run commands from a PyHealth source checkout. All executable example code is
included here. Generated cohorts, checkpoints and results are user outputs and
are not distributed with the example.

The tasks emit separate `duration` and `event` tensors of shape `[batch, 1]`,
matching PyHealth's SurvivalMDN and standard Trainer. The classical adapters
combine these into NumPy outcome arrays internally; the task schema stays the
same for all three models.

## Installation and a download-free run

Install PyHealth normally. MDN and calibration use its normal dependencies.
Cox/DeepSurv additionally need the optional example requirements:

```bash
pip install -e .
pip install -r examples/survival_benchmark/requirements.txt
python -m examples.survival_benchmark.run_survival_mdn \
  --dataset synthetic --splits 1 --epochs 3 --device cpu --out /tmp/mdn-demo
python -m examples.survival_benchmark.run_baselines \
  --dataset synthetic --splits 1 --epochs 3 --out /tmp/baseline-demo
python -m examples.survival_benchmark.calibrate_saved \
  /tmp/baseline-demo/cox/split_1/seed_1/run_*/predictions.npz \
  --out /tmp/cox-calibration.json
```

The synthetic cohort has six artificial patients and deterministic, disjoint
2/2/2 training/validation/test groups. It tests execution and interfaces;
its scores are not paper reproduction results. One censored patient is kept
in training; the tiny validation/test pairs have observed outcomes so their
curve metrics are defined. Only explicitly requested models
are fitted. Run `--help` on any module for its CLI arguments.

## GBSG and SUPPORT

Prepare public CSV inputs with the native dataset loaders, and obtain the
paper authors' released `data/gbsg` and `data/support` NPZ files separately from
[their reference repository](https://github.com/XintianHan/Survival-MDN).
This example reads those files and does not vendor them or the reference code.
See the native GBSG/SUPPORT API docs for source preparation and feature order.
The SUPPORT CSV's patient-number column must be named `sno`.

```bash
python -m examples.survival_benchmark.run_survival_mdn \
  --dataset gbsg --root /data/gbsg --released-dir /data/Survival-MDN/data/gbsg \
  --splits 1 --device cpu --out /runs/gbsg-mdn
python -m examples.survival_benchmark.run_baselines \
  --dataset gbsg --root /data/gbsg --released-dir /data/Survival-MDN/data/gbsg \
  --splits 1 --out /runs/gbsg-baselines
python -m examples.survival_benchmark.run_baselines \
  --dataset support2 --root /data/support2 \
  --released-dir /data/Survival-MDN/data/support --splits 1 \
  --out /runs/support-baselines
```

For MDN on SUPPORT use the same SUPPORT paths and `--dataset support2` with
`run_survival_mdn`. Select released partitions using `--splits 1 2 ... 10`.
Use a separate output root for every dataset and runner. Split membership and
input hashes are recorded. Model initialization uses `--seed` for baselines and
`--seeds` for MDN. Split IDs are not initialization seeds.

## Preprocessing, training and selection

- GBSG MDN uses raw released covariates and the paper's cohort-normalized time,
  `months / cohort_max + 0.002`. Each GBSG baseline fits feature standardization
  on training patients only and saves its means/scales. This is an explicit
  model-specific preprocessing choice, not identical arithmetic for every model.
  For a paired MDN preprocessing ablation, add `--standardize-gbsg`. This fits
  all seven feature means and population standard deviations on the selected
  split's training patients, applies them unchanged to validation/test, and
  saves them in the record. Constant features use a scale of one. Raw features
  remain the default; the option is rejected for other datasets.
- SUPPORT uses all 27 task features, standardized on the selected split's
  training patients, and `days / 365.25 + 0.001`. Baselines disable their extra
  feature transform because the task already standardizes the inputs.
- Cox wraps scikit-survival CoxPH with Breslow ties and an L2 penalty. DeepSurv
  wraps pycox CoxPH with full-risk-set batches, separate full validation batches,
  Adam, best-validation weights and training-only baseline hazards. Pycox's
  sorted cumulative-risk loss approximates tied event times. This is not exact
  parity with the original Theano DeepSurv implementation.
- Baseline curves use right-continuous training-hazard steps, equal one before
  the first knot and remain constant beyond the final knot. Records count test
  observations beyond the training time range. Predictions preserve row order.
- MDN uses the standard Trainer, full validation batches and singleton-safe
  training batches for BatchNorm. It restores the best validation checkpoint.
  Its default optimizer, epoch cap and patience follow the authors' defaults.
- Select baseline configurations by validation IBS and MDN configurations by
  validation negative log-likelihood. Neural checkpoint selection uses each
  model's validation loss. Test metrics are for reporting, never selection.

Baseline `--config` accepts JSON such as `{"cox": {"alpha": 0.1}, "deepsurv":
{"hidden": [32, 32], "lr": 0.001}}`. Epochs, patience, seed and device are CLI
options; `--device` applies to DeepSurv, while classical Cox runs on CPU.
MDN `--config` accepts `model` and `train` objects. For small ablations use
`--num-components 1 5 10 --transform softplus exp`; every requested combination
is fitted. Each output folder is named by configuration, split and seed.
Model options use the public constructor names, including `time_transform`,
`num_hidden_layers`, `hidden_dim` and `num_components`. For example:
`{"model": {"time_transform": "exp", "num_hidden_layers": 3}}`.

For raw-versus-standardized GBSG MDN, keep model settings, initialization seed,
split membership, time conversion and training limits fixed. Use separate
output roots for the two conditions, and add only `--standardize-gbsg` for the
second. For example, run this twice with the indicated differences:

```bash
python -m examples.survival_benchmark.run_survival_mdn \
  --dataset gbsg --root /data/gbsg --released-dir /data/Survival-MDN/data/gbsg \
  --splits 1 2 3 4 5 6 7 8 9 10 --seeds 1 --device cpu --out /runs/gbsg-mdn-raw
# Second condition: add --standardize-gbsg and use --out /runs/gbsg-mdn-standardized.
```

These are twenty new fits with the current upstream model. Keep them separate
from results produced with another model implementation. Compare all matched
splits and report the preprocessing effect without choosing conditions by test
performance. Early stopping still selects weights using validation loss only.

## Metrics and saved results

All models use public `survival_curve_metrics_fn` for aligned C^td, IBS and IBLL
at censoring-survival thresholds 1e-8, 0.2 and 0.4. A separately named
`test_metrics_reference_compatible` report deliberately follows the reference
evaluator's widest-horizon grid mismatch when the last observation is censored.
Use aligned metrics for consistent scoring; distinguish the compatibility report
when comparing published numbers. It does not change the public metric functions.

D-calibration uses exact `S_i(t_i)` values at each patient's observed event or
censoring time. It follows [Haider et al.](https://jmlr.org/papers/v21/18-772.html),
Appendix B.5 Algorithm 2, upper-inclusive probability bins and floor 1e-5.
Censored contributions total one per patient. The p-value is a distributional
diagnostic and is not proof of individual calibration. Tiny smoke cohorts have
limited calibration-test power.

Each fit writes `record.json`, `predictions.npz` and a checkpoint. Baselines also
write `split_ids.json`. Predictions contain patient IDs, observed times, events,
a plotting grid, curves and exact `survival_at_observed_time`. Records retain
settings, preprocessing, code/input fingerprints, validation/test scores, timings
and output fingerprints.

The runner automatically creates a directory such as
`cox/split_1/seed_1/run_0123456789abcdef/`. The final component is a 16-character
fingerprint of the fit settings, data hashes, relevant executable source bytes
and runtime identity, including the device. Reordering JSON keys, moving input
files without changing their contents, or editing this README does not change
the identity. Editing fingerprinted executable source does, even if the edit
does not change numerical behavior. Commit IDs and timestamps are provenance
only, so committing identical source does not invalidate a completed fit.

- An identical completed run is reused only after its record and saved files
  are verified. No fit files are rewritten.
- Changed settings, data, source or runtime automatically select a new
  directory under the same output root. Earlier runs are preserved.
- A directory without a completed record, or with missing, changed or unreadable
  outputs, raises an error naming the problem. Another process may still own an
  incomplete run. Inspect it before restarting; no automatic repair or overwrite
  occurs. For an explicit restart, first move that run directory outside the
  output root to a separate archive, then rerun the same command.
- Legacy files stored directly under `seed_1/`, without the fingerprint directory,
  remain untouched and are not reused or included in new summaries. The runner
  creates a new verified run automatically.

`summary.json`/`summary.csv` for MDN and `pilot_summary.json` for baselines describe
only the fits requested by the latest successful command. They are refreshed
views; the versioned fit records remain the source of historical results. Thus
older learning rates, runtimes or code versions cannot enter a new comparison
merely because they share the same output root. `record.json` includes the
relative `run_directory` for locating each selected baseline run. The CLI also
prints its directory when it reuses a run.

Cox restoration uses `CoxAdapter.load(checkpoint_dir)` and trusted joblib data.
DeepSurv uses `DeepSurvAdapter.load(checkpoint_dir, device="cpu")`, tensor state
weights and NPZ baseline hazards. Only load checkpoints from a trusted source.
MDN restores the recorded model configuration and `trainer/best.ckpt`; its
time transform and other constructor settings are saved in the record.
Saved exact probabilities can be scored without loading a model.
Use the printed run directory when selecting a specific
version; the shell wildcard in the quick-start command scores all matching
versions. Otherwise, regenerate them from a checkpoint and the
same patient features/preprocessing without retraining. Coarse plotting grids
are not silently substituted for exact calibration predictions.

## Tests

```bash
python -m unittest tests.core.test_survival_benchmark -v
```

The tests use small synthetic cohorts, exercise the real adapters and CLIs, and
require no downloads. Optional-backend cases skip when their packages are absent;
MDN, calibration and runner-integrity checks still run with normal PyHealth deps.
