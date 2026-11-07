# Neural Fine-Gray on PBC

This example predicts death and liver transplantation as competing first events
using [Neural Fine-Gray (Jeanselme et al., CHIL 2023)](https://proceedings.mlr.press/v209/jeanselme23a.html).
It connects `PBCDataset`, `CompetingRisksPBC`, train-only preprocessing, and
`NeuralFineGray` through PyHealth's `Trainer`.

## Setup

From the PyHealth checkout:

```sh
pip install -e .
pip install -r examples/pbc_neural_fine_gray/requirements-neural-fine-gray.txt
```

The optional requirements provide lifelines comparators and scikit-survival
historical metrics. Core dataset, task, model, and corrected metrics use
PyHealth's existing dependencies.

Supply a local `pbc2.csv` from the authors'
[DeepSurvivalMachines fork](https://github.com/Jeanselme/DeepSurvivalMachines/tree/e4b07b3f497f2266eaa71d0e182195e95663d367).
`--data-root` names the directory containing this file. Review its data-use
terms before running. The example neither downloads nor bundles records.

This source contains 1,945 visits for 312 participants. The task selects one
earliest whole baseline row per patient, with 25 encoded clinical columns.
Event labels are 0=censored, 1=death, and 2=transplantation. R's 418-patient
`pbc` object has different labels and is not interchangeable with this source.

## Run

Use an output directory outside the checkout. A synthetic smoke run exercises
the full integration without medical records:

```sh
python examples/pbc_neural_fine_gray/pbc_neural_fine_gray_replication.py \
  --synthetic --device cpu --folds 2 --trials 1 --epochs 2 --patience 1 \
  --output-dir /local/runs/pbc-synthetic
```

Run the five-fold study and all supplementary analyses:

```sh
python examples/pbc_neural_fine_gray/pbc_neural_fine_gray_replication.py \
  --data-root /local/datasets --device cuda:0 --historical \
  --folds 5 --trials 100 --epochs 1000 --patience 10 \
  --sensitivity-seeds 5 --paper-time --architecture-ablation \
  --output-dir /local/runs/pbc-full
```

Use `--device cpu` without CUDA. `--audit-only` checks aggregate cohort
information without fitting models. Supplementary options are disabled by
default. `--cox-penalties 0.1` restricts Cox selection to the original fixed
penalty; its default grid is 0.01, 0.1, 1, 10, and 100.

Repeating an identical command verifies and restores completed checkpoints,
then recomputes scores. Changes to data, source, dependencies, or protocol
require a new output directory. Interrupted fits without validated metadata
are retrained.

## Study design

Five event-stratified patient folds define the test sets. A 10% stratified
inner holdout selects models. Imputation, standardization, time scaling,
censoring weights, and comparator fits use inner-training patients only.
Numeric values are mean-imputed before the imputed matrix is standardized.
Native missingness is retained in observation indicators.

Each fold searches the original arm over encoder/head widths 25/50, depths
1-4, dropout 0/0.25/0.5/0.75, learning rates 0.001/0.0001, and batch sizes
100/250. Minimum validation NLL selects the best checkpoint. Early stopping
uses the requested patience and epoch ceiling. The grid contains 1,024
configurations; oversized trial requests exhaust this grid.

The other arms reuse the selected configuration:

| Arm | Input | Training intervention |
|---|---|---|
| Original | Clinical values | None |
| Constant | Clinical values and zero masks | Capacity control |
| Observed | Clinical values and observation masks | Indicators |
| Augmented | Clinical values and observation masks | 10% biomarker MCAR |

The three mask-input arms have equal parameter counts. Added missingness
affects bilirubin, albumin, and cholesterol only. All arms and comparators see
identical nested test masks at 0%, 10%, and 30%; existing missing values remain
missing.

Aalen-Johansen is input-independent. Cause-specific Cox models share an L2
penalty selected by summed validation partial log-likelihood, with no
validation refit. Their hazard increments use a probability-preserving
interval update to produce joint CIFs.

Corrected IPCW Brier and concordance score both causes at 3.19, 4.95, 5, and
7.45 unshifted years. Earlier competing events are known negatives. Exhausted
censoring support and missing comparable pairs are reported explicitly,
independently for each metric.
Primary, per-seed, and aggregate sensitivity reports retain their score column
and CSV headers even when every score is unsupported.

The primary H2 comparison requires lower mean five-year death Brier for
Augmented versus Constant at 10% and 30% added missingness, and no mean increase
versus Original at 0%. Paired-fold differences are descriptive comparisons.

## Supplementary analyses

- `--sensitivity-seeds N` refits all four arms with `N` fresh matched training
  seeds, keeping selected configurations, splits, preprocessing, and test
  masks fixed. Per-seed H2 decisions are reported separately from primary H2.
- `--paper-time` additionally scores 2.19, 3.95, and 6.45 unshifted years,
  accounting for the reference loader's one-year time shift. It uses the
  corrected metrics and does not replace the primary horizons.
- `--architecture-ablation` uses the Original arm and varies joint encoder/head
  width (25/50) or encoder dropout (0/0.25/0.5/0.75), keeping depths, learning rate, and batch
  size fixed. Held-out scores are reported for every case; they do not select
  a new configuration. Cases identical to the selected original model reuse
  its verified checkpoint.
- `--historical` applies the reference binary-cause scikit-survival convention
  to clean original predictions at 3.19, 4.95, and 7.45 years. Its estimand
  differs from corrected competing-risk scoring.

## Outputs

`manifest.json` records the protocol and source/data/dependency fingerprints.
`cohort_audit.json` contains aggregate cohort information. `metrics.csv`
retains every score or unsupported reason; `summary.csv`, `paired_effects.csv`,
and `h2.json` report the primary comparison. `completion.json` records fit and
metric counts.

Supplementary aggregate results are under `paper_time/`, `seed_sensitivity/`,
and `architecture/`. Fold directories contain preprocessing, Cox selection,
fit metadata, checkpoints, and curve checks.

Patient records, IDs, split definitions, predictions, caches, and checkpoints
must remain in controlled local storage. Only aggregate findings should be
shared.

## Results

The five-fold study with seed 2026 completed 638 neural fits: 515 primary,
100 matched-seed, and 23 distinct architecture fits. It also fitted 50 Cox
models and retained 9,060 metric rows, including unsupported scores.

Five-year death Brier scores are mean (sample SD) over five supported folds.
Lower is better; rates refer to added biomarker missingness.

| Arm | 0% | 10% | 30% |
|---|---|---|---|
| Original | 0.1190 (0.0318) | 0.1246 (0.0326) | 0.1289 (0.0290) |
| Constant | 0.1181 (0.0332) | 0.1237 (0.0315) | 0.1301 (0.0277) |
| Observed | 0.1215 (0.0308) | 0.1260 (0.0296) | 0.1331 (0.0266) |
| Augmented | 0.1333 (0.0292) | 0.1357 (0.0273) | 0.1391 (0.0247) |
| Aalen-Johansen | 0.2037 (0.0261) | 0.2037 (0.0261) | 0.2037 (0.0261) |
| Tuned Cox | 0.1159 (0.0254) | 0.1184 (0.0261) | 0.1222 (0.0235) |

H2 was not supported. Primary paired mean Brier differences were +0.014333
against Original at 0%, and +0.012063/+0.008970 against Constant at 10%/30%.
Across five fresh seeds (2027-2031), corresponding mean differences were
+0.007061, -0.001619, and -0.002203. Masked performance improved on average,
but clean performance worsened; none of the five seeds met the complete H2
criterion.

All Original-arm architecture cases are shown below at 0% added missingness.
Other selected settings remain fixed within each fold.

| Case | Mean five-year death Brier |
|---|---:|
| Selected configuration | 0.1190 |
| Joint width 25 | 0.1215 |
| Joint width 50 | 0.1289 |
| Dropout 0 | 0.1096 |
| Dropout 0.25 | 0.1371 |
| Dropout 0.5 | 0.1432 |
| Dropout 0.75 | 0.1656 |

These exploratory test comparisons do not replace the primary selected
configuration. Original-arm corrected death Brier at the supplementary
2.19/3.95/6.45-year horizons was 0.0913/0.1129/0.1371.

## Interpretation

- The example covers baseline PBC competing first events; transplantation
  ends this follow-up and does not imply that subsequent death is impossible.
- Baseline-only age, whole-row selection, unshifted durations, preprocessing,
  and metric/comparator conventions differ from the paper.
- The five-fold, 100-trial, 1,000-epoch budget follows Section 4.3. These shared
  settings do not make scores directly comparable with published values.
- Seed and architecture diagnostics are conditional on the original selected
  configurations; they do not repeat model selection for every seed.
