# Neural Fine-Gray (NeuralFG): Competing-Risk Survival Analysis in PyHealth

This directory contains the training, evaluation, and reproducibility utilities for a **PyHealth implementation of Neural Fine-Gray (NeuralFG)**, a neural competing-risk survival model inspired by *Neural Fine-Gray: Monotonic neural networks for competing risks* (Jeanselme et al., CHIL 2023).

Developed for the Georgia Tech **CSE 6250: Big Data for Health Informatics (Fall 2026)** team project. The experiments below are a **paper-aligned PyHealth reproduction**, **not** a bit-for-bit reproduction of the authors' original implementation.

## 1. Objective and model

For a subject with features $x$, NeuralFG estimates each event-specific cumulative incidence function (CIF):

$$
F_k(t \mid x) = P(T \leq t, J = k \mid x), \qquad k = 1, \ldots, K.
$$

The PyHealth implementation provides a shared feature embedding and event-specific monotonic networks, allowing prediction of nondecreasing cause-specific CIFs that remain nonnegative and whose sum is at most one. It supports censored observations and competing event types.

- Implementation: [`pyhealth/models/neural_fg.py`](../../pyhealth/models/neural_fg.py)
- PyHealth API documentation: [`docs/api/models/pyhealth.models.neural_fg.rst`](../../docs/api/models/pyhealth.models.neural_fg.rst)
- Main model tests: [`tests/test_neural_fg.py`](../../tests/test_neural_fg.py)

## 2. Dataset

Experiments use the **DeepHit synthetic competing-risks dataset** with **30,000 subjects**, **12 numeric features**, and **two competing event types**.

| Field | Meaning |
| --- | --- |
| `feature1`–`feature12` | Numeric predictors |
| `time` | Observed event or censoring time |
| `label = 0` | Right-censored observation |
| `label = 1` or `2` | Observed event type |
| `true_time`, `true_label` | Latent event outcomes used **only** for the controlled censoring simulations and their held-out evaluation |

The dataset is loaded by `load_data()` in [`data.py`](data.py). Check that module's source/path requirements if you run these scripts in a new checkout. Latent event columns must **never** be used as training features or for selecting models.

## 3. Setup

Run commands **from the PyHealth repository root** (the directory containing `pyhealth/`, `examples/`, and `tests/`):

```bash
conda activate neuralfg
python -m pip install -e .
```

The experiment scripts also use PyTorch, NumPy, pandas, SciPy, scikit-learn, and (for plots) Matplotlib. Ensure those dependencies are available in your environment. The completed experiments were run with the local macOS/Python 3.12 environment; hardware and library differences can change training trajectories.

A quick import check:

```bash
python -c "from pyhealth.models import NeuralFG; print('NeuralFG import: PASS')"
```

## 4. Quick start and tests

For a short **paper-runner integration smoke test** (600 rows, 2 folds, 2 candidate configurations, 2 epochs), run:

```bash
python examples/deephit_paper_reproduction.py --pilot
```

A separate, **non-paper-aligned** paired cross-validation smoke test is available:

```bash
python examples/deephit_competing_risks_cv.py --smoke
```

The original standalone NeuralFG demonstration is:

```bash
python examples/deephit_competing_risks_neuralfg.py
```

**Note:** This last script runs training with its in-file configuration (100 epochs by default); it is **not** the small/fast pilot and should not be used to reproduce the reported 100-trial results.

Relevant automated tests:

```bash
python -m pytest \
  tests/test_neural_fg.py \
  tests/test_neuralfg_submission_smoke.py \
  tests/test_neuralfg_metrics.py \
  tests/test_neuralfg_metrics_reference.py \
  tests/test_neuralfg_cross_validation.py \
  tests/test_finegray_baseline.py \
  tests/test_paper_reproduction.py -q
```

At the final local preflight, **36 selected tests plus 5 cross-validation tests passed (41 total)**. This is a report of the contributor's local checks, not a guarantee of passing a fresh upstream CI environment.

### Fresh-training reproducibility check

A separate pilot was run with a **new output directory** to avoid using cached evaluations:

```bash
python examples/deephit_paper_reproduction.py \
  --pilot \
  --output-root examples/neuralfg/results/pr_fresh_pilot_20261114
```

For subsequent fresh runs, choose a **different, unused `--output-root`** path; the runner can resume/reuse results from an existing output directory. The completed fresh pilot used **600 sampled subjects, 2 folds, 2 candidate configurations per fold, and 2 epochs per fit** (4 NeuralFG fits in total). Both folds performed actual training, followed by candidate selection, Fine–Gray evaluation, and saving of results. No `loading existing evaluation` messages appeared in this fresh run.

| Check | Previously cached pilot | Freshly trained pilot |
| --- | ---: | ---: |
| Fold 1 best tuning NLL | 0.59466 | 0.59466 |
| Fold 2 best tuning NLL | 0.59082 | 0.59082 |
| NeuralFG Risk 1, q50 C-index | 0.558888 | 0.558888 |
| NeuralFG Risk 2, q50 C-index | 0.580982 | 0.580982 |

These displayed results agreed under the **same local macOS/Python 3.12 environment and seed**. The pilot verifies the short end-to-end training/evaluation workflow and repeatability at the displayed precision; it is **not** a performance benchmark or evidence of bitwise reproducibility on other hardware. The main results below come from the full 30,000-subject, five-fold, 100-configuration experiment.

## 5. Main paper-aligned reproduction

The primary experiment uses:

- **Five outer stratified folds**: 24,000 development and 6,000 held-out test subjects per fold.
- Within each development partition: **19,200 training**, **2,400 hyperparameter-tuning**, and **2,400 early-stopping** subjects.
- **100 sampled hyperparameter configurations per outer fold**, selected by the lowest **tuning-set negative log likelihood (NLL)**; the distinct early-stopping set controls checkpoint selection.
- Up to **1,000 epochs**, early-stopping patience **3**, seed **0**; train-only feature standardization and time normalization.
- Independent **Fine–Gray** baseline fitted to the training observations for each fold.
- Evaluation at the observed training event-time quantiles **q25, q50, and q75**, reporting cause-specific time-dependent concordance and inverse-probability-of-censoring-weighted (IPCW) Brier scores using the supplied **reference-style** metric implementations.

### Reproduction command

```bash
python examples/deephit_paper_reproduction.py \
  --mode search \
  --trials 100 \
  --folds 5 \
  --epochs 1000 \
  --patience 3 \
  --seed 0 \
  --output-root examples/neuralfg/results/paper_reproduction_search_full
```

**Computational warning:** This entails **500 NeuralFG fits** (100 candidates × 5 folds) and can take considerable time. Use `--pilot` before launching a full run. To test a single architecture instead, use `--mode fixed`.

### Main results

Five-fold **mean ± sample SD** on held-out outer test folds. **Higher C-index is better; lower Brier score is better.** These values correspond to the completed 100-trial run, not to the simpler demonstration script.

| Risk | Horizon | NeuralFG C-index | Fine–Gray C-index | NeuralFG Brier | Fine–Gray Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | q25 | 0.7866 ± 0.0148 | 0.6109 ± 0.0175 | 0.0659 ± 0.0026 | 0.0739 ± 0.0024 |
| 1 | q50 | 0.7417 ± 0.0124 | 0.5963 ± 0.0090 | 0.1208 ± 0.0035 | 0.1436 ± 0.0024 |
| 1 | q75 | 0.6872 ± 0.0114 | 0.5849 ± 0.0112 | 0.1804 ± 0.0031 | 0.2062 ± 0.0015 |
| 2 | q25 | 0.7922 ± 0.0185 | 0.6305 ± 0.0138 | 0.0625 ± 0.0025 | 0.0701 ± 0.0027 |
| 2 | q50 | 0.7396 ± 0.0151 | 0.6007 ± 0.0166 | 0.1206 ± 0.0020 | 0.1426 ± 0.0025 |
| 2 | q75 | 0.6872 ± 0.0140 | 0.5887 ± 0.0168 | 0.1783 ± 0.0036 | 0.2029 ± 0.0020 |

NeuralFG had better mean concordance and Brier scores than the custom Fine–Gray baseline for both competing events at each reported horizon. This does **not** establish identical reproduction of the original authors' implementation or external validation of the Fine–Gray solver.

### Comparison with the original paper

The original publication, *Neural Fine-Gray: Monotonic neural networks for competing risks* (Jeanselme et al., CHIL 2023), reports the following **Synthetic, primary event (Risk 1)** results in **Table 2**. The comparison below uses its published five-fold **means**, alongside the corresponding means from our paper-aligned PyHealth implementation. The full table above reports our fold-to-fold standard deviations.

| Metric | Horizon | Published NeuralFG | Reproduced NeuralFG | Published Fine–Gray | Our Fine–Gray |
| --- | --- | ---: | ---: | ---: | ---: |
| C-index | q25 | 0.791 | 0.7866 | 0.611 | 0.6109 |
| C-index | q50 | 0.754 | 0.7417 | 0.587 | 0.5963 |
| C-index | q75 | 0.715 | 0.6872 | 0.568 | 0.5849 |
| Brier | q25 | 0.068 | 0.0659 | 0.078 | 0.0739 |
| Brier | q50 | 0.125 | 0.1208 | 0.159 | 0.1436 |
| Brier | q75 | 0.192 | 0.1804 | 0.241 | 0.2062 |

Our NeuralFG reproduced the broad published performance pattern, including higher concordance than Fine–Gray on this nonlinear synthetic benchmark. The largest NeuralFG C-index difference was at q75 (**0.6872 vs. 0.715**, or **−0.0278**). Brier scores were numerically lower in this implementation, but **different preprocessing, model selection, metric conventions, and baseline implementations prevent claims of direct performance superiority**. In particular, the custom Fine–Gray implementation has not been independently validated against the R `cmprsk::crr()` reference implementation.

Original authors' released code: [Jeanselme/NeuralFineGray](https://github.com/Jeanselme/NeuralFineGray).

Saved output files include `config.json`, `candidates.json`, per-fold `split_indices.csv`, `hyperparameter_search.csv`, `best_hyperparameters.json`, checkpoints/training histories, `fold_metrics_reference.csv`, and `cv_summary_reference.csv`.

## 6. Controlled experiments

The following extensions use the **same five outer split assignments** from the main reproduction and a **fixed architecture** (embedding width 50 × 3 layers; monotonic width 25 × 2 layers), Adam learning rate `0.001`, batch size `100`, maximum `500` epochs, and patience `3`. They do **not** repeat the 100-candidate search for each condition.

First, identify the full-search results directory (which contains `fold_1/`, `fold_2/`, etc.):

```bash
MAIN="examples/neuralfg/results/paper_reproduction_search_full/search_nfull_k5_trials100_ep1000_pat3_seed0_cfgaf6f4e53"
```

If your output folder has a different configuration hash, replace the path above with the actual folder name.

### 6.1 Independent right-censoring sensitivity

The experiment creates **25%, 50%, and 75%** independently censored training scenarios while keeping the same subject features, latent event times/types, and split assignments. **Observed times and labels change under each censoring regime, but the latent event outcomes and held-out subject membership stay fixed.**

```bash
python examples/deephit_extension_experiments.py \
  --experiment censoring --split-dir "$MAIN" \
  --folds 5 --epochs 500 --patience 3 \
  --seed 0 --censor-seed 2026
```

Observed-data IPCW evaluation becomes unreliable in the most severely censored late-horizon setting: estimated censoring survival at the late horizon was approximately **G(30) = 0.00089** for 75% censoring, producing extreme inverse-probability weights. For a clearer assessment of training effects, we also evaluated all saved models on the **same held-out uncensored latent outcomes**, without retraining NeuralFG:

```bash
CENSOR="examples/neuralfg/results/extension_experiments/censoring/emb50x3_mono25x2_lr0.001_bs100_ep500_pat3_seed0_cseed2026"
python examples/evaluate_neuralfg_latent_censoring.py \
  --results-dir "$CENSOR" --split-dir "$MAIN" --folds 5
```

This evaluation may refit the Fine–Gray baseline to reconstruct baseline subdistribution hazards for CIF prediction; it **does not** retrain NeuralFG.

**Common latent-outcome test results at q50 (five-fold means):**

| Training censoring | Risk 1 C-index | Risk 1 Brier | Risk 2 C-index | Risk 2 Brier |
| --- | ---: | ---: | ---: | ---: |
| 25% | 0.7307 | 0.1574 | 0.7274 | 0.1561 |
| 50% | 0.7019 | 0.1650 | 0.7059 | 0.1624 |
| 75% | 0.7182 | 0.1622 | 0.6315 | 0.1772 |

The effect was not uniformly monotone, but higher censoring particularly reduced performance for **Risk 2**. The latent test comparison separates degraded fitting from instability introduced by evaluating under severe censoring. These are results from one synthetic dataset and five folds.

#### Training convergence and checkpoint selection

To assess the proposal's planned **training-convergence sensitivity**, we also summarized checkpoint selection from the saved `training_summary.csv` and per-fold `training_history.csv` files for the full **500-epoch-cap** censoring experiment. Values below are **mean ± sample SD across the same five folds**.

| Training censoring | Best-checkpoint epoch | Epochs executed before stopping |
| --- | ---: | ---: |
| 25% | 89.6 ± 40.5 | 127.0 ± 80.9 |
| 50% | 178.8 ± 82.4 | 204.0 ± 85.1 |
| 75% | 130.6 ± 99.3 | 147.4 ± 104.7 |

No run reached the 500-epoch maximum. The 50% censoring condition required the most epochs on average, whereas the 75% condition showed substantially greater fold-to-fold variability in best-checkpoint timing. These observations indicate **nonmonotonic, fold-dependent training behavior** rather than a simple increase in training time with censoring. Best-checkpoint epochs and stopping times describe optimization behavior, but are **not mathematical proof of convergence**; likelihood values are also not directly comparable across changed censoring distributions.

### 6.2 Dropout ablation

We varied embedding-network dropout across **0, 0.25, 0.50, and 0.75**, keeping the architecture, other training hyperparameters, and fold assignments fixed:

```bash
python examples/deephit_extension_experiments.py \
  --experiment dropout --split-dir "$MAIN" \
  --folds 5 --epochs 500 --patience 3 --seed 0
```

**q50 results (five-fold means):**

| Dropout | Risk 1 C-index | Risk 1 Brier | Risk 2 C-index | Risk 2 Brier |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 0.7025 | 0.1308 | 0.7162 | 0.1271 |
| 0.25 | 0.6634 | 0.1394 | 0.6725 | 0.1382 |
| 0.50 | 0.6609 | 0.1400 | 0.6708 | 0.1389 |
| 0.75 | 0.6245 | 0.1426 | 0.5896 | 0.1418 |

With this fixed architecture, **zero dropout had the best mean results**; high dropout generally worsened discrimination and Brier-score accuracy. Do not compare these values directly with the hyperparameter-selected main experiment as though they came from the same model-selection protocol.

## 7. Source-file guide

| Location | Purpose |
| --- | --- |
| `pyhealth/models/neural_fg.py` | Public NeuralFG model implementation |
| `examples/deephit_competing_risks_neuralfg.py` | Standalone training/demonstration |
| `examples/deephit_paper_reproduction.py` | Paper-aligned pilot/fixed/random-search runner |
| `examples/deephit_extension_experiments.py` | Censoring and dropout experiments |
| `examples/evaluate_neuralfg_latent_censoring.py` | Common latent-outcome checkpoint evaluation |
| `examples/neuralfg/data.py` | DeepHit data loading / PyHealth dataset adapter |
| `examples/neuralfg/train.py` | Basic model training and splitting |
| `examples/neuralfg/cross_validation.py` | Paired standard cross-validation utilities |
| `examples/neuralfg/finegray_baseline.py` | Custom Python Fine–Gray baseline |
| `examples/neuralfg/metrics.py` | General evaluation utilities |
| `examples/neuralfg/metrics_reference.py` | Reference-style concordance and Brier metrics |
| `examples/neuralfg/paper_reproduction.py` | Fold splits, scaling, training, model selection, evaluation |
| `tests/test_neural_fg.py` | Model functionality tests |
| `tests/test_paper_reproduction.py` | Reproduction protocol tests |

Paths in this table are relative to the PyHealth repository root.

## 8. Reproducibility and interpretation notes

1. **Held-out tests:** Outer test subjects are never used for candidate selection or early stopping. Fold-specific scaling is learned from training observations.
2. **Model selection:** The early-stopping NLL and hyperparameter-tuning NLL are evaluated on distinct subsets. Do not substitute the smaller quick-start run for the main reproduction.
3. **Evaluation conventions:** The `metrics_reference.py` metrics follow a reference-style competing-risk implementation. They differ in some time-tie/competing-event conventions from alternative formulae or libraries; identify the metric definition when comparing values.
4. **Baseline status:** The custom Python Fine–Gray baseline has unit tests, but its coefficients and CIFs **have not yet been independently cross-checked** against R `cmprsk::crr()` on the same dataset and tie convention. Treat comparisons accordingly.
5. **Censoring:** The 75%-censored late-horizon IPCW evaluation has near-zero censoring survival; prioritize the common latent-outcome evaluation when interpreting robustness.
6. **Uncertainty:** Reported SDs are across five folds, not repeated independent dataset draws or random-seed replicates. Results are not a statistical significance claim.
7. **Artifacts:** Generated output under `examples/neuralfg/results/` (including checkpoints, histories, large tables, and figures) is intentionally excluded from Git by `.gitignore`. Regenerate the outputs using the documented commands.
8. **Fresh pilot:** The fresh-training smoke check used a newly named output directory, trained 4 fits from scratch, and matched the cached pilot's displayed summary values in the same environment. It does not replace an external clean-install/CI check.
9. **Convergence analysis:** Epoch summaries reflect the five saved censoring experiments' folds and show sensitivity to training conditions; they do not establish formal convergence.
10. **Original paper:** Architecture details, initialization, preprocessing, training logic, and baseline implementation are not identical to the authors' release. This is a **paper-aligned PyHealth reproduction**, not an exact replication.

## 9. Contribution and acknowledgements

Implemented as a team contribution to PyHealth for Georgia Tech CSE6250. See the final pull request for contributor attribution, repository version, and review discussion. The project builds on the Neural Fine-Gray research of Jeanselme and collaborators and the DeepHit synthetic competing-risk benchmark.
