# CET Model Ablation (MLHC 2020)

**Paper:** [Neural Conditional Event Time Models](https://proceedings.mlr.press/v126/engelhard20a.html) (Engelhard et al., 2020)

## Overview

This example demonstrates the PyHealth-native Conditional Event Time (CET) model on synthetic right-censored event data and the existing PyHealth SUPPORT2 dataset.

CET jointly predicts whether a clinical event will eventually occur and, conditional on occurrence, when it will occur. We evaluate how changing the number of Monte Carlo samples and the Gumbel-Sigmoid temperature affects model performance.

The experiments serve two purposes:

1. **Synthetic ablation:** Evaluate event-occurrence discrimination and censored likelihood across multiple random seeds, with known eventual-occurrence labels.
2. **SUPPORT2 ablation:** Demonstrate model integration with an existing PyHealth dataset and evaluate held-out censored likelihood on real clinical data.

## Ablation Configurations

| Arm | Concrete Samples | Temperature | Purpose |
|---|---:|---:|---|
| A | 8 | 0.3 | Low-sample Monte Carlo baseline |
| B | 32 | 0.3 | Evaluate additional Monte Carlo samples |
| C | 32 | 1.0 | Evaluate higher-temperature relaxation |

Within each experiment, all configurations use the same dataset split, initial random seed, optimizer, and neural network architecture. The configurations differ only in their sampling parameters.

## 1. Synthetic Dataset Experiment

### Experimental Setup

- **Dataset:** 2,000 synthetic patients with 6 input features and 2 target events
- **Split:** 1,200 training, 400 validation, 400 testing
- **Censoring:** Artificial right censoring, with cutoff times sampled uniformly between 0.2 and 2.5
- **Model:** 32 hidden units, dropout 0.2
- **Optimizer:** Adam, learning rate 0.0003
- **Training:** Up to 60 epochs, batch size 64
- **Model selection:** Lowest validation negative log-likelihood (NLL)
- **Random seeds:** 7, 8, 9

The synthetic generator preserves the true eventual-occurrence labels for evaluation. These labels are not provided to CET during training.

We evaluate performance using:

- **Test NLL:** Held-out censored negative log-likelihood, evaluating the joint occurrence and timing objective. Lower is better.
- **Eventual-occurrence AUC:** Area under the ROC curve for predicting true eventual event occurrence. Higher is better.

AUC is calculated separately for each event and averaged across the two target events.

The reported NLL is the model's Monte Carlo-relaxed censored training objective, rather than an exact marginal mixture-cure log-likelihood.

### Running the Synthetic Experiments

Run the following commands from the PyHealth repository root:

```bash
python examples/synthetic_event_time_cet.py --n 2000 --epochs 60 --seed 7 --output cet_larger_synthetic.json

python examples/synthetic_event_time_cet.py --n 2000 --epochs 60 --seed 8 --output cet_ablation_seed8.json

python examples/synthetic_event_time_cet.py --n 2000 --epochs 60 --seed 9 --output cet_ablation_seed9.json
```

Each run evaluates all three configurations on the same synthetic train/validation/test split for that seed.

Results can be aggregated across seeds using the accompanying summary script:

```bash
python examples/summarize_cet_ablation.py cet_larger_synthetic.json cet_ablation_seed8.json cet_ablation_seed9.json
```

The per-event test AUC values used to calculate the reported averages can be inspected in the `test_eventual_auc` field for each configuration under `ablation` in the generated JSON files. The array contains the AUCs for Event 0 and Event 1, respectively.

### Synthetic Ablation Results

The following table summarizes results from three independently generated synthetic datasets using seeds 7, 8, and 9.

| Arm | Samples | Temperature | Test NLL (mean ± SD) ↓ | Mean AUC (mean ± SD) ↑ |
|---|---:|---:|---:|---:|
| A | 8 | 0.3 | 0.377 ± 0.010 | 0.635 ± 0.032 |
| B | 32 | 0.3 | **0.368 ± 0.018** | **0.636 ± 0.029** |
| C | 32 | 1.0 | 0.379 ± 0.014 | 0.630 ± 0.037 |

Values represent the mean and sample standard deviation across the three random seeds. For each seed, eventual-occurrence AUC is first averaged across the two target events.

Configuration B achieved the lowest average test NLL and highest average occurrence AUC, although differences between configurations were relatively small.

### Synthetic Experiment Findings

**1. Increasing Monte Carlo samples modestly improved average likelihood but had little effect on AUC.**

Increasing the sample count from 8 to 32 at temperature 0.3 reduced average test NLL from 0.377 to 0.368. However, mean event-occurrence AUC remained nearly unchanged (0.635 versus 0.636). Additional samples therefore provided a modest average improvement in likelihood without a corresponding improvement in occurrence discrimination.

**2. Increasing Gumbel-Sigmoid temperature did not improve average performance.**

At 32 samples, increasing temperature from 0.3 to 1.0 increased average test NLL from 0.368 to 0.379 and slightly decreased mean occurrence AUC from 0.636 to 0.630. These differences are modest, and the three-seed experiment does not establish that one temperature is universally optimal.

**3. Occurrence discrimination differed between the synthetic targets.**

For configuration B, Event 0 achieved a mean test AUC of approximately 0.711 across seeds, compared with 0.562 for Event 1. These results describe a difference in predictive performance between the two generated targets; the sampling-parameter ablation does not isolate its cause.

**4. Censored likelihood and occurrence AUC capture different aspects of performance.**

Improvements in censored NLL did not always correspond to improvements in eventual-occurrence AUC. CET optimizes a joint objective involving both event occurrence and conditional event timing rather than directly optimizing occurrence AUC.

### Synthetic Experiment Interpretation

The experiments demonstrate that the PyHealth CET implementation can train on right-censored data and learn meaningful event-occurrence relationships across multiple synthetic datasets.

Configuration B (32 samples, temperature 0.3) achieved the best average results among the tested settings. However, the differences were small relative to variation across seeds, particularly for occurrence AUC.

These findings support the functionality of the implementation but do not establish a universally superior sampling configuration.

## 2. SUPPORT2 Dataset Experiment

### Dataset and Preprocessing

We additionally evaluated CET using the existing PyHealth `Support2Dataset`, accessed through the `examples/survival_mdn_data.py` helper.

SUPPORT2 contains clinical patient information and right-censored mortality outcomes. Unlike the synthetic experiment, true eventual mortality cannot be established for all censored observations, so we report held-out censored NLL rather than eventual-occurrence AUC.

The full SUPPORT2 CSV was used rather than the small three-record fixture located in `test-resources/core/support2/`.

The experiment produced the following processed dataset:

- **Total observations:** 8,873
- **Training:** 6,211
- **Validation:** 1,331
- **Testing:** 1,331

The original SUPPORT2 dataset contained 9,105 observations. During preprocessing, 232 records were excluded because of missing white blood cell count (wblc) or creatinine (crea) values, leaving 8,873 observations for model training and evaluation. The example automatically generates temporary patient identifiers when the source CSV does not provide them. The updated example prints the raw row count, count missing either measurement, and final encoded row count for audit.

### Experimental Setup

- **Dataset:** SUPPORT2, through PyHealth's dataset-loading pathway
- **Target:** Right-censored mortality outcome
- **Model:** CET, using the same three sampling configurations
- **Training:** Up to 60 epochs
- **Random seed:** 7
- **Model selection:** Lowest validation NLL
- **Evaluation:** Held-out censored NLL

### Running the SUPPORT2 Experiment

A local copy of the full SUPPORT2 CSV is required.

The updated CET example accepts the original SUPPORT2 CSV directly, even when it does not contain an explicit patient identifier. If needed, it generates an internal sequential `sno` value in a temporary file to satisfy the pre-existing `Support2Dataset` loader.

Run the ablation from the PyHealth repository root:

```bash
python examples/synthetic_event_time_cet.py --source support2 --data "data/support2/support2.csv" --epochs 60 --seed 7 --output cet_support2_60.json
```

At startup, inspect the SUPPORT2 cohort audit output and verify that raw_rows - rows_missing_wblc_or_crea equals encoded_rows. The results below were reproduced using the original SUPPORT2 CSV and the current example's automatic temporary patient ID handling, with 60 training epochs and random seed 7. All three ablation configurations produced results matching the values reported in the table.

The dataset-loading helper initializes the PyHealth SUPPORT2 dataset, prepares the censored survival targets, and splits the observations for model training and evaluation.

### SUPPORT2 Ablation Results

All three configurations completed training and evaluation on the processed SUPPORT2 dataset.

| Arm | Samples | Temperature | Best Epoch | Validation NLL ↓ | Test NLL ↓ |
|---|---:|---:|---:|---:|---:|
| A | 8 | 0.3 | 59 | 0.5578 | **0.5070** |
| B | 32 | 0.3 | 60 | **0.5475** | 0.5102 |
| C | 32 | 1.0 | 56 | 0.5490 | 0.5113 |

Results are from one experiment using random seed 7. Lower NLL indicates better fit under the censored likelihood objective.

### SUPPORT2 Experiment Findings

**1. Increasing Monte Carlo samples improved validation NLL but not test NLL.**

Increasing the sample count from 8 to 32 at temperature 0.3 reduced validation NLL from 0.5578 to 0.5475. However, test NLL increased slightly from 0.5070 to 0.5102.

Although additional samples improved validation likelihood, the improvement did not translate to the held-out test split in this experiment.

**2. Increasing Gumbel-Sigmoid temperature slightly worsened test likelihood.**

At 32 samples, increasing temperature from 0.3 to 1.0 increased test NLL from 0.5102 to 0.5113. The difference was small and does not establish a meaningful advantage for either temperature.

**3. The lowest validation NLL and test NLL occurred in different configurations.**

Arm B achieved the lowest validation NLL, while Arm A achieved the lowest test NLL. This illustrates that model selection based on validation likelihood does not guarantee that the same configuration will achieve the best test likelihood.

**4. The model successfully trained using an existing PyHealth dataset.**

All three configurations completed training and evaluation using the SUPPORT2 preprocessing pathway. This demonstrates that the contributed CET model can operate on an existing PyHealth clinical dataset in addition to synthetic data.

### SUPPORT2 Experiment Interpretation

The SUPPORT2 results provide an additional evaluation of CET on real right-censored clinical outcomes.

Arm A achieved the lowest test NLL (0.5070), but the differences between configurations were small. The synthetic experiment favored Arm B on average, indicating that the relative performance of stochastic sampling configurations can vary by dataset.

Because the SUPPORT2 experiment used a single random seed, these findings should be treated as exploratory rather than statistically conclusive.

The experiment demonstrates successful dataset integration and model execution, not superiority over existing survival models or numerical reproduction of the original paper.

## 3. Differences from the Original Paper

These experiments evaluate a reusable PyHealth CET model component rather than reproduce the original MIMIC-III experiment.

The original study uses 346 patient features, 10 clinical events, and a larger neural network architecture. The synthetic experiments presented here use 6 features, 2 target events, and a smaller model to make training and repeated evaluation practical.

The SUPPORT2 experiment uses a different clinical dataset and mortality outcome rather than the MIMIC-III laboratory collection events studied in the paper.

The implementation follows the central CET approach of jointly modeling eventual occurrence and conditional event time. However, these experiments do not establish numerical equivalence with the original implementation or reproduce the paper's published MIMIC-III results.

Together, the synthetic and SUPPORT2 ablations demonstrate that the PyHealth implementation supports configurable stochastic sampling, can train on right-censored event data, and can be evaluated using both synthetic observations and an existing PyHealth dataset.

No patient-level datasets or trained model checkpoints should be committed to the public repository.