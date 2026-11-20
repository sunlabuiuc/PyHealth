# SEER lung cancer survival reproduction

This example reproduces the 12-month lung cancer comparison from Hegselmann et
al. (2018) in PyHealth and measures the effect of removing stage and tumor
extent. [Original study](https://proceedings.mlr.press/v85/hegselmann18a.html) ·
[Authors' implementation](https://github.com/stefanhgm/MLHC2018-reproducible-survival-seer)

## Run the example

Install the package from the PyHealth checkout using `python -m pip install -e .`.
Run the following commands from that checkout.

```bash
# Generated synthetic data, no SEER account required
python examples/seer_survival_mlp.py

# Authorized local Case Listing and a cache directory on approved storage
python examples/seer_survival_mlp.py --data-dir /path/to/export \
    --cache-dir /path/to/cache --results /path/to/results.json
```

The real-data run was recorded on Windows using CPU, Python 3.13.17, PyTorch
2.15.0, and PyHealth 2.1.0 based on upstream commit `4df0611`. The saved
[aggregate JSON](results.json) includes the validation candidates, selected
checkpoints, held-out scores, and paired ablation differences.

## Cohort and label definition

The Case Listing used the November 2025 SEER Research Data submission, 17
registries (2000–2023). The
[SEER*Stat session](Lung_2004_2009_current.ss) stores the cohort and export
settings. Access to the real records requires individual authorization.
[SEER data access](https://seer.cancer.gov/data/access.html)

1. Select malignant, first-primary lung and bronchus tumors diagnosed in
   2004–2009 with known age. Apply the saved session's exclusions for autopsy
   or death-certificate-only cases and cases alive with no survival time.
2. Run the session as a Case Listing and export comma-delimited formatted
   text with headers as `seer_export.txt`. Keep the export and its cache in
   approved local storage. See the [dataset API](../../docs/api/datasets/pyhealth.datasets.SEERDataset.rst)
   for the accepted format.
3. Group by patient ID and track each tumor by record number. Retain one sample
   per eligible tumor. Assign label 0 to cancer deaths before 12 months and
   label 1 to records with at least 12 months of observed survival. Exclude
   early censoring, earlier deaths from competing causes, missing survival,
   and unknown or inapplicable cause of death. Identifiers and outcomes are
   outside the feature schema.

Of 217,176 exported tumors, 204,457 qualified for the task. This total
includes 114,588 cases with label 0 and 89,869 with label 1. The export contained
304 records with unknown cause of death, so the task applies this exclusion
explicitly. The raw export includes follow-up years after the December 2014
cutoff shown in the session's survival settings.

## Experimental protocol

The patient-level 80/10/10 split used seed 42 and contained 163,565 training,
20,446 validation, and 20,446 test patients. All configurations used these
partitions. Training rows alone determined the category dictionaries and
scaling parameters. The numeric baseline uses verified native codes and assigns
training-derived ordinal codes to formatted categories. One-hot encoding uses
indicators for categories and special size values alongside scaled continuous
values.

The majority baseline predicts the most frequent training label. The learned
configurations use PyHealth logistic regression and a 64-unit, two-hidden-layer
MLP with ReLU. For each model and representation, Adam used a batch size of
1024 and evaluated learning rates of 0.001 and 0.003 for up to 10 epochs.
Validation ROC-AUC plus F1 selected the learning rate and checkpoint. Test F1
treats reaching 12 months as the positive class at a 0.5 probability threshold.

## Held-out results

The reference columns contain the corresponding 12-month lung results from
Table 3 of the paper. Results are reported as fractions.

| Configuration | ROC-AUC | F1 | Accuracy | Paper ROC-AUC | Paper F1 | Paper accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.5000 | 0.0000 | 0.5596 | 0.5000 | 0.0000 | 0.5609 |
| Logistic regression, numeric | 0.7726 | 0.6633 | 0.7284 | 0.8209 | 0.6942 | 0.7552 |
| Logistic regression, one-hot | 0.8300 | 0.7117 | 0.7599 | 0.8419 | 0.7162 | 0.7696 |
| MLP, numeric | 0.8226 | 0.7057 | 0.7558 | 0.8347 | 0.7207 | 0.7639 |
| MLP, one-hot | **0.8349** | **0.7143** | **0.7627** | 0.8446 | 0.7265 | 0.7681 |

One-hot encoding improved all three metrics for both learned models. The
one-hot MLP led the PyHealth comparison and scored 0.0097 AUC below its
counterpart in the paper. The paper used the November 2016 ASCII release.
Differences in release, cohort coverage, observed follow-up, feature coverage,
numeric codes, and model tuning make this a partial reproduction.

## Grouped ablation

The formatted export has no case-level registry or state/county column. This
experiment removes AJCC stage, tumor size, tumor extension, lymph nodes, and
metastases together. The full and reduced inputs each train a new one-hot MLP
using the same patient partitions, initial seed, Adam learning rate of 0.001,
and eight epochs. The full model's validation run selected these settings. The
grouped retraining tests the predictive contribution of the available stage
and extent information.

Each cell reports full, reduced, and the drop from full to reduced.

| Model seed | ROC-AUC | F1 | Accuracy |
| --- | --- | --- | --- |
| 42 | 0.8349 / 0.7208 / 0.1141 | 0.7143 / 0.5636 / 0.1507 | 0.7627 / 0.6735 / 0.0893 |
| 43 | 0.8354 / 0.7206 / 0.1148 | 0.7059 / 0.5672 / 0.1386 | 0.7627 / 0.6727 / 0.0900 |
| 44 | 0.8354 / 0.7204 / 0.1149 | 0.7138 / 0.5672 / 0.1466 | 0.7620 / 0.6716 / 0.0904 |
| Mean | 0.8352 / 0.7206 / 0.1146 | 0.7113 / 0.5660 / 0.1453 | 0.7625 / 0.6726 / 0.0899 |

Removing the group lowered all three metrics for every seed.
