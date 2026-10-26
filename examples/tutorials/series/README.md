# PyHealth tutorial series

Colab notebooks for PyHealth 2.1, from a first model to contributing a
dataset, plus three short notebooks on what changed in 2.1. Each one installs
`pyhealth==2.1.0` and uses public or synthetic data, so no credentials are
needed. The saved outputs come from a full run on Colab, so you can read the
results here before running anything.

## Tutorial series

| # | Notebook | Covers |
|---|---|---|
| 00 | [Quickstart](00_quickstart.ipynb) | A complete in-hospital mortality study: data figures, Table 1, three models with CIs and calibration, SHAP, subgroups, exported results (T4 GPU) |
| 01 | [Datasets](01_datasets.ipynb) | Patients and events, YAML configs, caching, loading your own CSVs |
| 02 | [Tasks and processors](02_tasks_and_processors.ipynb) | Writing tasks, what each processor produces |
| 03 | [Training and evaluation](03_training_and_evaluation.ipynb) | Leakage-free splits, early stopping, class imbalance, calibration, bootstrap CIs |
| 04 | [Choosing a model](04_choosing_a_model.ipynb) | Logistic regression, MLP, RNN, Transformer, RETAIN and XGBoost on one split (T4 GPU) |
| 05 | [Interpreting predictions](05_interpreting_predictions.ipynb) | TreeSHAP, Integrated Gradients, checking faithfulness |
| 06 | [Medical codes](06_medical_codes.ipynb) | Lookups, ICD-9/ICD-10 translation, CCS/ATC groupers, `code_mapping` |
| 07 | [Clinical text](07_clinical_text.ipynb) | Fine-tuning language models; specialty classification and ICD coding; label distributions and baselines (T4 GPU) |
| 08 | [Contributing](08_contributing.ipynb) | Dataset classes, file-based data, tests, PR checklist |

Most notebooks run on a free Colab CPU runtime. Tutorials 00, 04 and 07 open on
a free T4 GPU runtime and also run on a CPU (04 and 07 with smaller settings).

## What's new in PyHealth 2.1

| # | Notebook | Covers |
|---|---|---|
| 1 | [Splits and reproducibility](whats_new_1_splits_and_reproducibility.ipynb) | `PatientSplit`, four-way splits with a calibration set, reproducible training |
| 2 | [Calibration and confidence intervals](whats_new_2_calibration_and_intervals.ipynb) | Calibration metrics, patient-level bootstrap intervals, `set_pos_weight` |
| 3 | [XGBoost and code mapping](whats_new_3_xgboost_and_code_mapping.ipynb) | `XGBoostModel` with exact TreeSHAP, ICD-9/ICD-10 and CCSR mapping, `nested_multihot` |

## Opening in Colab

Open a notebook from this folder with
`https://colab.research.google.com/github/sunlabuiuc/PyHealth/blob/master/examples/tutorials/series/<notebook>.ipynb`,
or use the maintained Colab copies listed on
[pyhealth.dev](https://pyhealth.dev/tutorials.html) and in the
[documentation](https://pyhealth.readthedocs.io/en/latest/tutorials.html).

## Editing

The Colab copies are the working versions; the notebooks here are exported
from them after a full run. To change a tutorial, edit it in Colab (or here),
run it end to end with **Runtime > Run all**, and commit the notebook with its
outputs.
