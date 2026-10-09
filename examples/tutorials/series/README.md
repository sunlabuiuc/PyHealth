# PyHealth tutorial series

Nine Colab-ready notebooks, from a first model to contributing a dataset. Each
runs on a free Colab CPU runtime with public or synthetic data, so no
credentials are needed.

| # | Notebook | Covers |
|---|---|---|
| 00 | [Quickstart](00_quickstart.ipynb) | The five steps: dataset, task, split, model, evaluation |
| 01 | [Datasets](01_datasets.ipynb) | Patients and events, YAML configs, caching, loading your own CSVs |
| 02 | [Tasks and processors](02_tasks_and_processors.ipynb) | Writing tasks, what each processor produces |
| 03 | [Training and evaluation](03_training_and_evaluation.ipynb) | Leakage-free splits, early stopping, class imbalance, calibration, bootstrap CIs |
| 04 | [Choosing a model](04_choosing_a_model.ipynb) | Logistic regression to Transformer, plus an XGBoost baseline, on one split |
| 05 | [Interpreting predictions](05_interpreting_predictions.ipynb) | TreeSHAP, Integrated Gradients, checking faithfulness |
| 06 | [Medical codes](06_medical_codes.ipynb) | Lookups, ICD-9/ICD-10 translation, CCS/ATC groupers, `code_mapping` |
| 07 | [Clinical text](07_clinical_text.ipynb) | Fine-tuning language models; specialty classification and ICD coding |
| 08 | [Contributing](08_contributing.ipynb) | Dataset classes, file-based data, tests, PR checklist |

Open one in Colab from
`https://colab.research.google.com/github/sunlabuiuc/PyHealth/blob/master/examples/tutorials/series/<notebook>.ipynb`.

## Editing

The notebooks are generated from `_source/tutorials/tNN_*.py`; edit those, then
rebuild (and optionally execute) with:

```bash
python examples/tutorials/series/_source/build.py            # all notebooks
python examples/tutorials/series/_source/build.py t04 --run   # rebuild and execute one
```

`--run` needs `nbclient` and a Jupyter kernel with PyHealth installed; it skips
the install cell and stops at the first failing cell.
