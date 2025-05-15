# Medication-Allergy Conflict Detection Task

This module adds a custom PyHealth dataset and task for detecting conflicts between prescribed medications and known patient allergies using MIMIC-IV discharge summaries.

## Features

- Parses free-text discharge notes to extract medications and allergies
- Normalizes terms using RxNorm APIs to get ingredient CUIs
- Detects conflicts based on matching CUIs between meds and allergies
- Provides a PyHealth task interface (`get_label()`, `__call__()`, `export_conflicts()`)

## Files

- `med_allergy_conflict_dataset.py` – loads and processes MIMIC-IV discharge summaries
- `allergy_conflict_task.py` – PyHealth task that detects CUI-level conflicts
- `test_allergy_conflict_task.py` – unit tests for the task and dataset

## How to Run

```bash
python pyhealth/tasks/allergy_conflict_task.py