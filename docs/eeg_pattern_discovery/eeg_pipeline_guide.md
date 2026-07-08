# EEGBCI Pipeline Guide

This guide explains the EEGBCI files added for the motor movement/imagery
pipeline, the order to call them, what each file does, and how to interpret the
current report output.

## Runtime Call Order

For the moment-report artifact, run only the example script:

```bash
.venv/bin/python examples/eeg/eegbci/eegbci_pattern_discovery.py \
  --subjects 1 \
  --runs 3 \
  --max-windows 20 \
  --download
```

Internally, that calls the pipeline in this order:

1. `examples/eeg/eegbci/eegbci_pattern_discovery.py`
2. `EEGBCIDataset(...)` from `pyhealth/datasets/eegbci.py`
3. `pyhealth/datasets/configs/eegbci.yaml`
4. `dataset.set_task(EEGBCIPatternDiscovery(compute_stft=False))`
5. `EEGBCIPatternDiscovery.__call__()` in `pyhealth/tasks/eegbci.py`
6. `EEGMotorImageryEEGBCI._base_samples_from_patient()` in
   `pyhealth/tasks/eegbci.py`
7. `iter_annotation_windows()`, channel selection, normalization, optional STFT,
   and Welch bandpower computation in `pyhealth/tasks/eegbci.py`
8. Example-owned report helpers in
   `examples/eeg/eegbci/eegbci_pattern_discovery.py`
9. Output files under `outputs/eegbci_pattern_discovery/`

The reusable PyHealth data path stops at step 5 or 6 with a `SampleDataset`.
Steps 8 and 9 are report-only and are not the normal model-training interface.

## Files And Responsibilities

### `pyhealth/datasets/eegbci.py`

Defines `EEGBCIDataset`, the dataset entry point. It selects requested subjects
and runs, optionally downloads PhysioNet EEGBCI EDF files through MNE, finds local
EDF files when download is disabled, and writes the metadata table
`eegbci-pyhealth.csv`.

The metadata table has one record per subject/run EDF file. It does not create
2-second windows itself. Windowing happens in the task layer.

### `pyhealth/datasets/configs/eegbci.yaml`

Tells `BaseDataset` how to read `eegbci-pyhealth.csv` as the `records` table.
It maps `patient_id`, `record_id`, `subject_id`, `run`, `run_type`,
`signal_file`, and `source` into PyHealth dataset events.

### `pyhealth/tasks/eegbci.py`

Contains the reusable EEGBCI task logic.

Key pieces:

- `run_type_for_run()`, `task_label_for_event()`, and
  `numeric_label_for_task()` decode EEGBCI run/event labels.
- `select_eegbci_channels()` selects either the 16-channel compatibility montage
  or all channels.
- `normalize_signal()` applies task-level signal normalization.
- `iter_annotation_windows()` converts MNE annotations into full 2-second
  windows.
- `EEGMotorImageryEEGBCI` produces model-ready samples with `signal`, optional
  `stft`, and multiclass `label`.
- `EEGBCIPatternDiscovery` extends the motor-imagery task by adding Welch
  bandpower metadata and cautious legacy frequency-profile interpretation.

This file is the correct reusable task layer for downstream PyHealth models.

### `examples/eeg/eegbci/eegbci_pattern_discovery.py`

This is the report generator. It is not a training script.

It creates an `EEGBCIDataset`, applies `EEGBCIPatternDiscovery`, converts
samples to rows, computes rest baselines across all requested rows before
`--max-windows` truncation, annotates each moment with report-level state
hypotheses, and writes:

- `outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv`
- `outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md`

The report-level fields are intentionally example-owned because they depend on
cross-window context such as rest baselines and task/state comparison.

### `examples/eeg/eegbci/README.md`

Short user-facing instructions for running the example and reading the generated
CSV and Markdown files.

### `tests/core/test_eegbci.py`

Unit coverage for EEGBCI dataset metadata, task windowing, bandpower behavior,
and the report helpers. Normal tests use synthetic data and do not download
EEGBCI. The real-data smoke test is opt-in through `PYHEALTH_RUN_REAL_EEGBCI=1`.

### API Documentation Files

These expose the reusable dataset and task APIs in generated docs:

- `docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst`
- `docs/api/tasks/pyhealth.tasks.eegbci.rst`
- `docs/api/datasets.rst`
- `docs/api/tasks.rst`

## Output Interpretation

The CSV is a moment-by-moment ledger. Each row is one emitted 2-second EEG
window. The most important columns are:

- `task_label`: what the experiment instructed in that window.
- `dominant_band` and `{band}_relative`: the raw frequency profile.
- `state_hypothesis`: report-level frequency-pattern state.
- `state_confidence` and `evidence_score`: strength of the deterministic
  frequency-pattern evidence.
- `rest_reference_scope`: which rest baseline was used.
- `rest_{band}_relative_delta`: how the row differs from the selected rest
  baseline for that band.
- `task_state_relation`: whether the frequency-pattern state supports, adds
  detail to, disagrees with, or is ambiguous relative to the task label.
- `task_state_confidence`: confidence in that deterministic task/state relation.
- `interpretation`: report-level text summarizing the row in terms of the
  moment-report state, raw dominant band, rest reference, and task/state
  relation.
- `is_low_confidence`, `is_possible_artifact`, and `is_mixed_or_ambiguous`:
  parseable quality flags for filtering.

The CSV intentionally omits legacy task-level columns such as
`brain_state_hypothesis`, `confidence`, and `quality_flags`. Those fields still
exist inside `EEGBCIPatternDiscovery` samples for reusable task compatibility,
but they were too redundant and confusing for the final report artifact.

The Markdown report summarizes those rows. It should be read as signal-pattern
metadata, not as a clinical or cognitive conclusion. For example, a
`slow_wave_dominant_pattern` means the short window's delta/theta evidence was
strong under the current heuristic. It does not diagnose a subject state.

The report is useful for asking:

- Which frequency-pattern states dominate this run?
- Are the states diverse or collapsed into one profile?
- Did rest normalization produce usable deltas?
- Which windows are representative enough to inspect manually?
- Which rows are low confidence, artifact-like, or ambiguous?

## Next PyHealth Stage

The next normal PyHealth stage after dataset/task construction is model
training or evaluation on a `SampleDataset`.

Use this path for model work:

```python
dataset = EEGBCIDataset(root=..., subjects=[...], runs=[...], download=True)
sample_dataset = dataset.set_task(EEGMotorImageryEEGBCI())
```

That produces samples shaped for PyHealth models:

- `signal`: EEG tensor
- optional `stft`: time-frequency tensor when `compute_stft=True`
- `label`: multiclass target
- metadata fields such as subject, run, event, and timing

The current CSV/Markdown report is not the format expected by the next PyHealth
model-training stage. It is an analysis artifact for researchers. If the next
stage is a PyHealth model, pass the `SampleDataset` returned by `dataset.set_task`
to the trainer/model stack rather than reading
`eegbci_pattern_windows.csv`.

If the next stage is offline analysis, dashboarding, or manual review, then the
CSV and Markdown are the right outputs.

## Which Output Should Feed What?

| Output | Intended consumer | Suitable for PyHealth model training? |
| --- | --- | --- |
| `EEGMotorImageryEEGBCI` `SampleDataset` | PyHealth models and trainers | Yes |
| `EEGBCIPatternDiscovery` `SampleDataset` | Signal inspection plus reusable task metadata | Partly, but primarily exploratory |
| `eegbci_pattern_windows.csv` | Analysis, filtering, audit, dashboards | No |
| `eegbci_pattern_summary.md` | Human-readable report | No |
