# EEGBCI Pattern Discovery

This example uses `EEGBCIDataset` and `EEGBCIPatternDiscovery` to create
2-second EEGBCI windows with task labels, Welch bandpower features, and cautious
frequency-profile interpretations.

The interpretations are exploratory signal metadata. They are not clinical
diagnoses and do not prove a subject's cognition.

Run a tiny real-data example:

```bash
python examples/eeg/eegbci/eegbci_pattern_discovery.py \
  --subjects 1 \
  --runs 3 \
  --max-windows 20 \
  --download
```

Outputs are written to `outputs/eegbci_pattern_discovery/` by default:

- `eegbci_pattern_windows.csv`
- `eegbci_pattern_summary.md`

The CSV has one row per emitted 2-second window. Key columns include subject/run
metadata, `event_code`, decoded `task_label`, raw EEGBCI numeric label
(`eegbci_label` / `label`), PyHealth model-local label (`model_label`),
absolute window timing, band powers, relative band powers, `dominant_band`,
frequency ratios, legacy `brain_state_hypothesis`, `confidence`,
`quality_flags`, and `interpretation`.

The moment-report columns add analysis-grade fields:

- `analysis_version`
- `state_hypothesis`, `state_confidence`, and `evidence_score`
- `evidence_summary`
- `rest_reference_scope` and rest-normalized relative band deltas
- `task_state_relation`, `task_state_rationale`, and `task_state_confidence`
- `is_low_confidence`, `is_possible_artifact`, and `is_mixed_or_ambiguous`

The Markdown report summarizes state counts, task-label/state agreement,
rest-normalized bandpower deltas, confidence and quality flags, representative
windows, limitations, and next checks. These labels are signal-pattern
summaries from short EEG windows, not clinical findings or evidence of a
subject's cognition.

Implementation details are tracked in
`docs/eeg_pattern_discovery/moment_report_implementation_plan.md`.

`--root` points to the local EEGBCI data directory. With `--download`, MNE
downloads any missing EDF files under that root. PyHealth task caches are stored
under the configured PyHealth cache directory and are keyed by the requested
subject/run selection.
