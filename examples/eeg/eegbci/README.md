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
metadata, `event_code`, decoded `task_label`, PyHealth task-class identifier
(`eegbci_label` / `label`), PyHealth model-local label (`model_label`),
absolute window timing, band powers, relative band powers, `dominant_band`,
frequency ratios, and `interpretation`.

The moment-report columns add analysis-grade fields:

- `analysis_version`
- `state_hypothesis`, `state_confidence`, and `evidence_score`
- `evidence_summary`
- `rest_reference_scope` and rest-normalized relative band deltas
- `task_state_relation`, `task_state_rationale`, and `task_state_confidence`
- `is_low_confidence`, `is_possible_artifact`, and `is_mixed_or_ambiguous`

The `interpretation` column is report-level text derived from these moment-report
fields. Legacy task-level fields such as `brain_state_hypothesis`, `confidence`,
and `quality_flags` are intentionally not written to the CSV.

The Markdown report summarizes state counts, task-label/state agreement,
rest-normalized bandpower deltas, confidence and quality flags, representative
windows, limitations, and next checks. These labels are signal-pattern
summaries from short EEG windows, not clinical findings or evidence of a
subject's cognition.

## Data source and citation

The example uses PhysioNet's [EEG Motor Movement/Imagery Dataset
(eegmmidb), version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
The dataset files are distributed under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

Please cite:

- Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset* (version 1.0.0).
  PhysioNet. https://doi.org/10.13026/C28G6P
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw,
  J. R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI)
  System. *IEEE Transactions on Biomedical Engineering, 51*(6), 1034-1043.
- Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov,
  P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley,
  H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new
  research resource for complex physiologic signals. *Circulation, 101*(23),
  e215-e220.

`--root` points to the local EEGBCI data directory. With `--download`, MNE
downloads any missing EDF files under that root. PyHealth task caches are stored
under the configured PyHealth cache directory and are keyed by the requested
subject/run selection.
