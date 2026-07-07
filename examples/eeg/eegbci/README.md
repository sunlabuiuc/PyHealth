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
