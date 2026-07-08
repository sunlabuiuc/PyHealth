# EEGBCI Pattern Discovery Test Verification Plan

Date: 2026-07-07
Branch: `eegbci-pattern-discovery`
Base branch: `master`
Python: `.venv/bin/python`

This plan verifies the EEGBCI pattern-discovery implementation against the
Correctness Oracle in `docs/eeg_pattern_discovery/implementation_plan.md`.
Normal/offline checks must not download PhysioNet data. Real EEGBCI checks are
explicitly opt-in and use `/tmp/pyhealth-eegbci-verification` as the data/output
root.

## Checklist and Commands

### 1. Offline Unit Tests

- [ ] Run the default EEGBCI test file without network access.
- [ ] Confirm the real-data smoke test is skipped by default.
- [ ] Confirm run-aware labels, channel selection, normalization, annotation
  windowing, task sample generation, bandpower, dataset metadata, and exports
  pass together.

```bash
PYHEALTH_RUN_REAL_EEGBCI=0 .venv/bin/python -m pytest tests/core/test_eegbci.py -v
```

### 2. Task and Schema Tests

- [ ] Verify `EEGMotorImageryEEGBCI` exposes the expected task name, input
  schema, and multiclass output schema.
- [ ] Verify `compute_stft=False` removes `stft` from `input_schema`.
- [ ] Verify `EEGBCIPatternDiscovery(compute_stft=False)` keeps the pattern task
  name and tensor signal input.
- [ ] Verify STFT generation receives the actual task sample rate.
- [ ] Verify generated supervised samples include fixed-shape `signal`,
  decoded `task_label`, numeric model `label`, raw `eegbci_label`, `trial_id`,
  timing fields, and `sample_rate`.
- [ ] Verify pattern-discovery samples add `bandpower`,
  `brain_state_hypothesis`, `confidence`, `quality_flags`, and
  `interpretation`.

```bash
.venv/bin/python -m pytest \
  tests/core/test_eegbci.py::TestEEGBCITasks::test_task_schema_attributes \
  tests/core/test_eegbci.py::TestEEGBCITasks::test_task_schema_without_stft \
  tests/core/test_eegbci.py::TestEEGBCITasks::test_pattern_discovery_schema_attributes \
  tests/core/test_eegbci.py::TestEEGBCITasks::test_motor_imagery_task_returns_samples_from_raw \
  tests/core/test_eegbci.py::TestEEGBCITasks::test_pattern_discovery_adds_bandpower_metadata \
  tests/core/test_eegbci.py::TestEEGBCITasks::test_stft_uses_current_sample_rate \
  -v
```

### 3. Synthetic Bandpower Tests

- [ ] Verify a synthetic 10 Hz signal produces `dominant_band == "alpha"` and
  high `alpha_relative`.
- [ ] Verify a synthetic 20 Hz signal produces `dominant_band == "beta"` and
  high `beta_relative`.
- [ ] Verify deterministic interpretation metadata remains cautious and
  non-clinical.

```bash
.venv/bin/python -m pytest \
  tests/core/test_eegbci.py::TestEEGBCIHelpers::test_compute_band_powers_detects_alpha_sinusoid \
  tests/core/test_eegbci.py::TestEEGBCIHelpers::test_compute_band_powers_detects_beta_sinusoid \
  tests/core/test_eegbci.py::TestEEGBCIHelpers::test_interpret_band_profile_returns_cautious_metadata \
  -v
```

### 4. Dataset Metadata Tests

- [ ] Verify `EEGBCIDataset.prepare_metadata()` writes one row per requested
  subject/run.
- [ ] Verify required metadata columns are present:
  `patient_id`, `record_id`, `subject_id`, `run`, `run_type`, `signal_file`,
  and `source`.
- [ ] Verify local-file discovery works without network access.
- [ ] Verify metadata is rebuilt when the same root is reused with a different
  subject/run selection.
- [ ] Verify an offline `EEGBCIDataset(...).set_task(...)` integration path
  works through `BaseDataset`.
- [ ] Verify `download=True` delegates to `mne.datasets.eegbci.load_data`.
- [ ] Verify missing local files fail clearly when `download=False`.
- [ ] Verify the default task is `EEGBCIPatternDiscovery`.

```bash
.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIDataset -v
```

### 5. Import/Export Smoke Tests

- [ ] Verify public imports work from `pyhealth.datasets`, `pyhealth.tasks`, and
  direct modules.
- [ ] Verify run-aware EEGBCI label semantics from the Correctness Oracle.
- [ ] Verify synthetic alpha bandpower from the Correctness Oracle.

```bash
.venv/bin/python - <<'PY'
import numpy as np

from pyhealth.datasets import EEGBCIDataset
from pyhealth.datasets.eegbci import EEGBCIDataset as DirectDataset
from pyhealth.tasks import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI
from pyhealth.tasks.eegbci import (
    compute_band_powers,
    task_label_for_event,
)

assert EEGBCIDataset is DirectDataset
assert EEGMotorImageryEEGBCI.task_name == "EEGBCI_motor_imagery"
assert EEGBCIPatternDiscovery.task_name == "EEGBCI_pattern_discovery"
assert task_label_for_event(3, "T1") == "execute_left_fist"
assert task_label_for_event(4, "T1") == "imagine_left_fist"

sfreq = 200.0
times = np.arange(0, 2, 1 / sfreq)
alpha = np.sin(2 * np.pi * 10 * times)
features = compute_band_powers(np.stack([alpha, alpha]), sfreq)
assert features["dominant_band"] == "alpha"
assert features["alpha_relative"] > 0.5
print("imports and oracle smoke checks ok")
PY
```

### 6. Example-Output Validation

- [ ] Run the example on subject `1`, run `3`, with a small window cap.
- [ ] Verify `eegbci_pattern_windows.csv` exists and has at least one row.
- [ ] Verify `eegbci_pattern_summary.md` exists.
- [ ] Verify at least one CSV row contains task label, dominant band,
  hypothesis, confidence, and non-clinical interpretation text.

```bash
.venv/bin/python examples/eeg/eegbci/eegbci_pattern_discovery.py \
  --root /tmp/pyhealth-eegbci-verification/data \
  --subjects 1 \
  --runs 3 \
  --output-dir /tmp/pyhealth-eegbci-verification/example-output \
  --max-windows 20 \
  --download
.venv/bin/python - <<'PY'
from pathlib import Path

import pandas as pd

out = Path("/tmp/pyhealth-eegbci-verification/example-output")
csv_path = out / "eegbci_pattern_windows.csv"
summary_path = out / "eegbci_pattern_summary.md"
assert csv_path.exists(), csv_path
assert summary_path.exists(), summary_path
df = pd.read_csv(csv_path)
assert len(df) >= 1
required = {
    "task_label",
    "label",
    "eegbci_label",
    "model_label",
    "dominant_band",
    "brain_state_hypothesis",
    "confidence",
    "interpretation",
}
assert required.issubset(df.columns), sorted(set(required) - set(df.columns))
assert df["task_label"].notna().any()
assert not df["label"].astype(str).str.contains("tensor").any()
assert df["eegbci_label"].notna().any()
assert df["model_label"].notna().any()
assert df["dominant_band"].notna().any()
assert df["brain_state_hypothesis"].notna().any()
assert df["confidence"].notna().any()
assert df["interpretation"].str.contains("not evidence of cognition|not clinical", case=False, regex=True).any()
print(f"validated {len(df)} example rows")
PY
```

### 7. Opt-In Real EEGBCI Smoke Test

- [ ] Run the skipped-by-default real-data smoke test explicitly.
- [ ] Verify it downloads or reuses subject `1`, run `3`.
- [ ] Verify it produces at least one 16-channel pattern-discovery sample.

```bash
PYHEALTH_RUN_REAL_EEGBCI=1 .venv/bin/python -m pytest \
  tests/core/test_eegbci.py::TestEEGBCIRealDataSmoke \
  -v
```

### 8. Docs/API Smoke Checks

- [ ] Verify API RST files exist.
- [ ] Verify dataset/task toctrees include EEGBCI pages.
- [ ] Verify Sphinx targets can import the referenced objects/modules.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from pyhealth.datasets import EEGBCIDataset
from pyhealth.tasks import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI
import pyhealth.tasks.eegbci as eegbci_module

dataset_page = Path("docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst")
task_page = Path("docs/api/tasks/pyhealth.tasks.eegbci.rst")
assert dataset_page.exists()
assert task_page.exists()
assert "datasets/pyhealth.datasets.EEGBCIDataset" in Path("docs/api/datasets.rst").read_text()
assert "tasks/pyhealth.tasks.eegbci" in Path("docs/api/tasks.rst").read_text()
assert EEGBCIDataset.__name__ == "EEGBCIDataset"
assert EEGMotorImageryEEGBCI.task_name == "EEGBCI_motor_imagery"
assert EEGBCIPatternDiscovery.task_name == "EEGBCI_pattern_discovery"
assert hasattr(eegbci_module, "compute_band_powers")
print("docs/api smoke checks ok")
PY
```

## Results

Status after execution: all required checks passed on 2026-07-07.

| Area | Command | Result | Notes |
| --- | --- | --- | --- |
| Offline unit tests | `PYHEALTH_RUN_REAL_EEGBCI=0 .venv/bin/python -m pytest tests/core/test_eegbci.py -v` | Passed | Final run: 24 passed, 1 skipped in 9.36s. Real-data smoke skipped by default. |
| Task/schema tests | Targeted `TestEEGBCITasks` command | Passed | 6 passed in 8.58s, including STFT sample-rate forwarding. |
| Synthetic bandpower tests | Targeted `TestEEGBCIHelpers` command | Passed | 3 passed in 5.45s. 10 Hz alpha and 20 Hz beta checks passed. |
| Dataset metadata tests | `.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIDataset -v` | Passed | 6 passed in 8.60s, including stale selection rebuild and offline `set_task()` integration. |
| Import/export smoke tests | Inline Python smoke script | Passed | Printed `imports and oracle smoke checks ok`. |
| Example-output validation | Example command plus CSV/Markdown validator | Passed | Downloaded/reused subject 1 run 3, wrote CSV and Markdown, validated 20 rows. |
| Opt-in real EEGBCI smoke test | `PYHEALTH_RUN_REAL_EEGBCI=1 ... TestEEGBCIRealDataSmoke` | Passed | 1 passed in 33.61s against subject 1 run 3. |
| Docs/API smoke checks | Inline Python docs/API smoke script | Passed | Printed `docs/api smoke checks ok`. |
| Syntax smoke | `.venv/bin/python -m py_compile ...` | Passed | Dataset, task, example, and test modules compile. |

## Failures and Fixes Needed

Independent code review found one critical issue and several important issues
before final status was marked complete. They were fixed and covered by
regression tests:

- Fixed stale metadata reuse when the same EEGBCI root is instantiated with a
  different subject/run selection. Existing `eegbci-pyhealth.csv` is reused only
  when it exactly matches the requested pairs.
- Fixed PyHealth cache identity by including the subject/run selection in the
  default dataset name passed to `BaseDataset`.
- Added raw `eegbci_label` to task samples and changed the example CSV so
  `label`/`eegbci_label` preserve EEGBCI semantics while `model_label` records
  the processed PyHealth label.
- Fixed STFT generation to pass the actual `sample_rate` into
  `get_stft_torch()`.
- Changed annotation onset conversion to use MNE `raw.time_as_index(...,
  use_rounding=True)`.
- Added offline `BaseDataset.set_task()` integration coverage.
- Expanded the example README with output schema, label caveat, root/download,
  and cache behavior.

No remaining failing checks.

## Correctness Oracle Status

Satisfied:

- Run-aware decoding: import/export smoke asserts
  `task_label_for_event(3, "T1") == "execute_left_fist"` and
  `task_label_for_event(4, "T1") == "imagine_left_fist"`.
- Synthetic 10 Hz alpha: synthetic bandpower and import/export smoke checks pass.
- Metadata rows and columns: dataset metadata tests pass, including changed
  subject/run selection rebuild.
- `EEGMotorImageryEEGBCI(compute_stft=False)` schema and sample fields: task
  tests and offline integration pass.
- `EEGBCIPatternDiscovery(compute_stft=False)` supervised fields plus
  bandpower/hypothesis/confidence/flags/interpretation: task tests pass.
- Default offline test command: passes with the real-data smoke skipped.
- Opt-in real EEGBCI smoke: passes against subject 1, run 3.
- Example artifacts: CSV and Markdown are written and validated with 20 rows,
  including task label, dominant band, hypothesis, confidence, and non-clinical
  interpretation text.

## Final Status

Complete. The Correctness Oracle is satisfied, the independent code-review
findings were addressed, and every command in this plan passed.
