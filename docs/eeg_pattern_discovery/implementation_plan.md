# EEG Pattern Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: ENGINEERING REVIEWED, ready for implementation
Date: 2026-07-07
Source docs:
- `docs/eeg_pattern_discovery/brainstorm.md`
- `docs/eeg_pattern_discovery/design.md`

**Goal:** Add a first-class EEGBCI dataset and two EEGBCI tasks to PyHealth so real PhysioNet motor movement/imagery windows can be used for supervised classification and CELM-style exploratory pattern discovery.

**High-Level Summary:** We are turning the standalone CELM EEG pattern-discovery pipeline into a reusable PyHealth dataset, task, and example for real PhysioNet EEGBCI motor movement/imagery data. The research question is whether simple frequency profiles in short EEG windows can surface moment-level brain-state hypotheses that task labels alone miss. The practical problem is that PyHealth has EEG models and task infrastructure, but no first-class EEGBCI path that produces labeled windows, bandpower features, cautious interpretation metadata, and real-data validation without forcing normal CI to download raw EDF files.

**Architecture:** `EEGBCIDataset` builds one metadata row per subject/run EDF, following the existing `TUABDataset` and `TUEVDataset` CSV pattern. `EEGMotorImageryEEGBCI` reads EDF annotations and emits fixed windows for task-label prediction; `EEGBCIPatternDiscovery` extends the same windows with Welch bandpower features and cautious interpretation metadata. Offline unit tests mock MNE and EDF reads; an opt-in smoke test downloads one real EEGBCI run.

**Tech Stack:** PyHealth `BaseDataset` and `BaseTask`, MNE EEGBCI loader, NumPy, SciPy Welch PSD, pandas, torch, pytest/unittest, Sphinx RST docs.

## Global Constraints

- Do not claim brain-state hypotheses are clinical diagnoses.
- Decode EEGBCI `T1` and `T2` using run number.
- Normal CI must not download PhysioNet data.
- Keep raw EEG downloads outside the repo.
- Preserve compatibility with PyHealth models by making channel and sampling strategy explicit.
- Use real EEGBCI data in examples and opt-in smoke tests.
- Default pattern-discovery windows are 2 seconds.
- Default model-facing channel mode is a stable 16-channel subset; `channel_mode="all"` keeps all EEG channels.
- Default resampling is `200 Hz`; `resample_rate=None` keeps the EDF sample rate.

---

## File Structure

Create:

- `pyhealth/datasets/eegbci.py`: EEGBCI metadata dataset. Calls `mne.datasets.eegbci.load_data` only when `download=True`, otherwise discovers existing `SxxxRxx.edf` files under `root`.
- `pyhealth/datasets/configs/eegbci.yaml`: one-table metadata config for `eegbci-pyhealth.csv`.
- `pyhealth/tasks/eegbci.py`: run-aware labels, channel selection, annotation windowing, bandpower extraction, interpretation rules, and task classes.
- `tests/core/test_eegbci.py`: offline unit tests plus skipped-by-default real-data smoke test.
- `examples/eeg/eegbci/README.md`: runnable example instructions, output schema, caveats.
- `examples/eeg/eegbci/eegbci_pattern_discovery.py`: PyHealth example that writes CELM-equivalent CSV and Markdown summary.
- `docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst`: dataset API page.
- `docs/api/tasks/pyhealth.tasks.eegbci.rst`: EEGBCI task API page.

Modify:

- `pyhealth/datasets/__init__.py`: export `EEGBCIDataset`.
- `pyhealth/tasks/__init__.py`: export `EEGMotorImageryEEGBCI` and `EEGBCIPatternDiscovery`.
- `docs/api/datasets.rst`: include the EEGBCI dataset page.
- `docs/api/tasks.rst`: include the EEGBCI task page.

Defer:

- `examples/eeg/eegbci/eegbci_embedding_comparison.py`: useful later, but out of scope for the first implementation. The first pass should ship dataset, tasks, tests, docs, and the CELM-equivalent example.

Resolved decisions:

- First implementation scope: dataset, tasks, tests, docs, and CELM-equivalent example only.
- Default channel mode: 16-channel compatibility, with `channel_mode="all"` available for 64-channel experiments.
- Dependency model: use existing project-level `mne~=1.10.0`; do not add an optional EEG extra in this pass.
- Real-data validation: no committed `.npz` fixture in the first pass; use mocked offline tests plus the opt-in real-data smoke test.
- Model boundary: the first pattern-discovery pipeline uses signal processing and deterministic interpretation, not a neural model. PyHealth model training and pretrained embeddings are enabled by the task outputs but deferred from the first deliverable.
- Analysis stage: the first analysis stage is the CELM-equivalent CSV and Markdown report produced by `examples/eeg/eegbci/eegbci_pattern_discovery.py`.

## Recommended Execution Path

Start from a feature branch, not `main`:

```bash
git checkout -b codex/eegbci-pattern-discovery
```

If the implementation agent uses an isolated worktree, create or switch to that branch inside the worktree before editing. Keep each task boundary commit on this branch so review and rollback stay clean.

Use one main implementation session to own the feature end to end. Do not split concurrent coding across multiple sub-agents because `pyhealth/tasks/eegbci.py` and `tests/core/test_eegbci.py` are shared by most tasks, and parallel edits would create interface drift.

Execute the plan sequentially:

1. Task 1: pure EEGBCI helpers and tests.
2. Task 2: `EEGBCIDataset`, metadata config, dataset export, and tests.
3. Task 3: EEGBCI task classes, annotation windowing, sample schema, task export, and tests.
4. Task 4: skipped-by-default real-data smoke test.
5. Task 5: CELM-equivalent analysis example.
6. Task 6: API docs and import smoke test.
7. Final verification: run the default test command, import smoke, optional real-data smoke test, optional example command, and `graphify update .`.

Use sub-agents only for bounded review or investigation after a section is implemented. Good sub-agent assignments:

- Review `pyhealth/tasks/eegbci.py` for label/window/channel contract drift.
- Inspect existing PyHealth docs/export patterns before Task 6.
- Investigate real EEGBCI channel names if the opt-in smoke test fails.
- Review the example output schema against the correctness oracle.

Do not give different sections to independent sessions unless each session starts from the previous section's committed result. If separate sessions are used, hand off after Tasks 1, 2, 3, 4, and 5 only, with tests passing and commits made at each boundary.

## Data Flow

1. User instantiates `EEGBCIDataset(root, subjects, runs, download)`.
2. Dataset writes or reuses `<root>/eegbci-pyhealth.csv` with one row per EDF run.
3. `BaseDataset` loads rows into patient events from `pyhealth/datasets/configs/eegbci.yaml`.
4. User calls `dataset.set_task(EEGMotorImageryEEGBCI(...))` or `dataset.set_task(EEGBCIPatternDiscovery(...))`.
5. Task reads each `signal_file` with MNE, picks EEG channels, filters, optionally resamples, decodes annotations, and slices full fixed-length windows.
6. Motor-imagery task returns supervised samples with `label`.
7. Pattern-discovery task returns the same samples plus `bandpower`, `brain_state_hypothesis`, `confidence`, `quality_flags`, and `interpretation`.
8. Example script converts samples to CSV rows and writes a Markdown summary grouped by task label and hypothesis.

## Correctness Oracle

An implementation is complete only when these checks pass:

- `task_label_for_event(3, "T1") == "execute_left_fist"` and `task_label_for_event(4, "T1") == "imagine_left_fist"`, proving run-aware EEGBCI decoding.
- Synthetic 10 Hz input produces `dominant_band == "alpha"` and high `alpha_relative`, proving bandpower extraction works.
- `EEGBCIDataset.prepare_metadata()` writes one row per requested subject/run with `patient_id`, `record_id`, `subject_id`, `run`, `run_type`, `signal_file`, and `source`.
- `EEGMotorImageryEEGBCI(compute_stft=False)` returns fixed-shape `signal` tensors, decoded `task_label`, numeric `label`, `trial_id`, timing fields, and `sample_rate`.
- `EEGBCIPatternDiscovery(compute_stft=False)` returns every supervised field plus `bandpower`, `brain_state_hypothesis`, `confidence`, `quality_flags`, and `interpretation`.
- Default `pytest tests/core/test_eegbci.py -v` passes without network access and skips the real-data smoke test.
- `PYHEALTH_RUN_REAL_EEGBCI=1 pytest tests/core/test_eegbci.py::TestEEGBCIRealDataSmoke -v` passes against subject `1`, run `3`.
- The example command writes `eegbci_pattern_windows.csv` and `eegbci_pattern_summary.md`, with at least one row containing task label, dominant band, hypothesis, confidence, and non-clinical interpretation text.

## Analysis Stage Contract

The first analysis stage lives in `examples/eeg/eegbci/eegbci_pattern_discovery.py`. It is not just a demo script. It is the artifact generator that proves the pipeline can answer the research question on real windows.

Inputs:

- `EEGBCIDataset`
- `EEGBCIPatternDiscovery`
- CLI flags: `--root`, `--subjects`, `--runs`, `--output-dir`, `--max-windows`, `--download`

Outputs:

- `eegbci_pattern_windows.csv`: one row per window, including subject/run metadata, event code, decoded task label, bandpower values, relative powers, dominant band, ratios, hypothesis, confidence, quality flags, and interpretation.
- `eegbci_pattern_summary.md`: aggregate counts by task label and brain-state hypothesis, plus the non-clinical caveat.

Question answered:

- Do the frequency-profile hypotheses agree with, sharpen, or flag possible disagreement with the experimental EEGBCI labels?

Deferred analysis:

- Neural embedding comparison.
- Clustering or atlas generation.
- TFM token motif reports.
- Subject-shift reliability reports.

## Shared Interfaces

These names are fixed across tasks:

```python
EEGBCI_RUN_TYPES: dict[int, str]
EEGBCI_COMPAT_CHANNELS: tuple[str, ...]
EEGBCI_LABELS: dict[str, int]

def normalize_eegbci_channel_name(name: str) -> str: ...
def run_type_for_run(run: int) -> str: ...
def label_family_for_run(run: int) -> str: ...
def task_label_for_event(run: int, event_code: str) -> str: ...
def numeric_label_for_task(task_label: str) -> int: ...
def select_eegbci_channels(data: np.ndarray, ch_names: list[str], channel_mode: str = "compat16") -> tuple[np.ndarray, list[str]]: ...
def iter_annotation_windows(raw: mne.io.BaseRaw, run: int, window_size: float = 2.0) -> list[dict[str, Any]]: ...
def compute_band_powers(data: np.ndarray, sfreq: float) -> dict[str, float | str]: ...
def interpret_band_profile(features: dict[str, float | str]) -> dict[str, str]: ...
def normalize_signal(signal: np.ndarray, mode: str | None) -> np.ndarray: ...
```

Task classes:

```python
class EEGMotorImageryEEGBCI(BaseTask):
    task_name = "EEGBCI_motor_imagery"
    input_schema = {"signal": "tensor", "stft": "tensor"}
    output_schema = {"label": "multiclass"}

class EEGBCIPatternDiscovery(EEGMotorImageryEEGBCI):
    task_name = "EEGBCI_pattern_discovery"
```

## Task 1: Pure EEGBCI Helpers

**Files:**

- Create: `pyhealth/tasks/eegbci.py`
- Create: `tests/core/test_eegbci.py`

**Interfaces:**

- Consumes: NumPy and SciPy signal processing.
- Produces: `run_type_for_run`, `label_family_for_run`, `task_label_for_event`, `numeric_label_for_task`, `select_eegbci_channels`, `compute_band_powers`, `interpret_band_profile`, `normalize_signal`.

- [x] **Step 1: Write failing tests for run-aware labels**

Add these tests to `tests/core/test_eegbci.py`:

```python
import unittest

from pyhealth.tasks.eegbci import (
    label_family_for_run,
    numeric_label_for_task,
    run_type_for_run,
    task_label_for_event,
)


class TestEEGBCIHelpers(unittest.TestCase):
    def test_run_type_for_run(self):
        self.assertEqual(run_type_for_run(3), "motor_execution_left_right")
        self.assertEqual(run_type_for_run(4), "motor_imagery_left_right")
        self.assertEqual(run_type_for_run(5), "motor_execution_fists_feet")
        self.assertEqual(run_type_for_run(6), "motor_imagery_fists_feet")
        self.assertEqual(run_type_for_run(14), "motor_imagery_fists_feet")

    def test_task_label_for_event_is_run_aware(self):
        self.assertEqual(task_label_for_event(3, "T0"), "rest")
        self.assertEqual(task_label_for_event(3, "T1"), "execute_left_fist")
        self.assertEqual(task_label_for_event(3, "T2"), "execute_right_fist")
        self.assertEqual(task_label_for_event(4, "T1"), "imagine_left_fist")
        self.assertEqual(task_label_for_event(4, "T2"), "imagine_right_fist")
        self.assertEqual(task_label_for_event(5, "T1"), "execute_both_fists")
        self.assertEqual(task_label_for_event(5, "T2"), "execute_both_feet")
        self.assertEqual(task_label_for_event(6, "T1"), "imagine_both_fists")
        self.assertEqual(task_label_for_event(6, "T2"), "imagine_both_feet")

    def test_label_family_and_numeric_labels(self):
        self.assertEqual(label_family_for_run(3), "motor_execution")
        self.assertEqual(label_family_for_run(4), "motor_imagery")
        self.assertEqual(numeric_label_for_task("rest"), 0)
        self.assertEqual(numeric_label_for_task("execute_left_fist"), 1)
        self.assertEqual(numeric_label_for_task("imagine_both_feet"), 8)

    def test_invalid_run_and_event_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Unsupported EEGBCI run"):
            run_type_for_run(2)
        with self.assertRaisesRegex(ValueError, "Unsupported EEGBCI event"):
            task_label_for_event(3, "BAD")
```

- [x] **Step 2: Run tests to verify import failure**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCIHelpers -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pyhealth.tasks.eegbci'`.

- [x] **Step 3: Implement label helpers**

Add to `pyhealth/tasks/eegbci.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


EEGBCI_RUN_TYPES = {
    3: "motor_execution_left_right",
    4: "motor_imagery_left_right",
    5: "motor_execution_fists_feet",
    6: "motor_imagery_fists_feet",
    7: "motor_execution_left_right",
    8: "motor_imagery_left_right",
    9: "motor_execution_fists_feet",
    10: "motor_imagery_fists_feet",
    11: "motor_execution_left_right",
    12: "motor_imagery_left_right",
    13: "motor_execution_fists_feet",
    14: "motor_imagery_fists_feet",
}

EEGBCI_LABELS = {
    "rest": 0,
    "execute_left_fist": 1,
    "execute_right_fist": 2,
    "imagine_left_fist": 3,
    "imagine_right_fist": 4,
    "execute_both_fists": 5,
    "execute_both_feet": 6,
    "imagine_both_fists": 7,
    "imagine_both_feet": 8,
}


def run_type_for_run(run: int) -> str:
    try:
        return EEGBCI_RUN_TYPES[int(run)]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEGBCI run: {run}") from exc


def label_family_for_run(run: int) -> str:
    run_type = run_type_for_run(run)
    if "execution" in run_type:
        return "motor_execution"
    if "imagery" in run_type:
        return "motor_imagery"
    return "baseline"


def task_label_for_event(run: int, event_code: str) -> str:
    code = str(event_code).strip()
    if code == "T0":
        return "rest"
    run_type = run_type_for_run(run)
    mapping = {
        "motor_execution_left_right": {
            "T1": "execute_left_fist",
            "T2": "execute_right_fist",
        },
        "motor_imagery_left_right": {
            "T1": "imagine_left_fist",
            "T2": "imagine_right_fist",
        },
        "motor_execution_fists_feet": {
            "T1": "execute_both_fists",
            "T2": "execute_both_feet",
        },
        "motor_imagery_fists_feet": {
            "T1": "imagine_both_fists",
            "T2": "imagine_both_feet",
        },
    }
    try:
        return mapping[run_type][code]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEGBCI event {event_code!r} for run {run}") from exc


def numeric_label_for_task(task_label: str) -> int:
    try:
        return EEGBCI_LABELS[task_label]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEGBCI task label: {task_label}") from exc
```

- [x] **Step 4: Run label tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCIHelpers -v
```

Expected: PASS for the four label tests.

- [x] **Step 5: Add failing tests for channel selection and normalization**

Append to `TestEEGBCIHelpers`:

```python
    def test_select_eegbci_channels_compat16(self):
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS, select_eegbci_channels

        ch_names = list(EEGBCI_COMPAT_CHANNELS) + ["EXTRA"]
        data = np.arange(len(ch_names) * 100, dtype=float).reshape(len(ch_names), 100)
        selected, selected_names = select_eegbci_channels(data, ch_names, "compat16")
        self.assertEqual(selected.shape, (16, 100))
        self.assertEqual(selected_names, list(EEGBCI_COMPAT_CHANNELS))
        np.testing.assert_allclose(selected[0], data[0])

    def test_select_eegbci_channels_all(self):
        from pyhealth.tasks.eegbci import select_eegbci_channels

        data = np.ones((64, 50))
        ch_names = [f"CH{i}" for i in range(64)]
        selected, selected_names = select_eegbci_channels(data, ch_names, "all")
        self.assertEqual(selected.shape, (64, 50))
        self.assertEqual(selected_names, ch_names)

    def test_select_eegbci_channels_missing_channel_raises(self):
        from pyhealth.tasks.eegbci import select_eegbci_channels

        with self.assertRaisesRegex(ValueError, "Missing EEGBCI channels"):
            select_eegbci_channels(np.ones((2, 20)), ["C3", "C4"], "compat16")

    def test_normalize_signal_95th_percentile(self):
        from pyhealth.tasks.eegbci import normalize_signal

        signal = np.array([[0.0, 1.0, 2.0, 100.0], [0.0, -2.0, 2.0, 4.0]])
        normalized = normalize_signal(signal, "95th_percentile")
        self.assertEqual(normalized.shape, signal.shape)
        self.assertLess(np.max(np.abs(normalized[0])), 2.0)
```

- [x] **Step 6: Implement channel and normalization helpers**

Add below the label helpers:

```python
EEGBCI_COMPAT_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP4",
    "CP6",
)


def normalize_eegbci_channel_name(name: str) -> str:
    clean = name.upper().replace(".", "").replace("EEG ", "").replace("-REF", "")
    aliases = {
        "T9": "FT9",
        "T10": "FT10",
    }
    return aliases.get(clean, clean)


def select_eegbci_channels(
    data: np.ndarray,
    ch_names: List[str],
    channel_mode: str = "compat16",
) -> Tuple[np.ndarray, List[str]]:
    if channel_mode == "all":
        return data, list(ch_names)
    if channel_mode != "compat16":
        raise ValueError("channel_mode must be one of {'compat16', 'all'}")

    normalized_to_index = {
        normalize_eegbci_channel_name(name): idx for idx, name in enumerate(ch_names)
    }
    missing = [ch for ch in EEGBCI_COMPAT_CHANNELS if ch not in normalized_to_index]
    if missing:
        raise ValueError(f"Missing EEGBCI channels for compat16 mode: {missing}")
    indices = [normalized_to_index[ch] for ch in EEGBCI_COMPAT_CHANNELS]
    return data[indices], list(EEGBCI_COMPAT_CHANNELS)


def normalize_signal(signal: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return signal
    if mode == "95th_percentile":
        scale = np.quantile(
            np.abs(signal), q=0.95, axis=-1, method="linear", keepdims=True
        )
        return signal / (scale + 1e-8)
    if mode == "div_by_100":
        return signal / 100.0
    raise ValueError("normalization must be one of {None, '95th_percentile', 'div_by_100'}")
```

- [x] **Step 7: Run helper tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCIHelpers -v
```

Expected: PASS.

- [x] **Step 8: Add failing tests for bandpower and interpretation**

Append to `TestEEGBCIHelpers`:

```python
    def test_compute_band_powers_detects_alpha_sinusoid(self):
        from pyhealth.tasks.eegbci import compute_band_powers

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        alpha = np.sin(2 * np.pi * 10 * times)
        data = np.stack([alpha, alpha])
        features = compute_band_powers(data, sfreq)
        self.assertEqual(features["dominant_band"], "alpha")
        self.assertGreater(features["alpha_relative"], 0.5)
        self.assertGreater(features["alpha_beta_ratio"], 1.0)

    def test_compute_band_powers_detects_beta_sinusoid(self):
        from pyhealth.tasks.eegbci import compute_band_powers

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        beta = np.sin(2 * np.pi * 20 * times)
        data = np.stack([beta, beta])
        features = compute_band_powers(data, sfreq)
        self.assertEqual(features["dominant_band"], "beta")
        self.assertGreater(features["beta_relative"], 0.5)

    def test_interpret_band_profile_returns_cautious_metadata(self):
        from pyhealth.tasks.eegbci import interpret_band_profile

        interpretation = interpret_band_profile(
            {
                "dominant_band": "alpha",
                "alpha_relative": 0.65,
                "beta_relative": 0.10,
                "theta_relative": 0.10,
                "gamma_relative": 0.05,
                "alpha_beta_ratio": 6.5,
                "theta_beta_ratio": 1.0,
            }
        )
        self.assertEqual(interpretation["brain_state_hypothesis"], "relaxed_or_idle")
        self.assertIn(interpretation["confidence"], {"low", "medium", "high"})
        self.assertIn("consistent with", interpretation["interpretation"])
```

- [x] **Step 9: Implement bandpower and interpretation helpers**

Add below the channel helpers:

```python
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_band_powers(data: np.ndarray, sfreq: float) -> Dict[str, float | str]:
    from scipy.signal import welch

    if data.ndim != 2:
        raise ValueError("data must have shape (channels, time)")
    nperseg = min(data.shape[-1], int(sfreq * 2))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    mean_psd = psd.mean(axis=0)

    features: Dict[str, float | str] = {}
    total_power = 0.0
    band_values: Dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        value = float(np.trapz(mean_psd[mask], freqs[mask])) if np.any(mask) else 0.0
        features[f"{band}_power"] = value
        band_values[band] = value
        total_power += value

    denom = total_power + 1e-12
    for band, value in band_values.items():
        features[f"{band}_relative"] = float(value / denom)

    features["dominant_band"] = max(band_values, key=band_values.get)
    features["alpha_beta_ratio"] = float(
        band_values["alpha"] / (band_values["beta"] + 1e-12)
    )
    features["theta_beta_ratio"] = float(
        band_values["theta"] / (band_values["beta"] + 1e-12)
    )
    return features


def interpret_band_profile(features: Dict[str, float | str]) -> Dict[str, str]:
    dominant = str(features["dominant_band"])
    alpha_rel = float(features.get("alpha_relative", 0.0))
    beta_rel = float(features.get("beta_relative", 0.0))
    theta_rel = float(features.get("theta_relative", 0.0))
    gamma_rel = float(features.get("gamma_relative", 0.0))
    alpha_beta = float(features.get("alpha_beta_ratio", 0.0))
    theta_beta = float(features.get("theta_beta_ratio", 0.0))

    quality_flags: List[str] = []
    hypothesis = "mixed_frequency_profile"
    confidence = "low"

    if dominant == "alpha" and alpha_rel >= 0.45 and alpha_beta >= 2.0:
        hypothesis = "relaxed_or_idle"
        confidence = "medium"
    elif dominant == "beta" and beta_rel >= 0.35:
        hypothesis = "active_sensorimotor_processing"
        confidence = "medium"
    elif dominant == "theta" and theta_rel >= 0.35 and theta_beta >= 1.5:
        hypothesis = "slow_wave_or_drowsy_pattern"
        confidence = "medium"
    elif dominant == "gamma" and gamma_rel >= 0.30:
        hypothesis = "high_frequency_or_artifact_pattern"
        confidence = "low"
        quality_flags.append("possible_muscle_artifact")

    if confidence == "low":
        quality_flags.append("low_confidence")

    return {
        "brain_state_hypothesis": hypothesis,
        "confidence": confidence,
        "quality_flags": ";".join(quality_flags) if quality_flags else "none",
        "interpretation": (
            f"The segment is consistent with {hypothesis} based on a "
            f"{dominant}-dominant frequency profile. This is exploratory signal "
            "metadata, not evidence of cognition or a clinical diagnosis."
        ),
    }
```

- [x] **Step 10: Run helper tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCIHelpers -v
```

Expected: PASS.

- [x] **Step 11: Commit task 1**

Run:

```bash
git add pyhealth/tasks/eegbci.py tests/core/test_eegbci.py
git commit -m "feat: add EEGBCI helper functions"
```

## Task 2: EEGBCI Dataset

**Files:**

- Create: `pyhealth/datasets/eegbci.py`
- Create: `pyhealth/datasets/configs/eegbci.yaml`
- Modify: `pyhealth/datasets/__init__.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**

- Consumes: helper `run_type_for_run` from task 1, MNE EEGBCI loader when `download=True`.
- Produces: `EEGBCIDataset(root, dataset_name=None, config_path=None, subjects=None, runs=None, download=False, **kwargs)`.
- Produces metadata file: `<root>/eegbci-pyhealth.csv`.

- [x] **Step 1: Add failing dataset metadata tests**

Append to `tests/core/test_eegbci.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets.eegbci import EEGBCIDataset


class TestEEGBCIDataset(unittest.TestCase):
    def test_prepare_metadata_with_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edf = root / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R03.edf"
            edf.parent.mkdir(parents=True)
            edf.write_bytes(b"")

            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = str(root)
            ds.subjects = [1]
            ds.runs = [3]
            ds.download = False
            ds.prepare_metadata()

            csv_path = root / "eegbci-pyhealth.csv"
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "patient_id"], "S001")
            self.assertEqual(df.loc[0, "record_id"], "R03")
            self.assertEqual(df.loc[0, "subject_id"], 1)
            self.assertEqual(df.loc[0, "run"], 3)
            self.assertEqual(df.loc[0, "run_type"], "motor_execution_left_right")
            self.assertEqual(df.loc[0, "source"], "physionet_eegbci")

    def test_prepare_metadata_download_uses_mne_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_path = root / "S001R04.edf"
            fake_path.write_bytes(b"")
            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = str(root)
            ds.subjects = [1]
            ds.runs = [4]
            ds.download = True

            with patch(
                "pyhealth.datasets.eegbci.mne.datasets.eegbci.load_data",
                return_value=[str(fake_path)],
            ) as load_data:
                ds.prepare_metadata()

            load_data.assert_called_once_with(1, [4], path=str(root))
            df = pd.read_csv(root / "eegbci-pyhealth.csv")
            self.assertEqual(df.loc[0, "record_id"], "R04")
            self.assertEqual(df.loc[0, "run_type"], "motor_imagery_left_right")

    def test_prepare_metadata_missing_local_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = EEGBCIDataset.__new__(EEGBCIDataset)
            ds.root = tmp
            ds.subjects = [1]
            ds.runs = [3]
            ds.download = False
            with self.assertRaisesRegex(FileNotFoundError, "download=True"):
                ds.prepare_metadata()

    def test_default_task_returns_pattern_discovery(self):
        from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery

        ds = EEGBCIDataset.__new__(EEGBCIDataset)
        self.assertIsInstance(ds.default_task, EEGBCIPatternDiscovery)
```

- [x] **Step 2: Run dataset tests to verify failure**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCIDataset -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pyhealth.datasets.eegbci'`.

- [x] **Step 3: Implement `EEGBCIDataset`**

Create `pyhealth/datasets/eegbci.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mne
import pandas as pd

from .base_dataset import BaseDataset
from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery, run_type_for_run

logger = logging.getLogger(__name__)


class EEGBCIDataset(BaseDataset):
    """PhysioNet EEG Motor Movement/Imagery metadata dataset."""

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subjects: Optional[list[int]] = None,
        runs: Optional[list[int]] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "eegbci.yaml"
        self.root = root
        self.subjects = subjects or [1, 2, 3]
        self.runs = runs or list(range(3, 15))
        self.download = download
        self.prepare_metadata()
        super().__init__(
            root=root,
            tables=["records"],
            dataset_name=dataset_name or "eegbci",
            config_path=config_path,
            **kwargs,
        )

    def _find_local_edf(self, subject: int, run: int) -> Path | None:
        root = Path(self.root)
        pattern = f"S{subject:03d}R{run:02d}.edf"
        matches = sorted(root.rglob(pattern))
        return matches[0] if matches else None

    def prepare_metadata(self) -> None:
        root = Path(self.root)
        csv_path = root / "eegbci-pyhealth.csv"
        if csv_path.exists():
            return

        rows: list[dict] = []
        for subject in self.subjects:
            paths_by_run: dict[int, Path] = {}
            if self.download:
                downloaded = mne.datasets.eegbci.load_data(
                    subject, self.runs, path=str(root)
                )
                for path in downloaded:
                    p = Path(path)
                    for run in self.runs:
                        if p.name == f"S{subject:03d}R{run:02d}.edf":
                            paths_by_run[run] = p
            for run in self.runs:
                signal_file = paths_by_run.get(run) or self._find_local_edf(subject, run)
                if signal_file is None:
                    raise FileNotFoundError(
                        f"Missing EEGBCI EDF for subject {subject}, run {run}. "
                        "Pass download=True to fetch it with MNE."
                    )
                rows.append(
                    {
                        "patient_id": f"S{subject:03d}",
                        "record_id": f"R{run:02d}",
                        "subject_id": int(subject),
                        "run": int(run),
                        "run_type": run_type_for_run(run),
                        "signal_file": str(signal_file),
                        "source": "physionet_eegbci",
                    }
                )

        df = pd.DataFrame(rows)
        df.sort_values(["subject_id", "run"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Wrote EEGBCI metadata to %s", csv_path)

    @property
    def default_task(self) -> EEGBCIPatternDiscovery:
        return EEGBCIPatternDiscovery()
```

- [x] **Step 4: Add dataset config**

Create `pyhealth/datasets/configs/eegbci.yaml`:

```yaml
version: "1.0.0"
tables:
  records:
    file_path: "eegbci-pyhealth.csv"
    patient_id: "patient_id"
    timestamp: null
    attributes:
    - "record_id"
    - "subject_id"
    - "run"
    - "run_type"
    - "signal_file"
    - "source"
```

- [x] **Step 5: Export dataset**

Modify `pyhealth/datasets/__init__.py`:

```python
from pyhealth.datasets.eegbci import EEGBCIDataset
```

Place the import beside other EEG dataset exports.

- [x] **Step 6: Run dataset tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCIDataset -v
```

Expected: PASS.

- [x] **Step 7: Commit task 2**

Run:

```bash
git add pyhealth/datasets/eegbci.py pyhealth/datasets/configs/eegbci.yaml pyhealth/datasets/__init__.py tests/core/test_eegbci.py
git commit -m "feat: add EEGBCI dataset"
```

## Task 3: EEGBCI Task Classes

**Files:**

- Modify: `pyhealth/tasks/eegbci.py`
- Modify: `pyhealth/tasks/__init__.py`
- Modify: `tests/core/test_eegbci.py`

**Interfaces:**

- Consumes: dataset metadata events with `signal_file`, `run`, `record_id`, `run_type`.
- Produces: `EEGMotorImageryEEGBCI` and `EEGBCIPatternDiscovery` samples.

- [x] **Step 1: Add task schema tests**

Append to `tests/core/test_eegbci.py`:

```python
from dataclasses import dataclass
from typing import List

import numpy as np

from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI


@dataclass
class _EEGBCIEvent:
    signal_file: str
    record_id: str = "R03"
    subject_id: int = 1
    run: int = 3
    run_type: str = "motor_execution_left_right"
    source: str = "physionet_eegbci"


class _EEGBCIPatient:
    def __init__(self, patient_id: str, events: List[_EEGBCIEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_EEGBCIEvent]:
        if event_type not in (None, "records"):
            return []
        return self._events


class TestEEGBCITasks(unittest.TestCase):
    def test_task_schema_attributes(self):
        task = EEGMotorImageryEEGBCI()
        self.assertEqual(task.task_name, "EEGBCI_motor_imagery")
        self.assertEqual(task.input_schema, {"signal": "tensor", "stft": "tensor"})
        self.assertEqual(task.output_schema, {"label": "multiclass"})

    def test_task_schema_without_stft(self):
        task = EEGMotorImageryEEGBCI(compute_stft=False)
        self.assertEqual(task.input_schema, {"signal": "tensor"})

    def test_pattern_discovery_schema_attributes(self):
        task = EEGBCIPatternDiscovery(compute_stft=False)
        self.assertEqual(task.task_name, "EEGBCI_pattern_discovery")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
```

- [x] **Step 2: Run schema tests to verify failure**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCITasks -v
```

Expected: FAIL because classes do not exist.

- [x] **Step 3: Implement task constructors**

Add to `pyhealth/tasks/eegbci.py`:

```python
import torch
import mne

from pyhealth.tasks import BaseTask


class EEGMotorImageryEEGBCI(BaseTask):
    task_name: str = "EEGBCI_motor_imagery"
    input_schema: Dict[str, str] = {"signal": "tensor", "stft": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        window_size: float = 2.0,
        resample_rate: float | None = 200,
        bandpass_filter: Tuple[float, float] | None = (0.5, 45.0),
        channel_mode: str = "compat16",
        normalization: str | None = "95th_percentile",
        compute_stft: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.channel_mode = channel_mode
        self.normalization = normalization
        self.compute_stft = compute_stft
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}


class EEGBCIPatternDiscovery(EEGMotorImageryEEGBCI):
    task_name: str = "EEGBCI_pattern_discovery"
```

- [x] **Step 4: Run schema tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCITasks -v
```

Expected: PASS for schema tests.

- [x] **Step 5: Add failing annotation window tests**

Append to `TestEEGBCITasks`:

```python
    def test_iter_annotation_windows_uses_full_2s_windows(self):
        import mne
        from pyhealth.tasks.eegbci import iter_annotation_windows

        sfreq = 200.0
        raw = mne.io.RawArray(
            np.zeros((2, int(sfreq * 6))),
            mne.create_info(["C3", "C4"], sfreq=sfreq, ch_types=["eeg", "eeg"]),
            verbose="error",
        )
        raw.set_annotations(
            mne.Annotations(onset=[0.5, 2.0], duration=[1.0, 3.0], description=["T0", "T1"])
        )
        windows = iter_annotation_windows(raw, run=3, window_size=2.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["event_code"], "T1")
        self.assertEqual(windows[0]["task_label"], "execute_left_fist")
        self.assertEqual(windows[0]["start_sample"], 400)
        self.assertEqual(windows[0]["end_sample"], 800)
```

- [x] **Step 6: Implement annotation windowing**

Add before the task classes:

```python
def iter_annotation_windows(
    raw: mne.io.BaseRaw,
    run: int,
    window_size: float = 2.0,
) -> List[Dict[str, Any]]:
    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(window_size * sfreq))
    windows: List[Dict[str, Any]] = []
    for idx, annotation in enumerate(raw.annotations):
        event_code = str(annotation["description"])
        if event_code not in {"T0", "T1", "T2"}:
            continue
        start_sample = int(round(float(annotation["onset"]) * sfreq))
        duration_samples = int(round(float(annotation["duration"]) * sfreq))
        n_full_windows = duration_samples // window_samples
        for window_idx in range(n_full_windows):
            s0 = start_sample + window_idx * window_samples
            s1 = s0 + window_samples
            windows.append(
                {
                    "trial_id": f"ann{idx:04d}_win{window_idx:03d}",
                    "event_code": event_code,
                    "task_label": task_label_for_event(run, event_code),
                    "label_family": label_family_for_run(run),
                    "label": numeric_label_for_task(task_label_for_event(run, event_code)),
                    "start_time": s0 / sfreq,
                    "end_time": s1 / sfreq,
                    "start_sample": s0,
                    "end_sample": s1,
                }
            )
    return windows
```

- [x] **Step 7: Run annotation tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCITasks::test_iter_annotation_windows_uses_full_2s_windows -v
```

Expected: PASS.

- [x] **Step 8: Add failing sample-generation tests**

Append to `TestEEGBCITasks`:

```python
    def test_motor_imagery_task_returns_samples_from_raw(self):
        import mne

        sfreq = 200.0
        raw = mne.io.RawArray(
            np.ones((16, int(sfreq * 5))),
            mne.create_info(
                list(__import__("pyhealth.tasks.eegbci", fromlist=["EEGBCI_COMPAT_CHANNELS"]).EEGBCI_COMPAT_CHANNELS),
                sfreq=sfreq,
                ch_types=["eeg"] * 16,
            ),
            verbose="error",
        )
        raw.set_annotations(mne.Annotations(onset=[0.0], duration=[2.0], description=["T1"]))
        patient = _EEGBCIPatient("S001", [_EEGBCIEvent(signal_file="dummy.edf")])
        task = EEGMotorImageryEEGBCI(compute_stft=False, resample_rate=None, bandpass_filter=None)

        with patch("pyhealth.tasks.eegbci.mne.io.read_raw_edf", return_value=raw):
            samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "S001")
        self.assertEqual(sample["record_id"], "R03")
        self.assertEqual(sample["event_code"], "T1")
        self.assertEqual(sample["task_label"], "execute_left_fist")
        self.assertEqual(sample["label"], 1)
        self.assertEqual(tuple(sample["signal"].shape), (16, 400))

    def test_pattern_discovery_adds_bandpower_metadata(self):
        import mne
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        alpha = np.sin(2 * np.pi * 10 * times)
        raw = mne.io.RawArray(
            np.tile(alpha, (16, 1)),
            mne.create_info(list(EEGBCI_COMPAT_CHANNELS), sfreq=sfreq, ch_types=["eeg"] * 16),
            verbose="error",
        )
        raw.set_annotations(mne.Annotations(onset=[0.0], duration=[2.0], description=["T0"]))
        patient = _EEGBCIPatient("S001", [_EEGBCIEvent(signal_file="dummy.edf")])
        task = EEGBCIPatternDiscovery(compute_stft=False, resample_rate=None, bandpass_filter=None)

        with patch("pyhealth.tasks.eegbci.mne.io.read_raw_edf", return_value=raw):
            samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["bandpower"]["dominant_band"], "alpha")
        self.assertEqual(sample["brain_state_hypothesis"], "relaxed_or_idle")
        self.assertIn("interpretation", sample)
```

- [x] **Step 9: Implement EDF reading and sample generation**

Add methods inside `EEGMotorImageryEEGBCI`:

```python
    def read_raw(self, signal_file: str) -> mne.io.BaseRaw:
        raw = mne.io.read_raw_edf(signal_file, preload=True, verbose="error")
        raw.pick_types(eeg=True, stim=False, exclude=[])
        if self.bandpass_filter is not None:
            raw.filter(
                l_freq=self.bandpass_filter[0],
                h_freq=self.bandpass_filter[1],
                verbose="error",
            )
        if self.resample_rate is not None:
            raw.resample(self.resample_rate, n_jobs=1, verbose="error")
        return raw

    def _base_samples_from_patient(self, patient: Any) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for event in patient.get_events("records"):
            raw = self.read_raw(event.signal_file)
            data = raw.get_data(units="uV")
            selected, selected_names = select_eegbci_channels(
                data, raw.ch_names, self.channel_mode
            )
            selected = normalize_signal(selected, self.normalization)
            sfreq = float(raw.info["sfreq"])
            for idx, window in enumerate(
                iter_annotation_windows(raw, int(event.run), self.window_size)
            ):
                signal_np = selected[:, window["start_sample"] : window["end_sample"]]
                if signal_np.shape[-1] != int(round(self.window_size * sfreq)):
                    continue
                signal = torch.FloatTensor(signal_np)
                sample = {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "subject_id": int(event.subject_id),
                    "run": int(event.run),
                    "run_type": event.run_type,
                    "signal_file": event.signal_file,
                    "trial_id": f"{patient.patient_id}_{event.record_id}_{idx:04d}",
                    "event_code": window["event_code"],
                    "task_label": window["task_label"],
                    "label_family": window["label_family"],
                    "label": int(window["label"]),
                    "signal": signal,
                    "channel_names": selected_names,
                    "start_time": window["start_time"],
                    "end_time": window["end_time"],
                    "sample_rate": sfreq,
                }
                if self.compute_stft:
                    from pyhealth.models.tfm_tokenizer import get_stft_torch

                    sample["stft"] = get_stft_torch(signal.unsqueeze(0)).squeeze(0)
                samples.append(sample)
            raw.close()
        return samples

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        return self._base_samples_from_patient(patient)
```

Override `__call__` in `EEGBCIPatternDiscovery`:

```python
    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = self._base_samples_from_patient(patient)
        for sample in samples:
            features = compute_band_powers(
                sample["signal"].detach().cpu().numpy(),
                float(sample["sample_rate"]),
            )
            interpretation = interpret_band_profile(features)
            sample["bandpower"] = features
            sample.update(interpretation)
        return samples
```

- [x] **Step 10: Export tasks**

Modify `pyhealth/tasks/__init__.py`:

```python
from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI
```

Place it near other EEG task exports.

- [x] **Step 11: Run task tests**

Run:

```bash
pytest tests/core/test_eegbci.py::TestEEGBCITasks -v
```

Expected: PASS.

- [x] **Step 12: Commit task 3**

Run:

```bash
git add pyhealth/tasks/eegbci.py pyhealth/tasks/__init__.py tests/core/test_eegbci.py
git commit -m "feat: add EEGBCI tasks"
```

## Task 4: Real-Data Smoke Test

**Files:**

- Modify: `tests/core/test_eegbci.py`

**Interfaces:**

- Consumes: `PYHEALTH_RUN_REAL_EEGBCI=1`.
- Produces: skipped-by-default network/data smoke test.

- [x] **Step 1: Add skipped-by-default smoke test**

Append to `tests/core/test_eegbci.py`:

```python
import os


@unittest.skipUnless(
    os.environ.get("PYHEALTH_RUN_REAL_EEGBCI") == "1",
    "Set PYHEALTH_RUN_REAL_EEGBCI=1 to download and test real EEGBCI data.",
)
class TestEEGBCIRealDataSmoke(unittest.TestCase):
    def test_real_eegbci_subject_1_run_3_pattern_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = EEGBCIDataset(root=tmp, subjects=[1], runs=[3], download=True)
            sample_dataset = dataset.set_task(
                EEGBCIPatternDiscovery(compute_stft=False, window_size=2.0)
            )
            self.assertGreater(len(sample_dataset), 0)
            sample = sample_dataset[0]
            self.assertIn("signal", sample)
            self.assertEqual(sample["signal"].shape[0], 16)
            self.assertIn(sample["task_label"], set(EEGBCI_LABELS))
            self.assertIn("bandpower", sample)
            self.assertIn("brain_state_hypothesis", sample)
```

- [x] **Step 2: Run default tests and confirm skip**

Run:

```bash
pytest tests/core/test_eegbci.py -v
```

Expected: PASS with `TestEEGBCIRealDataSmoke` skipped.

- [x] **Step 3: Run opt-in smoke test when network access is acceptable**

Run:

```bash
PYHEALTH_RUN_REAL_EEGBCI=1 pytest tests/core/test_eegbci.py::TestEEGBCIRealDataSmoke -v
```

Expected: PASS after MNE downloads subject `1`, run `3`.

- [x] **Step 4: Commit task 4**

Run:

```bash
git add tests/core/test_eegbci.py
git commit -m "test: add opt-in EEGBCI real-data smoke test"
```

## Task 5: CELM-Equivalent Example

**Files:**

- Create: `examples/eeg/eegbci/README.md`
- Create: `examples/eeg/eegbci/eegbci_pattern_discovery.py`

**Interfaces:**

- Consumes: `EEGBCIDataset` and `EEGBCIPatternDiscovery`.
- CLI flags: `--root`, `--subjects`, `--runs`, `--output-dir`, `--max-windows`, `--download`.
- Produces: `eegbci_pattern_windows.csv` and `eegbci_pattern_summary.md`.

- [x] **Step 1: Create example script**

Create `examples/eeg/eegbci/eegbci_pattern_discovery.py`:

```python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from pyhealth.datasets import EEGBCIDataset
from pyhealth.tasks import EEGBCIPatternDiscovery


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            items.extend(range(int(start), int(end) + 1))
        else:
            items.append(int(part))
    return items


def sample_to_row(sample: dict) -> dict:
    bandpower = sample["bandpower"]
    return {
        "patient_id": sample["patient_id"],
        "record_id": sample["record_id"],
        "subject_id": sample["subject_id"],
        "run": sample["run"],
        "run_type": sample["run_type"],
        "trial_id": sample["trial_id"],
        "event_code": sample["event_code"],
        "task_label": sample["task_label"],
        "label_family": sample["label_family"],
        "label": sample["label"],
        "start_time": sample["start_time"],
        "end_time": sample["end_time"],
        "dominant_band": bandpower["dominant_band"],
        "alpha_beta_ratio": bandpower["alpha_beta_ratio"],
        "theta_beta_ratio": bandpower["theta_beta_ratio"],
        "brain_state_hypothesis": sample["brain_state_hypothesis"],
        "confidence": sample["confidence"],
        "quality_flags": sample["quality_flags"],
        "interpretation": sample["interpretation"],
        **{key: value for key, value in bandpower.items() if key.endswith("_power")},
        **{key: value for key, value in bandpower.items() if key.endswith("_relative")},
    }


def write_summary(rows: list[dict], path: Path) -> None:
    task_counts = Counter(row["task_label"] for row in rows)
    hypothesis_counts = Counter(row["brain_state_hypothesis"] for row in rows)
    lines = [
        "# EEGBCI Pattern Discovery Summary",
        "",
        "Brain-state hypotheses are exploratory signal metadata, not clinical diagnoses.",
        "",
        f"Processed windows: {len(rows)}",
        "",
        "## Task Labels",
        "",
    ]
    for label, count in task_counts.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Brain-State Hypotheses", ""])
    for label, count in hypothesis_counts.most_common():
        lines.append(f"- {label}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/.cache/pyhealth/eegbci")
    parser.add_argument("--subjects", default="1,2,3")
    parser.add_argument("--runs", default="3-14")
    parser.add_argument("--output-dir", default="outputs/eegbci_pattern_discovery")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EEGBCIDataset(
        root=str(Path(args.root).expanduser()),
        subjects=parse_int_list(args.subjects),
        runs=parse_int_list(args.runs),
        download=args.download,
    )
    sample_dataset = dataset.set_task(EEGBCIPatternDiscovery(compute_stft=False))

    rows = []
    for idx, sample in enumerate(sample_dataset):
        if args.max_windows is not None and idx >= args.max_windows:
            break
        rows.append(sample_to_row(sample))

    csv_path = output_dir / "eegbci_pattern_windows.csv"
    summary_path = output_dir / "eegbci_pattern_summary.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_summary(rows, summary_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Create README**

Create `examples/eeg/eegbci/README.md`:

````markdown
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
````

- [x] **Step 3: Run example on a tiny subset**

Run:

```bash
python examples/eeg/eegbci/eegbci_pattern_discovery.py --subjects 1 --runs 3 --max-windows 20 --download
```

Expected:

```text
Wrote outputs/eegbci_pattern_discovery/eegbci_pattern_windows.csv
Wrote outputs/eegbci_pattern_discovery/eegbci_pattern_summary.md
```

- [x] **Step 4: Commit task 5**

Run:

```bash
git add examples/eeg/eegbci/README.md examples/eeg/eegbci/eegbci_pattern_discovery.py
git commit -m "docs: add EEGBCI pattern discovery example"
```

## Task 6: API Documentation

**Files:**

- Create: `docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst`
- Create: `docs/api/tasks/pyhealth.tasks.eegbci.rst`
- Modify: `docs/api/datasets.rst`
- Modify: `docs/api/tasks.rst`

**Interfaces:**

- Consumes: public classes and helpers from tasks 2 and 3.
- Produces: Sphinx API pages.

- [x] **Step 1: Add dataset API page**

Create `docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst`:

```rst
pyhealth.datasets.EEGBCIDataset
================================

.. autoclass:: pyhealth.datasets.EEGBCIDataset
    :members:
    :undoc-members:
    :show-inheritance:
```

- [x] **Step 2: Add task API page**

Create `docs/api/tasks/pyhealth.tasks.eegbci.rst`:

```rst
pyhealth.tasks.eegbci
=====================

.. automodule:: pyhealth.tasks.eegbci
    :members:
    :undoc-members:
    :show-inheritance:
```

- [x] **Step 3: Include pages in API indexes**

Add the dataset page to the relevant `.. toctree::` in `docs/api/datasets.rst`:

```rst
   datasets/pyhealth.datasets.EEGBCIDataset
```

Add the task page to the relevant `.. toctree::` in `docs/api/tasks.rst`:

```rst
   tasks/pyhealth.tasks.eegbci
```

- [x] **Step 4: Run docs import smoke**

Run:

```bash
python - <<'PY'
from pyhealth.datasets import EEGBCIDataset
from pyhealth.tasks import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI
print(EEGBCIDataset.__name__)
print(EEGMotorImageryEEGBCI.__name__)
print(EEGBCIPatternDiscovery.__name__)
PY
```

Expected:

```text
EEGBCIDataset
EEGMotorImageryEEGBCI
EEGBCIPatternDiscovery
```

- [x] **Step 5: Commit task 6**

Run:

```bash
git add docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst docs/api/tasks/pyhealth.tasks.eegbci.rst docs/api/datasets.rst docs/api/tasks.rst
git commit -m "docs: add EEGBCI API docs"
```

## Final Verification

Run after all tasks:

```bash
pytest tests/core/test_eegbci.py -v
python - <<'PY'
from pyhealth.datasets import EEGBCIDataset
from pyhealth.tasks import EEGBCIPatternDiscovery, EEGMotorImageryEEGBCI
print("imports ok")
PY
```

Optional network/data verification:

```bash
PYHEALTH_RUN_REAL_EEGBCI=1 pytest tests/core/test_eegbci.py::TestEEGBCIRealDataSmoke -v
python examples/eeg/eegbci/eegbci_pattern_discovery.py --subjects 1 --runs 3 --max-windows 20 --download
```

Graph update after code changes:

```bash
graphify update .
```

## Engineering Review

Review mode: GStack `/plan-eng-review`
Review date: 2026-07-07
Review inputs: `brainstorm.md`, `design.md`, existing PyHealth EEG dataset/task files, current plan.

### Architecture

Decision: keep the dataset as metadata-only and put EDF reading/windowing in tasks.

Reason: this matches `pyhealth/datasets/tuab.py`, `pyhealth/datasets/tuev.py`, and `pyhealth/tasks/temple_university_EEG_tasks.py`. It also keeps MNE Raw objects out of cached dataset metadata.

### Data Flow

Risk found: the previous draft did not specify whether annotations are windowed before or after resampling.

Resolution: read/filter/resample first, then window from `raw.annotations` using the post-resample `raw.info["sfreq"]`. MNE keeps annotation onsets in seconds, so sample indices must be computed only after the final sample rate is known.

### Channel Strategy

Risk found: defaulting to all 64 EEGBCI channels would break many existing EEG model assumptions.

Resolution: default `channel_mode="compat16"` with named central motor channels. Allow `channel_mode="all"` for research use. Do not reuse TUAB/TUEV bipolar montage code because EEGBCI channel names and montage semantics differ.

### Label Semantics

Risk found: `T1` and `T2` are easy to decode incorrectly because their meaning depends on run number.

Resolution: keep `task_label_for_event(run, event_code)` as a pure helper with direct unit tests. Do not inline this mapping in task code.

### Edge Cases

- Missing local EDF with `download=False`: raise `FileNotFoundError` that explicitly suggests `download=True`.
- Unsupported run outside `3-14`: raise `ValueError`.
- Non-`T0`/`T1`/`T2` annotations: skip.
- Annotation shorter than `window_size`: emit no sample.
- Last partial window inside an annotation: skip to preserve fixed tensor shapes.
- Missing compatibility channels: raise `ValueError` listing missing channel names.
- `resample_rate=None`: preserve original sample rate and record it in each sample.
- `compute_stft=False`: remove `stft` from `input_schema`.
- Normal CI: never runs the real-data smoke test unless `PYHEALTH_RUN_REAL_EEGBCI=1`.

### Test Coverage

The plan covers:

- Run-aware labels.
- Channel selection.
- Bandpower on synthetic alpha and beta sinusoids.
- Interpretation metadata and non-clinical wording.
- Dataset metadata generation with local files and mocked MNE download.
- Task sample schema with MNE `RawArray`.
- Skipped-by-default real data test.
- Example smoke path.

Remaining risk: MNE's real EEGBCI channel labels may vary slightly from the expected names. The opt-in real-data smoke test is the guardrail for that. If it fails, adjust `normalize_eegbci_channel_name` rather than weakening the default channel contract.

### Performance

The first implementation reads EDF files inside task execution, matching current PyHealth EEG tasks. This is acceptable for the small example and keeps the first pass simple. If users process many subjects/runs repeatedly, add a later cache for preprocessed windows rather than doing it in this first feature.

### Scope Decision

Approved first scope: Approach B only.

Do not include the optional embedding comparison in the initial implementation. It depends on model/checkpoint/channel compatibility decisions that should be made after the core dataset and task path works on real data.

## Progress Log

- 2026-07-07: Converted draft requirements/design into a concrete TDD implementation plan using `superpowers:writing-plans`.
- 2026-07-07: Ran automatic engineering review and incorporated decisions on dataset/task boundaries, channel strategy, annotation timing, edge cases, and test coverage.
- 2026-07-07: Task 1 complete on branch `eegbci-pattern-discovery`; helper tests pass with `.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIHelpers -v`.
- 2026-07-07: Task 2 complete; dataset metadata tests pass with `.venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIDataset -v`.
- 2026-07-07: Task 3 complete; `.venv/bin/python -m pytest tests/core/test_eegbci.py -v` passes with 21 tests.
- 2026-07-07: Task 4 complete; normal EEGBCI tests pass with 21 passed/1 skipped, and `PYHEALTH_RUN_REAL_EEGBCI=1 .venv/bin/python -m pytest tests/core/test_eegbci.py::TestEEGBCIRealDataSmoke -v` passes against subject 1 run 3.
- 2026-07-07: Task 5 complete; `.venv/bin/python examples/eeg/eegbci/eegbci_pattern_discovery.py --subjects 1 --runs 3 --max-windows 20 --download` writes a 20-row CSV and Markdown summary.
- 2026-07-07: Task 6 complete; docs import smoke prints `EEGBCIDataset`, `EEGMotorImageryEEGBCI`, and `EEGBCIPatternDiscovery`.
- 2026-07-07: Final verification complete; default EEGBCI tests pass with 21 passed/1 skipped, import smoke prints `imports ok`, opt-in real-data smoke passes, the example writes verified 20-row artifacts, and `graphify update .` refreshed the code graph.
