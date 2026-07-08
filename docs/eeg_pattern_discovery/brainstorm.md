# EEG Pattern Discovery Migration Brainstorm

Date: 2026-07-07
Status: living brainstorm and design scratchpad
Mode: Builder / research

## Source Context

The original CELM artifact lives at `/Users/vihaanagrawal/Research/CELM/eeg_pattern_discovery`.

The prior design note is `/Users/vihaanagrawal/.gstack/projects/office-hours-users-vihaanagrawal-gstack-repos/vihaanagrawal-unknown-design-20260629-143914.md`.

The target project is `/Users/vihaanagrawal/Research/PyHealth`.

Useful external references:

- PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0: https://physionet.org/content/eegmmidb/
- MNE EEGBCI loader: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html

## Original CELM Idea

The CELM project asks whether frequency-based EEG patterns reveal moment-level brain-state hypotheses that clinical or experimental labels do not fully capture. It uses MNE's PhysioNet EEG Motor Movement/Imagery loader, processes subjects `1`, `2`, and `3`, runs `3-14`, creates 2-second labeled windows, computes Welch band powers, and assigns cautious deterministic interpretations.

The core output is a row-per-window table with:

- subject, run, event code, task label, and label family
- delta, theta, alpha, beta, and gamma absolute power
- relative band powers
- dominant band
- alpha/beta and theta/beta ratios
- moment-level hypothesis
- confidence and quality flags
- plain-English interpretation

The interpretation layer is explicitly exploratory. It must say that a signal pattern is consistent with a state, not that it proves the subject's cognition or a clinical diagnosis.

CELM code shape:

- `run_analysis.py`: orchestrates subject/run processing, writes CSV and Markdown summary.
- `src/data.py`: downloads PhysioNet EEGBCI with MNE, reads EDF, standardizes channels, filters, and yields labeled 2-second windows.
- `src/labels.py`: run-aware mapping for `T0`, `T1`, and `T2`.
- `src/features.py`: Welch PSD bandpower extraction.
- `src/interpretation.py`: deterministic frequency-profile hypothesis engine.
- `src/report.py`: summary grouped by task label and inferred hypothesis.
- `tests/test_labels.py` and `tests/test_interpretation.py`: fast unit tests for pure logic.

Important correction to the phrase "same pretrained models": the CELM artifact does not appear to use pretrained EEG models. It uses a real pretrained ecosystem only in the loose sense that MNE downloads and parses an established public dataset. The PyHealth migration can expand the idea by adding pretrained PyHealth EEG embeddings from BIOT, ContraWR, SparcNet, and TFMTokenizer on top of the CELM bandpower baseline.

## Real Dataset Answer

Yes, there is a clean way to stop relying only on mocked filesystem and signal-processing tests.

Use a two-tier test strategy:

1. Fast unit tests stay synthetic and mocked. They should cover label decoding, event-window slicing, bandpower output shape, interpretation rules, metadata CSV generation, and schema contracts. These run in normal CI.
2. Real-data smoke tests run behind an explicit opt-in flag, for example `PYHEALTH_RUN_REAL_EEGBCI=1`. They download a tiny EEGBCI subset with MNE, for example subject `1`, run `3`, create real windows, compute real bandpowers, and verify at least one sample has a real signal tensor and interpretation metadata.

Why this split works:

- Normal CI should not hit PhysioNet or download EDF files.
- The actual example should use real EEGBCI data end-to-end.
- The optional smoke test catches the exact class of bugs mocks miss: MNE annotation names, channel names, EDF loading, sampling frequency, and label boundaries.

Existing PyHealth EEG datasets include `TUABDataset`, `TUEVDataset`, `SleepEDFDataset`, `SHHSDataset`, and `ISRUCDataset`. Those are real EEG datasets, but they do not match the CELM project as closely as EEGBCI. TUAB/TUEV are better for clinical abnormal/event detection; SleepEDF is better for sleep staging; EEGBCI is the clean fit for motor execution/imagery and the CELM label semantics.

## PyHealth Fit

PyHealth already has EEG datasets, tasks, examples, and models. The migration should use PyHealth's current `BaseDataset -> BaseTask -> SampleDataset -> Trainer` flow, not a standalone script folder.

Relevant PyHealth conventions:

- dataset classes live in `pyhealth/datasets/`
- dataset metadata is represented as `*-pyhealth.csv` files
- dataset config YAML files live in `pyhealth/datasets/configs/`
- tasks live in `pyhealth/tasks/`
- examples live under `examples/eeg/` or `examples/conformal_eeg/`
- unit tests live under `tests/core/`
- exports are added in `pyhealth/datasets/__init__.py` and `pyhealth/tasks/__init__.py`

Existing EEG patterns to follow:

- `pyhealth/datasets/tuab.py`
- `pyhealth/datasets/tuev.py`
- `pyhealth/datasets/sleepedf.py`
- `pyhealth/tasks/temple_university_EEG_tasks.py`
- `examples/eeg/eeg_models/`
- `examples/conformal_eeg/`

Important local gotchas:

- `TUEVDataset` and `TUABDataset` build `*-pyhealth.csv` metadata files from EDF paths, then `BaseDataset` reads those tables.
- `EEGEventsTUEV` and `EEGAbnormalTUAB` read EDF inside the task. That is the pattern to copy for EEGBCI.
- Current EEG tests mock raw EDF reading and signal conversion heavily. That is acceptable for CI, but not enough for this feature's confidence.
- TUEV/TUAB tasks normalize to 16 channels for model compatibility. EEGBCI starts as 64 channels, so channel adaptation is a first-class design decision, not a small detail.

## Likely Migration Shape

Add a first-class EEGBCI dataset and tasks:

- `pyhealth/datasets/eegbci.py`
- `pyhealth/datasets/configs/eegbci.yaml`
- `pyhealth/tasks/eegbci.py`
- `pyhealth/tasks/eeg_pattern_discovery.py` or helper module under `pyhealth/tasks/eegbci.py`
- `tests/core/test_eegbci.py`
- `tests/core/test_eeg_pattern_discovery.py`
- `examples/eeg/eegbci/`

The dataset should represent each downloaded subject/run EDF as metadata:

- `patient_id`
- `record_id`
- `subject_id`
- `run`
- `run_type`
- `signal_file`
- `source`
- `sfreq` if cheaply known, otherwise computed by the task

The task should parse annotations and emit window-level samples:

- `patient_id`
- `record_id`
- `signal_file`
- `run`
- `run_type`
- `trial_id`
- `event_code`
- `label`
- `task_label`
- `label_family`
- `signal`
- `start_time`
- `end_time`
- optional `stft`
- optional `bandpower`
- optional frequency interpretation metadata

### Proposed Dataset Contract

`EEGBCIDataset` should be the dataset wrapper for the PhysioNet EEG Motor Movement/Imagery Dataset.

Recommended constructor:

```python
dataset = EEGBCIDataset(
    root="~/.cache/pyhealth/eegbci",
    subjects=[1, 2, 3],
    runs=list(range(3, 15)),
    download=True,
)
```

Recommended metadata rows:

| Column | Meaning |
| --- | --- |
| `patient_id` | stable subject key, for example `S001` |
| `record_id` | run key, for example `R03` |
| `subject_id` | integer EEGBCI subject id |
| `run` | integer EEGBCI run id |
| `run_type` | baseline, motor execution left/right, motor imagery left/right, motor execution fists/feet, motor imagery fists/feet |
| `signal_file` | local EDF path downloaded by MNE |
| `source` | `physionet_eegbci` |

`download=False` should require files to already exist and should fail clearly if metadata cannot be built. `download=True` can call `mne.datasets.eegbci.load_data(...)`.

### Proposed Task Contracts

1. `EEGMotorImageryEEGBCI`
   - Purpose: supervised classification task using real EEGBCI event labels.
   - Input: raw `signal` tensor, optional `stft`.
   - Output: `label` as multiclass task label.
   - Windowing: fixed windows inside each annotation, default 2 seconds for parity with CELM.

2. `EEGPatternDiscoveryEEGBCI`
   - Purpose: exploratory moment-level pattern discovery.
   - Input: raw `signal` tensor and computed `bandpower` tensor/dict.
   - Output: `brain_state_hypothesis` as metadata or optional multiclass label.
   - Extra fields: `dominant_band`, `alpha_beta_ratio`, `theta_beta_ratio`, `confidence`, `quality_flags`, `interpretation`.

Keep these separate. Supervised task-label prediction and exploratory frequency interpretation are different jobs. Mixing them into one task will make the API confusing.

## Model Reuse

Existing EEG-capable models:

- `BIOT`: consumes raw `signal`, has `get_embeddings()` and `load_pretrained_weights()`.
- `ContraWR`: consumes one signal tensor, computes STFT internally, supports `embed=True`.
- `SparcNet`: consumes one signal tensor, supports `embed=True`.
- `TFMTokenizer`: uses `signal` and `stft`, supports pretrained/token workflows.

Important model mismatch:

EEGBCI is commonly 64-channel EEG, while several PyHealth EEG examples and pretrained checkpoints expect 16 or 18 channels. The design must choose a channel adaptation strategy:

- select a stable 16-channel 10-20 subset
- regional average/pool EEGBCI channels into a 16-channel montage
- add a small channel adapter model
- run 64-channel-compatible models only in the first version

The conservative PyHealth-aligned first version is likely 16-channel selection, because it keeps existing pretrained EEG models usable.

Recommended first channel strategy:

1. Preserve the full 64-channel signal in metadata or a configurable task mode.
2. Default the model-facing task output to a stable 16-channel 10-20 subset or montage compatible with existing PyHealth EEG models.
3. Document the selected channels and make the adapter function pure and testable.

Do not hide this inside model code. Channel mapping belongs near EEGBCI task preprocessing so every model sees the same input contract.

## Bigger Ideas

### 1. EEGBCI Moment Discovery Benchmark

Create a PyHealth benchmark where the same EEGBCI windows can be used for supervised task-label prediction and CELM-style frequency-pattern discovery. Compare bandpower-only features, BIOT embeddings, ContraWR embeddings, SparcNet embeddings, and TFM token features.

### 2. Pretrained Embedding Atlas

Use pretrained EEG model embeddings to cluster windows into learned neural motifs. Summarize clusters by subject, run family, task label, CELM hypothesis, and quality flags. This turns the project from rules-only interpretation into representation discovery.

### 3. TFM Token Report Cards

Use TFMTokenizer to extract discrete token patterns from EEG windows. Report which token motifs are enriched in rest, execution, imagery, artifact-like gamma, and slow-wave-heavy windows.

### 4. EEG Model Cards

For each evaluated window or cluster, generate a compact model card with prediction, confidence, band profile, model embedding neighborhood, salient channels/time regions where available, and caution flags.

### 5. Subject-Shift Reliability Suite

Use PyHealth's calibration/conformal examples to measure how well models and hypotheses transfer across subjects and run families. Report uncertain windows, artifact-heavy subjects, and subject-specific drift.

### 6. Brain-State Atlas Explorer

Generate a static Markdown/CSV/HTML artifact that treats each cluster as a "neural motif." Each motif gets:

- cluster size
- dominant CELM hypothesis
- top task labels
- subject/run enrichment
- representative windows
- model confidence spread
- quality flags

This is the most demoable bigger idea. It turns "we computed band powers" into "we discovered recurring motifs and can inspect where they appear."

### 7. Label Disagreement Mining

Rank windows where the experimental task label and model/frequency evidence disagree:

- task says rest but embedding neighbors look like motor execution
- task says imagery but bandpower looks artifact-heavy
- model predicts left/right confidently while bandpower hypothesis is low confidence
- cluster is subject-specific rather than task-specific

This is useful because the real research question is not just classification accuracy. It is whether moment-level signal structure contains information labels miss.

### 8. Real-Data Regression Fixture

Add a tiny recorded fixture generated from one real EEGBCI run, not raw private data:

- a small `.npz` containing one or two already-extracted 2-second windows
- expected label metadata
- expected bandpower keys and rough numerical ranges

This gives CI a real-signal-ish path without downloading PhysioNet. The full real-data smoke test remains opt-in.

## Open Design Decisions

1. Minimum viable contribution versus larger research module.
2. Channel adaptation strategy for EEGBCI to pretrained EEG models.
3. Whether bandpower features are part of `input_schema` or extra metadata.
4. Whether the default task predicts task labels, moment-state hypotheses, or both.
5. How much generated analysis output should be included in the repo versus created by examples.
6. Whether `EEGPatternDiscoveryEEGBCI` should be a PyHealth task, a task helper, or an example-only analysis layer.
7. Whether MNE should become a required dependency for this dataset or stay behind an EEG extra.

## Approaches Considered

### Approach A: Minimal Example-Only Port

Summary: Add `examples/eeg/eegbci/eeg_pattern_discovery.py` that imports CELM logic adapted to PyHealth imports, downloads EEGBCI through MNE, and writes CSV/Markdown outputs.

Effort: S
Risk: Low

Pros:

- Fastest way to get real EEGBCI data flowing.
- Small surface area.
- Good for proving that the CELM idea still works inside the PyHealth repo.

Cons:

- Not really "in PyHealth format."
- Harder for users to reuse in PyHealth training/calibration workflows.
- Tests still mostly sit around the example, not the library.

### Approach B: First-Class EEGBCI Dataset + Pattern Task

Summary: Add `EEGBCIDataset`, `EEGMotorImageryEEGBCI`, and `EEGPatternDiscoveryEEGBCI`, then add examples that use both the supervised labels and the exploratory frequency hypotheses.

Effort: M
Risk: Medium

Pros:

- Matches PyHealth architecture.
- Lets EEGBCI become reusable for future EEG work.
- Cleanly supports real-data examples and optional smoke tests.

Cons:

- Requires careful metadata generation and MNE dependency handling.
- Needs a clear channel adapter story.
- Slightly bigger API commitment.

### Approach C: EEG Representation Atlas

Summary: Build Approach B, then add an atlas example that extracts embeddings from BIOT, ContraWR, SparcNet, or TFMTokenizer, clusters windows, and reports neural motifs alongside CELM bandpower hypotheses.

Effort: L
Risk: Medium-High

Pros:

- The most creative version.
- Uses PyHealth's pretrained EEG model story.
- Produces a research artifact that is more interesting than a CSV.

Cons:

- Pretrained checkpoint availability and channel mismatch can slow this down.
- Clustering can become arbitrary if success criteria are vague.
- Needs careful caveats so it does not overclaim cognition.

## Current Recommendation

Use a two-layer design:

1. Core PyHealth integration: `EEGBCIDataset`, `EEGMotorImagery`, and `EEGBCIPatternDiscovery` with CELM bandpower and interpretation helpers.
2. Small creative layer: one optional example that compares CELM bandpower hypotheses with embeddings from one compatible PyHealth EEG model.

This keeps the core contribution maintainable while expanding the idea only a little beyond the original deterministic CSV/report artifact.

More concrete recommendation: choose Approach B as the implementation baseline. Do not build the full Approach C atlas unless the core migration feels too small after it works.

## Implementation Plan Sketch

### Task 1: Extract Pure EEGBCI Utilities

Create run-aware label mapping, event-window slicing, channel selection, and bandpower functions as testable pure helpers. These should be adapted from CELM, not copied blindly.

Likely files:

- `pyhealth/tasks/eegbci.py`
- `tests/core/test_eegbci.py`

Verification:

- unit tests for all EEGBCI run/event mappings
- unit tests that windows do not cross annotation boundaries
- unit tests for bandpower keys and quality flags

### Task 2: Add `EEGBCIDataset`

Implement dataset metadata generation around MNE's EEGBCI downloader.

Likely files:

- `pyhealth/datasets/eegbci.py`
- `pyhealth/datasets/configs/eegbci.yaml`
- `pyhealth/datasets/__init__.py`
- `docs/api/datasets.rst`
- `docs/api/datasets/pyhealth.datasets.EEGBCIDataset.rst`

Verification:

- synthetic metadata tests with mocked `mne.datasets.eegbci.load_data`
- no network calls in default CI

### Task 3: Add Supervised and Discovery Tasks

Add a supervised EEGBCI motor task and an exploratory pattern discovery task.

Likely files:

- `pyhealth/tasks/eegbci.py`
- `pyhealth/tasks/__init__.py`
- `docs/api/tasks/pyhealth.tasks.eegbci.rst`

Verification:

- mocked EDF tests for task sample schema
- pure helper tests for interpretation logic
- optional real-data smoke test behind `PYHEALTH_RUN_REAL_EEGBCI=1`

### Task 4: Add Real Examples

Create examples that a user can run locally.

Likely files:

- `examples/eeg/eegbci/eegbci_pattern_discovery.py`
- optional: `examples/eeg/eegbci/eegbci_embedding_comparison.py`
- `examples/eeg/eegbci/README.md`

Verification:

- example supports `--subjects 1 --runs 3 --max-windows 20`
- output CSV and Markdown summary are generated outside the source tree by default

### Task 5: Add Optional Tiny Pretrained Embedding Comparison

Add one small embedding comparison where channel shape permits. This should not become a multi-model benchmark, dashboard, or large clustering framework.

Likely files:

- `examples/eeg/eegbci/eegbci_embedding_comparison.py`

Verification:

- embedding extraction works with a toy model or one compatible PyHealth EEG model
- real pretrained checkpoint paths are CLI arguments, not hardcoded local paths

## Success Criteria

- PyHealth exposes `EEGBCIDataset`.
- PyHealth exposes at least one EEGBCI task that returns real 2-second windows from EDF annotations.
- The CELM label-mapping mistake risk is handled: `T1`/`T2` are decoded using run number.
- Bandpower and interpretation metadata can be produced for every window.
- Unit tests do not require network access.
- An opt-in smoke test runs against real MNE-downloaded EEGBCI data.
- A runnable example produces a CSV and Markdown summary similar to CELM.
- The optional expanded example can compare CELM bandpower hypotheses with pretrained or learned embeddings without becoming a full research platform.

## Premises To Validate Before Coding

1. EEGBCI should be added as a first-class PyHealth dataset, not just an example script.
2. The default EEGBCI task should emit 2-second windows because preserving the CELM question matters more than matching TUAB/TUEV window lengths.
3. Channel adaptation should be explicit and testable, with a conservative 16-channel default for pretrained PyHealth EEG models.
4. Real-data confidence should come from an opt-in smoke test plus runnable examples, not from making normal CI download PhysioNet data.
5. The discovery interpretation must remain cautious and non-clinical.

## Next Question

The main fork in the road is scope:

- If the goal is a quick migration, implement Approach A first.
- If the goal is "get it into PyHealth format," implement Approach B first.
- If the goal is a small but memorable research demo, implement Approach B and one tiny embedding comparison.

Recommendation: Approach B first. Add only the smallest embedding comparison if it clearly improves the story.
