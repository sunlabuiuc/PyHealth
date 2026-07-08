# Design: EEG Pattern Discovery in PyHealth

Generated: 2026-07-07
Branch: master
Repo: PyHealth
Status: APPROVED FOR IMPLEMENTATION
Mode: Builder / research

## Problem Statement

Migrate the CELM EEG pattern discovery idea into PyHealth's architecture.

The original question is still the right one:

> Can frequency-based EEG patterns reveal moment-level brain-state hypotheses that task or clinical labels do not fully capture?

In CELM, this was a standalone research artifact. In PyHealth, it should become a reusable EEG dataset/task path, with examples that can train, embed, cluster, and summarize real EEG windows.

High-level summary: We are turning the standalone CELM EEG pattern-discovery pipeline into a reusable PyHealth dataset, task, and example for real PhysioNet EEGBCI motor movement/imagery data. The research question is whether simple frequency profiles in short EEG windows can surface moment-level brain-state hypotheses that task labels alone miss. The practical problem is that PyHealth has EEG models and task infrastructure, but no first-class EEGBCI path that produces labeled windows, bandpower features, cautious interpretation metadata, and real-data validation without forcing normal CI to download raw EDF files.

## Source Artifact

The CELM artifact lives at `/Users/vihaanagrawal/Research/CELM/eeg_pattern_discovery`.

It is a real EEG pipeline, not just mocked signal processing:

- Dataset: PhysioNet EEG Motor Movement/Imagery through MNE EEGBCI.
- Subjects: `1`, `2`, `3`.
- Runs: `3-14`.
- Windowing: full 2-second windows inside MNE annotations.
- Signal processing: MNE EDF loading, standardization, EEG picking, `0.5-45 Hz` filtering, Welch PSD.
- Output: checked-in CSV with 2,160 processed segments.
- Models: no pretrained model or checkpoint. The original is classical signal processing plus deterministic interpretation rules.

That last point matters. The PyHealth expansion should not pretend CELM already used pretrained models. It should add them as the bigger creative layer.

## Why PyHealth Is A Good Fit

PyHealth already has the right abstractions:

- EEG datasets: `TUABDataset`, `TUEVDataset`, `SleepEDFDataset`, `SHHSDataset`, `ISRUCDataset`.
- EEG tasks: `EEGEventsTUEV`, `EEGAbnormalTUAB`, `SleepStagingSleepEDF`.
- EEG models: `BIOT`, `ContraWR`, `SparcNet`, `TFMTokenizer`.
- Calibration examples under `examples/conformal_eeg/`.

The migration should follow the existing pattern:

1. Dataset class builds metadata rows pointing at raw signal files.
2. Task class reads EDF, preprocesses signals, windows annotations, and returns PyHealth samples.
3. Examples run actual research workflows.
4. Tests mock network/EDF boundaries by default, with optional real-data smoke tests.

## Premises

1. EEGBCI should be a first-class PyHealth dataset, not only a script in `examples/`.
2. The default pattern-discovery window should stay 2 seconds to preserve the CELM research question.
3. `T1` and `T2` labels must be decoded using the run number. They do not mean the same thing in every EEGBCI run.
4. Brain-state hypotheses are heuristic metadata, not clinical labels.
5. Normal CI should not download PhysioNet data. Real-data coverage should be opt-in.
6. Channel adaptation is part of the API design. EEGBCI is 64-channel/160 Hz; several PyHealth EEG model examples assume 16 channels and often 200 Hz.

## Recommended Approach

Build the core PyHealth integration first, then add the creative atlas.

### Layer 1: Core PyHealth Integration

Add:

- `pyhealth/datasets/eegbci.py`
- `pyhealth/datasets/configs/eegbci.yaml`
- `pyhealth/tasks/eegbci.py`
- exports in `pyhealth/datasets/__init__.py` and `pyhealth/tasks/__init__.py`
- tests under `tests/core/`
- documentation under `docs/api/`

Core classes:

- `EEGBCIDataset`
- `EEGMotorImageryEEGBCI`
- `EEGBCIPatternDiscovery`

The dataset should produce metadata. The task should do EDF reading, annotation parsing, windowing, channel handling, optional STFT, bandpower extraction, and interpretation metadata.

### Layer 2: Small Optional Research Layer

Add examples under `examples/eeg/eegbci/`:

- `eegbci_pattern_discovery.py`: reproduce the CELM CSV/Markdown artifact using PyHealth objects.
- optional `eegbci_embedding_comparison.py`: extract embeddings from one compatible EEG model and compare them with CELM bandpower hypotheses.

Keep this small. The goal is not to create a large EEG research platform. The optional comparison should answer one question:

> Do learned EEG embeddings group windows in a way that agrees with, sharpens, or contradicts the simple bandpower hypotheses?

It can compare:

- CELM bandpower hypotheses.
- Supervised task labels.
- One pretrained or learned model embedding.

## Model Boundary

The first EEGBCI pattern-discovery pipeline does not require a neural model. It uses real EEGBCI data, MNE preprocessing, Welch bandpower features, and deterministic interpretation rules. The supervised `EEGMotorImageryEEGBCI` task should make model training possible through normal PyHealth models, and the dataset/task contract should be compatible with BIOT, ContraWR, SparcNet, and TFMTokenizer where channel and sampling assumptions fit. Pretrained model embeddings are a second-stage research layer, not a dependency for the first implementation.

## Analysis Stage

The first analysis stage should be a CELM-equivalent sample-level and aggregate report. It should convert `EEGBCIPatternDiscovery` samples into a CSV with task labels, bandpower features, ratios, hypotheses, confidence, and quality flags, then write a Markdown summary grouped by task label and inferred hypothesis. This answers whether frequency-profile hypotheses line up with, sharpen, or disagree with the experimental labels. The later atlas stage can add model embeddings and clustering, but the first pass should prove the dataset/task/analysis path end to end with real EEGBCI data.

## Dataset Contract

Recommended constructor:

```python
dataset = EEGBCIDataset(
    root="~/.cache/pyhealth/eegbci",
    subjects=[1, 2, 3],
    runs=list(range(3, 15)),
    download=True,
)
```

Recommended metadata columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | subject key, for example `S001` |
| `record_id` | run key, for example `R03` |
| `subject_id` | integer EEGBCI subject id |
| `run` | integer EEGBCI run id |
| `run_type` | baseline, motor execution, or motor imagery subtype |
| `signal_file` | local EDF path downloaded by MNE |
| `source` | `physionet_eegbci` |

`download=True` may call `mne.datasets.eegbci.load_data`. `download=False` should require local files/metadata and fail clearly if they are missing.

## Task Contracts

### `EEGMotorImageryEEGBCI`

Purpose: supervised task-label prediction.

Sample shape:

```python
{
    "patient_id": "S001",
    "signal_file": ".../S001R03.edf",
    "run": 3,
    "trial_id": "S001_R03_0001",
    "event_code": "T1",
    "task_label": "execute_left_fist",
    "label_family": "motor_execution",
    "label": 1,
    "signal": Tensor[C, T],
    "stft": Tensor[C, F, TT],  # if compute_stft=True
}
```

### `EEGBCIPatternDiscovery`

Purpose: exploratory moment-level signal interpretation.

Sample shape extends the supervised sample with:

```python
{
    "bandpower": {
        "delta_power": ...,
        "theta_power": ...,
        "alpha_power": ...,
        "beta_power": ...,
        "gamma_power": ...,
        "delta_relative": ...,
        "theta_relative": ...,
        "alpha_relative": ...,
        "beta_relative": ...,
        "gamma_relative": ...,
        "dominant_band": "alpha",
        "alpha_beta_ratio": ...,
        "theta_beta_ratio": ...,
    },
    "brain_state_hypothesis": "relaxed_or_idle",
    "confidence": "medium",
    "quality_flags": "low_confidence",
    "interpretation": "The segment is alpha-dominant...",
}
```

The discovery task should keep `task_label` and `label` available so PyHealth training, evaluation, and embedding extraction remain usable.

## Channel And Sampling Strategy

Default recommendation:

- Preserve `original_sample_rate=160` as metadata.
- Resample to `200 Hz` by default for compatibility with existing PyHealth EEG models.
- Offer `resample_rate=None` to keep the original signal.
- Default to a stable 16-channel adapter for pretrained-model compatibility.
- Offer `channel_mode="all"` for 64-channel experiments with compatible models.

Reasoning:

- BIOT can be configured for different channel counts, but pretrained 18-channel weights are not directly compatible with raw 64-channel EEGBCI.
- ContraWR and SparcNet are structurally friendlier to raw tensors, but still need sane window length and channel expectations.
- TFMTokenizer has stronger assumptions: it expects `signal` and `stft`, has 200 Hz-ish temporal assumptions, and the classifier currently uses a 16-channel embedding table.

Do not reuse TUAB/TUEV bipolar montage functions. They hard-code TUH channel names.

## Real Dataset Testing

Use three levels:

1. Unit tests, always on:
   - run-aware `T0`/`T1`/`T2` label mapping
   - fixed-window segmentation
   - bandpower feature keys and rough values on synthetic sinusoids
   - interpretation rules
   - dataset metadata generation with mocked MNE download
   - task sample schema with mocked EDF/Raw object

2. Fixture test, always on if fixture is accepted:
   - a tiny `.npz` with one or two extracted windows from EEGBCI
   - verifies real-shaped signal arrays without downloading data

3. Real-data smoke test, opt-in:
   - gated by `PYHEALTH_RUN_REAL_EEGBCI=1`
   - downloads subject `1`, run `3`
   - verifies at least one real window, signal tensor shape, decoded task label, and bandpower metadata

This is the right answer to the current testing concern. Mocking is fine for normal CI, but there should be one real-data path for the thing mocks cannot prove.

## Approaches Considered

### Approach A: Example-Only Port

Fastest. Add only an example script that wraps the CELM pipeline in PyHealth imports.

This proves the idea quickly, but it is not really PyHealth-format. It is a demo living inside the repo.

### Approach B: Dataset + Task + Example

Add EEGBCI as a dataset, add supervised and discovery tasks, then add a runnable example.

This is the recommended path. It fits PyHealth and creates reusable substrate for later EEG research.

### Approach C: EEG Representation Atlas

Build Approach B, then add a clustering/embedding report that compares bandpower hypotheses with BIOT, ContraWR, SparcNet, or TFMTokenizer embeddings.

This is the most interesting research artifact, but it is larger than the desired scope right now. Treat it as a later idea, not the current implementation target.

## Success Criteria

- `EEGBCIDataset` can build metadata for selected subjects/runs.
- `EEGMotorImageryEEGBCI` returns real labeled EEG windows.
- `EEGBCIPatternDiscovery` reproduces the CELM-style bandpower and interpretation schema.
- Default tests do not require network access.
- Optional real-data smoke test validates MNE/PhysioNet behavior.
- Example writes a CSV and Markdown summary from real EEGBCI data.
- Optional embedding comparison can run with at least one compatible model path.
- Documentation states that brain-state hypotheses are exploratory and non-clinical.

## Resolved Decisions

1. Stop the first implementation at dataset, tasks, tests, docs, and CELM-equivalent example. Defer the embedding comparison.
2. Use 16-channel compatibility as the default channel mode. Offer `channel_mode="all"` for 64-channel experiments.
3. Keep using `mne` as a normal project dependency. `pyproject.toml` already declares `mne~=1.10.0`, and existing EEG tasks import it directly.
4. Do not commit a tiny real-signal `.npz` fixture in the first pass. Use offline mocked tests plus the opt-in `PYHEALTH_RUN_REAL_EEGBCI=1` smoke test.
5. Treat the CELM-equivalent CSV and Markdown generator as the first analysis stage. Do not require a neural model for that stage.

## Recommendation

Implement Approach B first.

Stop there unless the implementation feels too small. If adding a creative layer, add only a tiny one-model embedding comparison, not a full atlas.

## Planning Update

2026-07-07: The approved design has been converted into a concrete implementation plan at `docs/eeg_pattern_discovery/implementation_plan.md`. GStack `/plan-eng-review` reviewed the plan and locked the first implementation scope to Approach B: dataset, tasks, tests, docs, and CELM-equivalent example. The embedding comparison remains deferred.
