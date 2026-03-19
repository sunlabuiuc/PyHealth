# Med-Flamingo Integration Todo

This checklist reflects the current branch state as of 2026-03-19 and is meant to help split work across teammates.

## Status Summary

- The repo already contains a Med-Flamingo MVP.
- The MVP appears to support VQA-RAD and Path-VQA loading, a generative VQA task, a Med-Flamingo wrapper, custom generative evaluation, and LoRA fine-tuning utilities.
- The biggest remaining gaps are native PyHealth trainer integration, paper-faithful leakage control, Visual USMLE support, and dependency/documentation cleanup.
- Important: this status is based on code inspection. Full validation was not completed in the current shell because core dependencies were missing locally.

## Believed Done

- [x] Add VQA-RAD dataset wrapper.
  - File: `pyhealth/datasets/vqa_rad.py`
  - Includes annotation autodetection, split normalization, metadata CSV generation, optional clean split overrides, and basic dedup hooks.

- [x] Add Path-VQA dataset wrapper.
  - File: `pyhealth/datasets/path_vqa.py`
  - Includes annotation autodetection, split normalization, metadata CSV generation, optional clean split overrides, and basic dedup hooks.

- [x] Add dataset config files for both VQA datasets.
  - Files:
    - `pyhealth/datasets/configs/vqa_rad.yaml`
    - `pyhealth/datasets/configs/path_vqa.yaml`

- [x] Add a generative medical VQA task.
  - File: `pyhealth/tasks/generative_vqa.py`
  - Emits raw `image_path`, `question`, `answer`, `split`, `question_id`, `image_id`, and `dataset` fields.

- [x] Add a Med-Flamingo model wrapper.
  - File: `pyhealth/models/med_flamingo.py`
  - Includes:
    - OpenFlamingo model construction
    - Med-Flamingo checkpoint download/loading
    - Few-shot prompt building
    - Generation path
    - LoRA configuration
    - LoRA save/load
    - Custom LoRA fit loop

- [x] Add generative metrics.
  - File: `pyhealth/metrics/generative.py`
  - Includes exact match and optional BERTScore F1.

- [x] Export new datasets/tasks/models/metrics through package init files.
  - Files:
    - `pyhealth/datasets/__init__.py`
    - `pyhealth/tasks/__init__.py`
    - `pyhealth/models/__init__.py`
    - `pyhealth/metrics/__init__.py`

- [x] Add example scripts.
  - Files:
    - `examples/vqarad_generativevqa_medflamingo.py`
    - `examples/med_flamingo_lora_train_eval.py`

- [x] Add unit and smoke-test coverage for the new MVP.
  - Files:
    - `tests/core/test_vqa_rad_dataset.py`
    - `tests/core/test_path_vqa_dataset.py`
    - `tests/core/test_generative_vqa_task.py`
    - `tests/core/test_med_flamingo.py`
    - `tests/core/test_generative_metrics.py`
    - `tests/integration/test_med_flamingo_lora_smoke.py`

- [x] Add docs/API stubs for the new components.
  - Files under `docs/api/`

## Still Needs To Be Done

### Core architecture

- [ ] Decide whether we are shipping an MVP or a fully native PyHealth integration.
  - Current code works through custom model helpers, not the generic PyHealth trainer.

- [ ] Add native generative-task support to PyHealth trainer/evaluation.
  - Current blocker: `pyhealth/trainer.py` only understands classification/regression modes and otherwise falls back to loss-only evaluation.
  - Goal: allow text predictions and generative metrics to flow through the standard trainer path.

- [ ] Decide whether `MedFlamingo.fit_lora()` stays as a project-specific training path or gets absorbed into core trainer abstractions.
  - Current state is functional but not fully aligned with the proposal language about using PyHealth's native engines.

### Benchmark completeness

- [ ] Add Visual USMLE dataset support.
  - No dataset/task/eval path for Visual USMLE was found.

- [ ] Add a benchmark runner that can evaluate across:
  - VQA-RAD
  - Path-VQA
  - Visual USMLE

- [ ] Add a unified result format for cross-dataset reporting.
  - Suggested outputs:
    - exact match
    - BERTScore F1
    - per-dataset prediction dumps
    - few-shot ablation table

### Leakage control / methodology fidelity

- [ ] Implement real leakage filtering closer to the paper.
  - Current FAISS path is a stub and only warns.
  - Current pHash fallback is helpful, but it is not the same as the paper's embedding-based dedup.

- [ ] Decide whether to use:
  - CLIP/ViT embeddings plus FAISS kNN
  - a documented simplified approximation for class project scope

- [ ] Document exactly which leakage-control approach is used in the final writeup.
  - This matters because the paper explicitly treats leakage control as part of the methodology.

### Dependency and environment cleanup

- [ ] Add missing Med-Flamingo-specific dependencies to packaging or extras.
  - Likely needed:
    - `open_flamingo`
    - `huggingface_hub`
    - optional `bitsandbytes`
    - optional `bert-score`
    - optional `faiss`

- [ ] Document environment setup in a stable place.
  - There is already a hand-test note, but install/runtime expectations should be made easier for teammates.

- [ ] Confirm model asset requirements.
  - Need clear instructions for:
    - `LLAMA_PATH`
    - Hugging Face token access
    - checkpoint filename/repo ID

### Validation

- [ ] Run the new unit tests in a working environment.
  - Current shell could not run them because key packages were missing.

- [ ] Run the heavy Med-Flamingo smoke test with real model assets.
  - File: `tests/integration/test_med_flamingo_lora_smoke.py`

- [ ] Manually test a real Colab-style path.
  - Few-shot inference first
  - Then optional LoRA adapter training

- [ ] Confirm example scripts work end to end with real data.
  - `examples/vqarad_generativevqa_medflamingo.py`
  - `examples/med_flamingo_lora_train_eval.py`

## Recommended Work Split

### Partner A: infrastructure / architecture

- [ ] Trainer support for generative tasks
- [ ] metrics integration in core eval path
- [ ] packaging and dependency cleanup
- [ ] test execution in a proper environment

### Partner B: datasets / benchmarking

- [ ] Visual USMLE dataset support
- [ ] benchmark runner across all datasets
- [ ] result export and few-shot ablation reporting
- [ ] leakage-control implementation or documented approximation

## Suggested Immediate Next Steps

- [ ] First, get the current MVP running end to end in one real environment.
- [ ] Second, decide whether the project goal is:
  - "working Med-Flamingo MVP inside PyHealth"
  - or "fully native PyHealth generative trainer integration"
- [ ] Third, assign one person to trainer/core work and one person to benchmark/data work.
- [ ] Fourth, add Visual USMLE only after the VQA-RAD and Path-VQA path is confirmed stable.

## Known Caveats

- This branch appears close to an MVP, but not yet a faithful reproduction of the full Med-Flamingo paper setup.
- The proposal text currently overstates integration with PyHealth's native trainer stack.
- The proposal text also overstates leakage-control completeness unless FAISS or another embedding-based dedup path is actually implemented.
