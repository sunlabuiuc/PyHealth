---
name: pyhealth-choose-a-model
description: Pick a PyHealth model that is compatible with a task's processor types — sequential, multimodal, drug-recommendation, graph, signal, and generative families — and read its real constructor signature. Use when selecting a model or when a model rejects a task's input schema.
---

# Choose a model

Use this to identify which model
to use and which file to read for its full `__init__` signature.

**Import path**: `from pyhealth.models import <ClassName>`
**Read source**: `pyhealth/models/<file>.py` — the `__init__` signature there is
the source of truth for constructor arguments. Do not guess kwargs.

---

## Processor Type Legend

| Symbol | Processor | Typical feature |
|--------|-----------|-----------------|
| `seq`    | SequenceProcessor | Visit-level code lists (diagnoses, meds, procedures) |
| `ts`     | TimeseriesProcessor | Timestamped lab values / vitals |
| `mh`     | MultiHotProcessor | Bag-of-codes per visit (binary presence vector) |
| `tensor` | TensorProcessor | Pre-computed float feature vectors |
| `image`  | ImageProcessor | 2D/3D image arrays |
| `stagenet` | StageNetProcessor / StageNetTensorProcessor | StageNet-specific temporal tensors |

---

## 1. EHR Clinical Prediction — Sequential Models

Models that operate on **sequential** inputs (`seq`, `ts`). Choose these when
the task uses only sequence or timeseries processors.

| Class | File | Processors | Tasks | Notes |
|-------|------|------------|-------|-------|
| `RNN` | rnn.py | `seq`, `ts` | binary, multiclass, multilabel | GRU/LSTM/RNN; fast baseline |
| `RETAIN` | retain.py | `seq`, `ts` | binary, multiclass, multilabel | Interpretable reverse-time attention |
| `AdaCare` | adacare.py | `seq`, `ts` | binary, multiclass, multilabel | Adaptive clinical feature weighting |
| `Deepr` | deepr.py | `seq` only | binary, multiclass, multilabel | Conv1d over code sequences; seq only |
| `Agent` | agent.py | `seq` only | binary, multiclass | Dual-agent RL with skip connections |
| `TCN` | tcn.py | `seq`, `ts`, `tensor` | binary, multiclass, multilabel | Dilated temporal convolutions |
| `StageNet` | stagenet.py | `mh`, `ts`, `stagenet` | binary, multiclass, multilabel | Stage-adaptive health risk; needs StageNetProcessor |
| `StageAttentionNet` | stagenet_mha.py | `mh`, `ts`, `stagenet` | binary, multiclass, multilabel | StageNet + multi-head attention |
| `SparcNet` | sparcnet.py | `ts` | seizure/EEG classification | DenseNet for EEG rhythmic patterns |

---

## 2. EHR Clinical Prediction — Multimodal Models

Models that handle **mixed** processor types (`seq` + `mh`, `ts` + `tensor`, etc.).
Use these when the task combines sequential and non-sequential features.

| Class | File | Processors | Tasks | Notes |
|-------|------|------------|-------|-------|
| `MultimodalRNN` | rnn.py | `seq`, `ts`, `mh`, `tensor` | binary, multiclass, multilabel | **Safest mixed-type choice**; same file as RNN |
| `MultimodalRETAIN` | retain.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | binary, multiclass, multilabel | Interpretable + mixed-type; same file as RETAIN |
| `MultimodalAdaCare` | adacare.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | binary, multiclass, multilabel | AdaCare + mixed-type; same file as AdaCare |
| `ConCare` | concare.py | `seq`, `ts`, `mh`, `tensor` | binary, multiclass, multilabel | Channel-wise GRU + multi-head self-attention |
| `GRASP` | grasp.py | `seq`, `ts`, `mh`, `tensor` | binary, multiclass, multilabel | Graph-based patient similarity |
| `Transformer` | transformer.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | binary, multiclass, multilabel | Multi-head self-attention over EHR visits |
| `EHRMamba` | ehrmamba.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | binary, multiclass, multilabel | Mamba SSM; linear complexity in sequence length |
| `JambaEHR` | jamba_ehr.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | binary, multiclass, multilabel | Hybrid Transformer-Mamba interleaved layers |
| `CNN` | cnn.py | `seq`, `ts`, `mh`, `tensor`, `image` | binary, multiclass, multilabel | 1D/2D/3D conv; adapts to processor spatial dim |
| `MICRON` | micron.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | drug recommendation | Medication change prediction via residual RNN |

---

## 3. Non-Sequential Models

Models for tasks where features are **static or bag-of-codes** (no temporal ordering).

| Class | File | Processors | Tasks | Notes |
|-------|------|------------|-------|-------|
| `MLP` | mlp.py | `seq`, `ts`, `tensor` | binary, multiclass, multilabel | Simple baseline; per-feature MLP + concat |
| `LogisticRegression` | logistic_regression.py | `seq`, `tensor` | binary, multiclass, regression | Linear baseline |

---

## 4. Drug Recommendation Models

Specialized models for **medication recommendation** tasks. These typically
require `mh` (MultiHotProcessor) inputs representing visit medication sets.

| Class | File | Processors | Tasks | Notes |
|-------|------|------------|-------|-------|
| `GAMENet` | gamenet.py | `mh` | drug recommendation | GCN + memory networks; DDI-aware |
| `SafeDrug` | safedrug.py | `mh` | drug recommendation | Molecular graph + DDI safety constraints |
| `MoleRec` | molerec.py | `mh` | drug recommendation | Molecular structure via GIN convolution |
| `MICRON` | micron.py | `seq`, `ts`, `mh`, `tensor`, `stagenet` | drug recommendation | Change-based prediction; also listed above |

---

## 5. Graph Neural Networks

For tasks with **explicit graph structure** (e.g., patient-disease graphs,
knowledge graphs). Require adjacency matrix construction.

| Class | File | Processors | Tasks | Notes |
|-------|------|------------|-------|-------|
| `GAT` | gnn.py | graph/`seq` | patient-level predictions | Graph Attention Network |
| `GCN` | gnn.py | graph/`seq` | patient-level predictions | Graph Convolutional Network |

---

## 6. Signal / EEG / Waveform Models

Specialized for **physiological signal** data (EEG, waveforms, time-series signals).

| Class | File | Processors | Tasks | Notes |
|-------|------|------------|-------|-------|
| `ContraWR` | contrawr.py | image/signal | sleep stage classification | STFT + 2D CNN self-supervised |
| `BIOT` | biot.py | signal | biosignal predictions | Foundation model for biomedical signals |

---

## 7. Generative / Embedding Models

Not for standard supervised prediction tasks.

| Class | File | Processors | Notes |
|-------|------|------------|-------|
| `GAN` | gan.py | image | ResBlock2D generator/discriminator |
| `VAE` | vae.py | image | CNN encoder/decoder; 32/64/128px |
| `TransformersModel` | transformers_model.py | text | HuggingFace wrapper |
| `TorchvisionModel` | torchvision_model.py | image | torchvision model wrapper |

---

## Quick Selection Guide

```
Task has mixed processors (seq/ts + mh/tensor)?
  → MultimodalRNN  (general purpose)
  → MultimodalRETAIN  (if interpretability needed)
  → MultimodalAdaCare  (if adaptive feature weighting needed)
  → ConCare / GRASP / Transformer / EHRMamba / JambaEHR  (all support mixed)

Task has only seq/ts processors?
  → RNN  (fast GRU baseline)
  → RETAIN  (interpretable)
  → AdaCare  (adaptive)
  → Transformer  (attention-based)
  → TCN  (dilated conv)

Task has only mh/tensor processors?
  → MLP  (standard baseline)
  → LogisticRegression  (linear baseline)

Drug recommendation task?
  → GAMENet / SafeDrug / MoleRec / MICRON

User requested a specific model (e.g. "use AdaCare")?
  → Check if it supports the task's processor types (seq/ts only vs mixed)
  → If incompatible (task has mh/tensor but model is seq-only), prefer the
    multimodal variant of that model, NOT an unrelated architecture:
      AdaCare  → MultimodalAdaCare
      RETAIN   → MultimodalRETAIN
      RNN      → MultimodalRNN
  → Use grep -rn "class <ModelName>" pyhealth/models to find its file
  → Use `pyhealth/models/<file>.py` to read __init__ signature
```

---

## Baseline Sanity Check

Before spending real compute, run a `dev=True` smoke test — roughly 1000
patients, two epochs. One successful epoch with a decreasing loss is enough to
confirm the pipeline is wired correctly.

```python
# Smoke test. Show the user the output, then re-run with dev=False for real numbers.
if __name__ == '__main__':
    dataset = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=sorted(required_tables),
        cache_dir=cache_dir,
        dev=True,   # small subset — for "does it run?", never for "is it good?"
    )
    samples = dataset.set_task(MyTask())
    assert len(samples) > 0, "set_task returned no samples — check task __call__"
    # then run 2 training epochs and confirm loss decreases
```

**Never report a number from a `dev=True` run.** The subset is far too small for
the metric to mean anything; once the smoke test passes, re-run with
`dev=False`.

**Cache safety**: use a different `cache_dir` for `dev=True` runs vs full runs, or always
assign a new `task_name` — `dev` and full caches are not interchangeable.

---

## Reading a Model's Source

To inspect any model before using it:

```python
# 1. Confirm the file
grep -rn "class AdaCare" pyhealth/models
# -> adacare.py:47: class AdaCare(BaseModel):

# 2. Read the implementation
`pyhealth/models/adacare.py`
# -> Full source with __init__ signature, supported processors, docstring

# 3. Read __init__ signature specifically
grep -rn "def __init__" pyhealth/models
```
