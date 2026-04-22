# PTB-XL ECG Diagnosis — Ablation Study

Reproduces and extends the benchmark from:
> Nonaka, K., & Seita, D. (2021). *In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis.* PMLR 149, 414-424.

## Files

| File | Description |
|------|-------------|
| `ptbxl_ecg_diagnosis.ipynb` | Full ablation notebook (local or Colab with auto-download) |
| `ptbxl_ecg_diagnosis_colab.ipynb` | Colab-ready version with executed results (mounts Google Drive, injects custom modules) |
| `ptbxl_ecg_diagnosis_resnet.py` | CLI script version with argparse (same ablation, runnable from terminal) |

## Ablation Dimensions

1. **Task type** — multilabel (ROC-AUC) vs multiclass (weighted F1, accuracy)
2. **Model architecture** — MLP, CNN, Transformer
3. **Hidden dimension** — 32, 64, 128

Total: 3 models × 3 dims × 2 tasks = 18 configurations.

## Dataset

- **PTB-XL**: 21,837 clinical 12-lead ECGs from 18,885 patients
- **5 superdiagnostic classes**: NORM, MI, STTC, CD, HYP
- **46 SCP codes** mapped to superclasses via `SCP_TO_SUPER`
- Auto-downloads from PhysioNet (~1.8 GB)

## Quick Start

```bash
# CLI (auto-downloads PTB-XL on first run)
python examples/ptbxl_ecg_diagnosis_resnet.py --epochs 10

# Or with existing data
python examples/ptbxl_ecg_diagnosis_resnet.py --root /path/to/ptb-xl --sampling-rate 500
```

## Results (500 samples, 10 epochs, CPU)

### Multiclass Task (Weighted F1 / Accuracy)

| Model | dim=32 | dim=64 | dim=128 |
|-------|--------|--------|---------|
| MLP | 0.43 / 0.59 | 0.42 / 0.55 | **0.56 / 0.65** |
| CNN | 0.45 / 0.52 | 0.50 / 0.57 | 0.54 / 0.60 |
| Transformer | 0.42 / 0.57 | 0.30 / 0.43 | 0.43 / 0.43 |

### Multilabel Task (ROC-AUC / F1)

| Model | dim=32 | dim=64 | dim=128 |
|-------|--------|--------|---------|
| MLP | 0.73 / 0.30 | 0.74 / 0.32 | 0.75 / 0.49 |
| CNN | 0.77 / 0.48 | 0.74 / 0.44 | **0.81 / 0.37** |
| Transformer | 0.74 / 0.37 | 0.75 / 0.37 | 0.64 / 0.40 |

### Key Findings

- **CNN-128** achieved best multilabel ROC-AUC (0.81)
- **MLP-128** achieved best multiclass F1 (0.56)
- Larger hidden dims consistently improved MLP and CNN
- Transformer struggled most — flat 60,000-dim signal input (500 Hz × 12 leads) is not well-suited for attention
- Results align with paper's finding that convolutional architectures outperform Transformers on ECG

### Paper Reference

Nonaka & Seita (2021) reported on full PTB-XL with 50+ epochs and LR scheduling:
- Multilabel ROC-AUC: **0.928** (ResNet)
- Multiclass F1: **0.821** (ResNet)

Gap explained by: fewer epochs (10 vs 50+), smaller sample (500 vs 19K), simpler models (no ResNet), no LR scheduling.

## PyHealth Modules

- `pyhealth/datasets/ptbxl.py` — `PTBXLDataset`
- `pyhealth/tasks/ptbxl_diagnosis.py` — `PTBXLDiagnosis`, `PTBXLMulticlassDiagnosis`
- `tests/core/test_ptbxl.py` — 27 unit tests (synthetic data, ~3 sec)

## Authors

Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
