# Wav2Sleep Ablation Study Report

## Research Protocol Implementation

This report presents results from systematic ablation studies following the research protocol:

1. **Model capacity evaluation** by varying hidden representation dimension (32, 64, 128)
2. **Regularization analysis** using dropout rates of 0.1 and 0.3
3. **Missing modality robustness** evaluation (All/ECG+PPG/ECG only)
4. **Attention-based visualization** techniques (extension)

## Model Capacity

| Hidden Dimension | Accuracy | F1-Score | Parameters | Complexity Analysis |
|------------------|----------|----------|------------|---------------------|
| 32 | 1.000 | 1.000 | 1,441,063 | Efficient model with minimal parameters, good for resource-constrained environments |
| 64 | 1.000 | 1.000 | 1,485,767 | Balanced model with moderate capacity, optimal for most scenarios |
| 128 | 1.000 | 1.000 | 1,679,815 | High-capacity model with many parameters, risk of overfitting on small datasets |

## Regularization

| Dropout Rate | Accuracy | F1-Score | Regularization Effect |
|--------------|----------|----------|------------------------|
| 0.1 | 1.000 | 1.000 | Minimal regularization - good baseline performance |
| 0.3 | 1.000 | 1.000 | Strong regularization - may under-fit on complex patterns |

## Missing Modality

| Configuration | Accuracy | F1-Score | Performance Drop | Clinical Viability |
|---------------|----------|----------|------------------|--------------------|
| All_Modalities | 1.000 | 1.000 | Baseline | Baseline |
| ECG_Only | 1.000 | 1.000 | Baseline | Viable |

## Attention Visualization


