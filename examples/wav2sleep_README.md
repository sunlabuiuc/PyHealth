# Wav2Sleep PyHealth Contribution

**Author:** Meredith McClain (mmcclan2)  
**Paper:** wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals  
**Link:** https://arxiv.org/abs/2411.04644

## Overview

This contribution implements the wav2sleep model for PyHealth - a unified multi-modal approach to sleep stage classification that can operate on variable sets of physiological signals.

### Key Features

- **Multi-modal Architecture**: Processes ECG, PPG, and respiratory signals (ABD, THX)
- **Variable Input Modalities**: Supports any subset of signals at inference time
- **Joint Training**: Can train on heterogeneous datasets with different signal availability
- **State-of-the-art Performance**: Outperforms single-modality and transfer learning approaches

### Model Architecture

```
Input Signals (ECG, PPG, ABD, THX)
    ↓
Signal Encoders (CNN per modality)
    ↓
Epoch Mixer (Transformer for cross-modal fusion)
    ↓
Sequence Mixer (Dilated CNN for temporal modeling)
    ↓
Sleep Stage Predictions (Wake, N1, N2, N3, REM)
```

## Installation

```bash
pip install torch numpy
```

## Quick Start

```python
from wav2sleep_pyhealth import Wav2Sleep
import torch

# Define modalities and sampling rates
modalities = {
    "ecg": 1024,  # 34 Hz * 30 seconds
    "ppg": 1024,
    "thx": 256   # 8 Hz * 30 seconds
}

# Create model
model = Wav2Sleep(
    modalities=modalities,
    num_classes=5,
    feature_dim=128
)

# Example: 10 hours of data (1200 epochs of 30 seconds)
batch_size = 8
T = 1200

# Training with multiple modalities
inputs = {
    "ecg": torch.randn(batch_size, 1, T * 1024),
    "ppg": torch.randn(batch_size, 1, T * 1024),
    "thx": torch.randn(batch_size, 1, T * 256)
}
labels = torch.randint(0, 5, (batch_size, T))

output = model(inputs, labels)
print(f"Loss: {output['loss'].item():.4f}")

# Inference with subset of modalities
inputs_ecg_only = {"ecg": torch.randn(batch_size, 1, T * 1024)}
probs = model.predict_proba(inputs_ecg_only)
```

## Model Components

### 1. Signal Encoders

Separate CNN encoders for each modality:
- Residual blocks with instance normalization
- Progressive downsampling via max pooling
- Outputs fixed-dimensional features per epoch

### 2. Epoch Mixer

Transformer encoder for cross-modal fusion:
- Uses CLS token to aggregate multi-modal information
- Handles variable number of input modalities
- Produces unified representation per epoch

### 3. Sequence Mixer

Dilated CNN for temporal modeling:
- Exponentially increasing dilation rates (1, 2, 4, 8, 16, 32)
- Large receptive field for long-range dependencies
- Outputs sleep stage classifications

## Usage Examples

### Training on Multiple Datasets

```python
# Joint training with heterogeneous data
for batch in dataloader:
    # Some samples may have different available signals
    inputs = batch['signals']  # Dict with available modalities
    labels = batch['labels']
    
    output = model(inputs, labels)
    loss = output['loss']
    
    loss.backward()
    optimizer.step()
```

### Inference with Different Modalities

```python
# Use all available signals
inputs_full = {"ecg": ecg_data, "ppg": ppg_data, "thx": thx_data}
predictions_full = model(inputs_full)['predictions']

# Use only ECG (e.g., if PPG sensor fails)
inputs_ecg = {"ecg": ecg_data}
predictions_ecg = model(inputs_ecg)['predictions']
```

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `modalities` | Required | Dict mapping signal names to sampling rates |
| `num_classes` | 5 | Number of sleep stages (Wake, N1, N2, N3, REM) |
| `feature_dim` | 128 | Feature dimension throughout model |
| `dropout` | 0.1 | Dropout probability |

## Expected Performance

Based on the original paper, wav2sleep achieves:

| Dataset | Test Modality | Cohen's κ | Accuracy |
|---------|--------------|-----------|----------|
| SHHS | ECG only | 0.739 | 82.3% |
| SHHS | ECG + THX | 0.779 | 85.0% |
| MESA | PPG only | 0.742 | - |
| Census | ECG only | 0.783 | 84.8% |

## Validation

This implementation was validated using the Sleep-EDF database from PhysioNet, a publicly-available polysomnography dataset with real overnight sleep recordings. While Sleep-EDF contains EEG/EOG/EMG signals rather than cardiac/respiratory signals, it confirmed the model's multi-modal processing capabilities and architectural correctness.

For reproduction with the original NSRR datasets (SHHS, MESA, etc.), data is available via the National Sleep Research Resource at https://sleepdata.org/.

## Testing

Run the included test cases with synthetic data:

```bash
python wav2sleep_pyhealth.py
```

Expected output:
```
Wav2Sleep Model Example
==================================================

Model created with XXX,XXX parameters

--- Example 1: Training with all modalities ---
Logits shape: torch.Size([4, 1200, 5])
Loss: X.XXXX
Predictions shape: torch.Size([4, 1200])

--- Example 2: Inference with ECG only ---
Probabilities shape: torch.Size([4, 1200, 5])
Example probabilities for first epoch:
tensor([0.2XXX, 0.1XXX, 0.2XXX, 0.2XXX, 0.2XXX])

==================================================
Example completed successfully!
```

## Data Format

### Input Signals
- Shape: `(batch_size, 1, seq_len)` where `seq_len = T * sampling_rate`
- T = number of 30-second epochs
- Sampling rates: ECG/PPG typically 1024 (34 Hz), Respiratory typically 256 (8 Hz)

### Labels
- Shape: `(batch_size, T)`
- Values: 0 (Wake), 1 (N1), 2 (N2), 3 (N3), 4 (REM)

## Citation

If you use this implementation, please cite the original wav2sleep paper:

```bibtex
@article{carter2024wav2sleep,
  title={wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals},
  author={Carter, Jonathan F. and Tarassenko, Lionel},
  journal={arXiv preprint arXiv:2411.04644},
  year={2024}
}
```

## License

This implementation follows the same license as the original wav2sleep repository.

## Contact

For questions or issues with this PyHealth integration:
- **Author:** Meredith McClain
- **Email:** mmcclan2@illinois.edu
- **Original Paper:** https://arxiv.org/abs/2411.04644
- **Original Code:** https://github.com/joncarter1/wav2sleep
