# Wav2Sleep: Multi-Modal Sleep Stage Classification

**Author:** Meredith McClain (mmcclan2)  
**Paper:** wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals  
**Link:** https://arxiv.org/abs/2411.04644  
**Year:** 2024

## Overview

Wav2Sleep is a unified multi-modal model for automatic sleep stage classification from physiological signals. Unlike traditional approaches that train separate models for each signal type, wav2sleep can operate on variable sets of inputs during both training and inference.

### Key Features

- **Multi-modal Architecture**: Processes ECG, PPG, abdominal (ABD), and thoracic (THX) respiratory signals
- **Variable Inputs**: Works with any subset of signals at test time
- **Joint Training**: Trains on heterogeneous datasets with different signal availability
- **State-of-the-art Performance**: Cohen's κ scores of 0.74-0.81 across multiple datasets

### Architecture
```
Input Signals → Signal Encoders → Epoch Mixer → Sequence Mixer → Sleep Stages
   (ECG, PPG,      (CNN per         (Transformer   (Dilated CNN    (Wake, N1,
    ABD, THX)       modality)        fusion)        temporal)       N2, N3, REM)
```

## Installation

Wav2Sleep is part of PyHealth. Install with:
```bash
pip install pyhealth
```

## Quick Start
```python
from pyhealth.models.wav2sleep import Wav2Sleep
import torch

# Define available signal types and sampling rates
modalities = {
    "ecg": 1024,  # 34 Hz × 30 seconds
    "ppg": 1024,
    "thx": 256    # 8 Hz × 30 seconds
}

# Create model
model = Wav2Sleep(
    modalities=modalities,
    num_classes=5,  # Wake, N1, N2, N3, REM
    feature_dim=128
)

# Example: 10 hours of data (1200 30-second epochs)
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

# Inference with subset (e.g., if PPG sensor fails)
inputs_ecg_only = {"ecg": torch.randn(batch_size, 1, T * 1024)}
probs = model.predict_proba(inputs_ecg_only)
```

## Model Components

### 1. Signal Encoders

Separate CNN encoders for each modality:
- Residual blocks with instance normalization
- Progressive channel expansion: [16, 32, 64, 128]
- Max pooling for downsampling
- Outputs: Feature vectors per 30-second epoch

### 2. Epoch Mixer

Transformer encoder for cross-modal fusion:
- Uses CLS token to aggregate information
- 2 layers, 8 attention heads
- Handles variable number of input modalities
- Outputs: Unified representation per epoch

### 3. Sequence Mixer

Dilated CNN for temporal modeling:
- 2 blocks of dilated convolutions
- Dilation rates: [1, 2, 4, 8, 16, 32]
- Large receptive field captures sleep cycles
- Outputs: Sleep stage classifications

## Training
```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch['signals']  # Dict with available modalities
        labels = batch['labels']
        
        output = model(inputs, labels)
        loss = output['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Stochastic Masking

During training, modalities can be randomly masked to improve generalization:
```python
# Mask probabilities (example)
mask_probs = {"ecg": 0.5, "ppg": 0.1, "abd": 0.7, "thx": 0.7}

# Randomly select subset of modalities for each batch
import random
masked_inputs = {
    k: v for k, v in inputs.items() 
    if random.random() > mask_probs.get(k, 0)
}

output = model(masked_inputs, labels)
```

## Performance

Results from Carter & Tarassenko (2024):

| Dataset | Test Modality | Cohen's κ | Accuracy |
|---------|--------------|-----------|----------|
| SHHS | ECG only | 0.739 | 82.3% |
| SHHS | ECG + THX | 0.779 | 85.0% |
| MESA | PPG only | 0.742 | - |
| MESA | ECG + THX | 0.783 | 86.1% |
| Census | ECG only | 0.783 | 84.8% |
| Census | ECG + THX | 0.812 | - |

## Data Format

### Input Signals
- **Shape:** `(batch_size, 1, seq_len)` 
- **seq_len:** `T × sampling_rate` where T = number of epochs
- **Sampling rates:**
  - ECG/PPG: 1024 samples/epoch (≈34 Hz)
  - ABD/THX: 256 samples/epoch (≈8 Hz)

### Labels
- **Shape:** `(batch_size, T)`
- **Values:** 
  - 0: Wake
  - 1: N1 (light sleep)
  - 2: N2 (light sleep)
  - 3: N3 (deep sleep)
  - 4: REM

## Datasets

The original paper uses seven datasets from the National Sleep Research Resource (NSRR):
- SHHS (Sleep Heart Health Study)
- MESA (Multi-Ethnic Study of Atherosclerosis)
- WSC (Wisconsin Sleep Cohort)
- CHAT (Childhood Adenotonsillectomy Trial)
- CFS (Cleveland Family Study)
- CCSHS (Cleveland Children's Sleep and Health Study)
- MROS (Osteoporotic Fractures in Men Study)

Total: 10,000+ overnight PSG recordings

## Citation

If you use Wav2Sleep, please cite:
```bibtex
@article{carter2024wav2sleep,
  title={wav2sleep: A Unified Multi-Modal Approach to Sleep Stage 
         Classification from Physiological Signals},
  author={Carter, Jonathan F. and Tarassenko, Lionel},
  journal={arXiv preprint arXiv:2411.04644},
  year={2024}
}
```

## References

- Original paper: https://arxiv.org/abs/2411.04644
- GitHub repository: https://github.com/joncarter1/wav2sleep
- PyHealth: https://github.com/sunlabuiuc/PyHealth

## License

This implementation follows the same license as the original wav2sleep repository.

## Contact

- **Author:** Meredith McClain
- **Email:** mmcclan2@illinois.edu
- **Course:** CS 598 Deep Learning for Healthcare, UIUC
