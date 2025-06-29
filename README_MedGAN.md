# MedGAN for Synthetic EHR Data Generation

This implementation provides a complete pipeline for generating synthetic Electronic Health Record (EHR) data using MedGAN, following the approach from the original research paper "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks" by Choi et al. (2017).

## Overview

This module provides a comprehensive pipeline for:
- **Preprocessing**: Transforming MIMIC-III data through ICD-9 → ICD-10 → PhecodeX → PhecodeXM conversions
- **MedGAN Model**: GAN architecture with autoencoder pretraining for generating synthetic medical records
- **Postprocessing**: Converting synthetic data back to standardized PhecodeXM format (594 codes)

## Research Background

The original MedGAN paper introduced a novel approach for generating synthetic patient records using:
- **Autoencoder pretraining**: To learn meaningful representations of the sparse binary medical data
- **Generator-Discriminator architecture**: With minibatch averaging for training stability
- **Binary cross-entropy loss**: Adapted for the discrete nature of medical codes

## Data Pipeline

### 1. Preprocessing Pipeline

The preprocessing follows a multi-stage transformation process:

#### Stage 1: MIMIC-III to ICD-9 Matrix
- **Input**: MIMIC-III DIAGNOSES_ICD table
- **Processing**: Extract ICD-9 codes and convert to 3-digit format (e.g., "250.01" → "250")
- **Output**: Binary matrix (patients × ICD-9 codes)
- **Shape**: Typically ~46,517 patients × ~1,070 ICD-9 codes

#### Stage 2: ICD-9 to ICD-10 Conversion
- **Mapping**: Uses official ICD-9 to ICD-10 crosswalk mappings
- **Process**: Each ICD-9 code maps to one or more ICD-10 codes
- **Output**: Binary matrix (patients × ICD-10 codes)
- **Shape**: ~46,517 patients × ~5,613 ICD-10 codes

#### Stage 3: ICD-10 to PhecodeX Conversion
- **Mapping**: Uses PhecodeX mappings (expanded phecode system)
- **Process**: Groups related ICD-10 codes into phenotypic categories
- **Output**: Binary matrix (patients × PhecodeX codes)
- **Shape**: ~46,517 patients × ~2,254 PhecodeX codes

#### Stage 4: PhecodeX to PhecodeXM Conversion (Postprocessing)
- **Mapping**: Uses PhecodeXM mappings (medical phecode system)
- **Process**: Further groups PhecodeX codes into medical categories
- **Output**: Binary matrix (patients × PhecodeXM codes)
- **Shape**: ~46,517 patients × 594 PhecodeXM codes

### Data Pipeline Visualization

```mermaid
flowchart TD
    A[MIMIC-III DIAGNOSES_ICD] --> B[ICD-9 Extraction]
    B --> C[3-digit Truncation]
    C --> D[ICD-9 Matrix<br/>46,517 × 1,070]
    
    D --> E[ICD-9 → ICD-10<br/>Crosswalk Mapping]
    E --> F[ICD-10 Matrix<br/>46,517 × 5,613]
    
    F --> G[ICD-10 → PhecodeX<br/>Phenotypic Mapping]
    G --> H[PhecodeX Matrix<br/>46,517 × 2,254]
    
    H --> I[MedGAN Training<br/>Autoencoder + GAN]
    I --> J[Synthetic PhecodeX<br/>1,000 × 2,254]
    
    J --> K[PhecodeX → PhecodeXM<br/>Medical Mapping]
    K --> L[Final Synthetic Data<br/>1,000 × 594]
    
    style A fill:#e1f5fe
    style D fill:#fff3e0
    style F fill:#fff3e0
    style H fill:#fff3e0
    style J fill:#f3e5f5
    style L fill:#e8f5e8
```

### 2. MedGAN Architecture

The MedGAN model consists of three main components:

#### Autoencoder
- **Purpose**: Pretrains to learn meaningful representations of sparse binary data
- **Architecture**: 
  - Encoder: Dense layers with ReLU activation
  - Decoder: Dense layers with Sigmoid activation
- **Loss**: Binary cross-entropy
- **Pretraining**: 100+ epochs before GAN training

#### Generator
- **Input**: Random noise (latent_dim = 128)
- **Architecture**: Dense layers with batch normalization and ReLU
- **Output**: Latent representations for autoencoder decoder
- **Purpose**: Generate fake patient representations

#### Discriminator
- **Input**: Real or synthetic patient data
- **Architecture**: Dense layers with minibatch averaging
- **Output**: Binary classification (real vs fake)
- **Purpose**: Distinguish between real and synthetic data

### MedGAN Architecture Visualization

```mermaid
graph TB
    subgraph "Training Phase 1: Autoencoder Pretraining"
        A1[Real Patient Data<br/>2,254 features] --> B1[Encoder<br/>Dense + ReLU]
        B1 --> C1[Latent Space<br/>128 dimensions]
        C1 --> D1[Decoder<br/>Dense + Sigmoid]
        D1 --> E1[Reconstructed Data<br/>2,254 features]
        A1 -.->|Binary Cross-Entropy Loss| E1
    end
    
    subgraph "Training Phase 2: GAN Training"
        A2[Random Noise<br/>128 dimensions] --> B2[Generator<br/>Dense + BN + ReLU]
        B2 --> C2[Fake Latent<br/>128 dimensions]
        C2 --> D2[Autoencoder Decoder<br/>Dense + Sigmoid]
        D2 --> E2[Fake Patient Data<br/>2,254 features]
        
        F2[Real Patient Data<br/>2,254 features] --> G2[Discriminator<br/>Dense + Minibatch Avg]
        E2 --> G2
        G2 --> H2[Real/Fake Classification]
    end
    
    subgraph "Generation Phase"
        A3[Random Noise<br/>128 dimensions] --> B3[Trained Generator]
        B3 --> C3[Fake Latent<br/>128 dimensions]
        C3 --> D3[Trained Decoder]
        D3 --> E3[Synthetic Data<br/>2,254 features]
        E3 --> F3[Threshold 0.5]
        F3 --> G3[Binary Synthetic Data<br/>2,254 features]
    end
    
    style A1 fill:#e3f2fd
    style E1 fill:#e8f5e8
    style A2 fill:#fff3e0
    style E2 fill:#f3e5f5
    style F2 fill:#e3f2fd
    style A3 fill:#fff3e0
    style G3 fill:#e8f5e8
```

### 3. Training Process

#### Phase 1: Autoencoder Pretraining
```python
# Pretrain autoencoder for 100+ epochs
autoencoder_losses = model.pretrain_autoencoder(
    dataloader=dataloader,
    epochs=100,
    lr=0.001,
    device=device
)
```

#### Phase 2: GAN Training
```python
# Train generator and discriminator alternately
for epoch in range(1000):
    # Train discriminator
    real_output = discriminator(real_data)
    fake_output = discriminator(fake_data.detach())
    d_loss = discriminator_loss(real_output, fake_output)
    
    # Train generator
    fake_output = discriminator(fake_data)
    g_loss = generator_loss(fake_output)
```

### 4. Postprocessing Pipeline

After generating synthetic data, the pipeline converts it back through the mapping chain:

#### Step 1: Synthetic PhecodeX → PhecodeXM
- **Input**: Synthetic matrix (patients × 2,254 PhecodeX codes)
- **Process**: Apply PhecodeX to PhecodeXM mappings
- **Output**: Synthetic matrix (patients × 594 PhecodeXM codes)

#### Step 2: Binarization
- **Threshold**: 0.5 (values ≥ 0.5 become 1, others become 0)
- **Result**: Binary synthetic patient records

## Usage

### Basic Training

```bash
# Train with phecode mapping (recommended)
python examples/synthetic_data_generation_mimic3_medgan.py \
    --data_path /path/to/mimic3 \
    --use_phecode_mapping \
    --postprocess \
    --autoencoder_epochs 100 \
    --gan_epochs 1000 \
    --batch_size 128
```

### Simple Training (Raw ICD codes)

```bash
# Train without phecode mapping (like original paper)
python examples/synthetic_data_generation_mimic3_medgan_simple.py \
    --data_path /path/to/mimic3 \
    --n_epochs 1000 \
    --n_epochs_pretrain 100
```

### Generation

```python
import torch
import numpy as np
from pyhealth.models.generators.medgan import MedGAN

# Load trained model
checkpoint = torch.load("medgan_results/medgan_final.pth")
config = checkpoint['model_config']

# Initialize model
dummy_matrix = np.zeros((100, config['input_dim']), dtype=np.float32)
model = MedGAN.from_phecode_matrix(dummy_matrix, **config)

# Load state dicts
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
model.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])

# Generate synthetic data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with torch.no_grad():
    synthetic_data = model.generate(1000, device)
    binary_data = model.sample_transform(synthetic_data, threshold=0.5)

synthetic_matrix = binary_data.cpu().numpy()
print(f"Generated {synthetic_matrix.shape[0]} synthetic patients")
```

## Key Components

### PhecodeDataset (`pyhealth/datasets/phecode_dataset.py`)
- **PhecodeTransformer**: Handles ICD-9 → ICD-10 → PhecodeX conversions
- **PhecodeDataset**: Main dataset class for preprocessing
- **PhecodeMatrixDataset**: PyTorch dataset wrapper for training

### MedGAN Model (`pyhealth/models/generators/medgan.py`)
- **MedGAN**: Main model class with autoencoder, generator, and discriminator
- **Training methods**: Pretraining and GAN training loops
- **Generation methods**: Sample generation and transformation

### Mapping Files (`pyhealth/datasets/phecode_mappings/`)
- **ICD9toICD10Mapping.json**: ICD-9 to ICD-10 crosswalk
- **icd10_to_phecodex_mapping.json**: ICD-10 to PhecodeX mappings
- **phecodex_to_phecodexm_mapping.json**: PhecodeX to PhecodeXM mappings
- **phecodexm_types.json**: Final 594 PhecodeXM code definitions

## Output Files

After training, the pipeline produces:

- **`phecode_matrix.npy`**: Preprocessed PhecodeX matrix (2,254 codes)
- **`synthetic_binary_matrix.npy`**: Raw synthetic data
- **`synthetic_phecodexm_matrix.npy`**: Final postprocessed data (594 codes)
- **`synthetic_phecodexm.csv`**: CSV format for compatibility
- **`medgan_final.pth`**: Trained model checkpoint
- **`loss_history.npy`**: Training loss curves

## Shape Differences

| Stage | Matrix Shape | Code Count | Description |
|-------|-------------|------------|-------------|
| ICD-9 | (46,517, 1,070) | 1,070 | Raw MIMIC-III ICD-9 codes |
| ICD-10 | (46,517, 5,613) | 5,613 | Expanded ICD-10 codes |
| PhecodeX | (46,517, 2,254) | 2,254 | Phenotypic groupings |
| PhecodeXM | (46,517, 594) | 594 | Medical categories |

## Notes

- **Sparsity**: Medical data is highly sparse (~98.5% zeros)
- **Stability**: Minibatch averaging helps prevent mode collapse
- **Privacy**: Synthetic data preserves statistical properties while protecting privacy
- **Compatibility**: Output format matches synthEHRella for easy integration

## References

1. Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017). Generating Multi-label Discrete Patient Records using Generative Adversarial Networks. arXiv preprint arXiv:1703.06490.
2. Denny, J. C., et al. (2013). PheWAS: demonstrating the feasibility of a phenome-wide scan to discover gene-disease associations. Bioinformatics, 29(9), 1205-1210. 