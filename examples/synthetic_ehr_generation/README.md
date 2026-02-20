# Synthetic EHR Generation Examples

This directory contains examples for training generative models on Electronic Health Records (EHR) data using PyHealth. These models can generate synthetic patient histories that preserve statistical properties of real EHR data while protecting patient privacy.

## Overview

The examples demonstrate how to:
1. Load and process MIMIC-III/IV data for generative modeling
2. Train various baseline models (GReaT, CTGAN, TVAE, Transformer)
3. Generate synthetic patient histories
4. Convert between different data representations (tabular, sequential, nested)

## Installation

### Core Requirements

```bash
pip install pyhealth
```

### Optional Dependencies (for baseline models)

For GReaT model:
```bash
pip install be-great
```

For CTGAN and TVAE:
```bash
pip install sdv
```

## Quick Start

### 1. Transformer-based Generation (Recommended)

Train a transformer model on MIMIC-III data:

```bash
python synthetic_ehr_mimic3_transformer.py \
    --mimic_root /path/to/mimic3 \
    --output_dir ./output \
    --epochs 50 \
    --batch_size 32 \
    --num_synthetic_samples 1000
```

### 2. Baseline Models

Train various baseline models:

```bash
# GReaT (Generative Relational Transformer)
python synthetic_ehr_baselines.py \
    --mimic_root /path/to/mimic3 \
    --train_patients /path/to/train_ids.txt \
    --test_patients /path/to/test_ids.txt \
    --output_dir ./synthetic_data \
    --mode great

# CTGAN (Conditional GAN)
python synthetic_ehr_baselines.py \
    --mimic_root /path/to/mimic3 \
    --train_patients /path/to/train_ids.txt \
    --test_patients /path/to/test_ids.txt \
    --output_dir ./synthetic_data \
    --mode ctgan

# TVAE (Variational Autoencoder)
python synthetic_ehr_baselines.py \
    --mimic_root /path/to/mimic3 \
    --train_patients /path/to/train_ids.txt \
    --test_patients /path/to/test_ids.txt \
    --output_dir ./synthetic_data \
    --mode tvae
```

## Architecture

### PyHealth Components

1. **Task**: `SyntheticEHRGenerationMIMIC3/MIMIC4`
   - Processes patient records into samples suitable for generative modeling
   - Creates nested sequences of diagnosis codes per visit
   - Located in: `pyhealth/tasks/synthetic_ehr_generation.py`

2. **Model**: `TransformerEHRGenerator`
   - Decoder-only transformer architecture (similar to GPT)
   - Learns to generate patient visit sequences autoregressively
   - Located in: `pyhealth/models/synthetic_ehr.py`

3. **Utilities**: `pyhealth.utils.synthetic_ehr_utils`
   - Functions for converting between data representations
   - Processes MIMIC data for different baseline models
   - Located in: `pyhealth/utils/synthetic_ehr_utils.py`

### Data Representations

The code supports three data representations:

1. **Nested Sequences** (PyHealth native):
   ```python
   [
       [['410', '250'], ['410', '401']],  # Patient 1: 2 visits
       [['250'], ['401', '430']],         # Patient 2: 2 visits
   ]
   ```

2. **Text Sequences** (for token-based models):
   ```
   "410 250 VISIT_DELIM 410 401"
   "250 VISIT_DELIM 401 430"
   ```

3. **Tabular/Flattened** (for CTGAN, TVAE, GReaT):
   ```
   SUBJECT_ID | 410 | 250 | 401 | 430
   ---------- | --- | --- | --- | ---
   0          |  2  |  1  |  1  |  0
   1          |  0  |  1  |  1  |  1
   ```

## Examples

### Example 1: Basic Training

```python
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import SyntheticEHRGenerationMIMIC3
from pyhealth.models import TransformerEHRGenerator
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.trainer import Trainer

# Load data
base_dataset = MIMIC3Dataset(
    root="/path/to/mimic3",
    tables=["DIAGNOSES_ICD"]
)

# Apply task
task = SyntheticEHRGenerationMIMIC3(min_visits=2)
sample_dataset = base_dataset.set_task(task)

# Split and create loaders
train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)

# Train model
model = TransformerEHRGenerator(
    dataset=sample_dataset,
    embedding_dim=256,
    num_heads=8,
    num_layers=6
)

trainer = Trainer(model=model, device="cuda")
trainer.train(train_loader, val_loader, epochs=50)
```

### Example 2: Generate Synthetic Data

```python
# Generate synthetic patient histories
model.eval()
synthetic_codes = model.generate(
    num_samples=1000,
    max_visits=10,
    temperature=1.0,
    top_k=50,
    top_p=0.95
)

# Convert to different formats
from pyhealth.utils.synthetic_ehr_utils import (
    nested_codes_to_sequences,
    sequences_to_tabular
)

# To text sequences
sequences = nested_codes_to_sequences(synthetic_codes)

# To tabular format
df = sequences_to_tabular(sequences)
df.to_csv("synthetic_ehr.csv", index=False)
```

### Example 3: Using Baseline Models

```python
from pyhealth.utils.synthetic_ehr_utils import (
    process_mimic_for_generation,
    create_flattened_representation
)

# Process MIMIC data
data = process_mimic_for_generation(
    mimic_data_path="/path/to/mimic3",
    train_patients_path="train_ids.txt",
    test_patients_path="test_ids.txt"
)

train_flattened = data["train_flattened"]

# Train CTGAN
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

metadata = Metadata.detect_from_dataframe(train_flattened)
synthesizer = CTGANSynthesizer(metadata, epochs=100, batch_size=64)
synthesizer.fit(train_flattened)

# Generate
synthetic_data = synthesizer.sample(num_rows=1000)
```

## Parameters

### TransformerEHRGenerator

- `embedding_dim`: Dimension of token embeddings (default: 256)
- `num_heads`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 6)
- `dim_feedforward`: Hidden dimension of feedforward network (default: 1024)
- `dropout`: Dropout probability (default: 0.1)
- `max_seq_length`: Maximum sequence length (default: 512)

### Generation Parameters

- `num_samples`: Number of synthetic patients to generate
- `max_visits`: Maximum visits per patient
- `temperature`: Sampling temperature (higher = more random)
- `top_k`: Keep only top k tokens for sampling (0 = disabled)
- `top_p`: Nucleus sampling threshold (1.0 = disabled)

## Output Format

Generated synthetic data is saved in multiple formats:

1. **CSV Format** (`synthetic_ehr.csv`):
   ```
   SUBJECT_ID,HADM_ID,ICD9_CODE
   0,0,41001
   0,0,25000
   0,1,41001
   ...
   ```

2. **Text Sequences** (`synthetic_sequences.txt`):
   ```
   41001 25000 VISIT_DELIM 41001 40199
   25000 VISIT_DELIM 40199 43001
   ...
   ```

3. **Model Checkpoints**: Saved in `output_dir/exp_name/`

## Evaluation

To evaluate synthetic data quality, you can use:

1. **Distribution Matching**: Compare code frequency distributions
2. **Downstream Tasks**: Train predictive models on synthetic data
3. **Privacy Metrics**: Measure memorization and privacy risks
4. **Clinical Validity**: Have clinical experts review synthetic patients

Example evaluation script (to be implemented):

```python
from pyhealth.metrics.synthetic import (
    evaluate_distribution_match,
    evaluate_downstream_task,
    evaluate_privacy_metrics
)
```

## Citation

If you use this code, please cite:

```bibtex
@software{pyhealth2024synthetic,
  title={PyHealth: A Python Library for Health Predictive Models},
  author={PyHealth Contributors},
  year={2024},
  url={https://github.com/sunlabuiuc/PyHealth}
}
```

For the reproducible synthetic EHR baseline:

```bibtex
@article{gao2024reproducible,
  title={Reproducible Synthetic EHR Generation},
  author={Gao, Chufan and others},
  year={2024}
}
```

## Contributing

To add new generative models:

1. Create a model class inheriting from `BaseModel`
2. Implement the `forward()` method
3. Implement a `generate()` method for sampling
4. Add example script to this directory

## References

- [PyHealth Documentation](https://pyhealth.readthedocs.io/)
- [MIMIC-III Database](https://mimic.mit.edu/)
- [GReaT Paper](https://arxiv.org/abs/2210.06280)
- [CTGAN Paper](https://arxiv.org/abs/1907.00503)
- [Reproducible Synthetic EHR](https://github.com/chufangao/reproducible_synthetic_ehr)
