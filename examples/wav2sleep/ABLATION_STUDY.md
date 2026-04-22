# Wav2Sleep Ablation Study

This document explains the systematic ablation study implemented for the Wav2Sleep model, including the research methodology, setup instructions, and result interpretation.

## Table of Contents

1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [Installation](#installation)
4. [Running the Ablation Study](#running-the-ablation-study)
5. [Understanding Results](#understanding-results)
6. [Clinical Interpretation](#clinical-interpretation)

---

## Overview

### What is an Ablation Study?

An ablation study is a systematic experimental approach used in machine learning to understand the contribution of individual model components or hyperparameters to overall performance. By systematically removing or varying specific components, researchers can:

- Determine optimal model configurations
- Understand regularization effects
- Assess robustness to missing data
- Identify clinically relevant model behaviors

### Wav2Sleep Model

Wav2Sleep is a deep learning model for automated sleep stage classification using multimodal physiological signals:

- **Input**: ECG (electrocardiogram) and PPG (photoplethysmography) signals
- **Architecture**: 1D CNN encoders + Transformer-based fusion
- **Output**: Sleep stage classification (Wake, N1, N2, N3, REM)

---

## Research Questions

The ablation study addresses four key research questions:

### 1. Model Capacity Evaluation

**Question**: How does model complexity (hidden dimension) affect performance?

**Method**: Test hidden dimensions [32, 64, 128]

**Hypothesis**: Larger hidden dimensions may initially improve performance but could lead to overfitting with limited data.

### 2. Regularization Analysis

**Question**: What is the effect of dropout regularization?

**Method**: Test dropout rates [0.1, 0.3]

**Hypothesis**: Higher dropout may prevent overfitting but could lead to underfitting with overly complex patterns.

### 3. Missing Modality Robustness

**Question**: How robust is the model to missing physiological signals?

**Method**: Test configurations:
- All modalities (ECG + PPG)
- ECG only

**Hypothesis**: Some degradation is expected when removing modalities, but the model should maintain reasonable performance.

### 4. Attention-Based Visualization

**Question**: Which physiological modalities does the model attend to during different sleep stages?

**Method**: Analyze attention patterns for each sleep stage

**Clinical Relevance**: Different sleep stages show characteristic physiological patterns that can validate model behavior.

---

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7+
- PyHealth package

### Step 1: Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv wav2sleep_env

# Activate the environment
source wav2sleep_env/bin/activate  # Linux/Mac
# or
wav2sleep_env\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

The ablation study requires the following packages:

```bash
# Core dependencies
pip install torch>=2.7.0
pip install numpy>=2.0.0
pip install scikit-learn>=1.7.0
pip install matplotlib>=3.4.0

# PyHealth and its dependencies
pip install pyhealth>=2.0.0

# Or install from source
cd PyHealth
pip install -e .
```

### Step 3: Verify Installation

```python
import torch
import numpy
from pyhealth.models.wav2sleep import Wav2Sleep

print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
```

---

## Running the Ablation Study

### Option 1: Run the Python Script

```bash
# Navigate to PyHealth directory
cd PyHealth

# Activate your virtual environment
source wav2sleep_env/bin/activate

# Run the ablation study
python examples/wav2sleep/sleep_multiclass_wav2sleep_corrected.py
```

### Option 2: Run the Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Open and run: examples/wav2sleep/wav2sleep_ablation_study.ipynb
```

### Option 3: Run Individual Studies

You can run specific ablation studies by modifying the script:

```python
<<<<<<< HEAD
# In sleep_multiclass_wav2sleep_comprehensive.py, comment out unwanted studies:
=======
# In sleep_multiclass_wav2sleep_corrected.py, comment out unwanted studies:
>>>>>>> a9f476b (Added jupyter notebook files for ablation studies, abaltion study guide explain the ablation apporach)

# Run only model capacity
all_results["model_capacity"] = run_model_capacity_ablation(base_samples)

# Comment out other studies
# all_results["regularization"] = run_regularization_ablation(base_samples)
# all_results["missing_modality"] = run_missing_modality_ablation(base_samples)
# all_results["attention"] = run_attention_visualization_extension(base_samples)
<<<<<<< HEAD

#  CommandLine command for abaltion : cd PyHealth && source .venv313/bin/activate && python examples/wav2sleep/sleep_multiclass_wav2sleep_comprehensive.py 2>&1
=======
>>>>>>> a9f476b (Added jupyter notebook files for ablation studies, abaltion study guide explain the ablation apporach)
```

---

## Understanding Results

### Output Structure

The ablation study produces the following outputs:

```
wav2sleep_experiment_results/
├── ablation_results.json    # Detailed numerical results
├── ablation_report.md      # Markdown summary
└── model_checkpoints/      # Saved models (if enabled)
```

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Percentage of correct predictions | Higher is better (max 1.0) |
| **F1-Score** | Harmonic mean of precision and recall | Accounts for class imbalance |
| **Parameters** | Total model parameters | Indicates model complexity |
| **Degradation** | Accuracy drop when removing modalities | Lower is better (more robust) |

### Expected Results

#### Model Capacity

| Hidden Dim | Expected Accuracy | Parameters |
|------------|-------------------|------------|
| 32 | ~0.95-1.0 | ~1.4M |
| 64 | ~0.95-1.0 | ~1.5M |
| 128 | ~0.85-0.95 | ~1.7M |

**Interpretation**: Hidden dimension 64 typically offers the best balance between performance and efficiency.

#### Regularization

| Dropout | Expected Effect |
|---------|-----------------|
| 0.1 | Good baseline performance |
| 0.3 | May underfit complex patterns |

**Interpretation**: Lower dropout (0.1) is recommended for this dataset size.

#### Missing Modality

| Configuration | Expected Accuracy | Degradation |
|---------------|-------------------|-------------|
| ECG + PPG | ~0.95-1.0 | Baseline |
| ECG Only | ~0.85-0.95 | <0.1 |

**Interpretation**: Model is robust to missing PPG; acceptable for single-sensor deployment.

---

<<<<<<< HEAD
## Visualization Analysis

The attention visualization shows:

1. **ECG dominance** in Wake and REM stages (expected due to HRV patterns)
2. **PPG dominance** in N2 and N3 stages (expected due to stable perfusion)
3. **Consistent attention patterns** across different model configurations

=======
>>>>>>> a9f476b (Added jupyter notebook files for ablation studies, abaltion study guide explain the ablation apporach)
## Clinical Interpretation

### Sleep Stage Physiology

| Stage | Characteristics | Dominant Signal |
|-------|-----------------|-----------------|
| **Wake** | High heart rate variability (HRV), irregular breathing | ECG |
| **N1** | Transition, HR begins to decrease | ECG |
| **N2** | Consolidated sleep, stable oxygen delivery | PPG |
| **N3** | Deep sleep, slow breathing, lowest HR | PPG |
| **REM** | Variable HR, irregular breathing, dreaming | ECG |

### Attention Patterns

The attention visualization reveals which physiological signals the model considers most important for each sleep stage:

- **ECG-dominant stages**: Wake, N1, REM (due to HRV patterns)
- **PPG-dominant stages**: N2, N3 (due to stable perfusion)

### Clinical Deployment Insights

1. **Single-sensor deployment**: ECG-only configuration maintains acceptable performance (~85-95% accuracy)

2. **Model efficiency**: Hidden dimension 64 provides optimal balance

3. **Regularization**: Standard dropout (0.1) is sufficient for this dataset size

4. **Attention interpretability**: Model attention aligns with clinical knowledge of sleep physiology

---

## Troubleshooting

### Common Issues

#### 1. Module Not Found Error

```bash
# Ensure PyHealth is installed
pip install -e PyHealth/
```

#### 2. CUDA Not Available

The script automatically falls back to CPU. For GPU acceleration:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### 3. Memory Issues

Reduce dataset size in the script:

```python
# Change these parameters
num_patients=10  # Reduced from 20
epochs_per_patient_range=(2, 4)  # Reduced from (4, 6)
```

---

## Extending the Study

### Adding New Experiments

To add new ablation experiments:

```python
def run_custom_ablation(base_samples):
    """Add your custom ablation study here."""
    
    # 1. Define experiment configuration
    # 2. Create model variants
    # 3. Train and evaluate
    # 4. Return results
    
    return results
```

### Adding New Modalities

To test additional physiological signals:

1. Modify data generation to include new signals
2. Update `SimpleDataset` class with new feature keys
3. Adjust model configuration

---

## References

<<<<<<< HEAD
- Wav2Sleep Paper: [[Citation](https://arxiv.org/abs/2411.04644)]
=======
- Wav2Sleep Paper: [Citation]
- PyHealth Documentation: [Link]
>>>>>>> a9f476b (Added jupyter notebook files for ablation studies, abaltion study guide explain the ablation apporach)
- Sleep Staging Standards: AASM Guidelines

---

## Contact

For questions or issues with the ablation study, please refer to the PyHealth repository or open an issue.

---

*Last Updated: April 2026*
