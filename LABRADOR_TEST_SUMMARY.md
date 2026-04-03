# LabradorModel Test

**Test File:** `tests/core/test_labrador.py`

## Overview

Minimal end-to-end smoke test for `LabradorModel` that verifies:
- Dataset creation with aligned lab codes and values
- Model initialization with correct hyperparameters
- Forward pass with expected output structure
- Backward pass with gradient computation
- Embedding extraction when requested

## Test Methods

| Test | Purpose |
|------|---------|
| `test_model_initialization` | Verifies model initializes with correct hyperparameters (embed_dim, num_heads, num_layers, feature keys, label key) |
| `test_model_forward` | Checks forward pass returns `loss`, `y_prob`, `y_true`, `logit` with correct shapes and batch sizes |
| `test_model_backward` | Confirms backward pass computes gradients on parameters |
| `test_model_with_embedding` | Validates embedding extraction returns correct shape `[batch_size, embed_dim]` |

---

## Key Implementation Details

## Test Data Structure

**Minimal Synthetic Dataset (2 samples, 4 labs each):**
```python
samples = [
    {
        "patient_id": "patient-0",
        "visit_id": "visit-0",
        "lab_codes": ["lab-1", "lab-2", "lab-3", "lab-4"],      # Categorical
        "lab_values": [1.0, 2.5, 3.0, 4.5],                     # Continuous
        "label": 0,
    },
    {
        "patient_id": "patient-1",
        "visit_id": "visit-1",
        "lab_codes": ["lab-1", "lab-2", "lab-3", "lab-4"],
        "lab_values": [2.1, 1.8, 2.9, 3.5],
        "label": 1,
    },
]
```

**Input Schema:**
```python
input_schema = {
    "lab_codes": "sequence",    # → Categorical tokens [batch, 4]
    "lab_values": "tensor",     # → Float values [batch, 4]
}
output_schema = {"label": "binary"}
```

**Model Configuration:**
```python
LabradorModel(
    dataset=dataset,
    code_feature_key="lab_codes",
    value_feature_key="lab_values",
    embed_dim=32,               # Lightweight for testing
    num_heads=2,
    num_layers=1,
)
```

---

## Running the Test

**Option 1: Using pixi (recommended)**
```bash
cd /home/leemh/PyHealth
make init   # First time only
make test
```

**Option 2: Direct unittest**
```bash
cd /home/leemh/PyHealth
python3 -m unittest tests.core.test_labrador.TestLabradorModel -v
```

**Option 3: Single test method**
```bash
python3 -m unittest tests.core.test_labrador.TestLabradorModel.test_model_forward -v
```
