# Integration Hooks for Dhruv & Nafis

This document outlines exactly where and how to integrate modality encoders (Dhruv) and fusion module (Nafis) into the Wav2Sleep model.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│   {'ecg': [B, T, raw_features], 'ppg': [...], 'resp': [...]}    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EmbeddingModel                                │
│              (PyHealth built-in, already done)                   │
│                  Output: [B, T, embedding_dim]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║           DHRUV'S HOOK: Modality Encoders                 ║  │
│  ║                                                           ║  │
│  ║   ECG Encoder:  [B, T, embedding_dim] → [B, T, D]         ║  │
│  ║   PPG Encoder:  [B, T, embedding_dim] → [B, T, D]         ║  │
│  ║   Resp Encoder: [B, T, embedding_dim] → [B, T, D]         ║  │
│  ║                                                           ║  │
│  ║   Location: wav2sleep.py lines 486-491                    ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║           NAFIS'S HOOK: Fusion Module                     ║  │
│  ║                                                           ║  │
│  ║   Input:  Dict[str, Tensor[B, T, D]]                      ║  │
│  ║   Output: Tensor[B, T, D]                                 ║  │
│  ║                                                           ║  │
│  ║   Location: wav2sleep.py lines 493-506                    ║  │
│  ║   (CLS-token Transformer already implemented as default)  ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Dilated CNN Sequence Mixer                          │
│                   (Already implemented)                          │
│                  Output: [B, T, hidden_dim]                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Classification Head                            │
│              Linear: hidden_dim → num_classes                    │
│                  Output: [B, T, 5]                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 DHRUV'S INTEGRATION: Modality Encoders

### Current Placeholder (wav2sleep.py:486-491)

```python
# Modality-specific encoders (placeholders for Dhruv's CNN encoders)
# These should output [B, T, embedding_dim]
self.modality_encoders = nn.ModuleDict()
for modality in self.available_modalities:
    # Placeholder encoder - outputs [B, T, embedding_dim]
    self.modality_encoders[modality] = nn.Identity()  # ← REPLACE THIS
```

### Expected Interface

```python
class ModalityEncoder(nn.Module):
    """Base interface for modality-specific encoders.
    
    Input shape:  [B, T, embedding_dim]  (from EmbeddingModel)
    Output shape: [B, T, D]              (D = embedding_dim typically)
    
    Note: The encoder processes ALREADY EMBEDDED features, not raw signals.
    The EmbeddingModel handles the initial feature processing.
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        # Your CNN architecture here
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedded features [B, T, input_dim]
            
        Returns:
            Encoded features [B, T, output_dim]
        """
        pass
```

### Integration Steps for Dhruv

**Step 1:** Create your encoder classes (e.g., in a separate file `encoders.py`)

```python
# pyhealth/models/encoders.py

import torch
import torch.nn as nn

class ECGEncoder(nn.Module):
    """CNN encoder for ECG modality."""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Example: 1D CNN for temporal features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → transpose → [B, D, T]
        x = x.transpose(1, 2)
        x = self.encoder(x)
        # [B, D, T] → transpose → [B, T, D]
        return x.transpose(1, 2)


class PPGEncoder(nn.Module):
    """CNN encoder for PPG modality."""
    # Similar structure to ECGEncoder
    ...


class RespEncoder(nn.Module):
    """CNN encoder for Respiratory modality."""
    # Similar structure to ECGEncoder
    ...
```

**Step 2:** Modify wav2sleep.py to use your encoders

```python
# In wav2sleep.py, replace lines 486-491:

# BEFORE (placeholder):
self.modality_encoders = nn.ModuleDict()
for modality in self.available_modalities:
    self.modality_encoders[modality] = nn.Identity()

# AFTER (Dhruv's encoders):
from pyhealth.models.encoders import ECGEncoder, PPGEncoder, RespEncoder

self.modality_encoders = nn.ModuleDict()
encoder_classes = {
    'ecg': ECGEncoder,
    'ppg': PPGEncoder,
    'resp': RespEncoder,
}
for modality in self.available_modalities:
    self.modality_encoders[modality] = encoder_classes[modality](
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        dropout=dropout,
    )
```

**Step 3:** (Optional) Add encoder configuration to __init__

```python
def __init__(
    self,
    dataset: SampleDataset,
    embedding_dim: int = 128,
    # ... existing params ...
    encoder_type: str = 'cnn',  # NEW: 'cnn', 'resnet', 'transformer', etc.
    encoder_layers: int = 2,     # NEW: depth of encoder
):
```

### Testing Your Encoders

```python
# Quick test for Dhruv's encoders
import torch
from pyhealth.models.encoders import ECGEncoder

encoder = ECGEncoder(input_dim=128, output_dim=128)
x = torch.randn(2, 10, 128)  # [B, T, D]
y = encoder(x)
print(f"Input: {x.shape} → Output: {y.shape}")
assert y.shape == x.shape, "Shape mismatch!"
```

---

## 🔧 NAFIS'S INTEGRATION: Fusion Module

### Current Implementation (wav2sleep.py:493-506)

The CLS-token Transformer fusion is **already implemented** as the default:

```python
# PAPER-FAITHFUL: CLS-token Transformer Fusion
if use_paper_faithful:
    self.fusion_module = CLSTokenTransformerFusion(
        embed_dim=embedding_dim,
        num_heads=num_fusion_heads,
        num_layers=num_fusion_layers,
        dropout=dropout,
        max_modalities=len(self.modality_keys),
    )
else:
    self.fusion_module = None
```

### Option A: Use Existing CLSTokenTransformerFusion

Nafis can use the existing implementation directly - it's already paper-faithful!

### Option B: Replace with Custom Fusion Module

If Nafis has a different/improved fusion approach:

**Expected Interface:**

```python
class FusionModule(nn.Module):
    """Base interface for multimodal fusion.
    
    Input:  Dict[str, Tensor[B, T, D]] - modality embeddings
    Output: Tensor[B, T, D]            - fused representation
    
    Must handle:
    - Variable number of modalities (1, 2, or 3)
    - Fixed output dimension regardless of input count
    """
    
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        
    def forward(
        self, 
        modality_embeddings: Dict[str, torch.Tensor],
        modality_order: List[str] = ['ecg', 'ppg', 'resp'],
    ) -> torch.Tensor:
        """
        Args:
            modality_embeddings: Dict mapping modality names to [B, T, D] tensors
            modality_order: Order of modalities for positional encoding
            
        Returns:
            Fused representation of shape [B, T, D]
        """
        pass
```

**Integration Steps for Nafis:**

**Step 1:** Create your fusion module

```python
# pyhealth/models/fusion.py

import torch
import torch.nn as nn
from typing import Dict, List

class NafisFusionModule(nn.Module):
    """Custom fusion module by Nafis."""
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        # ... your params ...
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Your fusion architecture here
        
    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        modality_order: List[str] = ['ecg', 'ppg', 'resp'],
    ) -> torch.Tensor:
        """
        Args:
            modality_embeddings: {'ecg': [B,T,D], 'ppg': [B,T,D], ...}
            
        Returns:
            Fused features [B, T, D]
        """
        # Your fusion logic here
        pass
```

**Step 2:** Replace in wav2sleep.py

```python
# In wav2sleep.py, replace lines 493-506:

# BEFORE (built-in CLS-token fusion):
if use_paper_faithful:
    self.fusion_module = CLSTokenTransformerFusion(...)

# AFTER (Nafis's fusion):
from pyhealth.models.fusion import NafisFusionModule

if use_paper_faithful:
    self.fusion_module = NafisFusionModule(
        embed_dim=embedding_dim,
        num_heads=num_fusion_heads,
        # ... Nafis's params ...
    )
```

### Testing Your Fusion Module

```python
# Quick test for Nafis's fusion
import torch
from pyhealth.models.fusion import NafisFusionModule

fusion = NafisFusionModule(embed_dim=128)

# Test with all modalities
modality_embs = {
    'ecg': torch.randn(2, 10, 128),
    'ppg': torch.randn(2, 10, 128),
    'resp': torch.randn(2, 10, 128),
}
fused = fusion(modality_embs)
print(f"All modalities → Output: {fused.shape}")
assert fused.shape == (2, 10, 128), "Shape mismatch!"

# Test with missing modalities
for combo in [['ecg'], ['ecg', 'ppg'], ['ppg', 'resp']]:
    partial_embs = {k: v for k, v in modality_embs.items() if k in combo}
    fused = fusion(partial_embs)
    print(f"{combo} → Output: {fused.shape}")
    assert fused.shape == (2, 10, 128), "Shape mismatch!"
```

---

## 📍 Exact Line Numbers Reference

| Component | Location | Lines |
|-----------|----------|-------|
| **Modality Encoders (Dhruv)** | `wav2sleep.py` | 486-491 |
| **Fusion Module (Nafis)** | `wav2sleep.py` | 493-506 |
| Forward pass - encoder usage | `wav2sleep.py` | 556-562 |
| Forward pass - fusion usage | `wav2sleep.py` | 567-578 |

---

## 🔄 Data Flow Through Integration Points

```python
def forward(self, **kwargs):
    # 1. Extract modality inputs
    available_inputs = {...}  # line 544-548
    
    # 2. EmbeddingModel (built-in)
    embedded_inputs = self.embedding_model(available_inputs)  # line 551
    
    # 3. ═══ DHRUV'S ENCODERS ═══
    modality_embeddings = {}
    for modality, embedded_data in embedded_inputs.items():
        encoded = self.modality_encoders[modality](embedded_data)  # line 560
        modality_embeddings[modality] = encoded
    
    # 4. Shape validation
    self._validate_modality_shapes(modality_embeddings)  # line 565
    
    # 5. ═══ NAFIS'S FUSION ═══
    if self.use_paper_faithful and self.fusion_module is not None:
        fused_features = self.fusion_module(
            modality_embeddings, 
            modality_order=self.modality_keys
        )  # lines 569-572
    
    # 6. Temporal modeling (Dilated CNN - already implemented)
    temporal_features = self.temporal_layer(fused_features)  # line 581
    
    # 7. Classification
    logits = self.classifier(temporal_features)  # line 584
```

---

## ✅ Integration Checklist

### For Dhruv (Encoders)
- [ ] Create `ECGEncoder` class with input/output shape `[B, T, D]`
- [ ] Create `PPGEncoder` class with input/output shape `[B, T, D]`
- [ ] Create `RespEncoder` class with input/output shape `[B, T, D]`
- [ ] Test each encoder independently with random input
- [ ] Replace `nn.Identity()` placeholders in wav2sleep.py
- [ ] Run full model test to verify integration

### For Nafis (Fusion)
- [ ] Review existing `CLSTokenTransformerFusion` implementation
- [ ] Decide: use existing or create custom fusion
- [ ] If custom: create module with same interface
- [ ] Test with variable number of modalities (1, 2, 3)
- [ ] Verify output shape is always `[B, T, D]`
- [ ] Run full model test to verify integration

---

## 🧪 Full Integration Test

After both integrations:

```python
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.wav2sleep import Wav2Sleep

# Create test dataset
samples = [...]  # Your sleep data samples
dataset = create_sample_dataset(samples, ...)

# Initialize model (now with Dhruv's encoders + Nafis's fusion)
model = Wav2Sleep(
    dataset=dataset,
    embedding_dim=128,
    hidden_dim=128,
    use_paper_faithful=True,
)

# Test forward pass
loader = get_dataloader(dataset, batch_size=2)
batch = next(iter(loader))
results = model(**batch)

print(f"Loss: {results['loss'].item():.4f}")
print(f"y_prob: {results['y_prob'].shape}")
print(f"y_true: {results['y_true'].shape}")

# Test backward pass
results['loss'].backward()
print("✓ Gradients computed successfully!")
```
