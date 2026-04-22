# RetinaUNet Notebook Results Summary

This file highlights key quantitative results from `examples/retina_unet.ipynb` so reviewers can see outcomes directly in PR diff view.

## Key Quantitative Results

### 1) Forward-pass sanity checks

- Baseline architecture sanity loss: **386.7198**
- P4-P5 removed architecture sanity loss: **369.0320**

### 2) Baseline training run (initial run)

- Model parameters: **20,457,112**
- Best checkpoint val losses observed:
  - 0.7554
  - 0.4634
  - 0.1600
  - 0.0709
- Final best `val_loss` in this run: **0.0709**

### 3) Extended training run (same architecture, longer training)

- Model parameters: **20,457,112**
- Best checkpoint val losses observed:
  - 0.0615
  - 0.0423
  - 0.0270
  - 0.0249
- Final best `val_loss` in this run: **0.0249**
- Reported detection metric: **mAP@0.1 = 1.0000**

### 4) Ablation: remove P4-P5 feature pyramid levels

- Model parameters: **12,299,800**
- Best checkpoint val losses observed:
  - 0.7444
  - 0.5426
  - 0.0966
  - 0.0474
- Final best `val_loss` in this run: **0.0474**

## Ablation Takeaway

- Removing P4-P5 reduces parameters from 20,457,112 to 12,299,800
  (about **39.9% fewer parameters**).
- However, best validation loss degrades versus the longer baseline run:
  - Baseline (longer run): 0.0249
  - P4-P5 removed: 0.0474
  - Absolute delta: **+0.0225**