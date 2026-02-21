# PyHealth Synthetic EHR - Quick Reference Card

## 🚀 Which Notebook Should I Use?

| Your Original Mode | Use This Notebook | Use This Guide |
|-------------------|-------------------|----------------|
| `--mode great` | `PyHealth_Synthetic_EHR_Colab.ipynb` | `COLAB_GUIDE.md` |
| `--mode ctgan` | `PyHealth_Synthetic_EHR_Colab.ipynb` | `COLAB_GUIDE.md` |
| `--mode tvae` | `PyHealth_Synthetic_EHR_Colab.ipynb` | `COLAB_GUIDE.md` |
| `--mode transformer_baseline` | `PyHealth_Transformer_Baseline_Colab.ipynb` | `TRANSFORMER_BASELINE_GUIDE.md` |

## 📋 Checklist Before Running

### Required Files in Google Drive

```
MyDrive/
├── mimic3_data/
│   ├── ADMISSIONS.csv              ✓ Required
│   ├── PATIENTS.csv                ✓ Required
│   ├── DIAGNOSES_ICD.csv           ✓ Required
│   ├── train_patient_ids.txt       ✓ Required
│   └── test_patient_ids.txt        ✓ Required
└── original_output/                ✓ Optional (for comparison)
    ├── great/
    │   └── great_synthetic_flattened_ehr.csv
    ├── ctgan/
    │   └── ctgan_synthetic_flattened_ehr.csv
    ├── tvae/
    │   └── tvae_synthetic_flattened_ehr.csv
    └── transformer_baseline/
        └── transformer_baseline_synthetic_ehr.csv
```

### Colab Settings

- [ ] Runtime type: **GPU** (or A100)
- [ ] Google Drive mounted
- [ ] Paths configured in notebook
- [ ] Expected runtime: 40-60 min (GReaT/CTGAN/TVAE) or 2-3 hours (Transformer)

## ⚙️ Configuration Template

Copy this into the config cell and update paths:

```python
# Data paths
MIMIC_DATA_PATH = "/content/drive/MyDrive/mimic3_data/"
TRAIN_PATIENTS_PATH = "/content/drive/MyDrive/mimic3_data/train_patient_ids.txt"
TEST_PATIENTS_PATH = "/content/drive/MyDrive/mimic3_data/test_patient_ids.txt"

# Original output (for comparison)
ORIGINAL_OUTPUT = "/content/drive/MyDrive/original_output"

# Model selection (for tabular models)
MODEL_MODE = "great"  # or "ctgan", "tvae"

# Or for transformer_baseline:
ORIGINAL_OUTPUT_CSV = "/content/drive/MyDrive/original_output/transformer_baseline/transformer_baseline_synthetic_ehr.csv"

# Output
PYHEALTH_OUTPUT = "/content/pyhealth_output"  # or save to Drive

# Training
NUM_EPOCHS = 2  # Quick test (use 10-50 for production)
BATCH_SIZE = 512  # (or 64 for transformer)
NUM_SYNTHETIC_SAMPLES = 10000
```

## ⏱️ Expected Timelines

### GReaT, CTGAN, TVAE (Tabular Models)

| Step | Time | Cumulative |
|------|------|------------|
| Setup | 5 min | 5 min |
| Data processing | 10 min | 15 min |
| Training (2 epochs) | 20 min | 35 min |
| Generation | 5 min | 40 min |
| Comparison | 2 min | 42 min |
| **TOTAL** | | **~40-45 min** |

### Transformer Baseline (Sequential Model)

| Step | Time | Cumulative |
|------|------|------------|
| Setup | 5 min | 5 min |
| Data processing | 10 min | 15 min |
| Tokenizer | 2 min | 17 min |
| Training (50 epochs) | 100 min | 117 min |
| Generation | 15 min | 132 min |
| Comparison | 2 min | 134 min |
| **TOTAL** | | **~2-2.5 hours** |

💡 **Speed Tips:**
- Use A100 GPU (Colab Pro) → 2x faster
- Reduce epochs for quick test → 10x faster
- Increase batch size (if memory allows) → 1.5x faster

## 🎯 Validation Checklist

After running, check these:

### For All Models

- [ ] Training completed without errors
- [ ] Synthetic data generated (10,000 samples)
- [ ] Output CSV file created
- [ ] Visualizations saved

### If Comparing with Original

- [ ] Original file found and loaded
- [ ] Statistical comparison table shows similar values
- [ ] Visual plots show overlapping distributions
- [ ] ≥3 out of 4 validation checks pass
- [ ] Code frequency correlation > 0.7

## 📊 Understanding Validation Results

### ✅ Excellent Match (100% confidence)
```
✓ PASS - All 4 validation checks
✓ Correlation > 0.85
✓ Visual distributions nearly identical
```
→ **PyHealth implementation is correct!**

### ⚠️ Good Match (95% confidence)
```
✓ PASS - 3/4 validation checks
✓ Correlation 0.7-0.85
⚠️  Some minor distribution differences
```
→ **Expected due to stochastic nature of models**

### ❌ Poor Match (investigate)
```
✗ FAIL - <3 validation checks
✗ Correlation < 0.6
✗ Very different distributions
```
→ **Check hyperparameters, data splits, or training**

## 🔧 Quick Fixes

### Runtime Disconnected
```python
# Change output to Drive (survives disconnection)
PYHEALTH_OUTPUT = "/content/drive/MyDrive/pyhealth_output"
```

### Out of Memory
```python
# Reduce memory usage
BATCH_SIZE = 256  # or 128, or 64
NUM_SYNTHETIC_SAMPLES = 5000  # instead of 10000
MAX_SEQ_LENGTH = 256  # for transformer (instead of 512)
```

### Training Too Slow
```python
# Quick test settings
NUM_EPOCHS = 2  # instead of 50
NUM_SYNTHETIC_SAMPLES = 1000  # instead of 10000
```

### Can't Find Original Output
```python
# Check exact path
!ls -la /content/drive/MyDrive/original_output/
# Update path in config cell
ORIGINAL_OUTPUT_CSV = "/content/drive/MyDrive/path/to/your/file.csv"
```

## 📥 What You'll Download

The zip file contains:

### For GReaT/CTGAN/TVAE
```
pyhealth_output/
├── great/  (or ctgan/ or tvae/)
│   ├── {model}_synthetic_flattened_ehr.csv  ← Main output
│   └── model files (*.pkl, *.pt, config.json)
├── synthetic_data_visualization.png
└── comparison_visualization.png
```

### For Transformer Baseline
```
pyhealth_transformer_output/
├── transformer_baseline_synthetic_ehr.csv   ← Main output
├── transformer_baseline_model_final/        ← Model checkpoint
├── checkpoints/                             ← Training checkpoints
├── synthetic_visualization.png
└── comparison_visualization.png
```

## 🆘 Troubleshooting Decision Tree

```
Problem?
├─ Training fails
│  ├─ "CUDA out of memory" → Reduce batch size
│  ├─ "RuntimeError" → Check GPU enabled
│  └─ Takes forever → Verify GPU, reduce epochs
│
├─ Generation fails
│  ├─ "Out of memory" → Reduce GEN_BATCH_SIZE
│  ├─ Invalid sequences → Check tokenizer
│  └─ All same output → Increase temperature
│
├─ Comparison fails
│  ├─ File not found → Check ORIGINAL_OUTPUT path
│  ├─ Low correlation → Check hyperparameters match
│  └─ Large differences → Check data splits match
│
└─ Runtime disconnects
   ├─ During training → Save checkpoints to Drive
   ├─ Keep tab active → Use Colab Pro
   └─ Long runtime → Split into multiple runs
```

## 📚 Documentation Links

| Resource | Use For |
|----------|---------|
| `COLAB_GUIDE.md` | Detailed Colab instructions for tabular models |
| `TRANSFORMER_BASELINE_GUIDE.md` | Detailed instructions for transformer |
| `README.md` | General PyHealth synthetic EHR overview |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |
| `compare_outputs.py` | Standalone comparison script |

## 💡 Pro Tips

1. **Start with quick test:** Use 2 epochs first to verify everything works
2. **Save to Drive:** Avoids data loss if runtime disconnects
3. **Monitor progress:** Watch GPU utilization with `!nvidia-smi`
4. **Match hyperparameters:** Use same settings as original for best comparison
5. **Document settings:** Note your configuration for reproducibility

## 🎓 Model Selection Guide

| Model | Speed | Quality | Memory | Use When |
|-------|-------|---------|--------|----------|
| **GReaT** | Slow | High | High | Best correlations needed |
| **CTGAN** | Medium | High | Medium | Balanced approach |
| **TVAE** | Fast | Good | Low | Quick experiments |
| **Transformer** | Slow | High | High | Sequential patterns important |

## ✨ Success Indicators

You know it's working when you see:

1. ✅ All cells run without errors
2. ✅ Training loss decreases over time
3. ✅ Synthetic data has realistic properties
4. ✅ Comparison shows high correlation (if applicable)
5. ✅ Validation checks pass
6. ✅ Download completes successfully

## 🎉 Final Checklist

Before finishing:

- [ ] Downloaded results zip file
- [ ] Checked validation results
- [ ] Saved any important settings/notes
- [ ] (Optional) Backed up to Drive for safekeeping
- [ ] (Optional) Shared with team if collaborative

---

**Need Help?** Check the detailed guides:
- Tabular models → `COLAB_GUIDE.md`
- Transformer → `TRANSFORMER_BASELINE_GUIDE.md`
- Issues → https://github.com/sunlabuiuc/PyHealth/issues
