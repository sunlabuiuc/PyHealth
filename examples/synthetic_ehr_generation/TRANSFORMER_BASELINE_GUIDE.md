# Transformer Baseline Comparison Guide

This guide explains how to run the PyHealth version of the `transformer_baseline` mode and compare it with your original results in Google Colab.

## What is Transformer Baseline?

The `transformer_baseline` mode from the original baselines.py script:
- Converts patient data into **text sequences** (not tabular)
- Trains a **GPT-2 style decoder** model
- Generates synthetic sequences autoregressively
- Converts back to tabular format

This is different from GReaT/CTGAN/TVAE which work on flattened tabular data.

## Quick Start in Google Colab

### Prerequisites

You should already have:
1. ✅ Original transformer_baseline results in Google Drive
2. ✅ MIMIC-III data files (ADMISSIONS.csv, PATIENTS.csv, DIAGNOSES_ICD.csv)
3. ✅ Train/test patient ID files

### Your Original Output Structure

```
MyDrive/
└── original_output/
    └── transformer_baseline/
        └── transformer_baseline_synthetic_ehr.csv  ← Your original results
```

### Step-by-Step Process

#### 1. Upload Notebook to Colab

- Go to https://colab.research.google.com/
- Click **File > Upload notebook**
- Upload `PyHealth_Transformer_Baseline_Colab.ipynb`

#### 2. Select GPU Runtime

⚠️ **CRITICAL:** Transformer training requires GPU!

- Click **Runtime > Change runtime type**
- Select **GPU** (or **A100** if available)
- Click **Save**

#### 3. Configure Paths

In the "Step 3: Configure Paths" cell, update:

```python
# Your MIMIC data
MIMIC_DATA_PATH = "/content/drive/MyDrive/mimic3_data/"
TRAIN_PATIENTS_PATH = "/content/drive/MyDrive/mimic3_data/train_patient_ids.txt"
TEST_PATIENTS_PATH = "/content/drive/MyDrive/mimic3_data/test_patient_ids.txt"

# YOUR ORIGINAL OUTPUT (important!)
ORIGINAL_OUTPUT_CSV = "/content/drive/MyDrive/original_output/transformer_baseline/transformer_baseline_synthetic_ehr.csv"

# Training settings (match your original if possible)
NUM_EPOCHS = 50  # Same as original
TRAIN_BATCH_SIZE = 64  # Same as original
NUM_SYNTHETIC_SAMPLES = 10000  # Same as original
```

#### 4. Run All Cells

- Click **Runtime > Run all**
- Authorize Google Drive when prompted
- Wait for completion (~2-3 hours for 50 epochs)

## Expected Timeline

With GPU (T4 or A100) and 50 epochs:

| Step | Duration | What's Happening |
|------|----------|------------------|
| Setup | ~5 min | Installing packages, cloning PyHealth |
| Data Processing | ~10 min | Loading and processing MIMIC data |
| Tokenizer Building | ~2 min | Creating vocabulary from medical codes |
| Training | ~90-120 min | Training transformer (50 epochs × ~2 min/epoch) |
| Generation | ~10-15 min | Generating 10,000 synthetic patients |
| Comparison | ~2 min | Statistical analysis and visualization |
| **Total** | **~2-3 hours** | Full pipeline |

💡 **Tip:** For quick testing, use `NUM_EPOCHS = 2` (takes ~15 minutes total)

## What the Notebook Does

### Automatic Pipeline

The notebook runs these steps automatically:

1. **✓ Mounts Google Drive** - Access your data and original results
2. **✓ Installs dependencies** - transformers, tokenizers, PyHealth
3. **✓ Processes MIMIC data** - Converts to sequential format
4. **✓ Builds tokenizer** - Word-level tokenizer for medical codes
5. **✓ Trains GPT-2 model** - Same architecture as original
6. **✓ Generates synthetic data** - 10,000 samples in batches
7. **✓ Compares with original** - Statistical tests and visualizations
8. **✓ Downloads results** - Zip file with all outputs

### Key Differences from Original

The PyHealth version:
- ✅ Uses PyHealth utility functions (`synthetic_ehr_utils`)
- ✅ Same model architecture (GPT-2)
- ✅ Same training procedure (HuggingFace Trainer)
- ✅ Same generation method (autoregressive sampling)
- ✅ Produces statistically similar outputs

## Understanding the Comparison

### What Gets Compared

The notebook compares:

#### 1. Basic Statistics
```
Metric                  Original    PyHealth
Total patients          10000       10000
Total visits            27543       27812
Total codes             145234      146891
Unique codes            4523        4487
Avg codes/patient       14.52       14.69
Avg visits/patient      2.75        2.78
Avg codes/visit         5.27        5.28
```

#### 2. Distribution Tests
- **Kolmogorov-Smirnov test** - Compares code distributions
- **Pearson correlation** - Measures code frequency similarity
- **Visual comparisons** - Histograms and scatter plots

#### 3. Validation Checks
- ✓ Similar number of patients (within 5%)
- ✓ Similar total codes (within 20%)
- ✓ Similar codes per patient (within 20%)
- ✓ High code frequency correlation (>0.7)

### Expected Results

#### ✅ If All Checks Pass:

```
VALIDATION CHECKS
==================
  ✓ PASS - Similar number of patients (within 5%)
  ✓ PASS - Similar total codes (within 20%)
  ✓ PASS - Similar codes per patient (within 20%)
  ✓ PASS - High code frequency correlation (>0.7)

Result: 4/4 checks passed

🎉 All checks passed! PyHealth implementation matches original.
```

**Interpretation:** The PyHealth implementation is working correctly and produces statistically equivalent outputs to the original baselines.py.

#### ⚠️ If Some Checks Fail:

**Common reasons (usually OK):**
- Different random seeds → Different specific samples (expected)
- Different training convergence → Slightly different distributions (OK)
- Fewer training epochs → Lower quality (use more epochs)

**When to worry:**
- Correlation < 0.5 → Major implementation difference
- >30% difference in any metric → Something is wrong

### Visualizations

The notebook creates two sets of plots:

#### 1. Synthetic Data Visualization
- Distribution of codes per patient
- Distribution of visits per patient
- Top 20 most frequent codes
- Distribution of codes per visit

#### 2. Comparison Visualization
- Side-by-side histograms (codes per patient)
- Side-by-side histograms (visits per patient)
- Scatter plot (code frequency correlation)
- Bar chart (top codes comparison)

## Output Files

After running, you'll have:

```
pyhealth_transformer_output/
├── transformer_baseline_synthetic_ehr.csv  ← Main synthetic data
├── transformer_baseline_model_final/       ← Trained model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_args.bin
├── checkpoints/                            ← Training checkpoints
│   └── checkpoint-XXXX/
├── synthetic_visualization.png             ← Data plots
└── comparison_visualization.png            ← Comparison plots
```

Download the zip file at the end to get everything.

## Troubleshooting

### Issue: Training is Slow

**Symptom:** Each epoch takes >5 minutes

**Solutions:**
1. Verify GPU is enabled: Run `!nvidia-smi` cell
2. Check batch size: Increase to 128 or 256
3. Reduce sequence length: Set `MAX_SEQ_LENGTH = 256`
4. Use A100 GPU (Colab Pro)

### Issue: Out of Memory

**Symptom:** "CUDA out of memory" error

**Solutions:**
1. Reduce `TRAIN_BATCH_SIZE` to 32 or 16
2. Reduce `MAX_SEQ_LENGTH` to 256
3. Reduce `GEN_BATCH_SIZE` to 256
4. Restart runtime and clear memory

### Issue: Generation is Slow

**Symptom:** Generation takes >30 minutes

**Solutions:**
1. This is normal for 10,000 samples
2. Reduce `NUM_SYNTHETIC_SAMPLES` for testing
3. Increase `GEN_BATCH_SIZE` if memory allows
4. Use A100 GPU for faster generation

### Issue: Comparison Shows Large Differences

**Symptom:** Validation checks fail, low correlation

**Possible causes:**
1. **Different number of epochs** - Original used 50, you used 2
   - Solution: Match the epoch count
2. **Different hyperparameters** - Check your original script settings
   - Solution: Match `EMBEDDING_DIM`, `NUM_LAYERS`, `NUM_HEADS`
3. **Different data split** - Train/test split doesn't match
   - Solution: Use exact same patient ID files
4. **Model not converged** - Training stopped too early
   - Solution: Train for more epochs

### Issue: Original CSV Not Found

**Symptom:** "Skipping comparison" message

**Solutions:**
1. Check path: Verify `ORIGINAL_OUTPUT_CSV` is correct
2. Check Drive mount: Ensure Drive is mounted properly
3. Check filename: Must be exactly `transformer_baseline_synthetic_ehr.csv`
4. Upload manually if needed

### Issue: Runtime Disconnected

**Symptom:** "Runtime disconnected" during training

**Solutions:**
1. **Save to Drive:** Set `PYHEALTH_OUTPUT` to a Drive path
2. **Use Colab Pro:** Longer runtime limits
3. **Keep tab active:** Don't close browser
4. **Resume from checkpoint:** Load last checkpoint if available

## Advanced: Matching Original Exactly

To get the closest match to your original results:

### 1. Match Hyperparameters

Check your original script and match:
```python
NUM_EPOCHS = 50  # Match original
TRAIN_BATCH_SIZE = 64  # Match original
EMBEDDING_DIM = 512  # Match original
NUM_LAYERS = 8  # Match original
NUM_HEADS = 8  # Match original
MAX_SEQ_LENGTH = 512  # Match original
```

### 2. Match Training Settings

In the training arguments cell, ensure:
```python
learning_rate=1e-4,  # Match original
lr_scheduler_type="cosine",  # Match original
```

### 3. Use Same Data Split

Use the **exact same** train_patient_ids.txt and test_patient_ids.txt files you used for the original run.

### 4. Match Generation Settings

In the generation cell:
```python
max_length=max_len_train,  # Same as training max
do_sample=True,
top_k=50,  # Match original
top_p=0.95,  # Match original
```

## Interpreting Results

### Good Results ✅

If you see:
- All validation checks pass
- Correlation > 0.8
- Visual distributions overlap closely
- Similar top codes

**→ PyHealth implementation is correct!**

### Acceptable Results ⚠️

If you see:
- 3/4 validation checks pass
- Correlation between 0.7-0.8
- Visual distributions similar but not identical
- Most top codes match

**→ Expected due to randomness in training/generation**

### Poor Results ❌

If you see:
- <2 validation checks pass
- Correlation < 0.6
- Very different distributions
- Completely different top codes

**→ Check hyperparameters and data splits**

## Key Metrics to Watch

### During Training

Monitor these in the training logs:
- **Loss should decrease** - From ~8-10 to ~2-3
- **No NaN losses** - Indicates training instability
- **Consistent progress** - Each epoch should improve

### During Generation

Watch for:
- **Valid sequences** - Not all padding or special tokens
- **Reasonable length** - Not all max length or all too short
- **Known codes** - Mostly codes from training vocabulary

### In Comparison

Focus on:
1. **Code frequency correlation** - Most important (>0.7 is good)
2. **Similar averages** - Codes/patient should be close
3. **Distribution shape** - Histograms should look similar
4. **Top codes overlap** - Top 20 should be mostly the same

## FAQ

**Q: Why does training take so long?**
A: 50 epochs × 2 min/epoch = ~100 minutes. This is normal for transformers. Use fewer epochs for testing.

**Q: Why are results not exactly the same?**
A: Generative models are stochastic. Different runs produce different samples, but statistics should be similar.

**Q: Can I use CPU instead of GPU?**
A: Not recommended. CPU training would take 10-20x longer (20+ hours).

**Q: How do I know if my comparison is successful?**
A: If 3+ validation checks pass and correlation > 0.7, you're good!

**Q: What if I don't have the original results?**
A: That's fine! The notebook will skip comparison and just show your PyHealth results.

**Q: Can I use MIMIC-IV instead of MIMIC-III?**
A: Yes! Just update the paths and use MIMIC-IV file structure.

## Next Steps

After successful comparison:

1. **✓ Use for research** - PyHealth version is production-ready
2. **Experiment** - Try different hyperparameters
3. **Evaluate quality** - Test on downstream tasks
4. **Scale up** - Generate larger synthetic cohorts
5. **Integrate** - Use in your PyHealth pipelines

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review the notebook's error messages
3. Compare with the working original
4. Open an issue: https://github.com/sunlabuiuc/PyHealth/issues

## Summary

The transformer_baseline mode is special because it's **sequential** (not tabular). The PyHealth notebook:

✅ Uses the same model architecture (GPT-2)
✅ Uses the same training procedure
✅ Uses the same generation method
✅ Produces statistically similar outputs
✅ Provides comprehensive comparison tools

If validation checks pass, your PyHealth implementation is working correctly! 🎉
