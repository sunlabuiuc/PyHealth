# Running PyHealth Synthetic EHR Generation in Google Colab

This guide explains how to run the PyHealth synthetic EHR generation code in Google Colab and compare it with the original baselines.py outputs.

## Quick Start (5 steps)

### 1. Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Upload notebook**
3. Upload `PyHealth_Synthetic_EHR_Colab.ipynb`

**Option B: From GitHub** (once merged)
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Open notebook > GitHub**
3. Enter: `sunlabuiuc/PyHealth`
4. Navigate to `examples/synthetic_ehr_generation/PyHealth_Synthetic_EHR_Colab.ipynb`

### 2. Select GPU Runtime

**IMPORTANT:** You need a GPU for reasonable training times.

1. Click **Runtime > Change runtime type**
2. Select **Hardware accelerator: GPU** (or **A100** if available)
3. Click **Save**

### 3. Prepare Your Data

You have two options for data access:

**Option A: Use Google Drive** (Recommended)
1. Upload your MIMIC data to Google Drive:
   ```
   MyDrive/
   └── mimic3_data/
       ├── ADMISSIONS.csv
       ├── PATIENTS.csv
       ├── DIAGNOSES_ICD.csv
       ├── train_patient_ids.txt
       └── test_patient_ids.txt
   ```

2. The notebook will mount your Drive automatically

**Option B: Direct Upload to Colab**
1. Run the upload cell in the notebook
2. Select and upload your files
3. Files will be at `/content/filename.csv`

⚠️ **Note:** Direct uploads are lost when runtime disconnects!

### 4. Configure Paths

In the notebook's "Step 3: Configure Paths" cell, update:

```python
# Update these paths
MIMIC_DATA_PATH = "/content/drive/MyDrive/mimic3_data/"
TRAIN_PATIENTS_PATH = "/content/drive/MyDrive/mimic3_data/train_patient_ids.txt"
TEST_PATIENTS_PATH = "/content/drive/MyDrive/mimic3_data/test_patient_ids.txt"

# If comparing with original outputs
ORIGINAL_OUTPUT = "/content/drive/MyDrive/original_output"

# Choose your model
MODEL_MODE = "great"  # Options: "great", "ctgan", "tvae"
```

### 5. Run All Cells

1. Click **Runtime > Run all**
2. Or run cells one-by-one with **Shift+Enter**
3. Grant permissions when prompted (for Drive access)

**Expected Runtime:**
- Setup: ~5 minutes
- Data processing: ~5-10 minutes
- Training (2 epochs): ~15-30 minutes
- Generation: ~5-10 minutes
- **Total: ~40-60 minutes**

## Detailed Workflow

### Step-by-Step Execution

#### Cell 1: Check GPU
```python
!nvidia-smi
```
**Expected Output:** GPU information (e.g., "Tesla T4", "A100")

#### Cell 2: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Action Required:** Click the authorization link and grant access

#### Cell 3-4: Install Dependencies
```python
!pip install -q polars pandas numpy scipy scikit-learn
!pip install -q be-great sdv
```
**Duration:** ~3-5 minutes

#### Cell 5-6: Clone PyHealth
```python
!git clone https://github.com/sunlabuiuc/PyHealth.git
```
**Duration:** ~1 minute

#### Cell 7: Configure Paths
**ACTION REQUIRED:** Update paths to match your setup!

#### Cell 8: Verify Files
**Expected Output:** All files should show ✓

#### Cell 9: Process MIMIC Data
**Duration:** ~5-10 minutes depending on data size
**Output:**
```
Admissions shape: (58976, 19)
Patients shape: (46520, 8)
Diagnoses shape: (651047, 5)
...
Train EHR shape: (X, 3)
Train flattened shape: (Y, Z)
```

#### Cell 10-12: Train Model
Choose one based on `MODEL_MODE`:
- Cell 10: GReaT model
- Cell 11: CTGAN model
- Cell 12: TVAE model

**Duration:** ~15-30 minutes
**Progress:** You'll see training progress bars

#### Cell 13-14: Inspect Results
**Outputs:**
- Synthetic data summary
- Visualization plots

#### Cell 15-16: Compare (Optional)
Only runs if you have original baseline outputs
**Outputs:**
- Statistical comparison table
- Correlation plots
- Validation check results

#### Cell 17: Download Results
Downloads a zip file with all outputs

## File Structure After Running

```
pyhealth_output/
├── great/  (or ctgan/ or tvae/)
│   ├── great_synthetic_flattened_ehr.csv
│   ├── model.pt
│   └── config.json
├── synthetic_data_visualization.png
└── comparison_visualization.png (if compared)
```

## Comparing with Original Baselines

### Prerequisites

1. You must have already run the original `baselines.py` script
2. Original outputs should be in Google Drive:
   ```
   MyDrive/
   └── original_output/
       └── great/
           └── great_synthetic_flattened_ehr.csv
   ```

### Comparison Process

The notebook automatically compares if it finds the original outputs. It will show:

1. **Statistical Comparison Table:**
   ```
   Metric          Original    PyHealth    Difference
   Mean            2.3456      2.3512      0.0056
   Std             1.2345      1.2398      0.0053
   Sparsity        87.23%      87.45%      0.22%
   ```

2. **Validation Checks:**
   - ✓ Similar dimensions (within 1%)
   - ✓ Similar sparsity (within 10%)
   - ✓ Similar mean (within 20%)

3. **Visualizations:**
   - Distribution comparison plots
   - Code frequency correlation scatter plot

### Expected Results

**✓ All checks should PASS** - This indicates:
- PyHealth processes data the same way
- Models produce statistically similar outputs
- Implementation is correct

**Some checks FAIL** - Possible reasons:
- Different random seeds (expected)
- Different number of training epochs
- Model not fully converged
- This is usually OK for generative models!

## Troubleshooting

### Issue: Runtime Disconnected

**Symptoms:**
- "Runtime disconnected" message
- Need to restart from beginning

**Solutions:**
1. Save outputs to Google Drive (not `/content/`)
2. Use Runtime > Manage sessions to monitor
3. Keep browser tab active
4. Consider Colab Pro for longer runtimes

### Issue: Out of Memory

**Symptoms:**
- "Cuda out of memory" error
- Training crashes

**Solutions:**
1. Reduce `BATCH_SIZE` (try 256 or 128)
2. Reduce `NUM_SYNTHETIC_SAMPLES`
3. Use smaller subset of data for testing
4. Upgrade to Colab Pro with more RAM

### Issue: Slow Training

**Symptoms:**
- Training takes >1 hour
- Progress is very slow

**Solutions:**
1. Verify GPU is being used: check `nvidia-smi` output
2. Reduce `NUM_EPOCHS` for faster testing
3. Reduce data size
4. Try different model (TVAE is usually faster than GReaT)

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'pyhealth'
```

**Solutions:**
1. Restart runtime and run all cells from top
2. Make sure clone cell completed successfully
3. Check that `sys.path.insert()` cell ran

### Issue: Files Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**
1. Verify Google Drive is mounted: run `!ls /content/drive/MyDrive/`
2. Check paths in config cell match your folder structure
3. Ensure files were uploaded completely

### Issue: Comparison Doesn't Run

**Symptoms:**
- Comparison cells show "Skipping comparison..."

**Solutions:**
1. Verify `ORIGINAL_OUTPUT` path is correct
2. Ensure original CSV exists at specified location
3. Check file naming matches exactly

## Tips for Best Results

### Training Quality
- **More epochs = better quality** (but slower)
  - Quick test: 2 epochs (~15 min)
  - Good quality: 10-20 epochs (~1-2 hours)
  - Best quality: 50+ epochs (~4-6 hours)

### Model Selection
- **GReaT**: Best for preserving correlations, slowest
- **CTGAN**: Good balance of speed and quality
- **TVAE**: Fastest, decent quality

### Data Size
- Start small for testing (1000 patients)
- Scale up once working (10000+ patients)

### Monitoring
- Watch GPU utilization: `!watch -n 1 nvidia-smi`
- Monitor training loss (should decrease)
- Check generated samples periodically

## Advanced Usage

### Using A100 GPU (Colab Pro)

If you have Colab Pro with A100 access:
1. Select **A100 GPU** in runtime settings
2. Increase batch size to 1024 or higher
3. Can handle larger datasets and more epochs

### Saving Checkpoints to Drive

To prevent data loss:
```python
# In config cell, change:
PYHEALTH_OUTPUT = "/content/drive/MyDrive/pyhealth_output"
```

This saves everything directly to Drive (survives disconnections).

### Running Multiple Models

To try all models:
1. Run notebook with `MODEL_MODE = "great"`
2. Download results
3. Change to `MODEL_MODE = "ctgan"`
4. Run again
5. Change to `MODEL_MODE = "tvae"`
6. Run again
7. Compare all three!

### Batch Processing

To generate multiple datasets:
```python
for num_samples in [1000, 5000, 10000]:
    NUM_SYNTHETIC_SAMPLES = num_samples
    # Run generation cell
    # Save with different name
```

## FAQ

**Q: Can I use MIMIC-IV instead of MIMIC-III?**
A: Yes! The code works with both. Just use the appropriate file structure.

**Q: How long does training take?**
A: With 2 epochs on GPU: 15-30 minutes. With 50 epochs: 4-6 hours.

**Q: Why are outputs different from original?**
A: Generative models are stochastic. Different runs produce different samples, but statistics should be similar.

**Q: Can I use free Colab?**
A: Yes! But you may hit runtime limits for long training. Colab Pro recommended for >20 epochs.

**Q: How much GPU memory do I need?**
A: 15GB is sufficient (T4 works). A100 is better for large datasets.

**Q: Can I pause and resume training?**
A: Yes, but you need to save model checkpoints to Drive first. The notebook saves models automatically.

## Next Steps

After successfully running the notebook:

1. **Evaluate Quality**
   - Run the comparison script
   - Check validation metrics
   - Visually inspect samples

2. **Experiment**
   - Try different models
   - Adjust hyperparameters
   - Test different epoch counts

3. **Use Synthetic Data**
   - Train downstream models
   - Test privacy metrics
   - Validate clinical feasibility

4. **Scale Up**
   - Use full dataset
   - Train for more epochs
   - Generate larger synthetic cohorts

## Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Review the notebook's error messages
3. Check PyHealth documentation: https://pyhealth.readthedocs.io/
4. Open an issue: https://github.com/sunlabuiuc/PyHealth/issues

## Citation

If you use this code, please cite:

```bibtex
@software{pyhealth2024,
  title={PyHealth: A Python Library for Health Predictive Models},
  author={PyHealth Contributors},
  year={2024},
  url={https://github.com/sunlabuiuc/PyHealth}
}
```
